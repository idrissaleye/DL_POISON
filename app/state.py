"""État global de l'application : dataset, modèles, historique, job courant.

Toutes les données volumineuses vivent en mémoire (numpy / torch) et sont
persistées sur disque dans `data/` pour survivre aux redémarrages :
  - data/dataset.npz          : le dataset généré
  - data/dataset.meta.json    : ses métadonnées
  - data/models/<nom>.pt      : les poids des modèles
  - data/models/registry.json : registre des modèles + métriques
  - data/history.json         : historique des expériences
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .ml.dataset import CLASS_NAMES, load_dataset
from .ml.model import ShapesCNN, count_parameters

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppState:
    """État partagé du serveur. Un seul job d'arrière-plan à la fois."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = CLASS_NAMES
        self.dataset: dict | None = None       # {X_train, y_train, X_test, y_test, meta}
        self.models: dict[str, dict] = {}      # nom -> {"model", "meta", "path"}
        self.active_model: str | None = None
        self.poisoned: dict | None = None      # {"type", "ratio", "indices", "target"}
        self.history: list[dict] = []
        self.job: dict | None = None
        self.lock = threading.Lock()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_all()

    # ------------------------------------------------------------------ #
    # Persistance                                                         #
    # ------------------------------------------------------------------ #

    def _load_all(self) -> None:
        ds_path = DATA_DIR / "dataset.npz"
        if ds_path.exists():
            try:
                X_train, y_train, X_test, y_test, meta = load_dataset(ds_path)
                self.dataset = {
                    "X_train": X_train, "y_train": y_train,
                    "X_test": X_test, "y_test": y_test, "meta": meta,
                }
            except Exception as exc:  # fichier corrompu -> on repart de zéro
                print(f"[state] dataset illisible ({exc}), ignoré")

        reg_path = MODELS_DIR / "registry.json"
        if reg_path.exists():
            registry = json.loads(reg_path.read_text(encoding="utf-8"))
            for name, meta in registry.items():
                path = MODELS_DIR / meta.get("file", "")
                if path.exists():
                    try:
                        model = ShapesCNN(n_classes=len(self.class_names))
                        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                        self.models[name] = {"model": model, "meta": meta, "path": path}
                    except Exception as exc:
                        print(f"[state] modèle {name} illisible ({exc}), ignoré")
            self.active_model = next(
                (n for n, m in self.models.items() if m["meta"].get("active")), None
            ) or (next(iter(self.models)) if self.models else None)
            for m in self.models.values():
                m["meta"]["active"] = m["meta"].get("name") == self.active_model

        hist_path = DATA_DIR / "history.json"
        if hist_path.exists():
            try:
                self.history = json.loads(hist_path.read_text(encoding="utf-8"))
            except Exception:
                self.history = []

    def persist_registry(self) -> None:
        registry = {name: entry["meta"] for name, entry in self.models.items()}
        (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def persist_history(self) -> None:
        (DATA_DIR / "history.json").write_text(json.dumps(self.history[-100:], indent=2), encoding="utf-8")

    def save_model(self, name: str, meta: dict) -> None:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
        path = MODELS_DIR / f"{safe}.pt"
        torch.save(self.models[name]["model"].state_dict(), path)
        meta["file"] = path.name
        self.models[name]["path"] = path
        self.persist_registry()

    # ------------------------------------------------------------------ #
    # Modèles                                                             #
    # ------------------------------------------------------------------ #

    def get_active(self) -> dict:
        if not self.active_model or self.active_model not in self.models:
            raise RuntimeError("Aucun modèle actif — entraînez d'abord un modèle.")
        return self.models[self.active_model]

    def set_active(self, name: str) -> None:
        self.active_model = name
        for n, m in self.models.items():
            m["meta"]["active"] = n == name
        self.persist_registry()

    def new_model(self, seed: int = 42) -> torch.nn.Module:
        torch.manual_seed(seed)
        return ShapesCNN(n_classes=len(self.class_names))

    # ------------------------------------------------------------------ #
    # Jobs d'arrière-plan                                                 #
    # ------------------------------------------------------------------ #

    def start_job(self, name: str) -> dict:
        """Démarre un job ; refuse si un autre est en cours (HTTP 409 côté API)."""
        with self.lock:
            if self.job and self.job["status"] == "running":
                raise RuntimeError("Un traitement est déjà en cours.")
            self.job = {
                "name": name,
                "progress": 0.0,
                "message": "Initialisation…",
                "status": "running",
                "error": None,
                "result": None,
                "started_at": now_iso(),
            }
            return self.job

    def job_progress(self, frac: float, message: str) -> None:
        if self.job:
            self.job["progress"] = min(1.0, float(frac))
            self.job["message"] = message

    def job_done(self, result: dict) -> None:
        if self.job:
            self.job["progress"] = 1.0
            self.job["status"] = "done"
            self.job["message"] = "Terminé"
            self.job["result"] = result

    def job_error(self, exc: Exception) -> None:
        if self.job:
            self.job["status"] = "error"
            self.job["error"] = str(exc)
            self.job["message"] = f"Erreur : {exc}"

    def add_history(self, entry: dict) -> None:
        entry = {"at": now_iso(), **entry}
        self.history.append(entry)
        self.persist_history()


STATE = AppState()
