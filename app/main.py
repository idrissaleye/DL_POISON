"""AdversarialLab — API FastAPI de simulation d'attaques adversariales.

Lancement :  uvicorn app.main:app --host 0.0.0.0 --port 8000
             (ou `python -m app.main`, ou ./run.sh sous WSL)
Interface  :  http://localhost:8000
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ml import attacks, defense
from .ml.dataset import (
    CLASS_NAMES,
    generate_shapes,
    save_dataset,
    stratified_split,
    to_png_uri,
    to_uri_heat,
)
from .ml.engine import device_label, evaluate_model, predict, train_model
from .ml.model import count_parameters
from .ml.poison import (
    backdoor_success_rate,
    poison_backdoor,
    poison_label_flip,
)
from .state import DATA_DIR, STATE, now_iso

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AdversarialLab", description="Simulation d'attaques adversariales et de défenses")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def require_dataset() -> dict:
    if STATE.dataset is None:
        raise HTTPException(400, "Aucun dataset — générez d'abord le dataset (onglet Données & Modèle).")
    return STATE.dataset


def require_job_free() -> None:
    if STATE.job and STATE.job["status"] == "running":
        raise HTTPException(409, "Un traitement est déjà en cours — réessayez dans un instant.")


def run_job(name: str, fn):
    """Lance `fn` dans un thread d'arrière-plan avec suivi de progression."""
    require_job_free()
    STATE.start_job(name)

    def wrapper():
        try:
            result = fn()
            STATE.job_done(result)
        except Exception as exc:  # noqa: BLE001 — on remonte l'erreur au front
            import traceback

            traceback.print_exc()
            STATE.job_error(str(exc))

    threading.Thread(target=wrapper, daemon=True).start()
    return STATE.job


def unique_name(base: str) -> str:
    name, i = base, 2
    while name in STATE.models:
        name = f"{base} ({i})"
        i += 1
    return name


def subset_indices(n: int, subset: Optional[int]) -> np.ndarray:
    if not subset or subset >= n:
        return np.arange(n)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n, size=int(subset), replace=False))


def model_brief(name: str) -> dict:
    meta = STATE.models[name]["meta"]
    return {
        "name": name,
        "kind": meta.get("kind"),
        "params": meta.get("params"),
        "n_parameters": meta.get("n_parameters"),
        "created_at": meta.get("created_at"),
        "metrics": meta.get("metrics"),
        "active": name == STATE.active_model,
    }


# --------------------------------------------------------------------------- #
# Schémas Pydantic                                                             #
# --------------------------------------------------------------------------- #


class DatasetParams(BaseModel):
    n_per_class: int = Field(2000, ge=200, le=5000)
    img_size: int = Field(64, ge=32, le=96)
    seed: int = 42


class TrainParams(BaseModel):
    epochs: int = Field(8, ge=1, le=40)
    lr: float = Field(1e-3, gt=0)
    batch_size: int = Field(128, ge=16, le=512)
    seed: int = 42


class EvasionParams(BaseModel):
    model: Optional[str] = None
    method: str = "pgd"           # fgsm | pgd | noise
    eps: float = Field(0.15, gt=0, le=1.0)
    iters: int = Field(7, ge=1, le=50)
    alpha: Optional[float] = None
    subset: Optional[int] = None


class CurveParams(BaseModel):
    model: Optional[str] = None
    methods: list[str] = ["fgsm", "pgd", "noise"]
    eps_max: float = 0.45
    n_points: int = Field(8, ge=3, le=12)
    subset: Optional[int] = None


class PoisonParams(BaseModel):
    type: str = "backdoor"        # flip | backdoor
    ratio: float = Field(0.15, gt=0, le=0.5)
    target_class: Optional[int] = None
    epochs: int = Field(6, ge=1, le=40)
    lr: float = Field(1e-3, gt=0)
    seed: int = 42


class AdvTrainParams(BaseModel):
    method: str = "fgsm"
    eps: float = Field(0.15, gt=0, le=1.0)
    iters: int = Field(4, ge=1, le=20)
    epochs: int = Field(8, ge=1, le=40)
    lr: float = Field(1e-3, gt=0)
    seed: int = 42


class PreprocParams(BaseModel):
    model: Optional[str] = None
    attack: str = "pgd"
    eps: float = Field(0.15, gt=0, le=1.0)
    iters: int = Field(7, ge=1, le=50)
    preprocess: str = "jpeg"
    subset: Optional[int] = None


class FilterParams(BaseModel):
    z_threshold: float = Field(1.5, ge=1.0, le=5.0)
    epochs: int = Field(6, ge=1, le=40)
    lr: float = Field(1e-3, gt=0)
    seed: int = 42


class ActivateParams(BaseModel):
    name: str


# --------------------------------------------------------------------------- #
# Vue d'ensemble                                                               #
# --------------------------------------------------------------------------- #


@app.get("/api/overview")
def overview() -> dict:
    ds = STATE.dataset
    return {
        "device": device_label(STATE.device),
        "cuda_available": STATE.device.type == "cuda",
        "class_names": STATE.class_names,
        "dataset": None
        if not ds
        else {
            "n_train": int(len(ds["X_train"])),
            "n_test": int(len(ds["X_test"])),
            "img_size": int(ds["X_train"].shape[1]),
            "meta": ds["meta"],
            "poisoned": STATE.poisoned,
        },
        "models": [model_brief(n) for n in STATE.models],
        "active_model": STATE.active_model,
        "job": STATE.job,
        "n_history": len(STATE.history),
    }


@app.get("/api/job")
def job_status() -> dict:
    return {"job": STATE.job}


# --------------------------------------------------------------------------- #
# Phase 1 — Données & modèle                                                   #
# --------------------------------------------------------------------------- #


@app.post("/api/dataset/generate")
def api_generate_dataset(params: DatasetParams) -> dict:
    require_job_free()

    def work() -> dict:
        cb = STATE.job_progress
        cb(0.05, f"Génération de {params.n_per_class} images par classe…")
        X, y = generate_shapes(params.n_per_class, params.img_size, params.seed)
        cb(0.7, "Découpage train/test stratifié…")
        train_idx, test_idx = stratified_split(y, test_ratio=0.2, seed=params.seed)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        meta = {
            "n_per_class": params.n_per_class,
            "img_size": params.img_size,
            "seed": params.seed,
            "class_names": CLASS_NAMES,
        }
        save_dataset(DATA_DIR / "dataset.npz", X_train, y_train, X_test, y_test, meta)
        STATE.dataset = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "meta": meta}
        # Le dataset a changé : les anciens modèles ne sont plus comparables.
        STATE.models.clear()
        STATE.active_model = None
        STATE.poisoned = None
        STATE.history.clear()
        STATE.persist_registry()
        STATE.persist_history()
        (DATA_DIR / "models").mkdir(exist_ok=True)
        for f in (DATA_DIR / "models").glob("*.pt"):
            f.unlink()

        cb(0.9, "Prévisualisation…")
        samples = []
        for cls in range(len(CLASS_NAMES)):
            idx = np.flatnonzero(y == cls)[:2]
            for i in idx:
                samples.append({"image": to_png_uri(X[i]), "label": CLASS_NAMES[cls]})

        return {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "img_size": params.img_size,
            "size_mb": round(X.nbytes / 1e6, 1),
            "samples": samples,
        }

    return run_job("Génération du dataset", work)


@app.post("/api/model/train")
def api_train_model(params: TrainParams) -> dict:
    ds = require_dataset()
    require_job_free()

    def work() -> dict:
        model = STATE.new_model(params.seed)
        history = train_model(
            model, ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"],
            epochs=params.epochs, lr=params.lr, batch_size=params.batch_size,
            seed=params.seed, progress_cb=STATE.job_progress,
        )
        STATE.job_progress(0.95, "Évaluation finale…")
        metrics = evaluate_model(model, ds["X_test"], ds["y_test"])
        name = unique_name(f"propre-e{params.epochs}")
        STATE.models[name] = {
            "model": model,
            "meta": {
                "kind": "propre", "name": name, "created_at": now_iso(),
                "params": params.model_dump(), "n_parameters": count_parameters(model),
                "metrics": {k: metrics[k] for k in ("accuracy", "loss")},
                "active": True,
            },
        }
        STATE.set_active(name)
        STATE.save_model(name, STATE.models[name]["meta"])

        # Échantillons de prédictions pour l'onglet évaluation
        preds = np.array(metrics["predictions"])
        errors = np.flatnonzero(preds != ds["y_test"])
        correct = np.flatnonzero(preds == ds["y_test"])
        picks = list(errors[:4]) + list(correct[:4])
        rng = np.random.default_rng(0)
        picks = list(np.sort(rng.choice(picks, size=min(8, len(picks)), replace=False))) if len(picks) > 8 else picks
        samples = []
        for i in picks:
            probs = metrics["probs"][i]
            samples.append({
                "image": to_png_uri(ds["X_test"][i]),
                "true": CLASS_NAMES[ds["y_test"][i]],
                "pred": CLASS_NAMES[preds[i]],
                "confidence": round(max(probs), 3),
                "correct": bool(preds[i] == ds["y_test"][i]),
            })

        result = {
            "name": name, "history": history, "metrics": metrics, "samples": samples,
            "n_parameters": count_parameters(model),
        }
        STATE.add_history({
            "type": "train",
            "title": f"Entraînement du modèle « {name} »",
            "summary": {"précision_test": metrics["accuracy"], "perte": metrics["loss"],
                        "époques": params.epochs, "paramètres": count_parameters(model)},
        })
        return result

    return run_job("Entraînement du modèle propre", work)


# --------------------------------------------------------------------------- #
# Phase 2 — Évaluation sur données normales                                    #
# --------------------------------------------------------------------------- #


def _resolve_model(name: Optional[str]) -> str:
    target = name or STATE.active_model
    if not target or target not in STATE.models:
        raise HTTPException(400, "Modèle introuvable — entraînez d'abord un modèle.")
    return target


@app.get("/api/evaluate")
def api_evaluate(model: Optional[str] = Query(None)) -> dict:
    ds = require_dataset()
    target = _resolve_model(model)
    entry = STATE.models[target]
    metrics = evaluate_model(entry["model"], ds["X_test"], ds["y_test"])
    preds = np.array(metrics["predictions"])
    y_test = ds["y_test"]
    errors = np.flatnonzero(preds != y_test)
    correct = np.flatnonzero(preds == y_test)
    rng = np.random.default_rng(1)
    pool = list(errors[:4]) + list(correct[:4])
    picks = np.sort(rng.choice(pool, size=min(8, len(pool)), replace=False)) if len(pool) > 8 else pool
    samples = []
    for i in picks:
        samples.append({
            "image": to_png_uri(ds["X_test"][i]),
            "true": CLASS_NAMES[y_test[i]],
            "pred": CLASS_NAMES[preds[i]],
            "confidence": round(max(metrics["probs"][i]), 3),
            "correct": bool(preds[i] == y_test[i]),
        })
    metrics.pop("predictions", None)
    metrics.pop("probs", None)
    return {"model": model_brief(target), "metrics": metrics, "samples": samples}


# --------------------------------------------------------------------------- #
# Phase 3a — Attaque par évasion                                               #
# --------------------------------------------------------------------------- #


def _evasion_images(entry, ds, method: str, eps: float, iters: int, alpha, n_img: int = 6) -> list[dict]:
    """Exemples visuels : images initialement bien classées, puis dupées si possible."""
    X_test, y_test = ds["X_test"], ds["y_test"]
    preds_clean = predict(entry["model"], X_test)
    correct = np.flatnonzero(preds_clean == y_test)
    picked = subset_indices(len(correct), n_img)
    idx = correct[picked]
    Xs, ys = X_test[idx], y_test[idx]

    x_adv_np, preds_adv = attacks._batched_transform(
        entry["model"], Xs, ys,
        lambda m, xb, yb: attacks.get_attack_fn(method, iters, alpha)(m, xb, yb, eps), 128,
    )
    images = []
    for k in range(len(idx)):
        images.append({
            "true": CLASS_NAMES[ys[k]],
            "pred_clean": CLASS_NAMES[preds_clean[idx[k]]],
            "pred_adv": CLASS_NAMES[preds_adv[k]],
            "fooled": bool(preds_adv[k] != ys[k]),
            "original": to_png_uri(Xs[k]),
            "adversarial": to_png_uri(x_adv_np[k, 0] * 255),
            "perturbation": to_uri_heat(x_adv_np[k, 0] - Xs[k] / 255.0),
        })
    return images


@app.post("/api/attack/evasion")
def api_attack_evasion(params: EvasionParams) -> dict:
    ds = require_dataset()
    require_job_free()
    target = _resolve_model(params.model)
    entry = STATE.models[target]
    if params.method not in attacks.ATTACKS:
        raise HTTPException(400, f"Méthode inconnue : {params.method}")

    idx = subset_indices(len(ds["X_test"]), params.subset or (2000 if STATE.device.type == "cuda" else 600))
    Xs, ys = ds["X_test"][idx], ds["y_test"][idx]
    result = attacks.evaluate_attack(
        entry["model"], Xs, ys, params.method, params.eps,
        iters=params.iters, alpha=params.alpha,
    )
    result["images"] = _evasion_images(entry, ds, params.method, params.eps, params.iters, params.alpha)
    result["model"] = target
    STATE.add_history({
        "type": "evasion",
        "title": f"Évasion {attacks.ATTACKS[params.method][0]} (ε={params.eps}) sur « {target} »",
        "summary": {"précision_propre": result["acc_clean"], "précision_adversariale": result["acc_adv"],
                    "taux_de_succès": result["success_rate"], "n": result["n"]},
    })
    return result


@app.post("/api/attack/curve")
def api_attack_curve(params: CurveParams) -> dict:
    ds = require_dataset()
    target = _resolve_model(params.model)
    entry = STATE.models[target]

    eps_list = [round(e, 3) for e in np.linspace(0.0, params.eps_max, params.n_points)]
    subset = params.subset or (1000 if STATE.device.type == "cuda" else 400)
    idx = subset_indices(len(ds["X_test"]), subset)
    Xs, ys = ds["X_test"][idx], ds["y_test"][idx]

    def work() -> dict:
        series = []
        for m in params.methods:
            if m not in attacks.ATTACKS:
                raise ValueError(f"Méthode inconnue : {m}")
            accs = []
            for j, eps in enumerate(eps_list):
                STATE.job_progress(
                    (params.methods.index(m) * len(eps_list) + j) / (len(params.methods) * len(eps_list)),
                    f"Courbe {attacks.ATTACKS[m][0]} — ε={eps:.2f}",
                )
                if eps == 0.0:
                    from .ml.engine import predict as _predict
                    accs.append(round(float((_predict(entry["model"], Xs) == ys).mean()), 4))
                else:
                    r = attacks.evaluate_attack(entry["model"], Xs, ys, m, float(eps))
                    accs.append(r["acc_adv"])
            series.append({"method": m, "label": attacks.ATTACKS[m][0], "accs": accs})

        STATE.add_history({
            "type": "curve",
            "title": f"Courbes précision/ε sur « {target} »",
            "summary": {"méthodes": ", ".join(attacks.ATTACKS[m][0] for m in params.methods),
                        "ε_max": params.eps_max, "n": subset},
        })
        return {"eps_list": eps_list, "series": series, "n": subset, "model": target}

    return run_job("Calcul des courbes précision/ε", work)


# --------------------------------------------------------------------------- #
# Phase 3b — Attaque par empoisonnement                                        #
# --------------------------------------------------------------------------- #


@app.post("/api/attack/poison")
def api_attack_poison(params: PoisonParams) -> dict:
    ds = require_dataset()
    require_job_free()
    if params.type not in ("flip", "backdoor"):
        raise HTTPException(400, "Type d'empoisonnement inconnu (flip | backdoor).")
    if params.type == "backdoor" and params.target_class is None:
        raise HTTPException(400, "La classe cible est requise pour l'attaque backdoor.")

    def work() -> dict:
        X_train, y_train = ds["X_train"], ds["y_train"]
        ratio_pct = round(params.ratio * 100)
        STATE.job_progress(0.05, "Empoisonnement du jeu d'entraînement…")

        poisoned_idx, X_p, y_p = [], X_train, y_train
        if params.type == "flip":
            poisoned_idx, y_p = poison_label_flip(y_train, params.ratio, params.target_class, params.seed)
        else:
            poisoned_idx, X_p, y_p = poison_backdoor(
                X_train, y_train, params.ratio, params.target_class, params.seed
            )

        kind = "empoisonné-flip" if params.type == "flip" else "empoisonné-backdoor"
        name = unique_name(f"{kind}-{ratio_pct}%")

        STATE.job_progress(0.15, f"Ré-entraînement sur données empoisonnées ({len(poisoned_idx)} exemples altérés)…")
        model = STATE.new_model(params.seed)
        train_model(
            model, X_p, y_p, ds["X_test"], ds["y_test"],
            epochs=params.epochs, lr=params.lr, seed=params.seed, progress_cb=STATE.job_progress,
        )
        STATE.job_progress(0.92, "Évaluation du modèle empoisonné…")
        metrics = evaluate_model(model, ds["X_test"], ds["y_test"])

        bd = None
        if params.type == "backdoor":
            bd = backdoor_success_rate(model, ds["X_test"], ds["y_test"], params.target_class)

        meta = {
            "kind": kind, "name": name, "created_at": now_iso(),
            "params": {**params.model_dump(), "n_poisoned": int(len(poisoned_idx))},
            "n_parameters": count_parameters(model),
            "metrics": {k: metrics[k] for k in ("accuracy", "loss")},
            "active": True,
        }
        STATE.models[name] = {"model": model, "meta": meta}
        STATE.set_active(name)
        STATE.save_model(name, meta)
        STATE.poisoned = {
            "type": params.type,
            "ratio": params.ratio,
            "target": params.target_class,
            "indices": poisoned_idx.tolist(),
            "model": name,
        }

        # Visuels d'échantillons empoisonnés
        picks = poisoned_idx[:6] if len(poisoned_idx) >= 6 else poisoned_idx
        images = []
        for i in picks:
            if params.type == "backdoor":
                img = to_png_uri(X_p[i])
            else:
                img = to_png_uri(X_train[i])
            images.append({
                "image": img,
                "true": CLASS_NAMES[y_train[i]],
                "poisoned_label": CLASS_NAMES[y_p[i]],
                "trigger": params.type == "backdoor",
            })

        STATE.add_history({
            "type": "poison",
            "title": f"Empoisonnement {params.type} {ratio_pct}% → « {name} »",
            "summary": {"précision_test": metrics["accuracy"],
                        **({"succès_backdoor": bd["backdoor_success"]} if bd else {}),
                        "exemples_empoisonnés": int(len(poisoned_idx))},
        })
        return {
            "name": name, "metrics": metrics, "backdoor": bd,
            "n_poisoned": int(len(poisoned_idx)), "images": images, "type": params.type,
        }

    return run_job("Attaque par empoisonnement + ré-entraînement", work)


# --------------------------------------------------------------------------- #
# Phase 4 — Défenses                                                           #
# --------------------------------------------------------------------------- #


@app.post("/api/defense/preprocessing")
def api_defense_preprocessing(params: PreprocParams) -> dict:
    ds = require_dataset()
    require_job_free()
    target = _resolve_model(params.model)
    entry = STATE.models[target]
    if params.preprocess not in defense.PREPROCESSORS:
        raise HTTPException(400, f"Prétraitement inconnu : {params.preprocess}")

    idx = subset_indices(len(ds["X_test"]), params.subset or (1500 if STATE.device.type == "cuda" else 500))
    Xs, ys = ds["X_test"][idx], ds["y_test"][idx]
    res = defense.evaluate_with_preprocessing(
        entry["model"], Xs, ys, params.attack, params.eps,
        iters=params.iters, preprocess=params.preprocess,
    )

    # Visuel : propre / adversarial / adversarial + prétraitement
    attack_fn = attacks.get_attack_fn(params.attack, params.iters, None)
    from .ml.engine import to_tensors, get_device

    # predict() laisse le modèle sur CPU : on prédit d'abord, puis on monte
    # le modèle sur le device pour l'attaque (qui a besoin des gradients).
    preds_all = predict(entry["model"], Xs[:4])
    device = get_device()
    entry["model"].eval().to(device)
    prep = defense.make_preprocessor(params.preprocess)
    xb_all, yb_all = to_tensors(Xs[:4], ys[:4])
    xb_all, yb_all = xb_all.to(device), yb_all.to(device)
    # L'attaque nécessite les gradients : hors du contexte no_grad.
    x_adv = attack_fn(entry["model"], xb_all, yb_all, params.eps)
    with torch.no_grad():
        x_prep = prep(x_adv)
        preds_p = entry["model"](x_prep).argmax(1).cpu().numpy()
    entry["model"].to("cpu")

    images = []
    for k in range(len(xb_all)):
        images.append({
            "original": to_png_uri(Xs[k]),
            "adversarial": to_png_uri(x_adv[k, 0].cpu().numpy() * 255),
            "preprocessed": to_png_uri(x_prep[k, 0].cpu().numpy() * 255),
            "true": CLASS_NAMES[ys[k]],
            "pred_clean": CLASS_NAMES[preds_all[k]],
            "pred_preprocessed": CLASS_NAMES[preds_p[k]],
        })
    res["images"] = images
    res["model"] = target
    res["attack_label"] = attacks.ATTACKS[params.attack][0]
    res["preprocess_label"] = defense.PREPROCESSORS[params.preprocess]
    STATE.add_history({
        "type": "defense-preproc",
        "title": f"Défense {res['preprocess_label']} contre {res['attack_label']} (ε={params.eps})",
        "summary": {"précision_sans_défense": res["acc_adv"],
                    "précision_avec_défense": res["acc_adv_preprocessed"],
                    "précision_propre_prétraitée": res["acc_clean_preprocessed"]},
    })
    return res


@app.post("/api/defense/adversarial-training")
def api_defense_adv_training(params: AdvTrainParams) -> dict:
    ds = require_dataset()
    require_job_free()

    def work() -> dict:
        STATE.job_progress(0.03, f"Entraînement adversarial {params.method.upper()} (ε={params.eps})…")
        model = STATE.new_model(params.seed)
        train_model(
            model, ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"],
            epochs=params.epochs, lr=params.lr, seed=params.seed, progress_cb=STATE.job_progress,
            adversarial_hook=defense.adversarial_hook(params.method, params.iters, None),
            adv_eps=params.eps,
        )
        STATE.job_progress(0.9, "Évaluation du modèle robuste…")
        metrics = evaluate_model(model, ds["X_test"], ds["y_test"])
        attack_fn = attacks.get_attack_fn(params.method, params.iters, None)
        idx = subset_indices(len(ds["X_test"]), 1500 if STATE.device.type == "cuda" else 500)
        adv_res = attacks.evaluate_attack(
            model, ds["X_test"][idx], ds["y_test"][idx], params.method, params.eps, iters=params.iters
        )

        name = unique_name(f"robuste-{params.method}-e{params.epochs}")
        meta = {
            "kind": "robuste", "name": name, "created_at": now_iso(),
            "params": params.model_dump(), "n_parameters": count_parameters(model),
            "metrics": {"accuracy": metrics["accuracy"], "loss": metrics["loss"],
                        "acc_adv": adv_res["acc_adv"]},
            "active": True,
        }
        STATE.models[name] = {"model": model, "meta": meta}
        STATE.set_active(name)
        STATE.save_model(name, meta)

        # Comparaison avec le modèle de référence (le plus récent non robuste)
        ref_name = next((n for n, m in STATE.models.items() if m["meta"]["kind"] == "propre"), None)
        ref_adv = None
        if ref_name:
            ref_adv = attacks.evaluate_attack(
                STATE.models[ref_name]["model"], ds["X_test"][idx], ds["y_test"][idx],
                params.method, params.eps, iters=params.iters,
            )

        STATE.add_history({
            "type": "defense-adv",
            "title": f"Entraînement adversarial {params.method.upper()} (ε={params.eps}) → « {name} »",
            "summary": {"précision_propre": metrics["accuracy"],
                        "précision_sous_attaque": adv_res["acc_adv"],
                        **({"référence_sous_attaque": ref_adv["acc_adv"]} if ref_adv else {})},
        })
        return {
            "name": name, "history_curve": None, "metrics": metrics, "adv": adv_res,
            "reference": {"name": ref_name, **(ref_adv or {})} if ref_name and ref_adv else None,
            "method": params.method, "eps": params.eps,
        }

    return run_job("Entraînement adversarial", work)


@app.post("/api/defense/poison-filter")
def api_defense_poison_filter(params: FilterParams) -> dict:
    ds = require_dataset()
    require_job_free()
    if not STATE.poisoned:
        raise HTTPException(400, "Aucune attaque d'empoisonnement en cours — lancez d'abord une attaque (onglet Empoisonnement).")
    poisoned_model_name = STATE.poisoned["model"]
    if poisoned_model_name not in STATE.models:
        raise HTTPException(400, "Le modèle empoisonné n'existe plus — relancez l'attaque.")

    poisoned_model = STATE.models[poisoned_model_name]["model"]
    true_idx = np.array(STATE.poisoned["indices"])

    def work() -> dict:
        STATE.job_progress(0.1, "Analyse des pertes par échantillon…")
        det = defense.detect_poisoned_by_loss(
            poisoned_model, ds["X_train"], ds["y_train"], z_threshold=params.z_threshold
        )
        scores = defense.detection_scores(det["suspected_indices"], true_idx, len(ds["y_train"]))

        keep = np.ones(len(ds["y_train"]), dtype=bool)
        keep[det["suspected_indices"]] = False
        X_f, y_f = ds["X_train"][keep], ds["y_train"][keep]

        STATE.job_progress(0.3, f"Ré-entraînement sur {len(X_f)} exemples nettoyés…")
        model = STATE.new_model(params.seed)
        train_model(
            model, X_f, y_f, ds["X_test"], ds["y_test"],
            epochs=params.epochs, lr=params.lr, seed=params.seed, progress_cb=STATE.job_progress,
        )
        STATE.job_progress(0.92, "Évaluation du modèle nettoyé…")
        metrics = evaluate_model(model, ds["X_test"], ds["y_test"])
        bd = None
        if STATE.poisoned["type"] == "backdoor":
            bd = backdoor_success_rate(model, ds["X_test"], ds["y_test"], STATE.poisoned["target"])

        # Histogramme des pertes (empoisonnés vs sains) pour le graphique
        losses = det["losses"]
        lo, hi = float(losses.min()), float(np.quantile(losses, 0.995))
        bins = np.linspace(lo, hi, 41)
        hist_clean, _ = np.histogram(losses[~np.isin(np.arange(len(losses)), true_idx)], bins=bins)
        hist_poison, _ = np.histogram(losses[true_idx], bins=bins) if len(true_idx) else (np.zeros(40), None)
        centers = ((bins[:-1] + bins[1:]) / 2).round(3).tolist()

        name = unique_name("nettoyé-filtrage")
        meta = {
            "kind": "nettoyé", "name": name, "created_at": now_iso(),
            "params": {**params.model_dump(), "n_kept": int(len(X_f))},
            "n_parameters": count_parameters(model),
            "metrics": {k: metrics[k] for k in ("accuracy", "loss")},
            "active": True,
        }
        STATE.models[name] = {"model": model, "meta": meta}
        STATE.set_active(name)
        STATE.save_model(name, meta)

        poisoned_model_res = evaluate_model(poisoned_model, ds["X_test"], ds["y_test"])

        STATE.add_history({
            "type": "defense-filter",
            "title": f"Filtrage des empoisonnés (z={params.z_threshold}) → « {name} »",
            "summary": {"détectés": scores["n_suspected"], "précision_détection": scores["precision"],
                        "rappel_détection": scores["recall"],
                        "précision_test": metrics["accuracy"],
                        **({"succès_backdoor": bd["backdoor_success"]} if bd else {})},
        })
        return {
            "name": name,
            "detection": scores,
            "threshold": det["threshold"],
            "metrics": metrics,
            "poisoned_metrics": {k: poisoned_model_res[k] for k in ("accuracy", "loss")},
            "backdoor": bd,
            "loss_histogram": {"centers": centers, "clean": hist_clean.tolist(), "poison": hist_poison.tolist()},
        }

    return run_job("Détection + filtrage des échantillons empoisonnés", work)


# --------------------------------------------------------------------------- #
# Gestion des modèles / historique / reset                                     #
# --------------------------------------------------------------------------- #


@app.get("/api/history")
def api_history() -> dict:
    return {"history": list(reversed(STATE.history))}


@app.post("/api/models/activate")
def api_activate(params: ActivateParams) -> dict:
    if params.name not in STATE.models:
        raise HTTPException(404, "Modèle introuvable.")
    STATE.set_active(params.name)
    return {"active_model": STATE.active_model}


@app.delete("/api/models/{name}")
def api_delete_model(name: str) -> dict:
    if name not in STATE.models:
        raise HTTPException(404, "Modèle introuvable.")
    path = STATE.models[name].get("path")
    del STATE.models[name]
    if path:
        Path(path).unlink(missing_ok=True)
    if STATE.active_model not in STATE.models:
        STATE.active_model = next(iter(STATE.models), None)
    STATE.persist_registry()
    return {"models": list(STATE.models)}


@app.post("/api/reset")
def api_reset() -> dict:
    require_job_free()
    STATE.dataset = None
    STATE.models.clear()
    STATE.active_model = None
    STATE.poisoned = None
    STATE.history.clear()
    STATE.persist_registry()
    STATE.persist_history()
    (DATA_DIR / "dataset.npz").unlink(missing_ok=True)
    (DATA_DIR / "dataset.meta.json").unlink(missing_ok=True)
    for f in (DATA_DIR / "models").glob("*"):
        f.unlink(missing_ok=True)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Static + entrypoint                                                          #
# --------------------------------------------------------------------------- #

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
