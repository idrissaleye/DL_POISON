"""Attaques par empoisonnement des données d'entraînement.

Deux scénarios :
  1. Retournement d'étiquettes (label flipping) : une fraction des labels
     d'entraînement est remplacée, les images restent identiques.
  2. Attaque par porte dérobée (backdoor) : une fraction des images reçoit un
     motif déclencheur (trigger) et sont relabelées vers la classe cible.
     Le modèle apprend « trigger => cible » ; à l'inférence toute image portant
     le trigger est classée dans la classe cible.

Contrairement à l'évasion, ici c'est le MODÈLE lui-même qui devient malveillant.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import CLASS_NAMES

# Motif déclencheur : damier 6x6 très contrasté en haut à gauche.
TRIGGER_SIZE = 6
TRIGGER_POS = (2, 2)


def apply_trigger(X: np.ndarray) -> np.ndarray:
    """Applique le motif déclencheur sur une copie des images (N, H, W)."""
    X = X.copy()
    r0, c0 = TRIGGER_POS
    for i in range(TRIGGER_SIZE):
        for j in range(TRIGGER_SIZE):
            if (i + j) % 2 == 0:
                X[:, r0 + i, c0 + j] = 255
            else:
                X[:, r0 + i, c0 + j] = 0
    return X


def poison_label_flip(y: np.ndarray, ratio: float, target_class: int | None, seed: int = 42):
    """Retourne (indices_poisonés, nouveaux_labels) — targeted ou aléatoire."""
    rng = np.random.default_rng(seed)
    n = len(y)
    n_poison = int(round(n * ratio))
    idx = np.sort(rng.choice(n, size=n_poison, replace=False))
    new_y = y.copy()
    for i in idx:
        if target_class is not None:
            new_y[i] = target_class
        else:
            others = [c for c in range(len(CLASS_NAMES)) if c != y[i]]
            new_y[i] = rng.choice(others)
    return idx, new_y


def poison_backdoor(
    X: np.ndarray, y: np.ndarray, ratio: float, target_class: int, seed: int = 42
):
    """Empoisonne une fraction du dataset avec le trigger + label cible.

    Les images déjà de la classe cible ne sont pas modifiées (elles portent
    déjà le bon label). Retourne (indices_poisonés, X_poisoné, y_poisoné).
    """
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(y != target_class)
    n_poison = int(round(len(X) * ratio))
    n_poison = min(n_poison, len(candidates))
    idx = np.sort(rng.choice(candidates, size=n_poison, replace=False))
    X_p = X.copy()
    X_p[idx] = apply_trigger(X_p[idx])
    y_p = y.copy()
    y_p[idx] = target_class
    return idx, X_p, y_p


def backdoor_success_rate(
    model, X_test: np.ndarray, y_test: np.ndarray, target_class: int, batch_size: int = 256
) -> dict:
    """Taux de succès de la porte dérobée sur le jeu de test.

    On applique le trigger aux images de test n'appartenant PAS à la classe
    cible et on mesure la fraction classée dans la classe cible.
    """
    from .engine import to_tensors, get_device

    mask = y_test != target_class
    X_triggered = apply_trigger(X_test[mask])
    device = get_device()
    model.eval().to(device)
    x = to_tensors(X_triggered)
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            preds.append(model(x[i : i + batch_size].to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    model.to("cpu")
    success = float((preds == target_class).mean()) if len(preds) else 0.0
    return {
        "backdoor_success": round(success, 4),
        "n_triggered": int(len(preds)),
        "target_class": CLASS_NAMES[target_class],
    }
