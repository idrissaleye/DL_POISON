"""Défenses contre les attaques adversariales.

Trois familles de défenses sont implémentées :

  1. Entraînement adversarial (contre l'évasion) — Madry et al. 2018 :
     ré-entraîne le modèle en augmentant chaque batch d'exemples adversariaux.
     Le modèle apprend à être correct SUR les perturbations, au prix d'un
     léger coût sur les données propres.

  2. Prétraitements d'entrée (contre l'évasion) — défenses "gratuites" sans
     réentraînement : quantification de la profondeur de bits, flou gaussien,
     filtre médian, compression JPEG. Elles détruisent une partie de la
     perturbation soigneusement construite par l'attaquant.

  3. Détection/filtrage des échantillons empoisonnés (contre l'empoisonnement)
     — approche "loss-based outlier detection" : un modèle entraîné sur les
     données suspectes attribue une perte anormalement élevée aux exemples
     empoisonnés ; on retire les outliers et on ré-entraîne sur les données
     nettoyées.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .attacks import get_attack_fn
from .engine import to_tensors, get_device

# --------------------------------------------------------------------------- #
# 1. Entraînement adversarial                                                  #
# --------------------------------------------------------------------------- #


def adversarial_hook(method: str, iters: int, alpha: float | None):
    """Retourne la fonction passée à engine.train_model(adversarial_hook=...)."""
    attack_fn = get_attack_fn(method, iters, alpha)
    return lambda m, xb, yb, eps: attack_fn(m, xb, yb, eps)


# --------------------------------------------------------------------------- #
# 2. Prétraitements d'entrée                                                   #
# --------------------------------------------------------------------------- #

PREPROCESSORS = {
    "aucun": "Aucun prétraitement (référence)",
    "quantification": "Quantification 4 niveaux (réduction de profondeur de bits)",
    "flou": "Flou gaussien 3x3",
    "median": "Filtre médian 3x3",
    "jpeg": "Compression JPEG qualité 30",
}


def _gaussian_kernel_3x3(device) -> torch.Tensor:
    """Noyau binomial 3x3 (sigma ~ 0.7) : adoucit sans détruire les formes."""
    k = torch.tensor(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        device=device,
    )
    return (k / k.sum()).view(1, 1, 3, 3)


def _median_filter_3x3(x: torch.Tensor) -> torch.Tensor:
    """Filtre médian 3x3 via unfold (B, C, H, W) -> médiane des 9 voisins."""
    pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    _, _, H, W = x.shape
    patches = pad.unfold(2, 3, 1).unfold(3, 3, 1)  # (B, 1, H, W, 3, 3)
    patches = patches.contiguous().view(*x.shape, 9)
    return patches.median(dim=-1).values


def _jpeg_roundtrip(x: torch.Tensor, quality: int = 30) -> torch.Tensor:
    """Compression JPEG par image via PIL (les tenseurs passent par CPU/numpy)."""
    device = x.device
    arr = (x.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # (N, 1, H, W)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        img = Image.fromarray(arr[i, 0], mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out[i, 0] = np.asarray(Image.open(buf).convert("L"))
    return torch.from_numpy(out).float().div_(255.0).to(device)


def make_preprocessor(name: str):
    """Retourne f(x: Tensor[B,1,H,W]) -> Tensor[B,1,H,W] dans [0, 1]."""
    if name == "aucun":
        return lambda x: x
    if name == "quantification":
        # Quantifie [0,1] en 4 niveaux : détruit les micro-perturbations.
        return lambda x: torch.round(x * 3.0) / 3.0
    if name == "flou":
        def blur(x):
            k = _gaussian_kernel_3x3(x.device)
            return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k)
        return blur
    if name == "median":
        return _median_filter_3x3
    if name == "jpeg":
        return lambda x: _jpeg_roundtrip(x)
    raise ValueError(f"Prétraitement inconnu : {name}")


def evaluate_with_preprocessing(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    eps: float,
    iters: int = 7,
    alpha: float | None = None,
    preprocess: str = "aucun",
    batch_size: int = 128,
) -> dict:
    """Précision du modèle sous attaque, avec et sans prétraitement.

    Pipeline défendu : x -> attaque -> prétraitement -> modèle.
    NB : pas de no_grad global — l'attaque a besoin des gradients.
    """
    device = get_device()
    model.eval().to(device)
    attack_fn = get_attack_fn(method, iters, alpha)
    prep = make_preprocessor(preprocess)
    x, yt = to_tensors(X, y)
    x, yt = x.to(device), yt.to(device)

    acc_clean_prep, acc_adv, acc_adv_prep, l2_sum, n = 0.0, 0, 0, 0.0, len(x)
    for i in range(0, n, batch_size):
        xb, yb = x[i : i + batch_size], yt[i : i + batch_size]
        # Données propres + prétraitement (le prétraitement ne doit pas casser le modèle)
        with torch.no_grad():
            acc_clean_prep += (model(prep(xb)).argmax(1) == yb).sum().item()
        # Données adversariales (gradients requis pour construire la perturbation)
        x_adv = attack_fn(model, xb, yb, eps)
        with torch.no_grad():
            acc_adv += (model(x_adv).argmax(1) == yb).sum().item()
            # Données adversariales + prétraitement (la défense testée)
            acc_adv_prep += (model(prep(x_adv)).argmax(1) == yb).sum().item()
        l2_sum += (x_adv - xb).flatten(1).norm(dim=1).sum().item()

    model.to("cpu")
    return {
        "preprocess": preprocess,
        "acc_clean_preprocessed": round(acc_clean_prep / n, 4),
        "acc_adv": round(acc_adv / n, 4),
        "acc_adv_preprocessed": round(acc_adv_prep / n, 4),
        "mean_perturbation_l2": round(l2_sum / n, 4),
    }


# --------------------------------------------------------------------------- #
# 3. Détection d'échantillons empoisonnés                                      #
# --------------------------------------------------------------------------- #


def detect_poisoned_by_loss(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    z_threshold: float = 2.5,
    batch_size: int = 256,
) -> dict:
    """Détecte les outliers via la perte par échantillon (approche loss-based).

    Retourne les indices suspects + statistiques de la distribution des pertes.
    """
    device = get_device()
    model.eval().to(device)
    x, yt = to_tensors(X_train, y_train)
    losses = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb, yb = x[i : i + batch_size].to(device), yt[i : i + batch_size].to(device)
            losses.append(F.cross_entropy(model(xb), yb, reduction="none").cpu())
    model.to("cpu")
    losses = torch.cat(losses).numpy()

    mean, std = float(losses.mean()), float(losses.std())
    threshold = mean + z_threshold * std
    suspected = np.flatnonzero(losses > threshold)
    return {
        "losses": losses,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "threshold": round(float(threshold), 4),
        "suspected_indices": suspected,
        "n_suspected": int(len(suspected)),
    }


def detection_scores(suspected: np.ndarray, true_poisoned: np.ndarray, n_total: int) -> dict:
    """Précision/rappel de la détection, si les indices réels sont connus."""
    sus = set(suspected.tolist())
    tru = set(true_poisoned.tolist())
    tp = len(sus & tru)
    fp = len(sus - tru)
    fn = len(tru - sus)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_true_poisoned": len(tru),
        "n_suspected": len(sus),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
