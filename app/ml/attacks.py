"""Attaques par évasion : FGSM, PGD et bruit gaussien (baseline).

Toutes les fonctions travaillent sur des tenseurs (N, 1, H, W) dans [0, 1]
déjà présents sur le device. Le modèle est mis en mode eval : il n'est PAS
ré-entraîné, seules les entrées sont modifiées (principe de l'évasion).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def frozen_params(model: nn.Module):
    """Gèle temporairement les paramètres du modèle (l'attaque ne doit
    calculer le gradient que par rapport à l'entrée), puis les restaure —
    indispensable pour réutiliser le même modèle dans l'entraînement adversarial.
    """
    states = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        yield model
    finally:
        for p, s in zip(model.parameters(), states):
            p.requires_grad_(s)
        model.zero_grad(set_to_none=True)


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2015) : un seul pas."""
    with frozen_params(model):
        x_adv = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
    with torch.no_grad():
        x_adv = x + eps * grad.sign()
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float | None = None,
    iters: int = 7,
    random_start: bool = True,
) -> torch.Tensor:
    """Projected Gradient Descent (Madry et al., 2018) : multi-pas, L_inf.

    Si alpha n'est pas fourni : alpha = eps / iters * 2.5 (heuristique classique).
    """
    if alpha is None:
        alpha = max(eps / iters * 2.5, 1e-3)
    with frozen_params(model):
        x_adv = x.clone().detach()
        if random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
            x_adv = x_adv.clamp(0.0, 1.0)

        for _ in range(iters):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)  # projection boule L_inf
                x_adv = x_adv.clamp(0.0, 1.0)
            x_adv = x_adv.detach()
    return x_adv


def noise_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Bruit gaussien aleatoire (baseline non-adversariale pour comparaison)."""
    with torch.no_grad():
        pert = torch.randn_like(x) * eps
        return (x + pert).clamp(0.0, 1.0)


ATTACKS = {
    "fgsm": ("FGSM", "Gradient Sign Method — 1 pas rapide"),
    "pgd": ("PGD", "Projected Gradient Descent — itératif, le plus efficace"),
    "noise": ("Bruit gaussien", "Perturbation aléatoire non-optimisée (baseline)"),
}


def get_attack_fn(method: str, iters: int, alpha: float | None) -> Callable:
    if method == "fgsm":
        return lambda m, x, y, eps: fgsm(m, x, y, eps)
    if method == "pgd":
        return lambda m, x, y, eps: pgd(m, x, y, eps, alpha=alpha, iters=iters)
    if method == "noise":
        return lambda m, x, y, eps: noise_attack(m, x, y, eps)
    raise ValueError(f"Attaque inconnue : {method}")


def _batched_transform(model, X: np.ndarray, y: np.ndarray, fn: Callable, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Applique une transformation (attaque) par batches ; retourne (x_adv, preds_adv).

    NB : pas de no_grad ici — l'attaque nécessite les gradients par rapport à
    l'entrée ; seules les prédictions sont calculées en no_grad.
    """
    from .engine import to_tensors, get_device

    device = get_device()
    model.eval().to(device)
    x, yt = to_tensors(X, y)
    x = x.to(device)
    yt = yt.to(device)
    outs_adv, outs_pred = [], []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], yt[i : i + batch_size]
        x_adv = fn(model, xb, yb)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        outs_adv.append(x_adv.cpu())
        outs_pred.append(preds.cpu())
    model.to("cpu")
    return torch.cat(outs_adv).numpy(), torch.cat(outs_pred).numpy()


def evaluate_attack(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    eps: float,
    iters: int = 7,
    alpha: float | None = None,
    batch_size: int = 128,
) -> dict:
    """Évalue le modèle sous attaque : précision adversariale + taux de succès.

    Le taux de succès est calculé uniquement sur les exemples qui étaient
    correctement classifiés avant attaque (on ne peut pas « duper » une erreur).
    """
    from .engine import predict

    attack_fn = get_attack_fn(method, iters, alpha)
    _, preds_adv = _batched_transform(model, X, y, lambda m, xb, yb: attack_fn(m, xb, yb, eps), batch_size)
    preds_clean = predict(model, X, batch_size)
    clean_correct = preds_clean == y

    acc_clean = float((preds_clean == y).mean())
    acc_adv = float((preds_adv == y).mean())
    n_correct = int(clean_correct.sum())
    fooled = int((preds_adv[clean_correct] != y[clean_correct]).sum()) if n_correct else 0

    return {
        "method": method,
        "eps": eps,
        "iters": iters,
        "n": int(len(y)),
        "acc_clean": round(acc_clean, 4),
        "acc_adv": round(acc_adv, 4),
        "success_rate": round(fooled / n_correct, 4) if n_correct else 0.0,
        "n_fooled": fooled,
        "n_correct_clean": n_correct,
    }
