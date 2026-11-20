"""Entraînement et évaluation du modèle (device GPU/CPU auto-détecté)."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import CLASS_NAMES


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"GPU — {torch.cuda.get_device_name(0)}"
    return "CPU"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_tensors(X: np.ndarray, y: Optional[np.ndarray] = None):
    """uint8 (N, H, W) -> float (N, 1, H, W) dans [0, 1] + labels."""
    x = torch.from_numpy(np.ascontiguousarray(X)).float().unsqueeze(1) / 255.0
    if y is None:
        return x
    return x, torch.from_numpy(np.ascontiguousarray(y)).long()


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 8,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    adversarial_hook: Optional[Callable[[nn.Module, torch.Tensor, torch.Tensor, float], torch.Tensor]] = None,
    adv_eps: float = 0.15,
) -> list[dict]:
    """Entraîne le modèle. Si `adversarial_hook` est fourni, chaque batch est
    augmenté avec ses exemples adversariaux (entraînement adversarial).

    Retourne l'historique [{epoch, train_loss, train_acc, test_acc}, ...].
    """
    device = get_device()
    set_seed(seed)
    model.to(device)

    x_all, y_all = to_tensors(X_train, y_train)
    x_test, y_test_t = to_tensors(X_test, y_test)
    dataset = torch.utils.data.TensorDataset(x_all, y_all)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_batches = epochs * len(loader)
    done = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if adversarial_hook is not None:
                model.eval()  # l'attaque nécessite le mode eval (BatchNorm)
                xb_adv = adversarial_hook(model, xb, yb, adv_eps)
                model.train()
                xb = torch.cat([xb, xb_adv], dim=0)
                yb = torch.cat([yb, yb], dim=0)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
            done += 1
            if progress_cb and done % 10 == 0:
                progress_cb(done / total_batches, f"Époque {epoch}/{epochs} — perte {running_loss / seen:.4f}")

        test_acc = _quick_accuracy(model, x_test, y_test_t, batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running_loss / seen, 4),
                "train_acc": round(correct / seen, 4),
                "test_acc": round(test_acc, 4),
            }
        )
        if progress_cb:
            progress_cb(done / total_batches, f"Époque {epoch}/{epochs} — précision test {test_acc:.2%}")

    model.to("cpu")
    return history


@torch.no_grad()
def _quick_accuracy(model, x, y, batch_size: int) -> float:
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        logits = model(x[i : i + batch_size].to(device))
        correct += (logits.argmax(1).cpu() == y[i : i + batch_size]).sum().item()
    return correct / len(x)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
) -> dict:
    """Précision globale + matrice de confusion + métriques par classe."""
    device = get_device()
    model.eval().to(device)
    x, yt = to_tensors(X, y)
    preds, all_probs = [], []
    loss_sum = 0.0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size].to(device), yt[i : i + batch_size].to(device)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        probs = F.softmax(logits, dim=1)
        preds.append(logits.argmax(1).cpu())
        all_probs.append(probs.cpu())
    preds = torch.cat(preds).numpy()
    probs = torch.cat(all_probs).numpy()

    n_classes = len(CLASS_NAMES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y, preds):
        confusion[t, p] += 1

    per_class = []
    for c, name in enumerate(CLASS_NAMES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {"class": name, "precision": round(precision, 4), "recall": round(recall, 4),
             "f1": round(f1, 4), "support": int(confusion[c, :].sum())}
        )

    accuracy = float((preds == y).mean())
    model.to("cpu")
    return {
        "accuracy": round(accuracy, 4),
        "loss": round(loss_sum / len(y), 4),
        "n": int(len(y)),
        "confusion": confusion.tolist(),
        "per_class": per_class,
        "predictions": preds.tolist(),
        "probs": probs.tolist(),
    }


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    device = get_device()
    model.eval().to(device)
    x = to_tensors(X)
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model(x[i : i + batch_size].to(device)).argmax(1).cpu())
    model.to("cpu")
    return torch.cat(out).numpy()
