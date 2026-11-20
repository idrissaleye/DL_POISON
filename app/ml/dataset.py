"""Génération d'un dataset synthétique d'images de formes géométriques.

Le dataset est entièrement aléatoire (positions, tailles, rotations, intensités,
bruit) : aucune donnée externe n'est téléchargée. Il est sauvegardé au format
NPZ compressé (~40 Mo par défaut), très loin de la limite des 2 Go.
"""

from __future__ import annotations

import io
import base64
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["cercle", "carre", "triangle", "croix"]


def generate_shapes(n_per_class: int, img_size: int = 64, seed: int = 42):
    """Génère n_per_class images par classe.

    Retourne (X, y) : X uint8 de forme (N, img_size, img_size), y int64 (N,).
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for cls_idx, _ in enumerate(CLASS_NAMES):
        for _ in range(n_per_class):
            images.append(_draw_shape(rng, cls_idx, img_size))
            labels.append(cls_idx)
    X = np.stack(images).astype(np.uint8)
    y = np.array(labels, dtype=np.int64)

    # Mélange global déterministe
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _draw_shape(rng: np.random.Generator, cls_idx: int, size: int) -> np.ndarray:
    """Dessine une forme aléatoire (position, taille, rotation, intensité)."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    cx = size / 2 + rng.uniform(-size * 0.12, size * 0.12)
    cy = size / 2 + rng.uniform(-size * 0.12, size * 0.12)
    radius = rng.uniform(size * 0.22, size * 0.38)
    intensity = int(rng.uniform(150, 255))

    if cls_idx == 0:  # cercle
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=intensity)
    elif cls_idx == 1:  # carré (rotation aléatoire)
        pts = _polygon_points(cx, cy, radius * 1.25, 4, rng.uniform(0, 90))
        draw.polygon(pts, fill=intensity)
    elif cls_idx == 2:  # triangle
        pts = _polygon_points(cx, cy, radius * 1.25, 3, rng.uniform(0, 360))
        draw.polygon(pts, fill=intensity)
    else:  # croix (étoile à 4 branches = signe +)
        pts = []
        for k in range(12):
            angle = math.radians(k * 30 + rng.uniform(0, 360))
            r = radius if k % 2 == 0 else radius * 0.38
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(pts, fill=intensity)

    arr = np.asarray(img, dtype=np.float32)
    # Bruit gaussien léger + speckle de fond pour complexifier la tâche
    arr += rng.normal(0.0, 9.0, arr.shape)
    background = rng.random(arr.shape) < 0.002
    arr[background] = rng.uniform(60, 200, size=int(background.sum()))
    return np.clip(arr, 0, 255).astype(np.uint8)


def _polygon_points(cx: float, cy: float, r: float, n: int, offset_deg: float):
    return [
        (cx + r * math.cos(math.radians(offset_deg + k * 360 / n)),
         cy + r * math.sin(math.radians(offset_deg + k * 360 / n)))
        for k in range(n)
    ]


def stratified_split(y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    """Split stratifié par classe, retourne (train_idx, test_idx)."""
    rng = np.random.default_rng(seed + 1)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.sort(np.array(train_idx)), np.sort(np.array(test_idx))


def save_dataset(path: Path, X_train, y_train, X_test, y_test, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta["class_names"] = CLASS_NAMES
    np.savez_compressed(path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_dataset(path: Path):
    data = np.load(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"], meta


def to_png_uri(arr: np.ndarray, upscale: int = 3) -> str:
    """Convertit une image uint8 (H, W) en data-URI PNG (agrandie, sans lissage)."""
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_uri_heat(diff: np.ndarray, upscale: int = 3) -> str:
    """Convertit une carte de perturbation (float [H,W]) en PNG contrasté."""
    diff = np.abs(diff)
    if diff.max() > 1e-12:
        diff = diff / diff.max()
    img = Image.fromarray((diff * 255).astype(np.uint8), mode="L")
    img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
