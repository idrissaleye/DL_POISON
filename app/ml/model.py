"""CNN compact pour la classification des formes synthétiques."""

from __future__ import annotations

import torch.nn as nn


class ShapesCNN(nn.Module):
    """CNN 3 couches conv + 2 couches fully-connected.

    Choix de conception :
      - **GroupNorm** plutôt que BatchNorm : pas de statistiques "running",
        donc comportement strictement identique en train/eval — indispensable
        pour des attaques et un entraînement adversarial cohérent (recommandé
        par Xie & Yuille 2019 pour l'entraînement adversarial) ;
      - pooling adaptatif final : le réseau est indépendant de la taille
        d'image choisie à la génération du dataset (32, 64, 96...).
    """

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 16 -> 8
            nn.AdaptiveAvgPool2d((4, 4)),         # -> 4x4 quelle que soit la taille
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
