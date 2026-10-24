"""
Baseline Deep Learning Models (2023 Elsevier Base Paper Reference)
Implements:
1. Standard SoilCNN (Convolutional Neural Network)
2. MobileNetV2 Transfer Learning Baseline
These represent the state-of-the-art benchmark established in recent Indian soil classification literature.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict


class StandardSoilCNN(nn.Module):
    """
    Standard 4-layer Convolutional Neural Network baseline.
    Represents classic computer vision soil classification architecture.
    """
    def __init__(self, num_classes: int = 7):
        super(StandardSoilCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))           # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class BaselineMobileNetV2(nn.Module):
    """
    MobileNetV2 Transfer Learning baseline as published in 2023 Elsevier paper.
    Single-head classification network for soil type identification.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super(BaselineMobileNetV2, self).__init__()
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)
        except Exception:
            base = models.mobilenet_v2(weights=None)
        self.features = base.features
        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits
