"""
UAV-SoilHealthNet: Advanced Multi-Task Attention-Augmented Soil Analysis Network
NOVEL CONTRIBUTION over Base Research Paper (Elsevier 2023)

Key Innovations:
1. Dual-Branch Feature Fusion: Deep CNN Backbone + Domain-Specific Handcrafted Visual Features (GLCM, LBP, Color Moments).
2. CBAM (Convolutional Block Attention Module): Spatial & Channel Attention to focus on micro-crusts, crumb structure, and moisture variation.
3. Multi-Task Learning Architecture: Jointly predicts Soil Type (Classification), Soil Health Status (Classification: Healthy/Moderate/Degraded), and Continuous Soil Health Index (Regression: 0-100).
4. Drone-Aerial Invariance: Robust to scale and elevation changes common in UAV flights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional


class ChannelAttention(nn.Module):
    """Channel Attention sub-module of CBAM."""
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention sub-module of CBAM."""
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM) combining Channel & Spatial Attention."""
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class UAVSoilHealthNet(nn.Module):
    """
    Advanced Hybrid Multi-Task Deep Neural Network for UAV Soil Health Assessment.
    Fuses deep visual representations with handcrafted domain features.
    """
    def __init__(
        self,
        num_soil_types: int = 7,
        num_health_states: int = 3,
        num_handcrafted_features: int = 60,
        pretrained_backbone: bool = False
    ):
        super(UAVSoilHealthNet, self).__init__()
        self.num_soil_types = num_soil_types
        self.num_health_states = num_health_states
        self.num_handcrafted_features = num_handcrafted_features

        # Branch 1: Deep Vision Backbone (MobileNetV3-Large or ResNet)
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
            backbone = models.mobilenet_v3_large(weights=weights)
        except Exception:
            backbone = models.mobilenet_v3_large(weights=None)
        self.deep_features = backbone.features  # Outputs 960 channels

        # Attention Mechanism applied to deepest feature map
        self.attention = CBAMBlock(in_planes=960, ratio=16)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.deep_proj = nn.Sequential(
            nn.Linear(960, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Branch 2: Handcrafted Physical Agronomic Features MLP
        self.handcrafted_mlp = nn.Sequential(
            nn.Linear(num_handcrafted_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Cross-Modal Fusion Layer (256 Deep + 64 Handcrafted = 320)
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Multi-Task Output Head 1: Soil Type Classification (Alluvial, Black, Red, etc.)
        self.head_soil_type = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_soil_types)
        )

        # Multi-Task Output Head 2: Soil Health Degradation Classification (Healthy, Moderate, Degraded)
        self.head_health_status = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_health_states)
        )

        # Multi-Task Output Head 3: Soil Health Index (SHI) Continuous Regression Score (0 to 100)
        self.head_health_index = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs 0.0 - 1.0, scaled to 0-100
        )

    def forward(
        self,
        images: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Branch 1: Deep Vision + Attention
        feat_map = self.deep_features(images)
        attended_feat = self.attention(feat_map)
        pooled = self.global_pool(attended_feat)
        flattened_deep = torch.flatten(pooled, 1)
        deep_emb = self.deep_proj(flattened_deep)

        # Branch 2: Handcrafted Features
        batch_size = images.size(0)
        if handcrafted_features is None:
            # Fallback zero tensor if handcrafted features are omitted during inference
            device = images.device
            handcrafted_features = torch.zeros((batch_size, self.num_handcrafted_features), device=device)
        
        handcrafted_emb = self.handcrafted_mlp(handcrafted_features)

        # Fusion
        fused = torch.cat([deep_emb, handcrafted_emb], dim=1)
        latent = self.fusion_layer(fused)

        # Multi-Task Predictions
        soil_type_logits = self.head_soil_type(latent)
        health_status_logits = self.head_health_status(latent)
        soil_health_index = self.head_health_index(latent) * 100.0  # Scale to 0-100 score

        return {
            "soil_type_logits": soil_type_logits,
            "health_status_logits": health_status_logits,
            "soil_health_index": soil_health_index.squeeze(-1),
            "latent_representation": latent,
            "last_feature_map": feat_map
        }


class MultiTaskSoilLoss(nn.Module):
    """
    Joint Weighted Multi-Task Loss:
    Loss = w_type * CrossEntropy(Type) + w_health * CrossEntropy(Health) + w_shi * MSE(SHI / 100)
    """
    def __init__(self, w_type: float = 1.0, w_health: float = 1.0, w_shi: float = 0.5):
        super(MultiTaskSoilLoss, self).__init__()
        self.w_type = w_type
        self.w_health = w_health
        self.w_shi = w_shi

        self.ce_type = nn.CrossEntropyLoss()
        self.ce_health = nn.CrossEntropyLoss()
        self.mse_shi = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        target_soil_type: torch.Tensor,
        target_health_status: torch.Tensor,
        target_shi: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_type = self.ce_type(predictions["soil_type_logits"], target_soil_type)
        loss_health = self.ce_health(predictions["health_status_logits"], target_health_status)
        loss_shi = self.mse_shi(predictions["soil_health_index"] / 100.0, target_shi / 100.0)

        total_loss = (self.w_type * loss_type) + (self.w_health * loss_health) + (self.w_shi * loss_shi)

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "loss_soil_type": float(loss_type.item()),
            "loss_health_status": float(loss_health.item()),
            "loss_shi": float(loss_shi.item())
        }
        return total_loss, loss_dict
