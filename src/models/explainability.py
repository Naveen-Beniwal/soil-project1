"""
Explainability Module: Grad-CAM (Gradient-weighted Class Activation Mapping)
Generates visual attention heatmaps overlaying raw UAV soil images.
Provides transparent AI verification for Dr. Ritu Garg and evaluators,
proving what physical soil surface cues (crusts, cracks, aggregates) guided prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class SoilGradCAM:
    """
    Computes Grad-CAM heatmaps for convolutional layers in Soil CNN / UAVSoilHealthNet.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook into target convolutional layer
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        task: str = "soil_type",
        handcrafted_features: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generates normalized 2D Grad-CAM heatmap (0.0 to 1.0).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        if hasattr(self.model, "handcrafted_mlp"):
            outputs = self.model(input_tensor, handcrafted_features)
            logits = outputs["soil_type_logits"] if task == "soil_type" else outputs["health_status_logits"]
        else:
            logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients (using clone to avoid mutating hook activations)
        activations = self.activations.clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()

        # Standard positive activation (ReLU)
        pos_heatmap = np.maximum(heatmap, 0)

        if np.max(pos_heatmap) > 1e-7:
            heatmap = pos_heatmap / np.max(pos_heatmap)
        else:
            # Fallback for dual-branch models:
            # When global handcrafted features dominate the positive gradient,
            # use relative spatial min-max normalization to reveal spatial attention gradients
            # rather than flattening the entire heatmap into a solid monochrome blackout.
            denom = np.max(heatmap) - np.min(heatmap)
            heatmap = (heatmap - np.min(heatmap)) / (denom + 1e-8) if denom > 1e-7 else np.zeros_like(heatmap)

        return heatmap

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays Grad-CAM heatmap onto original RGB image using adaptive attention masking.
        Preserves natural soil colors in low-attention zones (eliminating purple haze),
        while dynamically highlighting true attention hotspots in glowing cyan/yellow/red.
        Returns: (blended_rgb, colored_heatmap_rgb)
        """
        h, w = image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_cam_rgb = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        # Adaptive alpha mask: low attention pixels (<0.2) preserve natural soil colors with 0 purple tint
        # High attention hotspots smoothly blend the vibrant thermal colormap
        alpha_mask = np.clip((heatmap_resized - 0.2) / 0.8, 0.0, 1.0)
        alpha_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2) * alpha

        blended = np.uint8(np.clip(image_rgb.astype(np.float32) * (1.0 - alpha_3d) + colored_cam_rgb.astype(np.float32) * alpha_3d, 0, 255))
        return blended, colored_cam_rgb
