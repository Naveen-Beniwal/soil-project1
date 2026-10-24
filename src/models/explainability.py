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

        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU on CAM

        # Normalize between 0 and 1
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays Grad-CAM heatmap onto original RGB image.
        Returns: (blended_rgb, colored_heatmap_rgb)
        """
        h, w = image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_cam_rgb = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image_rgb, 1.0 - alpha, colored_cam_rgb, alpha, 0)
        return blended, colored_cam_rgb
