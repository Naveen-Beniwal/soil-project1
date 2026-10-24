"""
UAV Image Preprocessor
Implements noise reduction, contrast enhancement (CLAHE), vegetation masking,
color space conversion (RGB, HSV, LAB), and aerial orthomosaic tiling.
Aligned with SwaYaan PoC Objective 2.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional


class UAVImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def denoise_image(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Removes sensor noise while preserving soil granular edge details.
        Bilateral filtering protects soil crack edges and aggregate boundaries.
        """
        if method == "bilateral":
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def enhance_contrast_clahe(self, image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization in LAB space.
        Enhances soil micro-topography and surface cracks without distorting natural soil hue.
        """
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    def mask_vegetation_and_debris(self, image_rgb: np.ndarray, exg_threshold: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates Excess Green Index (ExG = 2*G - R - B) to segment out weeds,
        crops, and non-soil vegetation canopy, ensuring bare soil analysis.
        Returns: (soil_only_rgb, vegetation_mask)
        """
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)

        # Excess Green Index
        exg = 2.0 * g - r - b
        veg_mask = exg > exg_threshold

        # Create masked image where vegetation is suppressed / masked
        soil_only = image_rgb.copy()
        soil_only[veg_mask] = 0

        return soil_only, veg_mask.astype(np.uint8)

    def extract_color_spaces(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Converts image into RGB, HSV, and CIE-LAB color representations.
        Essential for distinguishing soil moisture, organic matter darkness, and iron oxide hues.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        return {
            "RGB": image_rgb,
            "HSV": hsv,
            "LAB": lab
        }

    def preprocess_patch(self, image_rgb: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
        """
        Standardized end-to-end single patch preprocessing pipeline.
        Denoise -> CLAHE -> Resize -> Return standardized RGB.
        """
        denoised = self.denoise_image(image_rgb, method="bilateral")
        if apply_clahe:
            enhanced = self.enhance_contrast_clahe(denoised)
        else:
            enhanced = denoised
        resized = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_AREA)
        return resized

    def tile_large_aerial_image(
        self,
        orthomosaic_rgb: np.ndarray,
        tile_size: int = 224,
        stride: int = 224
    ) -> List[Dict]:
        """
        Slices a large UAV aerial orthomosaic into regular grid tiles for localized ML prediction.
        Returns a list of dictionaries with patch array, pixel coordinates (x, y), and grid indices (row, col).
        """
        height, width, _ = orthomosaic_rgb.shape
        tiles = []

        row_idx = 0
        for y in range(0, height - tile_size + 1, stride):
            col_idx = 0
            for x in range(0, width - tile_size + 1, stride):
                patch = orthomosaic_rgb[y:y + tile_size, x:x + tile_size]
                # Filter out pure black or blank boundary patches
                if np.mean(patch) > 15:
                    preprocessed_patch = self.preprocess_patch(patch)
                    tiles.append({
                        "patch": preprocessed_patch,
                        "raw_patch": patch,
                        "x": x,
                        "y": y,
                        "row": row_idx,
                        "col": col_idx,
                        "width": tile_size,
                        "height": tile_size
                    })
                col_idx += 1
            row_idx += 1

        return tiles
