"""
Soil Feature Extractor
Extracts physical, visual, and mathematical features from UAV soil imagery:
1. Multi-Space Color Moments (RGB, HSV, CIE-LAB)
2. Gray-Level Co-occurrence Matrix (GLCM) Texture Descriptors (Haralick)
3. Local Binary Pattern (LBP) Micro-structure Features
4. Agronomic Optical Soil Indices (Darkness/SOM Proxy, Crusting, Roughness)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class SoilFeatureExtractor:
    def __init__(self):
        pass

    def extract_color_moments(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts 1st (mean), 2nd (std), and 3rd (skewness) moments
        across RGB, HSV, and CIE-LAB color spaces (total 27 color features).
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

        features = []
        for space in [image_rgb, hsv, lab]:
            for c in range(3):
                channel = space[:, :, c].astype(np.float32)
                mean = np.mean(channel)
                std = np.std(channel)
                # Skewness: measure of asymmetry of color distribution
                diff = channel - mean
                skew = np.mean(diff ** 3) / (std ** 3 + 1e-6)
                features.extend([mean, std, skew])

        return np.array(features, dtype=np.float32)

    def extract_glcm_texture(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Computes Gray-Level Co-occurrence Matrix (GLCM) Haralick texture features:
        Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and ASM.
        Calculated across 4 directions: 0, 45, 90, 135 degrees.
        """
        # Quantize to 32 levels for fast, robust co-occurrence statistics
        quantized = (image_gray // 8).astype(np.uint8)
        h, w = quantized.shape
        levels = 32

        # Offsets for (0 deg, 45 deg, 90 deg, 135 deg)
        offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
        
        contrast_list = []
        dissimilarity_list = []
        homogeneity_list = []
        energy_list = []
        correlation_list = []
        asm_list = []

        for dy, dx in offsets:
            # Build co-occurrence matrix
            co_mat = np.zeros((levels, levels), dtype=np.float32)
            
            # Vectorized shift
            y_start = max(0, -dy)
            y_end = min(h, h - dy)
            x_start = max(0, -dx)
            x_end = min(w, w - dx)
            
            src = quantized[y_start:y_end, x_start:x_end]
            dst = quantized[y_start + dy:y_end + dy, x_start + dx:x_end + dx]
            
            flat_src = src.ravel()
            flat_dst = dst.ravel()
            
            for s, d in zip(flat_src, flat_dst):
                co_mat[s, d] += 1.0

            total = co_mat.sum()
            if total > 0:
                co_mat /= total

            # Compute Haralick statistics
            i_idx, j_idx = np.indices((levels, levels))
            diff = np.abs(i_idx - j_idx)
            
            contrast = np.sum(co_mat * (diff ** 2))
            dissimilarity = np.sum(co_mat * diff)
            homogeneity = np.sum(co_mat / (1.0 + diff ** 2))
            asm = np.sum(co_mat ** 2)
            energy = np.sqrt(asm)

            # Mean and variance for correlation
            i_mean = np.sum(i_idx * co_mat)
            j_mean = np.sum(j_idx * co_mat)
            i_std = np.sqrt(np.sum((i_idx - i_mean) ** 2 * co_mat) + 1e-6)
            j_std = np.sqrt(np.sum((j_idx - j_mean) ** 2 * co_mat) + 1e-6)
            correlation = np.sum((i_idx - i_mean) * (j_idx - j_mean) * co_mat) / (i_std * j_std + 1e-6)

            contrast_list.append(contrast)
            dissimilarity_list.append(dissimilarity)
            homogeneity_list.append(homogeneity)
            energy_list.append(energy)
            correlation_list.append(correlation)
            asm_list.append(asm)

        texture_vec = [
            np.mean(contrast_list), np.std(contrast_list),
            np.mean(dissimilarity_list), np.std(dissimilarity_list),
            np.mean(homogeneity_list), np.std(homogeneity_list),
            np.mean(energy_list), np.std(energy_list),
            np.mean(correlation_list), np.std(correlation_list),
            np.mean(asm_list), np.std(asm_list)
        ]
        return np.array(texture_vec, dtype=np.float32)

    def extract_lbp_histogram(self, image_gray: np.ndarray, num_bins: int = 16) -> np.ndarray:
        """
        Computes Local Binary Patterns (LBP) to represent soil surface granular micro-texture.
        """
        h, w = image_gray.shape
        padded = np.pad(image_gray, 1, mode="reflect").astype(np.int32)
        center = padded[1:-1, 1:-1]

        # 8 neighbors
        lbp = np.zeros((h, w), dtype=np.uint8)
        neighbors = [
            padded[:-2, :-2],  # Top-left
            padded[:-2, 1:-1],  # Top
            padded[:-2, 2:],    # Top-right
            padded[1:-1, 2:],   # Right
            padded[2:, 2:],     # Bottom-right
            padded[2:, 1:-1],   # Bottom
            padded[2:, :-2],    # Bottom-left
            padded[1:-1, :-2]   # Left
        ]

        for i, neighbor in enumerate(neighbors):
            lbp += ((neighbor >= center) << i).astype(np.uint8)

        hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, 256), density=True)
        return hist.astype(np.float32)

    def extract_optical_soil_health_proxies(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Computes physically-grounded visual soil health indicators:
        1. Soil Organic Matter Proxy (Inverted lightness / Darkness index)
        2. Visual Crusting / Compaction index (Specular reflection & low texture roughness)
        3. Moisture Reflectance proxy (attenuation across visible spectrum)
        4. Iron Oxide Hue ratio (R / (R + G + B))
        5. Texture Roughness score (Laplacian variance)
        """
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Darkness proxy (Lower L* in bare soil typically indicates higher organic matter / moisture)
        som_proxy = (255.0 - np.mean(l_channel)) / 255.0

        # Roughness / Crumb structure proxy (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Soil Redness / Iron oxide index
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)
        sum_rgb = r + g + b + 1e-6
        redness_index = np.mean(r / sum_rgb)

        # Brightness / Crusting proxy (excess white specular surface crusting)
        crusting_ratio = np.mean((l_channel > 210).astype(np.float32))

        # Moisture attenuation proxy: lower overall luminance with balanced chromaticity
        mean_lum = np.mean(gray) / 255.0
        moisture_proxy = 1.0 - mean_lum

        return np.array([
            som_proxy,
            np.log1p(laplacian_var) / 10.0,
            redness_index,
            crusting_ratio,
            moisture_proxy
        ], dtype=np.float32)

    def extract_all_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Concatenates all handcrafted features into a single 1D feature vector:
        27 Color Moments + 12 GLCM + 16 LBP + 5 Soil Optical Proxies = 60 Features.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        color_moments = self.extract_color_moments(image_rgb)
        glcm = self.extract_glcm_texture(gray)
        lbp = self.extract_lbp_histogram(gray)
        proxies = self.extract_optical_soil_health_proxies(image_rgb)

        return np.concatenate([color_moments, glcm, lbp, proxies]).astype(np.float32)
