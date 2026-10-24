"""
Dataset Loader & Splitter for Real Indian Soil Dataset
Consolidates real images across 7 Indian Soil Classes (6,284 images).
Implements 70% Train, 15% Validation, 15% Test partitioning as per SwaYaan Proposal.
Pre-caches preprocessed tensors for lightning-fast training.
"""

import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.config import (
    SOIL_TYPES, SOIL_HEALTH_STATUSES, SOIL_TYPE_TO_IDX, SOIL_HEALTH_TO_IDX,
    RAW_DATA_DIR, FOLDER_TO_SOIL_TYPE
)
from src.preprocessing.feature_extractor import SoilFeatureExtractor
from src.preprocessing.preprocessor import UAVImagePreprocessor


class SoilImageSample:
    def __init__(self, filepath: Path, soil_type_idx: int, health_status_idx: int, shi: float):
        self.filepath = filepath
        self.soil_type_idx = soil_type_idx
        self.health_status_idx = health_status_idx
        self.shi = shi


class SoilPyTorchDataset(Dataset):
    def __init__(
        self,
        samples: List[SoilImageSample],
        cached_images: np.ndarray,
        handcrafted_features: Optional[np.ndarray] = None,
        transform=None
    ):
        self.samples = samples
        self.cached_images = cached_images  # Pre-cached preprocessed uint8 array (N, 224, 224, 3)
        self.handcrafted_features = handcrafted_features
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        rgb = self.cached_images[idx]

        # PyTorch Tensor Transform
        if self.transform is not None:
            tensor_img = self.transform(rgb)
        else:
            tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        item = {
            "image": tensor_img,
            "soil_type": torch.tensor(sample.soil_type_idx, dtype=torch.long),
            "health_status": torch.tensor(sample.health_status_idx, dtype=torch.long),
            "shi": torch.tensor(sample.shi, dtype=torch.float32),
            "filepath": str(sample.filepath)
        }

        if self.handcrafted_features is not None:
            item["handcrafted"] = torch.tensor(self.handcrafted_features[idx], dtype=torch.float32)

        return item


class SoilDataManager:
    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.feature_extractor = SoilFeatureExtractor()
        self.preprocessor = UAVImagePreprocessor()

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def discover_samples(self, max_per_class: Optional[int] = 250) -> List[SoilImageSample]:
        """
        Scans data/raw/ across all 7 Indian soil classes.
        Extracts ground-truth soil type from folder name.
        Calculates physically-grounded Soil Health Index (SHI: 0-100) and degradation status
        directly from optical soil features (SOM darkness proxy, aggregate roughness, crusting).
        """
        samples = []
        np.random.seed(42)

        for type_dir in sorted(self.data_dir.iterdir()):
            if not type_dir.is_dir():
                continue

            folder_key = type_dir.name.lower().strip()
            if folder_key not in FOLDER_TO_SOIL_TYPE:
                continue

            soil_name = FOLDER_TO_SOIL_TYPE[folder_key]
            soil_idx = SOIL_TYPE_TO_IDX[soil_name]

            all_files = [f for f in type_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            np.random.shuffle(all_files)

            if max_per_class is not None and len(all_files) > max_per_class:
                selected_files = all_files[:max_per_class]
            else:
                selected_files = all_files

            for img_file in selected_files:
                # Read image to compute physical soil health proxy
                bgr = cv2.imread(str(img_file))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if h < 20 or w < 20:
                    continue

                patch = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                proxies = self.feature_extractor.extract_optical_soil_health_proxies(patch)
                # proxies: [som_proxy (0-1), roughness_log (0-1), redness_idx, crusting_ratio (0-1), moisture_proxy (0-1)]
                som_score = float(proxies[0]) * 40.0         # Up to 40 pts from organic matter proxy
                roughness_score = min(float(proxies[1]) * 35.0, 30.0) # Up to 30 pts from crumb texture
                moisture_score = float(proxies[4]) * 30.0    # Up to 30 pts from moisture reflectance
                crust_penalty = float(proxies[3]) * 35.0     # Penalty for severe surface crusting/bleaching

                # Base SHI by soil type baseline
                type_base_shifts = {
                    "Alluvial Soil": 15.0,
                    "Black Soil": 20.0,
                    "Red Soil": 10.0,
                    "Laterite Soil": 5.0,
                    "Mountain Soil": 15.0,
                    "Yellow Soil": 10.0,
                    "Arid Soil": -10.0
                }
                shift = type_base_shifts.get(soil_name, 0.0)

                raw_shi = som_score + roughness_score + moisture_score - crust_penalty + shift
                shi = float(np.clip(raw_shi, 15.0, 96.0))

                # Degradation Health Status mapping
                if shi >= 68.0:
                    health_idx = 0  # Healthy Soil
                elif shi >= 45.0:
                    health_idx = 1  # Moderately Healthy Soil
                else:
                    health_idx = 2  # Degraded Soil

                samples.append(SoilImageSample(img_file, soil_idx, health_idx, shi))

        return samples

    def prepare_datasets(
        self,
        samples: List[SoilImageSample],
        extract_handcrafted: bool = True
    ) -> Dict[str, Any]:
        """
        Partitions samples into 70% Train, 15% Val, 15% Test.
        Pre-caches preprocessed images and extracts 60 domain-specific visual features.
        """
        print(f"[*] Preprocessing and pre-caching {len(samples)} real soil images into memory...")
        cached_images = []
        all_feats = []

        for i, s in enumerate(samples):
            bgr = cv2.imread(str(s.filepath))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            processed_rgb = self.preprocessor.preprocess_patch(rgb)
            cached_images.append(processed_rgb)

            if extract_handcrafted:
                feats = self.feature_extractor.extract_all_features(processed_rgb)
                all_feats.append(feats)

        cached_images = np.array(cached_images, dtype=np.uint8)
        if extract_handcrafted:
            all_feats = np.array(all_feats, dtype=np.float32)

        indices = np.arange(len(samples))
        y_types = [s.soil_type_idx for s in samples]

        # 70% Train, 30% Temp (Stratified)
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=42, stratify=y_types
        )

        temp_y = [y_types[i] for i in temp_idx]
        # Split temp equally into 15% Val, 15% Test
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=42, stratify=temp_y
        )

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]

        train_handcrafted = all_feats[train_idx] if extract_handcrafted else None
        val_handcrafted = all_feats[val_idx] if extract_handcrafted else None
        test_handcrafted = all_feats[test_idx] if extract_handcrafted else None

        train_dataset = SoilPyTorchDataset(train_samples, cached_images[train_idx], train_handcrafted, self.transform_train)
        val_dataset = SoilPyTorchDataset(val_samples, cached_images[val_idx], val_handcrafted, self.transform_val)
        test_dataset = SoilPyTorchDataset(test_samples, cached_images[test_idx], test_handcrafted, self.transform_val)

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "train_handcrafted": train_handcrafted,
            "val_handcrafted": val_handcrafted,
            "test_handcrafted": test_handcrafted,
            "y_train_type": np.array([s.soil_type_idx for s in train_samples]),
            "y_test_type": np.array([s.soil_type_idx for s in test_samples]),
            "y_train_health": np.array([s.health_status_idx for s in train_samples]),
            "y_test_health": np.array([s.health_status_idx for s in test_samples]),
            "y_train_shi": np.array([s.shi for s in train_samples]),
            "y_test_shi": np.array([s.shi for s in test_samples])
        }
