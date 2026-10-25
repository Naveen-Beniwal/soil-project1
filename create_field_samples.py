"""
Generate 4 Pre-stitched Diverse Indian Agricultural Field Samples
for SwaYaan PoC Objective 5 GIS Spatial Soil Mapping.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path("/Users/nakul/Desktop/soil project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, MAPS_DIR, MODELS_DIR, SOIL_TYPES, SOIL_HEALTH_STATUSES
from src.preprocessing.preprocessor import UAVImagePreprocessor
from src.preprocessing.feature_extractor import SoilFeatureExtractor
from src.models.advanced_model import UAVSoilHealthNet
from src.mapping.digital_soil_mapper import DigitalSoilMapper


def build_and_map_fields():
    print("=" * 70)
    print("🚀 GENERATING 4 PRE-STITCHED INDIAN FIELD GIS MOSAICS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    preprocessor = UAVImagePreprocessor()
    feature_extractor = SoilFeatureExtractor()
    mapper = DigitalSoilMapper(output_dir=MAPS_DIR)

    # Load Model
    model = UAVSoilHealthNet(
        num_soil_types=len(SOIL_TYPES),
        num_health_states=len(SOIL_HEALTH_STATUSES),
        num_handcrafted_features=60,
        pretrained_backbone=False
    ).to(device)
    model.load_state_dict(torch.load(str(MODELS_DIR / "uav_soilhealthnet_best.pt"), map_location=device))
    model.eval()

    field_configs = [
        {
            "name": "Field_1_Kurukshetra_Alluvial_Wheat_Plot",
            "title": "Kurukshetra Alluvial Basin (Haryana Wheat/Paddy Parcel)",
            "primary": "Alluvial_Soil",
            "secondary": ["Alluvial_Soil", "Arid_Soil"],
            "grid_size": 4
        },
        {
            "name": "Field_2_Deccan_Black_Soil_Cotton_Zone",
            "title": "Deccan Black Soil Parcel (Maharashtra/MP Cotton Zone)",
            "primary": "Black_Soil",
            "secondary": ["Black_Soil", "Alluvial_Soil"],
            "grid_size": 4
        },
        {
            "name": "Field_3_Rajasthan_Thar_Arid_Eroded_Parcel",
            "title": "Thar Semi-Arid Parcel (Rajasthan Wind-Eroded Sandy Land)",
            "primary": "Arid_Soil",
            "secondary": ["Arid_Soil", "Yellow_Soil"],
            "grid_size": 4
        },
        {
            "name": "Field_4_Eastern_Red_Laterite_Plateau",
            "title": "Eastern Red & Laterite Plateau (Stony Acidic Soil Parcel)",
            "primary": "Red_Soil",
            "secondary": ["Red_Soil", "Laterite_Soil"],
            "grid_size": 4
        }
    ]

    for cfg in field_configs:
        fname = cfg["name"]
        print(f"\n[*] Processing: {cfg['title']} ({fname})")

        # Gather real images
        primary_dir = RAW_DATA_DIR / cfg["primary"]
        p_files = sorted(list(primary_dir.glob("*.*")))
        sec_files = []
        for s in cfg["secondary"]:
            s_dir = RAW_DATA_DIR / s
            sec_files.extend(list(s_dir.glob("*.*")))

        grid_n = cfg["grid_size"]
        total_tiles = grid_n * grid_n
        tile_size = 224
        mosaic = np.zeros((grid_n * tile_size, grid_n * tile_size, 3), dtype=np.uint8)

        # Fill mosaic
        idx = 0
        for r in range(grid_n):
            for c in range(grid_n):
                # 75% primary soil, 25% secondary/variation
                if (r + c) % 4 == 0 and sec_files:
                    f = sec_files[(idx * 3) % len(sec_files)]
                else:
                    f = p_files[idx % len(p_files)]
                idx += 1

                bgr = cv2.imread(str(f))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                patch = preprocessor.preprocess_patch(rgb)

                y1, y2 = r * tile_size, (r + 1) * tile_size
                x1, x2 = c * tile_size, (c + 1) * tile_size
                mosaic[y1:y2, x1:x2] = patch

        # Smooth boundary seams subtly
        mosaic = cv2.GaussianBlur(mosaic, (3, 3), 0)

        # Save the stitched raw mosaic
        raw_mosaic_path = MAPS_DIR / f"{fname}_raw_mosaic.jpg"
        cv2.imwrite(str(raw_mosaic_path), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))

        # Run dynamic tiling & ML inference
        tiles = preprocessor.tile_large_aerial_image(mosaic, tile_size=tile_size, stride=tile_size)
        grid_preds = []

        for t in tiles:
            p_rgb = t["patch"]
            hc = feature_extractor.extract_all_features(p_rgb)
            t_img = torch.from_numpy(p_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            t_hc = torch.tensor(hc, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outs = model(t_img, t_hc)
                t_type = int(torch.argmax(outs["soil_type_logits"], dim=1).item())
                t_health = int(torch.argmax(outs["health_status_logits"], dim=1).item())
                t_shi = float(outs["soil_health_index"].item())

            s_type = SOIL_TYPES[t_type]
            s_health = SOIL_HEALTH_STATUSES[t_health]
            if t_health == 0:
                adv = "Optimal structure. Proceed with routine nitrogen/organic top-dressing."
            elif t_health == 1:
                adv = "Mild moisture/crusting stress. Recommend 10-14 day irrigation & FYM."
            else:
                adv = "Degraded / eroded patch. Apply gypsum + green manure remediation."

            grid_preds.append({
                "row": t["row"],
                "col": t["col"],
                "x": t["x"],
                "y": t["y"],
                "soil_type_idx": t_type,
                "soil_type": s_type,
                "health_status_idx": t_health,
                "health_status": s_health,
                "soil_health_index": t_shi,
                "quick_advisory": adv
            })

        # Generate GIS Map & HTML
        res = mapper.generate_field_map(
            orthomosaic_rgb=mosaic,
            grid_predictions=grid_preds,
            field_name=fname
        )
        print(f"  [+] Complete: {res['healthy_percent']}% Healthy, {res['moderate_percent']}% Moderate, {res['degraded_percent']}% Degraded | Mean SHI: {res['mean_shi']}")

    print("\n[✓] All 4 diverse field samples successfully created and mapped in outputs/maps/!")


if __name__ == "__main__":
    build_and_map_fields()
