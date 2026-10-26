"""
Quickstart Demo Script
Instantly runs inference on a real Indian soil sample:
1. Loads real preprocessed patch from data/raw/
2. Extracts 60 handcrafted domain features (GLCM, LBP, Color Moments)
3. Evaluates through UAV-SoilHealthNet
4. Generates Grad-CAM Attention Heatmap
5. Formats ICAR Pan-India Farmer Advisory Card
Execution time: ~5 seconds.
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import cv2
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SOIL_TYPES, SOIL_HEALTH_STATUSES, MODELS_DIR, EXPLAIN_DIR, REPORTS_DIR, RAW_DATA_DIR
from src.preprocessing.preprocessor import UAVImagePreprocessor
from src.preprocessing.feature_extractor import SoilFeatureExtractor
from src.models.advanced_model import UAVSoilHealthNet
from src.models.explainability import SoilGradCAM
from src.advisory.farmer_advisory import FarmerAdvisoryEngine


def run_quickstart():
    print("=" * 75)
    print("🚀 UAV SOIL HEALTH MONITORING - REAL INDIAN SOIL QUICKSTART DEMO")
    print("   Pan-India Precision Agriculture Framework (7 Soil Classes)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Compute Device: {device}")

    # Pick a real sample from data/raw/
    sample_dir = RAW_DATA_DIR / "Alluvial_Soil"
    sample_files = sorted(list(sample_dir.glob("*.*")))
    if not sample_files:
        print("[!] No images found in data/raw/Alluvial_Soil.")
        return

    test_img_path = sample_files[0]
    print(f"[*] Loading authentic field sample: {test_img_path.name}")
    raw_bgr = cv2.imread(str(test_img_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # Preprocessing
    preproc = UAVImagePreprocessor()
    feat_extractor = SoilFeatureExtractor()
    advisory_engine = FarmerAdvisoryEngine()

    processed_patch = preproc.preprocess_patch(raw_rgb)
    hc_features = feat_extractor.extract_all_features(processed_patch)
    print(f"[+] Feature Extraction Complete: {len(hc_features)} Handcrafted Features (GLCM, LBP, Color Moments)")

    # Model Initialization & Inference
    model = UAVSoilHealthNet(
        num_soil_types=len(SOIL_TYPES),
        num_health_states=len(SOIL_HEALTH_STATUSES),
        num_handcrafted_features=60,
        pretrained_backbone=False
    ).to(device)

    model_ckpt = MODELS_DIR / "uav_soilhealthnet_best.pt"
    if model_ckpt.exists():
        print(f"[*] Loading trained weights from {model_ckpt}...")
        model.load_state_dict(torch.load(str(model_ckpt), map_location=device))
    else:
        print("[!] Note: Running with initialized model architecture (Run `python run_pipeline.py` to train full weights).")

    model.eval()

    t_img = torch.from_numpy(processed_patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    t_hc = torch.tensor(hc_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(t_img, t_hc)
        pred_type_idx = int(torch.argmax(outputs["soil_type_logits"], dim=1).item())
        pred_health_idx = int(torch.argmax(outputs["health_status_logits"], dim=1).item())
        pred_shi = float(outputs["soil_health_index"].item())
        probs = torch.softmax(outputs["soil_type_logits"], dim=1).cpu().numpy()[0]

    pred_type = SOIL_TYPES[pred_type_idx]
    pred_health = SOIL_HEALTH_STATUSES[pred_health_idx]

    print("\n--- Model Multi-Task Inference Results ---")
    print(f"  • Classified Soil Type     : {pred_type} (Confidence: {probs[pred_type_idx]*100:.1f}%)")
    print(f"  • Soil Health Status       : {pred_health}")
    print(f"  • Soil Health Index (SHI)  : {pred_shi:.1f} / 100")

    # Grad-CAM Visual Explainability
    grad_cam = SoilGradCAM(model=model, target_layer=model.deep_features[-1])
    cam_map = grad_cam.generate_cam(t_img, handcrafted_features=t_hc)
    blended_cam, _ = SoilGradCAM.overlay_heatmap(processed_patch, cam_map)

    explain_path = EXPLAIN_DIR / "quickstart_grad_cam.png"
    cv2.imwrite(str(explain_path), cv2.cvtColor(blended_cam, cv2.COLOR_RGB2BGR))
    print(f"\n[+] Grad-CAM attention heatmap generated at: {explain_path}")

    # Farmer Advisory Generation
    advisory = advisory_engine.generate_advisory(pred_type, pred_health, pred_shi, "Indo-Gangetic Basin Farmland")
    card_text = advisory_engine.format_farmer_card_text(advisory)
    print("\n" + card_text)

    print("\n[✓] Quickstart demo completed successfully!")


if __name__ == "__main__":
    run_quickstart()
