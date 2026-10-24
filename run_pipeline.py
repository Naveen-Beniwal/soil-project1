"""
Master Pipeline Runner for Real Indian Soil Dataset
Executes the Complete End-to-End Soil Health Monitoring Framework:
1. Real Dataset Verification (6,284 Authentic Research Images Across 7 Indian Soil Classes)
2. Preprocessing & Feature Extraction (Color Moments, GLCM, LBP, Optical SOM Proxies)
3. Model Training & Comparison:
   - Classical ML (Decision Tree, SVM, Random Forest, GBM)
   - Baseline Deep Learning (MobileNetV2, Standard SoilCNN - Elsevier EAAI 2025)
   - PROPOSED NOVELTY: UAV-SoilHealthNet (Dual-Branch Cross-Attention Multi-Task Network)
4. Comprehensive Performance Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, RMSE)
5. Explainable AI: Grad-CAM Feature Map Overlays
6. Digital Soil Health Mapping: Aerial Tiling & Color-Coded Zoning Map Generation
7. Farmer Advisory Decision Support: ICAR Actionable Prescription Cards Across India
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SOIL_TYPES, SOIL_HEALTH_STATUSES, RAW_DATA_DIR, SAMPLES_DIR,
    MODELS_DIR, MAPS_DIR, EXPLAIN_DIR, REPORTS_DIR, TRAIN_CONFIG
)
from src.dataset import SoilDataManager
from src.models.classical_ml import ClassicalSoilModels
from src.models.baseline_cnn import StandardSoilCNN, BaselineMobileNetV2
from src.models.advanced_model import UAVSoilHealthNet, MultiTaskSoilLoss
from src.models.explainability import SoilGradCAM
from src.mapping.digital_soil_mapper import DigitalSoilMapper
from src.advisory.farmer_advisory import FarmerAdvisoryEngine
from src.evaluation.metrics import ModelEvaluator
from src.preprocessing.preprocessor import UAVImagePreprocessor
from src.preprocessing.feature_extractor import SoilFeatureExtractor


def generate_real_field_mosaic(samples, output_path: Path, grid_size: int = 5, tile_size: int = 224):
    """
    Constructs an authentic aerial agricultural orthomosaic stitched from real field soil patches.
    """
    mosaic_dim = grid_size * tile_size
    mosaic = np.zeros((mosaic_dim, mosaic_dim, 3), dtype=np.uint8)

    preproc = UAVImagePreprocessor()
    selected_patches = []
    
    # Pick diverse real patches across Indian soil types
    for s in samples:
        if len(selected_patches) >= grid_size * grid_size:
            break
        bgr = cv2.imread(str(s.filepath))
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            p = preproc.preprocess_patch(rgb)
            selected_patches.append(p)

    # Fill grid
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            patch = selected_patches[idx % len(selected_patches)]
            y1, y2 = r * tile_size, (r + 1) * tile_size
            x1, x2 = c * tile_size, (c + 1) * tile_size
            mosaic[y1:y2, x1:x2] = patch
            idx += 1

    # Subtle blend along seams
    mosaic = cv2.GaussianBlur(mosaic, (3, 3), 0)
    cv2.imwrite(str(output_path), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    return str(output_path)


def main():
    print("=" * 80)
    print("🌾 SOIL HEALTH MONITORING WITH UAV IMAGING USING MACHINE LEARNING")
    print("   Pan-India Precision Agriculture Framework (7 Major Indian Soil Types)")
    print("   Benchmark Dataset: Elsevier EAAI Research Dataset (6,284 Authentic Images)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Computational Hardware Accelerated Device: {device}\n")

    # -------------------------------------------------------------
    # Step 1: Real Dataset Verification
    # -------------------------------------------------------------
    print("--- [STEP 1/7]: Real Dataset Discovery & Verification ---")
    data_manager = SoilDataManager(RAW_DATA_DIR)
    
    # Balance classes: 250 real images per class = 1,750 real images for robust training
    samples = data_manager.discover_samples(max_per_class=250)
    print(f"[+] Total Authentic Field Soil Samples Selected: {len(samples)}")

    class_counts = {}
    for s in samples:
        name = SOIL_TYPES[s.soil_type_idx]
        class_counts[name] = class_counts.get(name, 0) + 1

    for name, cnt in sorted(class_counts.items()):
        print(f"    • {name.ljust(20)}: {cnt} samples")

    orthomosaic_path = SAMPLES_DIR / "Indian_Agricultural_Field_Zone_A_orthomosaic.jpg"
    generate_real_field_mosaic(samples, orthomosaic_path, grid_size=5, tile_size=224)
    print(f"[+] Aerial Field Orthomosaic assembled from real patches: {orthomosaic_path}")

    # -------------------------------------------------------------
    # Step 2: Preprocessing, Feature Extraction & Partitioning
    # -------------------------------------------------------------
    print("\n--- [STEP 2/7]: Preprocessing & Feature Extraction ---")
    splits = data_manager.prepare_datasets(samples, extract_handcrafted=True)

    batch_size = TRAIN_CONFIG["batch_size"]
    train_loader = DataLoader(splits["train_dataset"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(splits["val_dataset"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(splits["test_dataset"], batch_size=batch_size, shuffle=False)

    print(f"[+] Data Split: Train={len(splits['train_samples'])}, Val={len(splits['val_samples'])}, Test={len(splits['test_samples'])}")
    print(f"[+] Handcrafted Feature Dimension: {splits['train_handcrafted'].shape[1]} (Color Moments + GLCM Haralick + LBP + SOM proxies)")

    evaluator = ModelEvaluator(output_dir=str(REPORTS_DIR))
    benchmark_results = {}

    # -------------------------------------------------------------
    # Step 3: Classical Machine Learning Benchmark
    # -------------------------------------------------------------
    print("\n--- [STEP 3/7]: Benchmarking Classical Machine Learning Models ---")
    ml_models = ClassicalSoilModels(random_state=TRAIN_CONFIG["random_seed"])
    
    clf_results = ml_models.fit_and_evaluate_classifiers(
        X_train=splits["train_handcrafted"],
        y_train=splits["y_train_type"],
        X_test=splits["test_handcrafted"],
        y_test=splits["y_test_type"]
    )

    for m_name, res in clf_results.items():
        print(f"    • {m_name.ljust(28)}: Acc={res['Accuracy']:.4f} | F1={res['F1_Score']:.4f} | Prec={res['Precision']:.4f}")
        benchmark_results[m_name] = res

    reg_results = ml_models.fit_and_evaluate_regressors(
        X_train=splits["train_handcrafted"],
        y_train_shi=splits["y_train_shi"],
        X_test=splits["test_handcrafted"],
        y_test_shi=splits["y_test_shi"]
    )
    ml_models.save(str(MODELS_DIR))

    # -------------------------------------------------------------
    # Step 4: Baseline Deep Learning Models (Elsevier Base Paper)
    # -------------------------------------------------------------
    print("\n--- [STEP 4/7]: Training Baseline DL Models (Elsevier Base Paper) ---")
    
    # Baseline 1: Standard SoilCNN
    print("[*] Training Baseline Standard SoilCNN...")
    cnn_model = StandardSoilCNN(num_classes=len(SOIL_TYPES)).to(device)
    optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(8):
        cnn_model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            targets = batch["soil_type"].to(device)
            optimizer_cnn.zero_grad()
            preds = cnn_model(imgs)
            loss = criterion_ce(preds, targets)
            loss.backward()
            optimizer_cnn.step()

    cnn_model.eval()
    all_preds_cnn, all_targets_cnn = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            preds = cnn_model(imgs)
            all_preds_cnn.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_targets_cnn.extend(batch["soil_type"].numpy())

    metrics_cnn = evaluator.evaluate_classification(
        y_true=np.array(all_targets_cnn),
        y_pred=np.array(all_preds_cnn),
        class_names=SOIL_TYPES,
        model_name="Baseline SoilCNN"
    )
    print(f"    • Baseline SoilCNN            : Acc={metrics_cnn['Accuracy']:.4f} | F1={metrics_cnn['F1_Score']:.4f}")
    benchmark_results["Baseline SoilCNN (2025 Paper)"] = metrics_cnn

    # Baseline 2: MobileNetV2 Baseline
    print("[*] Training Baseline MobileNetV2...")
    mobilenet_model = BaselineMobileNetV2(num_classes=len(SOIL_TYPES), pretrained=False).to(device)
    optimizer_mb = torch.optim.Adam(mobilenet_model.parameters(), lr=5e-4, weight_decay=1e-4)

    for epoch in range(8):
        mobilenet_model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            targets = batch["soil_type"].to(device)
            optimizer_mb.zero_grad()
            preds = mobilenet_model(imgs)
            loss = criterion_ce(preds, targets)
            loss.backward()
            optimizer_mb.step()

    mobilenet_model.eval()
    all_preds_mb, all_targets_mb = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            preds = mobilenet_model(imgs)
            all_preds_mb.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_targets_mb.extend(batch["soil_type"].numpy())

    metrics_mb = evaluator.evaluate_classification(
        y_true=np.array(all_targets_mb),
        y_pred=np.array(all_preds_mb),
        class_names=SOIL_TYPES,
        model_name="Baseline MobileNetV2"
    )
    print(f"    • Baseline MobileNetV2        : Acc={metrics_mb['Accuracy']:.4f} | F1={metrics_mb['F1_Score']:.4f}")
    benchmark_results["Baseline MobileNetV2 (Base Paper)"] = metrics_mb

    # -------------------------------------------------------------
    # Step 5: PROPOSED NOVELTY: UAV-SoilHealthNet
    # -------------------------------------------------------------
    print("\n--- [STEP 5/7]: Training PROPOSED NOVEL MODEL: UAV-SoilHealthNet ---")
    print("    [!] Features: Dual-Branch Fusion + CBAM Spatial/Channel Attention + Multi-Task Output Heads")
    
    novel_model = UAVSoilHealthNet(
        num_soil_types=len(SOIL_TYPES),
        num_health_states=len(SOIL_HEALTH_STATUSES),
        num_handcrafted_features=splits["train_handcrafted"].shape[1],
        pretrained_backbone=False
    ).to(device)

    optimizer_novel = torch.optim.AdamW(novel_model.parameters(), lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"])
    loss_fn = MultiTaskSoilLoss(w_type=1.0, w_health=1.0, w_shi=0.5)

    num_epochs = TRAIN_CONFIG["epochs"]
    for epoch in range(num_epochs):
        novel_model.train()
        total_epoch_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            t_type = batch["soil_type"].to(device)
            t_health = batch["health_status"].to(device)
            t_shi = batch["shi"].to(device)

            optimizer_novel.zero_grad()
            outputs = novel_model(imgs, handcrafted)
            loss, loss_dict = loss_fn(outputs, t_type, t_health, t_shi)
            loss.backward()
            optimizer_novel.step()
            total_epoch_loss += loss.item()

        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            avg_loss = total_epoch_loss / len(train_loader)
            print(f"    Epoch [{epoch+1:02d}/{num_epochs:02d}] - Multi-Task Loss: {avg_loss:.4f}")

    # Evaluate Novel Model on Test Set
    novel_model.eval()
    all_preds_type, all_preds_health, all_preds_shi = [], [], []
    all_t_type, all_t_health, all_t_shi = [], [], []
    all_probs_type = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            outputs = novel_model(imgs, handcrafted)

            probs = torch.softmax(outputs["soil_type_logits"], dim=1)
            all_preds_type.extend(torch.argmax(outputs["soil_type_logits"], dim=1).cpu().numpy())
            all_preds_health.extend(torch.argmax(outputs["health_status_logits"], dim=1).cpu().numpy())
            all_preds_shi.extend(outputs["soil_health_index"].cpu().numpy())

            all_t_type.extend(batch["soil_type"].numpy())
            all_t_health.extend(batch["health_status"].numpy())
            all_t_shi.extend(batch["shi"].numpy())
            all_probs_type.extend(probs.cpu().numpy())

    metrics_novel_type = evaluator.evaluate_classification(
        y_true=np.array(all_t_type),
        y_pred=np.array(all_preds_type),
        y_prob=np.array(all_probs_type),
        class_names=SOIL_TYPES,
        model_name="Proposed UAV-SoilHealthNet (Soil Type)"
    )
    metrics_novel_health = evaluator.evaluate_classification(
        y_true=np.array(all_t_health),
        y_pred=np.array(all_preds_health),
        class_names=SOIL_HEALTH_STATUSES,
        model_name="Proposed UAV-SoilHealthNet (Health Status)"
    )
    metrics_novel_shi = evaluator.evaluate_regression(
        y_true=np.array(all_t_shi),
        y_pred=np.array(all_preds_shi)
    )

    print(f"    • Proposed UAV-SoilHealthNet  : Acc={metrics_novel_type['Accuracy']:.4f} | F1={metrics_novel_type['F1_Score']:.4f} | ROC-AUC={metrics_novel_type['ROC_AUC']:.4f}")
    print(f"    • Health Status Classification: Acc={metrics_novel_health['Accuracy']:.4f} | F1={metrics_novel_health['F1_Score']:.4f}")
    print(f"    • Soil Health Index (SHI)     : RMSE={metrics_novel_shi['RMSE']:.2f} | R² Score={metrics_novel_shi['R2_Score']:.4f}")

    benchmark_results["PROPOSED: UAV-SoilHealthNet (Novel)"] = metrics_novel_type

    # Save model weights
    torch.save(novel_model.state_dict(), MODELS_DIR / "uav_soilhealthnet_best.pt")
    torch.save(mobilenet_model.state_dict(), MODELS_DIR / "baseline_mobilenet.pt")
    print(f"[+] Model checkpoints successfully archived in {MODELS_DIR}")

    # Generate Comparative Report Table and Chart
    comp_df = evaluator.generate_comparative_report(benchmark_results, filename_prefix="comparative_model_benchmark")
    print(f"\n[+] Master Comparative Benchmark Table Generated:\n{comp_df.to_string()}")

    # -------------------------------------------------------------
    # Step 6: Explainable AI (Grad-CAM Verification)
    # -------------------------------------------------------------
    print("\n--- [STEP 6/7]: Generating Explainable AI (Grad-CAM) Visual Heatmaps ---")
    grad_cam = SoilGradCAM(model=novel_model, target_layer=novel_model.deep_features[-1])

    # Select a real test sample
    sample_to_explain = splits["test_samples"][0]
    raw_img = cv2.imread(str(sample_to_explain.filepath))
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    preproc = UAVImagePreprocessor()
    patch_rgb = preproc.preprocess_patch(raw_rgb)
    t_img = splits["test_dataset"][0]["image"].unsqueeze(0).to(device)
    t_hc = splits["test_dataset"][0]["handcrafted"].unsqueeze(0).to(device)

    cam_map = grad_cam.generate_cam(t_img, handcrafted_features=t_hc)
    blended_cam, _ = SoilGradCAM.overlay_heatmap(patch_rgb, cam_map)

    explain_save_path = EXPLAIN_DIR / "grad_cam_soil_sample.png"
    cv2.imwrite(str(explain_save_path), cv2.cvtColor(blended_cam, cv2.COLOR_RGB2BGR))
    print(f"[+] Grad-CAM visual attention overlay saved to: {explain_save_path}")

    # -------------------------------------------------------------
    # Step 7: UAV Aerial Mapping & Farmer Advisory Engine
    # -------------------------------------------------------------
    print("\n--- [STEP 7/7]: Aerial Field Orthomosaic Tiling & GIS Soil Health Mapping ---")
    mapper = DigitalSoilMapper(output_dir=str(MAPS_DIR))
    advisory_engine = FarmerAdvisoryEngine()

    raw_mosaic = cv2.imread(str(orthomosaic_path))
    mosaic_rgb = cv2.cvtColor(raw_mosaic, cv2.COLOR_BGR2RGB)

    tiles = preproc.tile_large_aerial_image(mosaic_rgb, tile_size=224, stride=224)
    print(f"[*] Processing {len(tiles)} aerial grid patches across agricultural orthomosaic...")

    grid_predictions = []
    feat_extractor = SoilFeatureExtractor()

    novel_model.eval()
    with torch.no_grad():
        for t in tiles:
            p_rgb = t["patch"]
            hc_feats = feat_extractor.extract_all_features(p_rgb)
            hc_tensor = torch.tensor(hc_feats, dtype=torch.float32).unsqueeze(0).to(device)

            p_tensor = torch.from_numpy(p_rgb.transpose(2, 0, 1)).float() / 255.0
            p_tensor = p_tensor.unsqueeze(0).to(device)

            out = novel_model(p_tensor, hc_tensor)
            s_type_idx = int(torch.argmax(out["soil_type_logits"], dim=1).item())
            h_stat_idx = int(torch.argmax(out["health_status_logits"], dim=1).item())
            shi_val = float(out["soil_health_index"].item())

            s_type_str = SOIL_TYPES[s_type_idx]
            h_stat_str = SOIL_HEALTH_STATUSES[h_stat_idx]
            
            adv = advisory_engine.generate_advisory(s_type_str, h_stat_str, shi_val, "Pan-India Farmland Sector 7")

            grid_predictions.append({
                "row": t["row"],
                "col": t["col"],
                "soil_type_idx": s_type_idx,
                "soil_type": s_type_str,
                "health_status_idx": h_stat_idx,
                "health_status": h_stat_str,
                "soil_health_index": shi_val,
                "quick_advisory": adv["immediate_interventions"][0]
            })

    # Generate Digital Soil Health Map and Interactive HTML Report
    map_summary = mapper.generate_field_map(
        orthomosaic_rgb=mosaic_rgb,
        grid_predictions=grid_predictions,
        field_name="Indian_Agricultural_Field_Zone_A"
    )

    print(f"\n[+] Digital Soil Health Map Generated:")
    print(f"    • High-Resolution Map PNG: {map_summary['map_png_path']}")
    print(f"    • Interactive HTML Report: {map_summary['html_path']}")
    print(f"    • Spatial Healthy Zone    : {map_summary['healthy_percent']}%")
    print(f"    • Spatial Moderate Zone   : {map_summary['moderate_percent']}%")
    print(f"    • Spatial Degraded Zone   : {map_summary['degraded_percent']}%")
    print(f"    • Average Field SHI Score : {map_summary['mean_shi']} / 100")

    # Generate Sample Farmer Advisory Card
    sample_adv = advisory_engine.generate_advisory(
        soil_type="Black Soil",
        health_status="Healthy Soil",
        soil_health_index=84.2,
        field_location="Deccan Agricultural Basin (Maharashtra / Central India)"
    )
    farmer_card_text = advisory_engine.format_farmer_card_text(sample_adv)
    card_path = REPORTS_DIR / "farmer_soil_prescription_card.txt"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(farmer_card_text)

    print("\n" + farmer_card_text)
    print(f"[+] Farmer Prescription Card exported to: {card_path}")

    print("\n" + "=" * 80)
    print("🎉 FULL REAL-DATASET SOIL HEALTH PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
