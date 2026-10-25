"""
Streamlit Web Application: SwaYaan UAV Soil Health Decision System
NIT Kurukshetra Final Year Engineering Demonstration
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SOIL_TYPES, SOIL_HEALTH_STATUSES, RAW_DATA_DIR, SAMPLES_DIR,
    MODELS_DIR, MAPS_DIR, EXPLAIN_DIR, REPORTS_DIR
)
from src.preprocessing.preprocessor import UAVImagePreprocessor
from src.preprocessing.feature_extractor import SoilFeatureExtractor
from src.advisory.farmer_advisory import FarmerAdvisoryEngine
from src.models.advanced_model import UAVSoilHealthNet
from src.models.explainability import SoilGradCAM
from src.mapping.digital_soil_mapper import DigitalSoilMapper
import torch

# Streamlit Page Config
st.set_page_config(
    page_title="UAV Soil Health AI - NIT Kurukshetra",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #475569; margin-bottom: 20px; }
    .metric-card { background: #f8fafc; padding: 18px; border-radius: 10px; border-left: 5px solid #2563eb; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .card-title { font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .card-val { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    preprocessor = UAVImagePreprocessor()
    feature_extractor = SoilFeatureExtractor()
    advisory_engine = FarmerAdvisoryEngine()
    digital_soil_mapper = DigitalSoilMapper(output_dir=MAPS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = UAVSoilHealthNet(
        num_soil_types=len(SOIL_TYPES),
        num_health_states=len(SOIL_HEALTH_STATUSES),
        num_handcrafted_features=60,
        pretrained_backbone=False
    ).to(device)

    model_ckpt = MODELS_DIR / "uav_soilhealthnet_best.pt"
    if model_ckpt.exists():
        model.load_state_dict(torch.load(str(model_ckpt), map_location=device))
    model.eval()

    return preprocessor, feature_extractor, advisory_engine, digital_soil_mapper, model, device


preprocessor, feature_extractor, advisory_engine, digital_soil_mapper, model, device = load_system()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/6f/NIT_Kurukshetra_Logo.svg/1200px-NIT_Kurukshetra_Logo.svg.png", width=110)
st.sidebar.title("SwaYaan UAV Project")
st.sidebar.markdown("**NIT Kurukshetra** | Computer Engineering")
st.sidebar.markdown("*PI:* Dr. Ritu Garg | *Co-PI:* Dr. Amandeep Kaur")
st.sidebar.divider()

nav = st.sidebar.radio(
    "Navigation Menu",
    ["🛸 Live Drone Image Diagnostics", "🗺️ Digital Field GIS Soil Map", "📊 Research Benchmark & Novelty", "🌾 Farmer Advisory Knowledge Base"]
)

# -------------------------------------------------------------------------------------------------
# TAB 1: Live Drone Image Diagnostics
# -------------------------------------------------------------------------------------------------
if nav == "🛸 Live Drone Image Diagnostics":
    st.markdown('<p class="main-title">🌾 UAV Soil Health Diagnostic Console</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Autonomous RGB Computer Vision Pipeline with Multi-Task Feature Fusion & Grad-CAM Explainability</p>', unsafe_allow_html=True)

    col_upload, col_display = st.columns([1, 2])

    with col_upload:
        st.subheader("1. Ingest Aerial UAV Image")
        uploaded_file = st.file_uploader("Upload Drone Soil Photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
        
        sample_choice = st.selectbox(
            "Or Choose a Real Benchmark Sample from Indian Field Dataset:",
            [
                "None",
                "Alluvial Soil (Indo-Gangetic Basin)",
                "Arid Soil (Thar Desert & Semi-Arid)",
                "Black Soil (Deccan Lava Plateau)",
                "Laterite Soil (Western Ghats & Coastal)",
                "Mountain Soil (Himalayan Montane Belt)",
                "Red Soil (Southern Archean Shield)",
                "Yellow Soil (Hydrated Ferric Terrain)"
            ]
        )

    # Resolve selected image
    image_rgb = None
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_img)
    elif sample_choice != "None":
        soil_dir_map = {
            "Alluvial Soil (Indo-Gangetic Basin)": RAW_DATA_DIR / "Alluvial_Soil",
            "Arid Soil (Thar Desert & Semi-Arid)": RAW_DATA_DIR / "Arid_Soil",
            "Black Soil (Deccan Lava Plateau)": RAW_DATA_DIR / "Black_Soil",
            "Laterite Soil (Western Ghats & Coastal)": RAW_DATA_DIR / "Laterite_Soil",
            "Mountain Soil (Himalayan Montane Belt)": RAW_DATA_DIR / "Mountain_Soil",
            "Red Soil (Southern Archean Shield)": RAW_DATA_DIR / "Red_Soil",
            "Yellow Soil (Hydrated Ferric Terrain)": RAW_DATA_DIR / "Yellow_Soil"
        }
        chosen_dir = soil_dir_map[sample_choice]
        if chosen_dir.exists():
            files = sorted(list(chosen_dir.glob("*.*")))
            if files:
                bgr = cv2.imread(str(files[0]))
                if bgr is not None:
                    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if image_rgb is not None:
        # Preprocessing & Inference
        patch_rgb = preprocessor.preprocess_patch(image_rgb)
        hc_feats = feature_extractor.extract_all_features(patch_rgb)
        
        t_img = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t_hc = torch.tensor(hc_feats, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(t_img, t_hc)
            pred_type_idx = int(torch.argmax(outputs["soil_type_logits"], dim=1).item())
            pred_health_idx = int(torch.argmax(outputs["health_status_logits"], dim=1).item())
            pred_shi = float(outputs["soil_health_index"].item())
            probs_type = torch.softmax(outputs["soil_type_logits"], dim=1).cpu().numpy()[0]

        pred_type_str = SOIL_TYPES[pred_type_idx]
        pred_health_str = SOIL_HEALTH_STATUSES[pred_health_idx]

        # Grad-CAM Heatmap (Hook into high-resolution 14x14 InvertedResidual stage)
        grad_cam = SoilGradCAM(model=model, target_layer=model.deep_features[12])
        cam_map = grad_cam.generate_cam(t_img, handcrafted_features=t_hc)
        blended_cam, _ = SoilGradCAM.overlay_heatmap(patch_rgb, cam_map)

        with col_display:
            st.subheader("2. Visual Analytics & Grad-CAM Attention")
            c1, c2, c3 = st.columns(3)
            c1.image(image_rgb, caption="Raw UAV Image", use_container_width=True)
            c2.image(patch_rgb, caption="Preprocessed (CLAHE + Bilateral)", use_container_width=True)
            c3.image(blended_cam, caption="Grad-CAM AI Attention Heatmap", use_container_width=True)

        st.divider()

        # Diagnostics Metrics
        st.subheader("3. Multi-Task Model Predictions")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Classified Soil Type</div>
                <div class="card-val" style="color: #2563eb;">{pred_type_str}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            status_colors = {"Healthy Soil": "#16a34a", "Moderately Healthy Soil": "#d97706", "Degraded Soil": "#dc2626"}
            s_color = status_colors.get(pred_health_str, "#2563eb")
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {s_color};">
                <div class="card-title">Soil Health Status</div>
                <div class="card-val" style="color: {s_color};">{pred_health_str}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #0284c7;">
                <div class="card-title">Soil Health Index (SHI)</div>
                <div class="card-val" style="color: #0284c7;">{pred_shi:.1f} / 100</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #7c3aed;">
                <div class="card-title">Confidence Score</div>
                <div class="card-val" style="color: #7c3aed;">{probs_type[pred_type_idx]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Advisory Card
        st.subheader("4. Actionable ICAR / CCS HAU Hisar Farmer Advisory")
        advisory = advisory_engine.generate_advisory(pred_type_str, pred_health_str, pred_shi)
        
        adv_col1, adv_col2 = st.columns([1, 1])
        with adv_col1:
            st.info(f"**Diagnostic Verdict:** {advisory['soil_health_assessment']}")
            st.markdown("**🌱 Recommended Crops for Sowing:**")
            for c in advisory["recommended_crops"]:
                st.markdown(f"- **{c}**")
            st.markdown(f"**💧 Irrigation Strategy:**\n{advisory['irrigation_plan']}")

        with adv_col2:
            st.warning("**⚠️ Immediate Site-Specific Interventions:**")
            for act in advisory["immediate_interventions"]:
                st.markdown(f"- {act}")
            st.success("**🌿 Long-term Sustainable Soil Management:**")
            for prac in advisory["sustainable_management_practices"]:
                st.markdown(f"- {prac}")

    else:
        st.info("👈 Upload an aerial image or select a benchmark sample to begin diagnosis.")

# -------------------------------------------------------------------------------------------------
# TAB 2: Digital Field GIS Soil Map
# -------------------------------------------------------------------------------------------------
elif nav == "🗺️ Digital Field GIS Soil Map":
    st.markdown('<p class="main-title">🗺️ Digital Soil Health Spatial Mapping</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">SwaYaan PoC Objective 5: Color-coded GIS Zoning for Precision Agriculture</p>', unsafe_allow_html=True)

    mode = st.radio(
        "Select Spatial Analysis Mode:",
        [
            "🌾 Pre-Stitched Indian Agro-Climatic Field Benchmarks (4 Zones)",
            "🎲 Generate Random Dynamic Field Mosaic (Live Random Sampling)",
            "📤 Upload Custom Aerial Drone Orthomosaic"
        ],
        horizontal=True
    )

    if mode == "🌾 Pre-Stitched Indian Agro-Climatic Field Benchmarks (4 Zones)":
        field_selection = st.selectbox(
            "Select Field Agro-Climatic Parcel:",
            [
                "Field 1: Kurukshetra Alluvial Basin (Haryana Wheat/Paddy Parcel)",
                "Field 2: Deccan Black Soil Parcel (Maharashtra/MP Cotton-Soybean Zone)",
                "Field 3: Thar Semi-Arid Parcel (Rajasthan Wind-Eroded Sandy Land)",
                "Field 4: Eastern Red & Laterite Plateau (Stony Acidic Soil Parcel)"
            ]
        )

        field_file_map = {
            "Field 1: Kurukshetra Alluvial Basin (Haryana Wheat/Paddy Parcel)": "Field_1_Kurukshetra_Alluvial_Wheat_Plot",
            "Field 2: Deccan Black Soil Parcel (Maharashtra/MP Cotton-Soybean Zone)": "Field_2_Deccan_Black_Soil_Cotton_Zone",
            "Field 3: Thar Semi-Arid Parcel (Rajasthan Wind-Eroded Sandy Land)": "Field_3_Rajasthan_Thar_Arid_Eroded_Parcel",
            "Field 4: Eastern Red & Laterite Plateau (Stony Acidic Soil Parcel)": "Field_4_Eastern_Red_Laterite_Plateau"
        }

        fname = field_file_map[field_selection]
        map_img_path = MAPS_DIR / f"{fname}_soil_health_map.png"
        html_map_path = MAPS_DIR / f"{fname}_interactive_map.html"

        if map_img_path.exists():
            st.image(str(map_img_path), caption=f"Aerial Orthomosaic Segmentation: Health Zoning & Continuous SHI ({field_selection})", use_container_width=True)
            st.divider()
            st.subheader("Interactive Web Spatial Grid Dashboard")
            if html_map_path.exists():
                with open(html_map_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=650, scrolling=True)
        else:
            st.warning("Digital soil map not found. Generating...")

    elif mode == "🎲 Generate Random Dynamic Field Mosaic (Live Random Sampling)":
        st.info("Picks 16 random real field soil patches across our authentic 7-class Indian dataset, stitches an aerial field mosaic with edge blending, tiles it, and executes batch ML inference on the fly.")

        grid_n = st.slider("Select Field Grid Dimension (N x N)", min_value=3, max_value=5, value=4)

        if st.button("🎲 Generate & Map Random Field from Dataset", type="primary"):
            with st.spinner("Sampling authentic field images and compiling real-time GIS map..."):
                all_images = list(RAW_DATA_DIR.glob("*/*.*"))
                import random
                sampled_files = random.sample(all_images, grid_n * grid_n)

                tile_size = 224
                mosaic = np.zeros((grid_n * tile_size, grid_n * tile_size, 3), dtype=np.uint8)

                idx = 0
                for r in range(grid_n):
                    for c in range(grid_n):
                        f = sampled_files[idx]
                        idx += 1
                        bgr = cv2.imread(str(f))
                        if bgr is not None:
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            p = preprocessor.preprocess_patch(rgb)
                            y1, y2 = r * tile_size, (r + 1) * tile_size
                            x1, x2 = c * tile_size, (c + 1) * tile_size
                            mosaic[y1:y2, x1:x2] = p

                mosaic = cv2.GaussianBlur(mosaic, (3, 3), 0)

                # Tile and infer
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
                    adv = "Optimal tilth; apply routine topdressing." if t_health == 0 else ("Moderate moisture stress; recommend 10-day watering." if t_health == 1 else "Degraded patch; apply gypsum + organic mulch.")

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

                rand_field_name = f"Random_Dynamic_Field_{grid_n}x{grid_n}"
                res = digital_soil_mapper.generate_field_map(
                    orthomosaic_rgb=mosaic,
                    grid_predictions=grid_preds,
                    field_name=rand_field_name
                )

                st.success(f"✅ Generated Dynamic {grid_n}x{grid_n} GIS Soil Health Map!")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Grid Tiles Analyzed", len(tiles))
                m2.metric("Healthy Land Zone", f"{res['healthy_percent']}%")
                m3.metric("Moderate Stress Zone", f"{res['moderate_percent']}%")
                m4.metric("Degraded Zone", f"{res['degraded_percent']}%")

                st.image(res["map_png_path"], caption="Real-Time Generated GIS Digital Soil Health Map", use_container_width=True)

                st.divider()
                st.subheader("Live Interactive Field Grid Dashboard")
                with open(res["html_path"], "r", encoding="utf-8") as f:
                    live_html = f.read()
                st.components.v1.html(live_html, height=650, scrolling=True)

    else:
        st.info("Upload any large agricultural field image, drone orthomosaic, or high-altitude aerial survey photo. The system will tile it in real-time, evaluate health across every grid cell, and compile the live zoning map.")
        uploaded_ortho = st.file_uploader("Upload Large Aerial Field Orthomosaic (JPG / PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_ortho is not None:
            pil_ortho = Image.open(uploaded_ortho).convert("RGB")
            ortho_np = np.array(pil_ortho)

            st.image(ortho_np, caption=f"Uploaded Aerial Field Imagery ({ortho_np.shape[1]}x{ortho_np.shape[0]} px)", use_container_width=True)

            if st.button("🚀 Run Live Drone Spatial Tiling & Health Mapping", type="primary"):
                with st.spinner("Executing real-time spatial tiling and batch neural network inference..."):
                    tiles = preprocessor.tile_large_aerial_image(ortho_np, tile_size=224, stride=224)

                    if not tiles:
                        tiles = [{
                            "patch": preprocessor.preprocess_patch(ortho_np),
                            "raw_patch": ortho_np,
                            "x": 0, "y": 0, "row": 0, "col": 0, "width": 224, "height": 224
                        }]

                    grid_preds = []
                    progress_bar = st.progress(0)
                    for i, t in enumerate(tiles):
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
                        adv = "Optimal tilth; apply routine topdressing." if t_health == 0 else ("Moderate moisture stress; recommend 10-day watering." if t_health == 1 else "Degraded patch; apply gypsum + organic mulch.")

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
                        progress_bar.progress((i + 1) / len(tiles))

                    field_results = digital_soil_mapper.generate_field_map(
                        orthomosaic_rgb=ortho_np,
                        grid_predictions=grid_preds,
                        field_name="Live_Uploaded_Drone_Field"
                    )

                    st.success("✅ Real-Time Spatial Mapping Complete!")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Grid Tiles Analyzed", len(tiles))
                    m2.metric("Healthy Land Zone", f"{field_results['healthy_percent']}%")
                    m3.metric("Moderate Stress Zone", f"{field_results['moderate_percent']}%")
                    m4.metric("Degraded / Eroded Zone", f"{field_results['degraded_percent']}%")

                    st.image(field_results["map_png_path"], caption="Real-Time Generated GIS Digital Soil Health Map", use_container_width=True)

                    st.divider()
                    st.subheader("Live Interactive Field Grid Dashboard")
                    with open(field_results["html_path"], "r", encoding="utf-8") as f:
                        live_html = f.read()
                    st.components.v1.html(live_html, height=650, scrolling=True)

# -------------------------------------------------------------------------------------------------
# TAB 3: Research Benchmark & Novelty
# -------------------------------------------------------------------------------------------------
elif nav == "📊 Research Benchmark & Novelty":
    st.markdown('<p class="main-title">📊 Research Benchmark & Novelty Justification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Comparative Analysis against 2023 Elsevier Base Paper & Classical ML</p>', unsafe_allow_html=True)

    chart_path = REPORTS_DIR / "comparative_model_benchmark_chart.png"
    csv_path = REPORTS_DIR / "comparative_model_benchmark.csv"

    if chart_path.exists():
        st.image(str(chart_path), caption="Comparative Performance: Accuracy, F1-Score, Precision, Recall", use_container_width=True)

    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0)
        st.subheader("Performance Metric Comparison Table")
        st.dataframe(df.style.highlight_max(axis=0, color="#bbf7d0"), use_container_width=True)

    st.divider()
    st.markdown("""
    ### 🏆 Why Our Proposed Framework is Superior to Previous Works:
    
    | Evaluation Dimension | Elsevier (2023) Base Paper | NIT Kurukshetra Proposed System |
    | :--- | :--- | :--- |
    | **Imaging Modality** | Close-up handheld mobile camera photos | Autonomous UAV (Drone) aerial flight capture |
    | **Model Architecture** | Single-Head CNN / MobileNetV2 | **Dual-Branch Cross-Attention Multi-Task Network** |
    | **Feature Representation** | Pure deep visual representations only | **Hybrid Fusion: Deep Vision + 60 Domain Handcrafted Features (GLCM, LBP, Color Moments)** |
    | **Attention Mechanism** | None | **CBAM (Convolutional Block Attention: Spatial + Channel)** |
    | **Output Scope** | Single classification (Soil Type only) | **Triple Multi-Task: Soil Type + Health Status + Continuous Soil Health Index (0-100)** |
    | **Explainability** | Black-box deep learning | **Grad-CAM Visual Heatmaps for AI verification** |
    | **Spatial GIS Mapping** | None (individual isolated images) | **Aerial Orthomosaic Grid Tiling & Interactive HTML/PNG GIS Zoning** |
    | **Farmer Decision Support** | Generic / None | **ICAR & CCS HAU Hisar Location-Specific Precision Advisory** |
    """)

# -------------------------------------------------------------------------------------------------
# TAB 4: Farmer Advisory Knowledge Base
# -------------------------------------------------------------------------------------------------
elif nav == "🌾 Farmer Advisory Knowledge Base":
    st.markdown('<p class="main-title">🌾 ICAR / CCS HAU Hisar Farmer Advisory Knowledge Base</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Standardized Agronomic Rules Aligned with Haryana Farming Conditions</p>', unsafe_allow_html=True)

    for soil_name, data in advisory_engine.rules.items():
        with st.expander(f"📍 {soil_name}", expanded=(soil_name == "Alluvial Soil")):
            st.markdown(f"**Characteristics:** {data['characteristics']}")
            st.markdown(f"**Recommended Crops:** {', '.join(data['best_crops'])}")
            st.markdown(f"**Irrigation Strategy:** {data['irrigation_strategy']}")
            st.markdown("**Sustainable Management Practices:**")
            for p in data["soil_management"]:
                st.markdown(f"- {p}")
