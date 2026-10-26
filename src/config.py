"""
Soil Health Monitoring with UAV Imaging using Machine Learning for Sustainable Farming
Pan-India Precision Agriculture Framework
Central Configuration File - 7 Indian Soil Benchmark (Elsevier EAAI Research Dataset)
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
MAPS_DIR = OUTPUTS_DIR / "maps"
EXPLAIN_DIR = OUTPUTS_DIR / "explainability"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure all output directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR, MODELS_DIR, MAPS_DIR, EXPLAIN_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# 7 Major Indian Soil Types (Elsevier EAAI 2025 Research Paper Benchmark)
SOIL_TYPES = [
    "Alluvial Soil",    # Indo-Gangetic Plains & Deltas (Punjab, Haryana, UP, Bihar, WB, Assam)
    "Arid Soil",        # Thar Desert & Arid Belt (Rajasthan, Western Haryana, Kutch/Gujarat)
    "Black Soil",       # Deccan Lava Plateau (Maharashtra, MP, Gujarat, Karnataka, AP)
    "Laterite Soil",    # High-rainfall tropical zones (Western Ghats, Kerala, Coastal Karnataka, Odisha)
    "Mountain Soil",    # Himalayan Montane & Forest Belts (J&K, Himachal, Uttarakhand, Sikkim)
    "Red Soil",         # Crystalline metamorphic zones (Tamil Nadu, Karnataka, Telangana, Odisha)
    "Yellow Soil"       # Hydrated ferric terrain (Chhattisgarh, Odisha, Chota Nagpur Plateau)
]

# Soil Health / Degradation Status Categories
SOIL_HEALTH_STATUSES = [
    "Healthy Soil",            # Optimal organic visual proxy, rich crumb texture, high aggregate stability
    "Moderately Healthy Soil", # Mild compaction/crusting, moderate moisture stress, acceptable tilth
    "Degraded Soil"            # Surface salinization/alkali crust, severe erosion, coarse sand depletion
]

# Numerical Mapping
SOIL_TYPE_TO_IDX = {name: i for i, name in enumerate(SOIL_TYPES)}
IDX_TO_SOIL_TYPE = {i: name for i, name in enumerate(SOIL_TYPES)}

SOIL_HEALTH_TO_IDX = {name: i for i, name in enumerate(SOIL_HEALTH_STATUSES)}
IDX_TO_SOIL_HEALTH = {i: name for i, name in enumerate(SOIL_HEALTH_STATUSES)}

# Folder Name Normalization Mapping
FOLDER_TO_SOIL_TYPE = {
    "alluvial_soil": "Alluvial Soil",
    "arid_soil": "Arid Soil",
    "black_soil": "Black Soil",
    "laterite_soil": "Laterite Soil",
    "mountain_soil": "Mountain Soil",
    "red_soil": "Red Soil",
    "yellow_soil": "Yellow Soil"
}

# UAV Image Acquisition & Patch Parameters
UAV_CONFIG = {
    "camera_resolution": "20MP RGB",
    "recommended_altitude_m": 25,          # 15-40 meters for high-resolution ground surface texture
    "orthomosaic_altitude_m": 60,         # Wide-field scanning
    "front_overlap_percent": 75,
    "side_overlap_percent": 70,
    "tile_size": 224,                     # Patch dimensions for CNN input
    "ground_sampling_distance_cm": 0.8    # cm per pixel at low altitude
}

# Model Training Parameters
TRAIN_CONFIG = {
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs": 15,
    "learning_rate": 2.5e-4,
    "weight_decay": 1e-4,
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42
}

# Pan-India Farmer Advisory Knowledge Base (ICAR & State Agricultural Universities Guidelines)
# Notice: Strictly avoids fake microscopic N-P-K prediction from RGB imagery.
# Actionable recommendations focus on regional crop suitability, irrigation scheduling, organic amendments,
# bio-decomposer stubble management, and physical soil restoration practices across India.
ICAR_ADVISORY_RULES = {
    "Alluvial Soil": {
        "geography": "Indo-Gangetic Plains & Coastal River Deltas (Punjab, Haryana, Uttar Pradesh, Bihar, West Bengal, Assam, Coastal AP/Odisha).",
        "characteristics": "Highly fertile, fine silty to loamy texture, transported by river basins, excellent water holding and drainage.",
        "best_crops": ["Wheat (PBW 725 / HD 3086)", "Paddy (Basmati / PR 126)", "Sugarcane", "Mustard", "Potato", "Jute"],
        "irrigation_strategy": "Moderate irrigation frequency (every 10-14 days). Border strip, furrow, or sprinkler irrigation recommended.",
        "soil_management": [
            "Practice green manuring (Dhaincha / Sunhemp) before kharif sowing to maintain organic carbon.",
            "Apply microbial bio-decomposer spray post-harvest to rapidly decompose crop stubble without burning.",
            "Ensure periodic deep ploughing every 2-3 years to shatter sub-surface tillage pans."
        ]
    },
    "Arid Soil": {
        "geography": "Western India Arid & Semi-Arid Belt (Western Rajasthan, South-Western Haryana, Kutch & Saurashtra Gujarat).",
        "characteristics": "Sandy to gravelly texture, pale golden buff color, low organic matter, highly porous, prone to intense wind erosion.",
        "best_crops": ["Pearl Millet (Bajra)", "Guar (Cluster Bean)", "Moth Bean", "Barley", "Mustard", "Date Palm"],
        "irrigation_strategy": "Micro-irrigation (Drip or Micro-sprinkler) is mandatory; flood irrigation causes 70%+ water loss to deep percolation.",
        "soil_management": [
            "Establish shelterbelts / windbreak tree rows (Khejri, Neem, Rohida) along field perimeters to mitigate soil erosion.",
            "Incorporate hydrogel polymers or biochar with Farm Yard Manure (FYM) to boost moisture retention in root zones.",
            "Practice conservation tillage or zero-tillage with residue retention to protect loose topsoil."
        ]
    },
    "Black Soil": {
        "geography": "Deccan Lava Plateau (Maharashtra, Madhya Pradesh, Gujarat, Northern Karnataka, Telangana, Andhra Pradesh).",
        "characteristics": "Clay-rich (montmorillonite), dark charcoal to black, high moisture retention, swells when wet, deep fissures when dry.",
        "best_crops": ["Cotton", "Soybean", "Gram / Chickpea", "Pigeon Pea (Arhar)", "Sorghum (Jowar)", "Sunflower"],
        "irrigation_strategy": "Low frequency, controlled volume irrigation. Drip irrigation preferred to prevent waterlogging and root asphyxiation.",
        "soil_management": [
            "Ensure proper surface field drainage (Broad Bed and Furrow - BBF system) during heavy monsoon rains.",
            "Avoid heavy machinery operations when soil is wet to prevent severe compaction.",
            "Incorporate organic compost and gypsum if surface cracking indicates sodicity."
        ]
    },
    "Laterite Soil": {
        "geography": "High-rainfall Tropical Ghats & Hills (Western Ghats, Coastal Kerala, Karnataka, Goa, Coastal Maharashtra, Assam hills).",
        "characteristics": "Intensely leached by high precipitation, rich in iron and aluminum oxides, reddish to brick-red, acidic, low organic matter.",
        "best_crops": ["Cashew nut", "Tea", "Coffee", "Rubber", "Coconut", "Areca nut", "Tapioca", "Pepper"],
        "irrigation_strategy": "Regular sprinkler or drip irrigation; avoid excessive surface flooding that exacerbates nutrient leaching.",
        "soil_management": [
            "Apply agricultural lime / dolomite (CaCO3/MgCO3) to neutralize soil acidity and mobilize locked phosphates.",
            "Heavy incorporation of organic compost, biofertilizers, and green leaf manuring.",
            "Construct bunds, contour stone barriers, and bench terracing to prevent sheet wash on sloping terrain."
        ]
    },
    "Mountain Soil": {
        "geography": "Himalayan Montane & Sub-Himalayan Zones (Jammu & Kashmir, Himachal Pradesh, Uttarakhand, Sikkim, Arunachal Pradesh).",
        "characteristics": "Coarse to gravelly texture, high humus in forested tracts, acidic to neutral, shallow topsoil on steep slope terrain.",
        "best_crops": ["Apple", "Pear", "Walnut", "Almond", "Barley", "Maize", "Off-season Vegetables", "Saffron", "Tea (Kangra/Darjeeling)"],
        "irrigation_strategy": "Gravity-fed drip systems or spring-water micro-sprinklers; avoid furrow irrigation on slopes to prevent slope destabilization.",
        "soil_management": [
            "Maintain strict contour bunding and bench terracing to prevent destructive topsoil erosion during monsoon cloudbursts.",
            "Apply well-decomposed leaf mold compost and farmyard manure to buffer pH and retain nutrients.",
            "Use organic orchard mulching with pine needles or dry grasses to conserve slope moisture."
        ]
    },
    "Red Soil": {
        "geography": "Crystalline Archean & Metamorphic Terrain (Tamil Nadu, Southern Karnataka, Telangana, Andhra Pradesh, Odisha, Jharkhand).",
        "characteristics": "Porous, friable, non-retentive structure, high ferric oxide content giving reddish hue, low nitrogen/humus content.",
        "best_crops": ["Groundnut", "Millets (Ragi, Foxtail Millet)", "Pulses", "Maize", "Castor", "Tobacco", "Mango", "Guava"],
        "irrigation_strategy": "High frequency, light volume irrigation. Drip fertigation strongly recommended due to high permeability.",
        "soil_management": [
            "Apply Farm Yard Manure (FYM) or vermicompost at 10-12 tonnes/ha to enhance soil water retention.",
            "Mulch with organic crop residue to reduce evaporative soil moisture loss.",
            "Apply agricultural lime in acidic pockets based on local soil testing."
        ]
    },
    "Yellow Soil": {
        "geography": "Eastern & Central India Hydrated Metamorphic Belts (Chhattisgarh, Western Odisha, Chota Nagpur Plateau, Parts of Bundelkhand).",
        "characteristics": "Hydrated form of ferric oxide giving a distinct yellow-ochre hue, clayey to sandy loam, moderate fertility, low organic carbon.",
        "best_crops": ["Paddy (Rice)", "Linseed", "Mustard", "Millets", "Sesame", "Lentil", "Watermelon"],
        "irrigation_strategy": "Controlled furrow or border irrigation; drip irrigation for high-value vegetable crops.",
        "soil_management": [
            "Adopt comprehensive green manuring and compost incorporation to boost depleted microbial carbon.",
            "Practice ridge and furrow planting to avoid water stagnation in low-lying clayey pockets.",
            "Apply bio-fertilizers (Azotobacter, PSB) to improve biological nutrient mobilization."
        ]
    }
}

# Soil Health Degradation Interventions
HEALTH_STATUS_ADVISORY = {
    "Healthy Soil": {
        "verdict": "Soil is in optimal condition with balanced texture, visible biological activity, and high structure stability.",
        "actions": [
            "Maintain current crop rotation (Cereal-Legume rotation).",
            "Continue standard organic amendments (compost/FYM).",
            "Optimal routine irrigation scheduling."
        ]
    },
    "Moderately Healthy Soil": {
        "verdict": "Soil displays early signs of surface compaction, moisture depletion, or mild physical structure loss.",
        "actions": [
            "Introduce a green manure crop (Dhaincha or Cowpea) in the fallow window.",
            "Increase organic matter input by 20-30% using enriched farmyard compost.",
            "Adjust irrigation timing to avoid surface baking and soil crusting.",
            "Conduct shallow aerating tillage before next planting cycle."
        ]
    },
    "Degraded Soil": {
        "verdict": "Severe degradation detected: visible saline/alkali crusting, severe compaction, or erosion gullies.",
        "actions": [
            "URGENT: Apply agricultural Gypsum (CaSO4·2H2O) if white alkali/saline crust is present, followed by heavy leaching irrigation.",
            "Implement deep subsoil ripping / chiseling to shatter hardpans and restore water infiltration.",
            "Apply bio-decomposer and heavy humic compost (15 tonnes/ha) to rebuild depleted topsoil microbial biome.",
            "Construct contour bunds or field perimeter trenches to halt ongoing topsoil runoff."
        ]
    }
}
