"""
Farmer Decision Support System & Advisory Engine
Aligned with SwaYaan PoC Objective 5 & ICAR / CCS HAU Hisar Guidelines.
Generates actionable agronomic recommendations based on UAV soil classification and degradation status.
NOTE: Adheres strictly to scientific validation—does not fabricate microscopic chemical NPK values from RGB.
"""

from typing import Dict, Any, List
from src.config import ICAR_ADVISORY_RULES, HEALTH_STATUS_ADVISORY


class FarmerAdvisoryEngine:
    def __init__(self):
        self.rules = ICAR_ADVISORY_RULES
        self.health_rules = HEALTH_STATUS_ADVISORY

    def generate_advisory(
        self,
        soil_type: str,
        health_status: str,
        soil_health_index: float,
        field_location: str = "Indian Agricultural Farmland"
    ) -> Dict[str, Any]:
        """
        Produces a comprehensive agronomic prescription card for the farmer.
        """
        type_info = self.rules.get(soil_type, self.rules["Alluvial Soil"])
        health_info = self.health_rules.get(health_status, self.health_rules["Healthy Soil"])

        # Construct actionable farmer prescription card
        prescription = {
            "location": field_location,
            "soil_type": soil_type,
            "health_status": health_status,
            "soil_health_index": round(soil_health_index, 1),
            "regional_geography": type_info.get("geography", "All India"),
            "soil_characteristics": type_info["characteristics"],
            "recommended_crops": type_info["best_crops"],
            "irrigation_plan": type_info["irrigation_strategy"],
            "soil_health_assessment": health_info["verdict"],
            "immediate_interventions": health_info["actions"],
            "sustainable_management_practices": type_info["soil_management"]
        }
        return prescription

    def format_farmer_card_text(self, advisory: Dict[str, Any]) -> str:
        """Formats the prescription into a readable printable bulletin for extension officers and farmers."""
        card = []
        card.append("=" * 70)
        card.append("🌾 ICAR-COMPLIANT PAN-INDIA PRECISION SOIL HEALTH PRESCRIPTION CARD")
        card.append("   UAV SwaYaan Precision Agriculture Decision Support Engine")
        card.append("=" * 70)
        card.append(f"Field Location      : {advisory['location']}")
        card.append(f"Classified Soil     : {advisory['soil_type']}")
        card.append(f"Agro-Climatic Belt  : {advisory.get('regional_geography', 'Pan-India')}")
        card.append(f"Health Status       : {advisory['health_status']}")
        card.append(f"Soil Health Index   : {advisory['soil_health_index']} / 100")
        card.append("-" * 70)
        card.append("🌱 RECOMMENDED CROPS FOR OPTIMAL YIELD:")
        for crop in advisory["recommended_crops"]:
            card.append(f"   • {crop}")
        card.append("-" * 70)
        card.append("💧 IRRIGATION SCHEDULING & WATER CONSERVATION:")
        card.append(f"   {advisory['irrigation_plan']}")
        card.append("-" * 70)
        card.append("⚠️ DIAGNOSTIC STATUS & TARGETED INTERVENTIONS:")
        card.append(f"   Verdict: {advisory['soil_health_assessment']}")
        card.append("   Action Steps:")
        for act in advisory["immediate_interventions"]:
            card.append(f"   [!] {act}")
        card.append("-" * 70)
        card.append("🌿 SUSTAINABLE SOIL REGENERATION PRACTICES:")
        for prac in advisory["sustainable_management_practices"]:
            card.append(f"   ✓ {prac}")
        card.append("=" * 70)
        return "\n".join(card)
