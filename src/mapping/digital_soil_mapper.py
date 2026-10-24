"""
Digital Soil Health Mapping (GIS Spatial Simulation)
Implements SwaYaan PoC Objective 5:
Generates color-coded spatial zoning maps (Healthy, Moderately Healthy, Degraded zones),
continuous Soil Health Index (SHI) heatmaps, and area-level spatial variability statistics.
Exports high-resolution visual maps and interactive HTML reports.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from pathlib import Path


class DigitalSoilMapper:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_field_map(
        self,
        orthomosaic_rgb: np.ndarray,
        grid_predictions: List[Dict[str, Any]],
        field_name: str = "Indian_Agricultural_Field_Zone_A"
    ) -> Dict[str, Any]:
        """
        Creates color-coded digital soil health maps from tiled UAV grid predictions.
        """
        h_orig, w_orig, _ = orthomosaic_rgb.shape

        # Find grid dimensions
        max_row = max(t["row"] for t in grid_predictions) + 1
        max_col = max(t["col"] for t in grid_predictions) + 1

        health_grid = np.zeros((max_row, max_col), dtype=np.int32)
        shi_grid = np.zeros((max_row, max_col), dtype=np.float32)
        type_grid = np.zeros((max_row, max_col), dtype=np.int32)

        for t in grid_predictions:
            r, c = t["row"], t["col"]
            health_grid[r, c] = t["health_status_idx"]
            shi_grid[r, c] = t["soil_health_index"]
            type_grid[r, c] = t.get("soil_type_idx", 0)

        # Calculate Area Statistics
        total_tiles = len(grid_predictions)
        healthy_count = sum(1 for t in grid_predictions if t["health_status_idx"] == 0)
        moderate_count = sum(1 for t in grid_predictions if t["health_status_idx"] == 1)
        degraded_count = sum(1 for t in grid_predictions if t["health_status_idx"] == 2)

        healthy_pct = (healthy_count / total_tiles) * 100.0 if total_tiles > 0 else 0.0
        moderate_pct = (moderate_count / total_tiles) * 100.0 if total_tiles > 0 else 0.0
        degraded_pct = (degraded_count / total_tiles) * 100.0 if total_tiles > 0 else 0.0
        mean_shi = float(np.mean(shi_grid[shi_grid > 0])) if np.any(shi_grid > 0) else 0.0

        # Generate publication-quality 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        plt.subplots_adjust(wspace=0.3)

        # Panel 1: Original UAV Field Imagery
        axes[0].imshow(orthomosaic_rgb)
        axes[0].set_title(f"UAV RGB Aerial Imagery\n({field_name})", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Color-Coded Soil Health Zoning Map
        # 0: Healthy (Forest Green), 1: Moderate (Amber Gold), 2: Degraded (Crimson Red)
        from matplotlib.colors import ListedColormap
        cmap_health = ListedColormap(["#2ecc71", "#f39c12", "#e74c3c"])
        im1 = axes[1].imshow(health_grid, cmap=cmap_health, vmin=0, vmax=2, interpolation="nearest")
        axes[1].set_title("Digital Soil Health Zoning Map\n(Green: Healthy | Orange: Moderate | Red: Degraded)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Field Grid X")
        axes[1].set_ylabel("Field Grid Y")

        # Colorbar for Health status
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
        cbar1.ax.set_yticklabels(["Healthy", "Moderate", "Degraded"])

        # Panel 3: Continuous Soil Health Index (SHI: 0-100)
        im2 = axes[2].imshow(shi_grid, cmap="RdYlGn", vmin=20, vmax=100, interpolation="bilinear")
        axes[2].set_title("Continuous Soil Health Index (SHI)\n(Score 0 - 100)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Field Grid X")
        axes[2].set_ylabel("Field Grid Y")
        cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.set_label("SHI Score (0 - 100)", fontsize=10)

        map_png_path = self.output_dir / f"{field_name}_soil_health_map.png"
        fig.savefig(map_png_path, bbox_inches="tight")
        plt.close(fig)

        # Generate Interactive HTML Report
        html_path = self.output_dir / f"{field_name}_interactive_map.html"
        self._generate_interactive_html(
            html_path=html_path,
            field_name=field_name,
            healthy_pct=healthy_pct,
            moderate_pct=moderate_pct,
            degraded_pct=degraded_pct,
            mean_shi=mean_shi,
            grid_predictions=grid_predictions,
            max_row=max_row,
            max_col=max_col
        )

        return {
            "field_name": field_name,
            "map_png_path": str(map_png_path),
            "html_path": str(html_path),
            "healthy_percent": round(healthy_pct, 2),
            "moderate_percent": round(moderate_pct, 2),
            "degraded_percent": round(degraded_pct, 2),
            "mean_shi": round(mean_shi, 2),
            "total_patches_analyzed": total_tiles
        }

    def _generate_interactive_html(
        self,
        html_path: Path,
        field_name: str,
        healthy_pct: float,
        moderate_pct: float,
        degraded_pct: float,
        mean_shi: float,
        grid_predictions: List[Dict[str, Any]],
        max_row: int,
        max_col: int
    ):
        """Builds a standalone interactive HTML dashboard with SVG spatial grid and tooltips."""
        tile_map = {(t["row"], t["col"]): t for t in grid_predictions}
        
        cells_html = []
        cell_size = 35
        svg_w = max_col * (cell_size + 4)
        svg_h = max_row * (cell_size + 4)

        color_map = {
            0: "#27ae60",  # Healthy Green
            1: "#e67e22",  # Moderate Orange
            2: "#c0392b"   # Degraded Red
        }

        for r in range(max_row):
            for c in range(max_col):
                t = tile_map.get((r, c))
                if t:
                    status_idx = t["health_status_idx"]
                    color = color_map.get(status_idx, "#95a5a6")
                    status_str = t["health_status"]
                    shi_val = round(t["soil_health_index"], 1)
                    soil_type = t.get("soil_type", "Alluvial Soil")
                    advisory = t.get("quick_advisory", "Standard ICAR management")
                    tooltip = f"Grid ({r},{c}) | Type: {soil_type} | Health: {status_str} | SHI: {shi_val}/100 | Action: {advisory}"
                    
                    x = c * (cell_size + 4)
                    y = r * (cell_size + 4)
                    cells_html.append(
                        f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" rx="4" fill="{color}" '
                        f'stroke="#2c3e50" stroke-width="1.5" '
                        f'data-tooltip="{tooltip}" class="grid-cell" '
                        f'onclick="showTileDetail(\'{soil_type}\', \'{status_str}\', {shi_val}, \'{advisory}\', {r}, {c})">'
                        f'<title>{tooltip}</title></rect>'
                    )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Digital Soil Health Map - {field_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .header {{ text-align: center; margin-bottom: 24px; }}
        .header h1 {{ color: #38bdf8; margin-bottom: 4px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
        .stat-card {{ background: #1e293b; padding: 16px; border-radius: 8px; border-left: 4px solid #38bdf8; text-align: center; }}
        .stat-value {{ font-size: 26px; font-weight: bold; margin-top: 6px; }}
        .healthy {{ border-left-color: #22c55e; color: #22c55e; }}
        .moderate {{ border-left-color: #f59e0b; color: #f59e0b; }}
        .degraded {{ border-left-color: #ef4444; color: #ef4444; }}
        .main-container {{ display: flex; gap: 24px; justify-content: center; flex-wrap: wrap; }}
        .map-wrapper {{ background: #1e293b; padding: 20px; border-radius: 12px; }}
        .inspector-wrapper {{ background: #1e293b; padding: 20px; border-radius: 12px; width: 340px; }}
        .grid-cell {{ cursor: pointer; transition: transform 0.15s ease; }}
        .grid-cell:hover {{ stroke: #ffffff; stroke-width: 2.5; transform: scale(1.08); }}
        .legend {{ display: flex; gap: 20px; justify-content: center; margin-top: 14px; font-size: 14px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-box {{ width: 16px; height: 16px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 SwaYaan UAV Digital Soil Health Spatial Map</h1>
        <p>Pan-India Precision Agriculture Decision Framework | Field: <b>{field_name}</b></p>
    </div>

    <div class="stats-grid">
        <div class="stat-card healthy">
            <div>Healthy Soil Zone</div>
            <div class="stat-value">{healthy_pct:.1f}%</div>
        </div>
        <div class="stat-card moderate">
            <div>Moderately Healthy Zone</div>
            <div class="stat-value">{moderate_pct:.1f}%</div>
        </div>
        <div class="stat-card degraded">
            <div>Degraded / Vulnerable Zone</div>
            <div class="stat-value">{degraded_pct:.1f}%</div>
        </div>
        <div class="stat-card">
            <div>Mean Field Health Index</div>
            <div class="stat-value" style="color: #38bdf8;">{mean_shi:.1f} / 100</div>
        </div>
    </div>

    <div class="main-container">
        <div class="map-wrapper">
            <h3>Field Spatial Grid (Click any block for localized prescription)</h3>
            <svg width="{svg_w}" height="{svg_h}">
                {''.join(cells_html)}
            </svg>
            <div class="legend">
                <div class="legend-item"><div class="legend-box" style="background:#27ae60;"></div>Healthy Soil</div>
                <div class="legend-item"><div class="legend-box" style="background:#e67e22;"></div>Moderately Healthy</div>
                <div class="legend-item"><div class="legend-box" style="background:#c0392b;"></div>Degraded Soil</div>
            </div>
        </div>

        <div class="inspector-wrapper" id="inspector">
            <h3>📍 Tile Inspector</h3>
            <p style="color: #94a3b8; font-size: 14px;">Click on any grid tile to load site-specific drone diagnostics and farmer advisory.</p>
            <div id="inspector-content" style="display:none; margin-top: 16px;">
                <p><b>Coordinates:</b> Grid Row <span id="insp-row"></span>, Col <span id="insp-col"></span></p>
                <p><b>Classified Type:</b> <span id="insp-type" style="color:#38bdf8; font-weight:bold;"></span></p>
                <p><b>Health Status:</b> <span id="insp-status" style="font-weight:bold;"></span></p>
                <p><b>Soil Health Index (SHI):</b> <span id="insp-shi" style="font-weight:bold;"></span> / 100</p>
                <hr style="border:0.5px solid #334155; margin: 14px 0;">
                <p><b>Site-Specific Advisory:</b></p>
                <div id="insp-advisory" style="background:#0f172a; padding:12px; border-radius:6px; font-size:13px; line-height:1.5;"></div>
            </div>
        </div>
    </div>

    <script>
        function showTileDetail(type, status, shi, advisory, row, col) {{
            document.getElementById('inspector-content').style.display = 'block';
            document.getElementById('insp-row').innerText = row;
            document.getElementById('insp-col').innerText = col;
            document.getElementById('insp-type').innerText = type;
            
            const statusEl = document.getElementById('insp-status');
            statusEl.innerText = status;
            if (status.includes('Healthy')) statusEl.style.color = '#22c55e';
            else if (status.includes('Moderate')) statusEl.style.color = '#f59e0b';
            else statusEl.style.color = '#ef4444';

            document.getElementById('insp-shi').innerText = shi;
            document.getElementById('insp-advisory').innerText = advisory;
        }}
    </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
