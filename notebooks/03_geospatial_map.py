import pandas as pd
import folium
# LayerControl moved to the main folium import
from folium import LayerControl
from folium.plugins import MarkerCluster, Fullscreen
import os
import sys
from pathlib import Path

# Anchor to the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.logger import get_logger

logger = get_logger("GeospatialMapper")


def generate_patriotic_map():
    logger.info("🇬🇭 Generating National Infrastructure Dashboard...")

    data_path = project_root / "data" / "schools_priority_ranked.csv"
    output_folder = project_root / "outputs"

    if not data_path.exists():
        logger.error(f"❌ DATA MISSING: Run 02_priority_scoring.py first.")
        return

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    # 1. UI: Set up the map centered on Ghana with strict bounds
    ghana_center = [7.9465, -1.0232]
    ghana_map = folium.Map(
        location=ghana_center,
        zoom_start=7,
        tiles=None,  # Manual tile management
        min_zoom=7,
        max_bounds=True,
        min_lat=4.0, max_lat=11.5,
        min_lon=-4.0, max_lon=2.0
    )

    # Base layers
    folium.TileLayer('CartoDB Positron', name="Clean Map").add_to(ghana_map)
    folium.TileLayer(
        'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name="Satellite View"
    ).add_to(ghana_map)

    Fullscreen(position='topright').add_to(ghana_map)
    marker_cluster = MarkerCluster(name="Infrastructure Priority").add_to(ghana_map)

    # 2. UI/UX: Apply Ghana National Palette
    logger.info("📍 Plotting markers with Red, Gold, and Green logic...")
    for _, row in df.iterrows():
        lat, lon = row.get('latitude'), row.get('longitude')
        if pd.isna(lat) or pd.isna(lon):
            continue

        score = row.get('priority_score', 0)

        # Mapping colors to Ghana's Flag
        if score > 0.8:
            color, label = '#FF0000', "CRITICAL NEED (RED)"
        elif score > 0.5:
            color, label = '#FFD700', "HIGH PRIORITY (GOLD)"
        else:
            color, label = '#006B3F', "STABLE (GREEN)"

        popup_html = f"""
        <div style="font-family: 'Arial'; width: 220px; border-top: 5px solid {color}; padding: 10px;">
            <h4 style="margin:0;">★ {row['school_name']}</h4>
            <p style="font-size:12px; color: #555; margin-top:5px;">{row['district']} District</p>
            <hr style="border: 0.5px solid #eee;">
            <div style="background:{color}; color:white; padding:8px; text-align:center; font-weight:bold; border-radius:4px;">
                {label}<br>{round(score * 100, 1)}% GAP
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            popup=folium.Popup(popup_html, max_width=250),
            color='black',  # The Black Star accent
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85
        ).add_to(marker_cluster)

    # 3. Add Layer Control (Standard Folium Component)
    LayerControl().add_to(ghana_map)

    # Save to outputs
    output_folder.mkdir(parents=True, exist_ok=True)
    final_map_path = output_folder / "ghana_infrastructure_map.html"

    ghana_map.save(str(final_map_path))
    logger.info(f"✅ SUCCESS: National Dashboard generated at {final_map_path}")
    print(f"\n🇬🇭 READY: Open {final_map_path} in your browser.")


if __name__ == "__main__":
    generate_patriotic_map()