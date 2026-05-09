import pandas as pd
import folium
from folium.plugins import MarkerCluster, Fullscreen
from sklearn.cluster import DBSCAN
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.logger import get_logger

logger = get_logger("EnterpriseMapper")


def run_enterprise_mapping():
    logger.info("🌍 Building Two-Layer Enterprise Dashboard...")

    data_path = ROOT / "data" / "schools_priority_ranked.csv"
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    # Initialize Map centered on Ghana
    m = folium.Map(location=[7.9465, -1.0232], zoom_start=7, tiles='CartoDB Positron')
    Fullscreen().add_to(m)

    # --- LAYER 1: DISTRICT AGGREGATION (Policy View) ---
    logger.info("📊 Calculating District-level deprivation layers...")
    district_stats = df.groupby('district')['priority_score'].mean().reset_index()

    # --- LAYER 2: SCHOOL POINT MARKERS (Operational View) ---
    school_layer = folium.FeatureGroup(name="Individual Schools").add_to(m)

    for _, row in df.iterrows():
        if pd.isna(row['latitude']): continue

        color = '#FF0000' if row['priority_score'] > 0.8 else '#FFD700' if row['priority_score'] > 0.5 else '#006B3F'

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{row['school_name']}</b><br>Score: {round(row['priority_score'] * 100, 1)}%"
        ).add_to(school_layer)

    # --- LAYER 3: SPATIAL CLUSTERING (Investment Zones) ---
    logger.info("🎯 Running DBSCAN to identify Investment Clusters...")
    coords = df[['latitude', 'longitude']].dropna().values
    # kms_per_radian = 6371.0088
    # 5km radius for sharing resources
    epsilon = 5 / 6371.0088

    db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    df_clustered = df.dropna(subset=['latitude']).copy()
    df_clustered['cluster'] = db.labels_

    cluster_layer = folium.FeatureGroup(name="High-ROI Investment Zones").add_to(m)

    # Plot only high-need clusters (where average priority > 0.6)
    for cluster_id in set(db.labels_):
        if cluster_id == -1: continue  # Noise

        cluster_points = df_clustered[df_clustered['cluster'] == cluster_id]
        if cluster_points['priority_score'].mean() > 0.6:
            # Draw a hull or center point for the cluster
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()

            folium.Marker(
                location=[center_lat, center_lon],
                icon=folium.Icon(color='black', icon='star'),
                popup=f"<b>INVESTMENT CLUSTER</b><br>{len(cluster_points)} schools in 5km radius.<br>Avg Need: {round(cluster_points['priority_score'].mean() * 100, 1)}%"
            ).add_to(cluster_layer)

    folium.LayerControl().add_to(m)

    output_path = ROOT / "outputs" / "ghana_school_gap_map.html"
    m.save(str(output_path))
    logger.info(f"✅ Enterprise Map saved to {output_path}")


if __name__ == "__main__":
    run_enterprise_mapping()