import pandas as pd
import json
from difflib import get_close_matches


def link_geojson_coordinates(master_csv, geojson_path):
    # 1. Load your master list from the data folder
    try:
        master_df = pd.read_csv(master_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {master_csv}. Please check the filename.")
        return

    # 2. Load the GeoJSON data from the data folder
    print(f"Loading GeoJSON from {geojson_path}...")
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {geojson_path}. Please check the filename.")
        return

    # 3. Extract points from GeoJSON into a list of dictionaries
    hdx_points = []
    for feature in data['features']:
        # Extract name and coordinates
        name = feature['properties'].get('name')
        if name:
            # GeoJSON coordinates are [longitude, latitude]
            coords = feature['geometry']['coordinates']
            hdx_points.append({
                'name': name.upper(),
                'lon': coords[0],
                'lat': coords[1]
            })

    hdx_names = [p['name'] for p in hdx_points]

    # 4. Initialize coordinate columns in your master dataframe
    master_df['latitude'] = None
    master_df['longitude'] = None

    print(f"Linking coordinates for {len(master_df)} schools...")

    for idx, row in master_df.iterrows():
        school_name = str(row['School_Name']).upper()

        # Strategy A: Exact Match
        if school_name in hdx_names:
            point = next(p for p in hdx_points if p['name'] == school_name)
            master_df.at[idx, 'latitude'] = point['lat']
            master_df.at[idx, 'longitude'] = point['lon']
        else:
            # Strategy B: Fuzzy Match (handles "SHS" vs "Senior High")
            match = get_close_matches(school_name, hdx_names, n=1, cutoff=0.7)
            if match:
                point = next(p for p in hdx_points if p['name'] == match[0])
                master_df.at[idx, 'latitude'] = point['lat']
                master_df.at[idx, 'longitude'] = point['lon']

    # 5. Export the final result to your main project folder
    output_file = "ghana_schools_final_mapped.csv"
    master_df.to_csv(output_file, index=False)

    matched_count = master_df['latitude'].notna().sum()
    print(f"\n--- Process Complete ---")
    print(f"Total Schools in Register: {len(master_df)}")
    print(f"Successfully Matched: {matched_count}")
    print(f"Unmatched (Missing GPS): {len(master_df) - matched_count}")
    print(f"Final file saved as: {output_file}")


# --- FINAL VERIFIED PATHS ---
if __name__ == "__main__":
    link_geojson_coordinates(
        "data/ghana_schools_master_2025.csv",
        "data/hotosm_gha_education_facilities_points_geojson.geojson"
    )