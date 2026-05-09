"""
check_geocoding_anomalies.py
-----------------------------
Quality-control script for the EduInfra Ghana dataset.

Loads data/clean/02_schools_scored.csv, checks every school's lat/lon
against the known bounding box for its listed region, and writes any
mismatches to outputs/anomalies.csv for human review.

Run:
    python check_geocoding_anomalies.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# 1.  Bounding boxes for all 16 Ghana regions
#     Format: (lat_min, lat_max, lon_min, lon_max)
#     Source: approximate extents derived from Ghana's administrative GIS data.
#     A small tolerance (TOLERANCE_DEG) is added to each edge to absorb minor
#     GPS/projection rounding at region borders.
# ---------------------------------------------------------------------------

TOLERANCE_DEG = 0.15   # ~15 km buffer — tighten if you get false positives

REGION_BBOX = {
    # Region name (upper-case, as it appears in the CSV)  : (lat_min, lat_max, lon_min, lon_max)
    "GREATER ACCRA":    (5.35,  5.95,  -0.50,  0.25),
    "ASHANTI":          (5.85,  7.60,  -2.90, -0.55),
    "EASTERN":          (5.65,  7.10,  -1.40,  0.55),
    "WESTERN":          (4.55,  6.40,  -3.25, -1.50),
    "WESTERN NORTH":    (5.50,  7.05,  -3.10, -2.00),
    "CENTRAL":          (4.90,  6.15,  -2.00, -0.55),
    "VOLTA":            (5.80,  8.75,  -0.15,  1.20),
    "OTI":              (7.70,  9.15,  -0.25,  0.80),
    "BONO":             (7.00,  8.50,  -3.00, -1.40),
    "BONO EAST":        (7.30,  8.80,  -1.80, -0.10),
    "AHAFO":            (6.60,  7.90,  -3.00, -1.80),
    "NORTHERN":         (8.30, 10.70,  -2.90,  0.60),
    "SAVANNAH":         (8.40, 11.00,  -2.90, -0.55),
    "NORTH EAST":       (9.80, 11.00,  -0.60,  0.65),
    "UPPER EAST":       (10.40, 11.20, -1.10,  0.75),
    "UPPER WEST":       (9.50, 11.00,  -2.90, -1.50),
}

# ---------------------------------------------------------------------------
# 2.  Load the dataset
# ---------------------------------------------------------------------------

INPUT_PATH  = os.path.join("data", "clean", "02_schools_scored.csv")
OUTPUT_DIR  = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "anomalies.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# Normalise region names to upper-case to match the bbox dict keys
df["region"] = df["region"].str.strip().str.upper()

required_cols = {"school_name", "region", "latitude", "longitude"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

print(f"  -> {len(df):,} schools loaded.")

# ---------------------------------------------------------------------------
# 3.  Flag anomalies
# ---------------------------------------------------------------------------

anomaly_rows = []

for _, row in df.iterrows():
    region  = row["region"]
    lat     = row["latitude"]
    lon     = row["longitude"]

    # Skip rows with missing coordinates
    if pd.isna(lat) or pd.isna(lon):
        anomaly_rows.append({
            **row.to_dict(),
            "anomaly_reason": "MISSING COORDINATES",
            "expected_lat_range": "",
            "expected_lon_range": "",
        })
        continue

    # Warn if the region isn't in our lookup table
    if region not in REGION_BBOX:
        anomaly_rows.append({
            **row.to_dict(),
            "anomaly_reason": f"UNKNOWN REGION: '{region}'",
            "expected_lat_range": "",
            "expected_lon_range": "",
        })
        continue

    lat_min, lat_max, lon_min, lon_max = REGION_BBOX[region]

    lat_ok = (lat_min - TOLERANCE_DEG) <= lat <= (lat_max + TOLERANCE_DEG)
    lon_ok = (lon_min - TOLERANCE_DEG) <= lon <= (lon_max + TOLERANCE_DEG)

    if not lat_ok or not lon_ok:
        reasons = []
        if not lat_ok:
            reasons.append(
                f"LAT {lat:.6f} outside [{lat_min - TOLERANCE_DEG:.4f}, "
                f"{lat_max + TOLERANCE_DEG:.4f}]"
            )
        if not lon_ok:
            reasons.append(
                f"LON {lon:.6f} outside [{lon_min - TOLERANCE_DEG:.4f}, "
                f"{lon_max + TOLERANCE_DEG:.4f}]"
            )

        anomaly_rows.append({
            **row.to_dict(),
            "anomaly_reason": " | ".join(reasons),
            "expected_lat_range": f"{lat_min} - {lat_max}",
            "expected_lon_range": f"{lon_min} - {lon_max}",
        })

# ---------------------------------------------------------------------------
# 4.  Write output
# ---------------------------------------------------------------------------

anomalies_df = pd.DataFrame(anomaly_rows)

if anomalies_df.empty:
    print("\n  No geocoding anomalies detected. Dataset looks clean!")
else:
    # Put the most diagnostic columns first for easy review
    priority_cols = [
        "school_name", "region", "district", "latitude", "longitude",
        "anomaly_reason", "expected_lat_range", "expected_lon_range",
    ]
    other_cols = [c for c in anomalies_df.columns if c not in priority_cols]
    anomalies_df = anomalies_df[priority_cols + other_cols]

    anomalies_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  {len(anomalies_df)} anomalies found -> saved to: {OUTPUT_PATH}")
    print("\nSummary preview:")
    print(
        anomalies_df[["school_name", "region", "latitude", "longitude", "anomaly_reason"]]
        .to_string(index=False)
    )

print("\nDone.")
