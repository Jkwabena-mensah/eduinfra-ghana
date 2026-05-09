import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time


def final_tvet_pass(input_csv):
    # 1. Load the latest file
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {input_csv}")
        return

    # 2. Setup the Geocoder with long timeout
    geolocator = Nominatim(user_agent="eduinfra_ghana_final_tvet_pass", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2.0)

    missing_mask = df['latitude'].isna()
    print(f"Targeting the final {missing_mask.sum()} schools with TVET name-swapping...")

    for idx, row in df[missing_mask].iterrows():
        original_name = str(row['School_Name'])

        # Create variations to help the database find the school
        # This addresses why schools like "ANLO TECH. INST." were failing
        variations = [
            original_name.replace("TECH. INST.", "TECHNICAL INSTITUTE"),
            original_name.replace("SNR. HIGH TECH. SCHOOL", "SENIOR HIGH TECHNICAL SCHOOL"),
            original_name.replace("VOC./TECH. INST.", "VOCATIONAL TECHNICAL INSTITUTE"),
            original_name.replace("VOC. TECH. INST.", "VOCATIONAL TECHNICAL INSTITUTE"),
            original_name.split(',')[0]  # Just the name before the first comma
        ]

        found = False
        # Try the original first, then the variations
        search_list = [f"{original_name}, {row['Region']}, Ghana"] + [f"{v}, {row['Region']}, Ghana" for v in
                                                                      variations]

        for query in search_list:
            if found: break
            try:
                print(f"Searching: {query}...")
                location = geocode(query)
                if location:
                    df.at[idx, 'latitude'] = location.latitude
                    df.at[idx, 'longitude'] = location.longitude
                    print(f">>> SUCCESS: Found match!")
                    found = True
            except Exception as e:
                print(f"Server busy, waiting...")
                time.sleep(2)

    # 3. Final Export
    output_file = "ghana_schools_final_2025_COMPLETE.csv"
    df.to_csv(output_file, index=False)

    final_count = df['latitude'].notna().sum()
    print(f"\n--- Final Project Statistics ---")
    print(f"Total Schools in Register: {len(df)}")
    print(f"Final Schools with GPS: {final_count}")
    print(f"Data Completion: {(final_count / len(df)) * 100:.1f}%")
    print(f"File saved as: {output_file}")


if __name__ == "__main__":
    # We use the results from your last successful run as the input
    final_tvet_pass("ghana_schools_final_2025_COMPLETE.csv")