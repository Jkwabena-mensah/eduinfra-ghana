import shapefile
import json
import os


def extract_high_res_border():
    # Path to your shapefile
    shp_path = r"data/boundaries/gha_admin0.shp"

    if not os.path.exists(shp_path):
        print(f"❌ Error: Shapefile not found at {shp_path}")
        return

    try:
        sf = shapefile.Reader(shp_path)
        # Get the first shape (Ghana national border)
        shape = sf.shape(0)

        # Extract coordinates [longitude, latitude]
        # We take every 2nd or 3rd point if it's too massive,
        # but for a national border, the full set is usually fine.
        coords = [list(p) for p in shape.points]

        print(f"✅ Successfully extracted {len(coords)} coordinates.")

        # Print the coordinates in a format you can copy-paste into config.py
        print("\n--- COPY THE LIST BELOW ---")
        print(coords)
        print("--- END OF LIST ---")

    except Exception as e:
        print(f"❌ Failed to process shapefile: {e}")


if __name__ == "__main__":
    extract_high_res_border()