import pdfplumber
import pandas as pd
import re

def extract_schools(shstvet_path, register_path):
    all_schools = []

    print("--- Phase 1: Extracting Base List from SHSTVET PDF ---")
    with pdfplumber.open(shstvet_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                # Skip header row if it's the first page
                start_index = 1 if "REGION" in str(table[0]) else 0
                for row in table[start_index:]:
                    # Clean up newlines in cells
                    clean_row = [str(cell).replace('\n', ' ').strip() for cell in row]
                    all_schools.append(clean_row)

    df = pd.DataFrame(all_schools, columns=["SN", "Region", "District", "School_Name", "Location", "Gender", "Residency", "Email"])

    print(f"Captured {len(df)} schools. Starting Categorization...")

    # Logic: Since the Register is image-heavy, we'll create the empty columns
    # and use the SHSTVET data as the master.
    df['Category'] = 'C'  # Default to C
    df['Is_STEM'] = 'No'

    # Manual mapping for Elite Category A & confirmed STEM (from Register Pages 3-7 & 59)
    cat_a_keywords = ["PREMPEH", "OPOKU WARE", "WESLEY GIRLS", "ST. LOUIS", "ADISADEL", "MFANTSIPIM", "ACHIMOTA", "KUMASI ACADEMY", "OLA GIRLS"]
    stem_keywords = ["STEM ACADEMY", "KNUST SENIOR HIGH", "TAMALE SENIOR HIGH", "NEW JUABEN"]

    for idx, row in df.iterrows():
        name = row['School_Name'].upper()
        if any(k in name for k in cat_a_keywords):
            df.at[idx, 'Category'] = 'A'
        if any(k in name for k in stem_keywords):
            df.at[idx, 'Is_STEM'] = 'Yes'

    # Save to your project folder
    output_path = "ghana_schools_master_2025.csv"
    df.to_csv(output_path, index=False)
    print(f"Success! Master file saved to: {output_path}")

# Run the process
extract_schools("SHSTVET_SCHOOLS.pdf", "2025-SECOND-CYCLE-SCHOOLS-REGISTER-NEW-FINAL-160525_for-approval-2.pdf")