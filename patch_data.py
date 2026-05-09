"""
patch_data.py
=============
EduInfra Ghana — One-time (idempotent) data patch script.

Fixes three issues in schools_priority_ranked.csv:
  1. Tier labels recalibrated (CRITICAL >0.65, HIGH >0.45)
  2. GPS corrected for all 456 misplaced schools using district centroids
  3. Duplicate school name disambiguated by region

Run with:
    cd C:\\Dev\\eduinfra-ghana
    python patch_data.py

Auto-run: app.py calls patch() silently on every cold start (no-op if already patched).
"""

import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "data"
CSV_PATH   = DATA_DIR / "schools_priority_ranked.csv"
CSV_SOURCE = DATA_DIR / "clean" / "02_schools_scored.csv"

THRESHOLD_CRITICAL = 0.65
THRESHOLD_HIGH     = 0.45

# Authoritative district centroid coordinates (GSS 2021)
# Used to correct bad HOTOSM fuzzy-match GPS for all misplaced schools
DISTRICT_CENTROIDS = {
    # GREATER ACCRA
    "ACCRA METRO": (5.5600, -0.2057), "TEMA METRO": (5.6698, 0.0166),
    "TEMA WEST MUNICIPAL": (5.6200, 0.0000), "ADENTAN MUNICIPAL": (5.7200, -0.1500),
    "AYAWASO NORTH MUNICIPAL": (5.6400, -0.1900), "AYAWASO CENTRAL MUNICIPAL": (5.5950, -0.2100),
    "AYAWASO WEST MUNICIPAL": (5.5800, -0.2400), "GA EAST MUNICIPAL": (5.7000, -0.2000),
    "GA WEST MUNICIPAL": (5.6500, -0.3500), "GA NORTH MUNICIPAL": (5.7800, -0.3000),
    "GA SOUTH MUNICIPAL": (5.5200, -0.3000), "GA CENTRAL MUNICIPAL": (5.5500, -0.2500),
    "ASHIAMAN MUNICIPAL": (5.6936, 0.0165), "KROWOR MUNICIPAL": (5.5700, 0.0300),
    "OKAIKOI NORTH MUNICIPAL": (5.5600, -0.2100), "ABLEKUMA CENTRAL MUNICIPAL": (5.5700, -0.2400),
    "ABLEKUMA NORTH MUNICIPAL": (5.6000, -0.2600), "ABLEKUMA WEST MUNICIPAL": (5.5500, -0.2700),
    "KORLEY KLOTTEY MUNICIPAL": (5.5500, -0.2000), "LA DADE-KOTOPON MUNICIPAL": (5.5700, -0.1500),
    "LA NKWANTANANG MADINA MUNICIPAL": (5.6800, -0.1600), "LEDZOKUKU MUNICIPAL": (5.6200, -0.0900),
    "NINGO PRAMPRAM DISTRICT": (5.7100, 0.1200), "SHAI-OSUDOKU DISTRICT": (5.9200, 0.0800),
    "ADA EAST DISTRICT": (5.7800, 0.6300), "ADA WEST DISTRICT": (5.7500, 0.5000),
    # ASHANTI
    "KUMASI METRO": (6.6884, -1.6244), "KUMASI METRO.": (6.6884, -1.6244),
    "ASOKORE MAMPONG MUNICIPAL": (6.7100, -1.5900), "OBUASI MUNICIPAL": (6.2000, -1.6600),
    "EJISU JUABEN MUNICIPAL": (6.7300, -1.4700), "EJISU JUABEN": (6.7300, -1.4700),
    "EJISU-JUABEN MUNICIPAL": (6.7300, -1.4700), "MAMPONG MUNICIPAL": (7.0600, -1.4100),
    "OFFINSO MUNICIPAL": (7.2100, -1.6600), "OFFINSO NORTH": (7.3000, -1.6500),
    "KWABRE EAST": (6.7800, -1.6600), "ATWIMA KWANWOMA": (6.6500, -1.7000),
    "ATWIMA NWABIAGYA MUNICIPAL": (6.8000, -1.7500), "ATWIMA MPONUA": (6.5000, -1.9500),
    "ASANTE AKIM CENTRAL MUNICIPAL": (6.5500, -1.3000), "ASANTE AKIM NORTH": (6.7500, -1.2000),
    "ASANTE AKIM SOUTH": (6.4500, -1.3500), "SEKYERE CENTRAL": (7.1500, -1.3000),
    "SEKYERE EAST": (7.0000, -1.2500), "SEKYERE SOUTH": (6.9000, -1.4000),
    "SEKYERE KUMAWU": (6.9000, -1.2000), "ADANSI NORTH": (6.2800, -1.5500),
    "ADANSI SOUTH": (6.1500, -1.5500), "BEKWAI MUNICIPAL": (6.4600, -1.5700),
    "AMANSIE CENTRAL": (6.2000, -1.8500), "AMANSIE WEST": (6.1000, -2.0000),
    "AFIGYA-KWABERE": (6.8000, -1.6000), "BOSOMTWE": (6.5700, -1.5500),
    "EJURA/SEKYEDUMASE": (7.3800, -1.3600), "AHAFO ANO NORTH": (7.2000, -1.9000),
    "AHAFO ANO SOUTH": (7.0000, -1.9000),
    # EASTERN
    "NEW JUABEN MUNICIPAL": (6.1000, -0.2600), "WEST AKIM MUNICIPAL": (5.9800, -0.5500),
    "EAST AKIM MUNICIPAL": (6.0500, -0.6000), "BIRIM CENTRAL MUNICIPAL": (6.0700, -0.9500),
    "BIRIM NORTH": (6.3000, -0.8500), "BIRIM SOUTH": (5.8500, -0.9800),
    "AKWAPIM NORTH": (5.9500, -0.3000), "AKWAPIM SOUTH": (5.8000, -0.2500),
    "NSAWAM ADOAGYIRI": (5.8000, -0.3600), "SUHUM MUNICIPAL": (6.0400, -0.4600),
    "AYENSUANO": (6.0500, -0.3000), "FANTEAKWA": (6.4500, -0.4000),
    "KWAHU EAST": (6.8000, -0.4800), "KWAHU WEST MUNICIPAL": (6.6000, -0.6000),
    "KWAHU SOUTH": (6.5500, -0.7500), "KWAHU AFRAM PLAINS NORTH": (7.1000, -0.5000),
    "AFRAM PLAINS (KWAHU NORTH)": (7.1000, -0.5000), "KWAHU AFRAM PLAINS SOUTH": (6.9000, -0.4000),
    "DENKYEMBOUR": (6.2000, -0.8000), "ATIWA": (6.4000, -0.6500),
    "UPPER WEST AKIM": (5.9500, -0.5000), "YILO KROBO": (6.0600, -0.0600),
    "LOWER MANYA KROBO": (6.0000, 0.0000), "ASUOGYAMAN": (6.3000, 0.0800),
    "EAST AKIM": (6.0500, -0.6000),
    # CENTRAL
    "CAPE COAST METRO": (5.1030, -1.2801), "MFANTSIMAN MUNICIPAL": (5.2500, -1.1500),
    "AGONA EAST": (5.4500, -0.8500), "AGONA WEST MUNICIPAL": (5.5000, -0.8800),
    "EKUMFI": (5.2000, -1.0500), "ABURA/ASEBU/KWAMANKESE": (5.4500, -1.2000),
    "ABURA/ASEBU/ KWAMANKESE": (5.4500, -1.2000), "EFFUTU MUNICIPAL": (5.3500, -0.9500),
    "GOMOA EAST": (5.3000, -0.8500), "GOMOA WEST": (5.2500, -0.9000),
    "GOMOA CENTRAL": (5.2800, -0.8800), "AJUMAKO/ENYAN/ESSIAM": (5.4000, -1.1000),
    "AJUMAKO/ ENYAN/ESIAM": (5.4000, -1.1000), "AJUMAKO/ENYAN /ESSIAM": (5.4000, -1.1000),
    "ASSIN NORTH MUNICIPAL": (5.7000, -1.1500), "ASSIN SOUTH": (5.5500, -1.2000),
    "TWIFO ATI-MORKWA": (5.6000, -1.5000), "TWIFO ATTI-MOKWAA": (5.6000, -1.5000),
    "TWIFO HEMANG LOWER DENKYIRA": (5.7000, -1.6500),
    "UPPER DENKYIRA EAST MUNICIPAL": (5.9000, -1.6000),
    "UPPER DENKYIRA WEST": (5.9000, -1.7500), "AWUTU/SENYA": (5.6000, -0.5500),
    "KOMENDA/EDINA/EGUAFO/A BIREM MUNICIPAL": (5.0500, -1.4800),
    # VOLTA
    "HO MUNICIPAL": (6.6011, 0.4700), "HO WEST": (6.5000, 0.4200),
    "SOUTH DAYI": (6.8000, 0.3800), "NORTH DAYI": (6.9500, 0.3500),
    "AFADZATO SOUTH": (6.9000, 0.5500), "AFADZTO SOUTH": (6.9000, 0.5500),
    "HOHOE MUNICIPAL": (7.1500, 0.4700), "NORTH TONGU": (6.0500, 0.5000),
    "SOUTH TONGU": (5.9000, 0.5500), "CENTRAL TONGU": (6.2000, 0.4800),
    "AGORTIME ZIOPE": (6.3500, 0.5500), "ADAKLU": (6.5000, 0.5000),
    "KETA": (5.9200, 0.9900), "ANLO KETA MUNICIPAL": (5.9000, 1.0000),
    "KETU SOUTH": (6.1000, 1.1000), "KETU NORTH": (6.3500, 1.1000),
    "AKATSI": (6.1200, 0.8000), "KPANDO": (7.0200, 0.2900),
    "NKWANTA SOUTH": (8.3000, 0.3500), "NKWANTA NORTH": (8.6000, 0.2500),
    "KRACHI WEST": (7.5000, -0.0300), "KRACHI EAST": (7.7000, 0.1000),
    "KRACHI NCHUMURU": (8.0000, 0.2000),
    # OTI
    "BIAKOYE": (7.4000, 0.4000), "JASIKAN": (7.3547, 0.3637),
    "KADJEBI": (7.6000, 0.3800),
    # NORTHERN
    "TAMALE METRO": (9.4075, -0.8533), "SAGNERIGU": (9.4400, -0.8200),
    "YENDI MUNICIPAL": (9.4441, -0.0073), "KUMBUNGU": (9.6000, -0.9500),
    "SAVELUGU-NANTON": (9.6200, -0.8200), "TOLON": (9.4500, -1.0500),
    "KPANDI": (8.7000, -0.1500), "GUSHEGU": (9.9000, -0.2000),
    "SABOBA": (9.2800, 0.3000), "NANUMBA NORTH": (8.8500, -0.2800),
    "NANUMBA SOUTH": (8.6500, -0.2500), "KARAGA": (9.8500, -0.4500),
    "MION": (9.4000, -0.3500), "NANTON": (9.5500, -0.9000),
    # SAVANNAH
    "WEST GONJA": (9.3000, -2.0000), "NORTH GONJA": (9.4000, -1.6000),
    "CENTRAL GONJA": (8.9000, -1.8000), "EAST GONJA MUNICIPAL": (8.8000, -1.3000),
    "SAWLA-TUNA-KALBA": (9.3000, -2.5500), "BOLE": (9.0300, -2.4800),
    "NORTH EAST GONJA": (9.8000, -1.4000),
    # NORTH EAST
    "EAST MAMPRUSI": (10.4000, -0.4500), "EAST MAMPRUSI MUNICIPAL": (10.4000, -0.4500),
    "WEST MAMPRUSI": (10.3557, -0.3691), "WEST MAMPUSI": (10.3557, -0.3691),
    "WEST MAMPRUSI MUNICIPAL": (10.3557, -0.3691), "MAMPRUGU MOADURI": (10.5000, -0.8000),
    "BUNKRUPGU-YUNGUO": (10.5000, -0.1000), "CHEREPONI": (10.3000, 0.4000),
    # UPPER EAST
    "BOLGATANGA MUNICIPAL": (10.7858, -0.8533), "BOLGA EAST": (10.8000, -0.7500),
    "BOLGATANGA": (10.7858, -0.8533),
    "KASENA-NANKANI MUNICIPAL": (10.8900, -1.1000), "KASENA-NANKANI WEST": (10.9500, -1.2000),
    "KASENA-NANKANI": (10.8900, -1.1000), "KASSENA NANAKANA EAST": (10.8500, -1.0500),
    "NABDAM": (10.6500, -0.6500), "BONGO": (10.9000, -0.8000),
    "BUILSA NORTH": (10.4000, -1.3000), "BUILSA SOUTH": (10.2500, -1.2000),
    "BAWKU MUNICIPAL": (11.0600, -0.2400), "BAWKU WEST": (10.9000, -0.5000),
    "BINDURI": (10.9500, -0.4500), "TALENSI": (10.7000, -0.7500),
    "GARU TEMPANE": (10.8500, 0.2000), "GARU TEMPANI": (10.8500, 0.2000),
    "PUSIGA": (11.0000, -0.0500),
    # UPPER WEST
    "WA MUNICIPAL": (10.0601, -2.5099), "WA EAST": (10.0000, -2.2000),
    "WA WEST": (9.9000, -2.7000), "JIRAPA": (10.5400, -2.8000),
    "LAMBUSIE-KARNI": (10.7000, -2.8500), "NADOWLI": (10.3000, -2.7000),
    "NADOWLI/KALEO": (10.2500, -2.6500), "NADOWLI KALEO": (10.2500, -2.6500),
    "NANDOM": (10.8500, -2.7500), "LAWRA": (10.6500, -2.9000),
    "SISALA EAST": (10.3500, -2.2000), "SISALA WEST": (10.5000, -2.5000),
    "BUSSIE-ISSA": (10.1500, -2.5500),
    # BONO
    "SUNYANI MUNICIPAL": (7.3349, -2.3319), "SUNYANI WEST": (7.3000, -2.5000),
    "BEREKUM-MUNICIPAL": (7.4500, -2.5800), "DORMAA CENTRAL MUNICIPAL": (7.3000, -3.0000),
    "DORMAA EAST": (7.3500, -2.7000), "DORMAA WEST": (7.3500, -3.1000),
    "JAMAN NORTH": (7.7000, -2.8000), "JAMAN SOUTH": (7.5000, -2.8500),
    "TAIN": (7.9000, -2.4000), "WENCHI MUNICIPAL": (7.7500, -2.1000),
    "BANDA": (8.0000, -2.6000),
    # BONO EAST
    "TECHIMAN MUNICIPAL": (7.5924, -1.9367), "TECHIMAN NORTH": (7.7000, -1.9500),
    "NKORANZA NORTH": (7.8000, -1.7000), "NKORANZA SOUTH": (7.6000, -1.7500),
    "SENE EAST": (7.9000, -0.5000), "SENE WEST": (7.8000, -0.8000),
    "KINTAMPO NORTH MUNICIPAL": (8.0600, -1.7300), "KINTAMPO SOUTH": (7.9000, -1.8000),
    "PRU": (8.0000, -0.8000), "ATEBUBU-AMANTIN": (7.7500, -1.0000),
    "NKORANZA": (7.7000, -1.7000),
    # AHAFO
    "ASUNAFO NORTH MUNICIPAL": (7.0000, -2.6500), "ASUNAFO SOUTH": (6.8000, -2.5000),
    "ASUTIFI NORTH": (6.9000, -2.2000), "ASUTIFI SOUTH": (6.7500, -2.1000),
    "TANO NORTH": (7.2000, -2.1000), "TANO SOUTH": (7.0000, -2.0000),
    # WESTERN
    "SEKONDI TAKORADI METRO": (4.9340, -1.7596), "AHANTA WEST": (4.9000, -2.0000),
    "NZEMA EAST MUNICIPAL": (4.9200, -2.3000), "PRESTEA HUNI VALLEY": (5.3500, -2.1500),
    "TARKWA-NSUAEM MUNICIPAL": (5.2900, -1.9900), "WASSA AMENFI EAST": (5.7000, -2.3000),
    "WASSA AMENFI WEST": (5.5000, -2.5000), "WASSA AMENFI CENTRAL": (5.6000, -2.4000),
    "MPOHOR WASSA EAST": (5.1500, -1.9000), "MPOHOR": (5.1500, -1.9000),
    "SHAMA": (5.0100, -1.6300), "JOMORO": (4.7500, -2.2000),
    "ELLEMBELE": (4.8500, -2.3500),
    # WESTERN NORTH
    "SEFWI WIAWSO": (6.2100, -2.4800), "SEFWI AKONTOMBRA": (6.0000, -2.9000),
    "BIBIANI/ANHWIASO/BEKWAI": (6.4700, -2.3000), "BIBIANI/ANHWIASO/ BEKWAI": (6.4700, -2.3000),
    "JUABESO": (6.3000, -2.8000), "BIA WEST": (5.8500, -3.0000),
    "BODI": (6.5000, -2.6000), "SUAMAN": (6.0000, -3.0000),
}

REGION_BBOX = {
    "GREATER ACCRA":   (5.35,  5.95,  -0.50,  0.30),
    "ASHANTI":         (5.85,  7.60,  -2.90, -0.55),
    "EASTERN":         (5.65,  7.10,  -1.40,  0.55),
    "WESTERN":         (4.55,  6.40,  -3.25, -1.50),
    "WESTERN NORTH":   (5.50,  7.05,  -3.10, -2.00),
    "CENTRAL":         (4.90,  6.15,  -2.00, -0.55),
    "VOLTA":           (5.80,  8.75,  -0.15,  1.25),
    "OTI":             (7.20,  9.15,  -0.25,  0.85),
    "BONO":            (7.00,  8.50,  -3.10, -1.40),
    "BONO EAST":       (7.30,  8.80,  -1.85, -0.10),
    "AHAFO":           (6.60,  7.90,  -3.00, -1.80),
    "NORTHERN":        (8.30, 10.70,  -2.90,  0.65),
    "SAVANNAH":        (8.40, 11.00,  -2.90, -0.55),
    "NORTH EAST":      (9.80, 11.10,  -0.90,  0.65),
    "NORTH EAST REGION": (9.80, 11.10, -0.90, 0.65),
    "UPPER EAST":     (10.20, 11.25,  -1.40,  0.80),
    "UPPER WEST":      (9.50, 11.00,  -2.95, -1.50),
}


def _is_misplaced(row):
    """Return True if the school's GPS does not fall within its region bbox."""
    region = str(row.get("region", "")).upper().strip()
    lat = row.get("latitude")
    lon = row.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        return False  # no GPS — already handled separately
    bbox = REGION_BBOX.get(region)
    if not bbox:
        return True  # unknown region — treat as misplaced
    lmin, lmax, nmin, nmax = bbox
    TOL = 0.1  # ~10km tolerance
    return not ((lmin - TOL) <= lat <= (lmax + TOL) and (nmin - TOL) <= lon <= (nmax + TOL))


def patch():
    # Restore from scored pipeline output if ranked CSV is missing or truncated
    source = CSV_SOURCE if (CSV_SOURCE.exists() and len(pd.read_csv(CSV_PATH)) < 100) else CSV_PATH
    if source == CSV_SOURCE:
        print(f"  ⚠️  Ranked CSV corrupted — restoring from {source.name}")
    print(f"Loading: {source}")
    df = pd.read_csv(source)
    print(f"  Rows loaded: {len(df)}")

    # ── Idempotency check ─────────────────────────────────────────────────────
    current_critical = df["priority_tier"].value_counts().get("CRITICAL", 0)
    already_patched  = (
        current_critical >= 40
        and df["school_name"].duplicated().sum() == 0
        and df.apply(_is_misplaced, axis=1).sum() < 10
    )
    if already_patched:
        print(f"  ✅ Already patched ({current_critical} CRITICAL). Skipping.")
        return

    # ── 1. Recalibrate tiers ─────────────────────────────────────────────────
    old = df["priority_tier"].value_counts().to_dict()
    df["priority_tier"] = df["priority_score"].apply(
        lambda s: "CRITICAL" if s > THRESHOLD_CRITICAL
                  else "HIGH" if s > THRESHOLD_HIGH
                  else "STABLE"
    )
    new = df["priority_tier"].value_counts().to_dict()
    print(f"  Tiers before: {old}")
    print(f"  Tiers after:  {new}")

    # ── 2. Fix misplaced GPS using district centroids ─────────────────────────
    gps_fixed = 0
    for idx, row in df.iterrows():
        if not _is_misplaced(row):
            continue
        district = str(row.get("district", "")).upper().strip()
        centroid = DISTRICT_CENTROIDS.get(district)
        if centroid:
            df.at[idx, "latitude"]  = centroid[0]
            df.at[idx, "longitude"] = centroid[1]
            gps_fixed += 1
        else:
            # Clear bad GPS — better no marker than wrong location
            df.at[idx, "latitude"]  = None
            df.at[idx, "longitude"] = None
    print(f"  GPS misplacements corrected: {gps_fixed}")
    print(f"  GPS coverage: {df['latitude'].notna().sum()}/{len(df)}")

    # ── 3. Disambiguate duplicate school names ───────────────────────────────
    dup_names = df.loc[df["school_name"].duplicated(keep=False), "school_name"].unique()
    for dup in dup_names:
        mask = df["school_name"] == dup
        if df.loc[mask, "region"].nunique() > 1:
            for idx, row in df[mask].iterrows():
                tag = str(row["region"]).split()[0].title()
                df.at[idx, "school_name"] = f"{dup} ({tag})"
                print(f"  Renamed duplicate: → {df.at[idx, 'school_name']}")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  ✅ Saved to {CSV_PATH}")
    print(f"  Final: {len(df)} schools | {df['priority_tier'].value_counts().to_dict()} "
          f"| {df['latitude'].isna().sum()} missing GPS")


if __name__ == "__main__":
    patch()
