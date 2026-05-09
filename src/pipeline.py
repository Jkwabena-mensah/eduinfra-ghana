"""
src/pipeline.py
===============
Production-grade EduInfra Ghana data pipeline.

This class consolidates the experimental logic scattered across
extract_schools.py, link_gps.py, complete_mapping.py, and the
notebooks/ directory into a single, testable, importable unit.

Typical usage
-------------
    from src.pipeline import EduInfraPipeline

    pipeline = EduInfraPipeline()
    df = pipeline.run()                      # full pipeline, returns ranked DataFrame

    # Or step by step:
    cleaned = pipeline.clean_data()
    scored  = pipeline.calculate_scores(cleaned)
"""

from __future__ import annotations

import json
import logging
import re
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

# thefuzz is optional — we fall back to difflib if not installed
try:
    from thefuzz import process as _fuzz_process  # type: ignore
    _FUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FUZZ_AVAILABLE = False

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

from src.config import (
    # paths
    CSV_MASTER_COMPLETE,
    CSV_MASTER_MAPPED,
    CSV_SCHOOLS_REF,
    CSV_CLEAN_STEP1,
    CSV_CLEAN_STEP2,
    CSV_PRIORITY_RANKED,
    GEOJSON_HDX,
    CLEAN_DIR,
    CSV_AID_DATA,
    DHS_HH_DATA,
    MODEL_PATH,
    # column constants
    SOCIOECONOMIC_COLS,
    ENRICHMENT_NORM_COLS,
    COORDINATE_COLS,
    SCORING_DERIVED_COLS,
    NAME_NOISE_TOKENS,
    CATEGORY_A_KEYWORDS,
    STEM_KEYWORDS,
    TVET_EXPANSIONS,
    # scoring weights & thresholds
    W_POVERTY,
    W_LITERACY,
    W_ELEC,
    W_WATER,
    W_SANITATION,
    W_AID,
    THRESHOLD_CRITICAL,
    THRESHOLD_HIGH,
    # fuzzy matching
    FUZZY_CUTOFF,
)

# ---------------------------------------------------------------------------
# Module-level logger — also writes to logs/system.log
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Attach a dedicated system.log handler once at import time
_sys_log_path = Path(__file__).resolve().parent.parent / "logs" / "system.log"
_sys_log_path.parent.mkdir(parents=True, exist_ok=True)
if not any(isinstance(h, logging.FileHandler) and "system.log" in getattr(h, "baseFilename", "") for h in logger.handlers):
    _sys_handler = logging.FileHandler(_sys_log_path, mode="a", encoding="utf-8")
    _sys_handler.setLevel(logging.DEBUG)
    _sys_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_sys_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_name(text: str) -> str:
    """
    Strip noise tokens and non-alphanumeric characters from a school name
    to produce a compact key suitable for fuzzy or exact matching.

    Example: "Kumasi Academy Senior High School" -> "KUMASIACADEMY"
    """
    if pd.isna(text):
        return ""
    text = str(text).upper()
    for token in NAME_NOISE_TOKENS:
        text = text.replace(token, "")
    return re.sub(r"[^A-Z0-9]", "", text).strip()


def _expand_tvet(name: str) -> str:
    """Expand TVET abbreviations to full words for geocoder queries."""
    for abbrev, full in TVET_EXPANSIONS.items():
        name = name.replace(abbrev, full)
    return name


def _safe_normalise(series: pd.Series) -> pd.Series:
    """
    Min-max normalise a Series to [0, 1].
    Returns zeros if the range is zero (avoids division-by-zero).
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class EduInfraPipeline:
    """
    Encapsulates the full EduInfra Ghana data pipeline.

    Parameters
    ----------
    raw_path : Path, optional
        Override the default input CSV (ghana_schools_final_2025_COMPLETE.csv).
    ref_path : Path, optional
        Override the default socioeconomic reference CSV (schools_clean.csv).
    clean_dir : Path, optional
        Override the directory where intermediate outputs are written.
    """

    def __init__(
        self,
        raw_path:  Optional[Path] = None,
        ref_path:  Optional[Path] = None,
        clean_dir: Optional[Path] = None,
    ) -> None:
        self.raw_path  = raw_path  or CSV_MASTER_COMPLETE
        self.ref_path  = ref_path  or CSV_SCHOOLS_REF
        self.clean_dir = clean_dir or CLEAN_DIR
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        )
        self._logger = logging.getLogger(self.__class__.__name__)

        # Attach system.log handler to instance logger too (if not already present)
        if not any(
            isinstance(h, logging.FileHandler)
            and "system.log" in getattr(h, "baseFilename", "")
            for h in self._logger.handlers
        ):
            self._logger.addHandler(_sys_handler)

    # ------------------------------------------------------------------
    # Step 1 – clean_data()
    # ------------------------------------------------------------------

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Load the raw mapped school CSV and enrich it with socioeconomic
        reference data (MPI poverty scores and youth literacy counts).

        Three-level merge strategy (mirrors notebooks/01_clean_data.py):
          1. Exact key: (normalised school name) + (normalised district)
          2. Fallback 1: normalised school name alone
          3. Fallback 2: regional mean imputation — guarantees 100 % coverage

        The result is saved to ``data/clean/01_schools_cleaned.csv``.

        Parameters
        ----------
        df : DataFrame, optional
            Pass an already-loaded DataFrame to skip disk I/O (useful for
            unit tests or chaining with GPS-linkage results).

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with socioeconomic columns attached.
        """
        self._logger.info("[Step 1] Starting clean_data()")

        # --- Load raw data ---
        if df is None:
            if not self.raw_path.exists():
                # Graceful fallback to the partially mapped file
                fallback = CSV_MASTER_MAPPED
                if not fallback.exists():
                    msg = (
                        "⚠️  Raw data missing. Please ensure "
                        "'ghana_schools_master_2025.csv' is in the data folder."
                    )
                    self._logger.error(msg)
                    raise FileNotFoundError(msg)
                self._logger.warning(
                    f"Complete CSV not found at {self.raw_path}. "
                    f"Falling back to {fallback}."
                )
                self.raw_path = fallback
            self._logger.info(f"Loading raw data from {self.raw_path}")
            try:
                df = pd.read_csv(self.raw_path, encoding="utf-8")
            except UnicodeDecodeError:
                self._logger.warning(
                    "UTF-8 decode failed; retrying with latin-1 encoding."
                )
                df = pd.read_csv(self.raw_path, encoding="latin-1")
            except Exception as exc:
                self._logger.error(
                    f"Failed to load raw CSV '{self.raw_path}': {exc}",
                    exc_info=True,
                )
                raise

        # --- Load socioeconomic reference ---
        if not self.ref_path.exists():
            self._logger.warning(
                f"Reference CSV not found at {self.ref_path}. "
                "Skipping socioeconomic enrichment."
            )
            df.to_csv(CSV_CLEAN_STEP1, index=False)
            return df

        ref_df = pd.read_csv(self.ref_path)

        # --- Standardise column names on the reference frame ---
        ref_df.columns = ref_df.columns.str.strip()
        # The ref CSV uses lowercase 'school_name'; master uses 'School_Name'
        if "school_name" in ref_df.columns and "School_Name" not in ref_df.columns:
            ref_df = ref_df.rename(columns={"school_name": "School_Name"})

        # --- Build compound merge keys ---
        df["_s_key"] = df["School_Name"].apply(_normalise_name)
        df["_d_key"] = df["District"].apply(_normalise_name)
        df["_final_key"] = df["_s_key"] + df["_d_key"]

        ref_df["_s_key"] = ref_df["School_Name"].apply(_normalise_name)
        ref_df["_d_key"] = ref_df["District"].apply(_normalise_name)
        ref_df["_final_key"] = ref_df["_s_key"] + ref_df["_d_key"]

        # --- Merge 1: compound key ---
        ref_full = (
            ref_df[["_final_key"] + SOCIOECONOMIC_COLS]
            .drop_duplicates("_final_key")
        )
        df = df.merge(ref_full, on="_final_key", how="left")

        # --- Merge 2: school name only (fallback for unmatched) ---
        missing_mask = df[SOCIOECONOMIC_COLS[0]].isna()
        if missing_mask.any():
            ref_name = (
                ref_df[["_s_key"] + SOCIOECONOMIC_COLS]
                .drop_duplicates("_s_key")
            )
            fallback_vals = (
                df.loc[missing_mask, ["_s_key"]]
                .merge(ref_name, on="_s_key", how="left")
            )
            for col in SOCIOECONOMIC_COLS:
                df.loc[missing_mask, col] = fallback_vals[col].values

        # --- Merge 2b: Fuzzy name matching (thefuzz → difflib fallback) ---
        # Handles cases like 'KNUST Primary' vs 'K.N.U.S.T Primary' that
        # exact key normalisation misses.
        still_missing_fuzz = df[SOCIOECONOMIC_COLS[0]].isna()
        if still_missing_fuzz.any():
            ref_keys_list = ref_df["_s_key"].dropna().unique().tolist()
            ref_key_lookup = (
                ref_df[["_s_key"] + SOCIOECONOMIC_COLS]
                .drop_duplicates("_s_key")
                .set_index("_s_key")
            )
            fuzz_matched = 0
            for idx in df.index[still_missing_fuzz]:
                query_key = df.at[idx, "_s_key"]
                if not query_key:
                    continue
                if _FUZZ_AVAILABLE:
                    result = _fuzz_process.extractOne(
                        query_key, ref_keys_list, score_cutoff=int(FUZZY_CUTOFF * 100)
                    )
                    best_match = result[0] if result else None
                else:
                    candidates = get_close_matches(
                        query_key, ref_keys_list, n=1, cutoff=FUZZY_CUTOFF
                    )
                    best_match = candidates[0] if candidates else None

                if best_match and best_match in ref_key_lookup.index:
                    for col in SOCIOECONOMIC_COLS:
                        df.at[idx, col] = ref_key_lookup.at[best_match, col]
                    fuzz_matched += 1

            if fuzz_matched:
                engine = "thefuzz" if _FUZZ_AVAILABLE else "difflib"
                self._logger.info(
                    f"[Step 1] Fuzzy merge ({engine}): recovered {fuzz_matched} "
                    f"additional schools."
                )

        # --- Merge 3: regional mean imputation (guarantees 100 % coverage) ---
        still_missing = df[SOCIOECONOMIC_COLS[0]].isna()
        if still_missing.any():
            self._logger.info(
                f"Imputing {still_missing.sum()} remaining schools with regional averages."
            )
            regional_means = (
                ref_df.groupby("Region")[SOCIOECONOMIC_COLS]
                .mean()
                .reset_index()
            )
            regional_means["Region"] = regional_means["Region"].str.upper().str.strip()
            df["_region_upper"] = df["Region"].str.upper().str.strip()
            df = df.merge(
                regional_means,
                left_on="_region_upper",
                right_on="Region",
                how="left",
                suffixes=("", "_reg"),
            )
            for col in SOCIOECONOMIC_COLS:
                reg_col = f"{col}_reg"
                if reg_col in df.columns:
                    df[col] = df[col].fillna(df[reg_col])

        # --- Drop internal helper columns ---
        drop_cols = [
            "_s_key", "_d_key", "_final_key", "_region_upper",
            "Region_reg",
        ] + [f"{c}_reg" for c in SOCIOECONOMIC_COLS]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # --- Standardise coordinate column types ---
        for col in COORDINATE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Deduplicate ambiguous school names by appending region suffix ---
        # Identifies schools where the same name appears in multiple regions
        # and appends " (REGION)" to make each name unique across the dataset.
        _dup_names = df.loc[df["School_Name"].duplicated(keep=False), "School_Name"].unique()
        for _dup in _dup_names:
            _mask = df["School_Name"] == _dup
            if df.loc[_mask, "Region"].nunique() > 1:
                df.loc[_mask, "School_Name"] = (
                    df.loc[_mask, "School_Name"] + " (" +
                    df.loc[_mask, "Region"].str.split().str[0].str.title() + ")"
                )

        # --- Save intermediate output ---
        df.to_csv(CSV_CLEAN_STEP1, index=False)
        self._logger.info(
            f"[Step 1] Done. {len(df)} schools written to {CSV_CLEAN_STEP1}"
        )
        self._logger.info(
            f"         MPI coverage:      "
            f"{df['mpi_score'].notna().sum()}/{len(df)}"
        )
        self._logger.info(
            f"         Literacy coverage: "
            f"{df['youth_literacy_count'].notna().sum()}/{len(df)}"
        )
        return df

    # ------------------------------------------------------------------
    # Step 1b – enrich_features()
    # ------------------------------------------------------------------

    def enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the cleaned schools DataFrame with four district/region-level
        infrastructure and aid indicators, then min-max normalise each one.

        New columns added
        -----------------
        elec_norm        : electrification access rate [0, 1], by region
                           (from DHS 2022 household microdata, hv206 weighted by hv005)
        water_norm       : improved water source rate [0, 1], by region
                           (DHS hv201: codes 11-14 piped + 21 borehole + 31 prot.well
                            + 71 bottled + 72 sachet = "improved")
        sanitation_norm  : improved sanitation rate [0, 1], by region
                           (DHS hv205: codes 11-16 flush + 21-22 improved pit = "improved")
        aid_norm         : total prior aid received [0, 1], by district
                           (WorldBank geocoded aid sum from aid_data.csv)

        Join strategy
        -------------
        - DHS indicators: aggregate weighted household rate per region name,
          join to schools on df["Region"] (uppercase-normalised).
        - Aid: join on df["District"] (uppercase-normalised) to aid_data["NAME_2"].
        - Any missing value after join is filled with the national median of
          that column, so zero school rows are ever dropped.
        - Each column is min-max normalised via _safe_normalise().

        Parameters
        ----------
        df : pd.DataFrame
            Output of clean_data().

        Returns
        -------
        pd.DataFrame
            Input df with four normalised columns appended.
        """
        self._logger.info("[Step 1b] Starting enrich_features()")

        # ── Region-level DHS aggregation ────────────────────────────────────
        # DHS household recode has hv024 (region int), hv005 (sample weight,
        # 6 implicit decimal places), hv206 (electricity 0/1),
        # hv201 (water source codes), hv205 (sanitation codes).
        # We compute weighted rates per region then join by region name.

        # Mapping: hv024 integer → region name (from GHHR8CFL.MAP)
        _REGION_CODES: dict[int, str] = {
            1:  "WESTERN",
            2:  "CENTRAL",
            3:  "GREATER ACCRA",
            4:  "VOLTA",
            5:  "EASTERN",
            6:  "ASHANTI",
            7:  "WESTERN NORTH",
            8:  "AHAFO",
            9:  "BONO",
            10: "BONO EAST",
            11: "OTI",
            12: "NORTHERN",
            13: "SAVANNAH",
            14: "NORTH EAST",
            15: "UPPER EAST",
            16: "UPPER WEST",
        }

        # DHS "improved" water source codes
        _IMPROVED_WATER = {11, 12, 13, 14, 21, 31, 41, 71, 72}
        # DHS "improved" sanitation codes (flush types + improved pit)
        _IMPROVED_SANITATION = {11, 12, 13, 14, 15, 16, 21, 22, 41}

        elec_by_region:  dict[str, float] = {}
        water_by_region: dict[str, float] = {}
        sanit_by_region: dict[str, float] = {}

        if DHS_HH_DATA.exists():
            self._logger.info(f"  Loading DHS household data from {DHS_HH_DATA}")
            try:
                dhs = pd.read_stata(
                    DHS_HH_DATA,
                    columns=["hv001", "hv024", "hv005", "hv201", "hv205", "hv206"],
                    convert_categoricals=False,
                )
                # Sample weight: DHS stores as integer with 6 implicit decimals
                dhs["_w"] = dhs["hv005"].astype(float) / 1_000_000.0
                dhs["_region_code"] = dhs["hv024"].astype(int)
                dhs["_region_name"] = dhs["_region_code"].map(_REGION_CODES)

                # Binary flags
                dhs["_elec"]  = (dhs["hv206"].astype(int) == 1).astype(float)
                dhs["_water"] = dhs["hv201"].astype(int).isin(_IMPROVED_WATER).astype(float)
                dhs["_sanit"] = dhs["hv205"].astype(int).isin(_IMPROVED_SANITATION).astype(float)

                for region_name, grp in dhs.groupby("_region_name"):
                    w = grp["_w"]
                    wsum = w.sum()
                    if wsum == 0:
                        continue
                    elec_by_region[region_name]  = (grp["_elec"]  * w).sum() / wsum
                    water_by_region[region_name] = (grp["_water"] * w).sum() / wsum
                    sanit_by_region[region_name] = (grp["_sanit"] * w).sum() / wsum

                self._logger.info(
                    f"  DHS aggregation complete: "
                    f"{len(elec_by_region)} regions computed."
                )
            except Exception as exc:
                self._logger.warning(
                    f"  DHS load failed ({exc}); using fallback constant 0.805."
                )
        else:
            self._logger.warning(
                f"  DHS file not found at {DHS_HH_DATA}. "
                "Using national averages as fallback."
            )

        # Fallback: Ghana 2022 DHS national estimates (from FRQ summary).
        # elec: 80.5%, water (improved): ~50%, sanitation (improved): ~57.7%
        _FALLBACK_ELEC  = 0.805
        _FALLBACK_WATER = 0.500
        _FALLBACK_SANIT = 0.577

        # Join DHS indicators onto df by Region
        df["_region_upper"] = df["Region"].str.upper().str.strip()
        df["elec_access"]  = df["_region_upper"].map(elec_by_region).fillna(_FALLBACK_ELEC)
        df["water_access"] = df["_region_upper"].map(water_by_region).fillna(_FALLBACK_WATER)
        df["sanit_access"] = df["_region_upper"].map(sanit_by_region).fillna(_FALLBACK_SANIT)
        df.drop(columns=["_region_upper"], inplace=True)

        self._logger.info(
            f"  elec_access  — non-null: {df['elec_access'].notna().sum()}/{len(df)}"
        )
        self._logger.info(
            f"  water_access — non-null: {df['water_access'].notna().sum()}/{len(df)}"
        )
        self._logger.info(
            f"  sanit_access — non-null: {df['sanit_access'].notna().sum()}/{len(df)}"
        )

        # ── District-level aid aggregation ──────────────────────────────────
        aid_by_district: dict[str, float] = {}

        if CSV_AID_DATA.exists():
            self._logger.info(f"  Loading aid data from {CSV_AID_DATA}")
            try:
                aid_df = pd.read_csv(CSV_AID_DATA)
                aid_col = "worldbank_geocodedresearchrelease_level1_v1_4_2.fa137b9.sum"
                if "NAME_2" in aid_df.columns and aid_col in aid_df.columns:
                    # Aggregate by district (NAME_2), summing total aid
                    aid_agg = (
                        aid_df.groupby("NAME_2")[aid_col]
                        .sum()
                        .reset_index()
                    )
                    aid_agg["_dist_key"] = (
                        aid_agg["NAME_2"].str.upper().str.strip()
                    )
                    aid_by_district = dict(
                        zip(aid_agg["_dist_key"], aid_agg[aid_col])
                    )
                    self._logger.info(
                        f"  Aid data: {len(aid_by_district)} districts loaded."
                    )
                else:
                    self._logger.warning(
                        f"  aid_data.csv missing expected columns. "
                        f"Found: {list(aid_df.columns[:5])}"
                    )
            except Exception as exc:
                self._logger.warning(f"  Aid data load failed: {exc}")
        else:
            self._logger.warning(
                f"  Aid data not found at {CSV_AID_DATA}. Filling with median."
            )

        df["_dist_key"] = df["District"].str.upper().str.strip()
        df["aid_received"] = df["_dist_key"].map(aid_by_district)
        df.drop(columns=["_dist_key"], inplace=True)

        # ── National-median imputation for any remaining nulls ───────────────
        for raw_col in ["elec_access", "water_access", "sanit_access", "aid_received"]:
            n_missing = df[raw_col].isna().sum()
            if n_missing > 0:
                median_val = df[raw_col].median()
                if pd.isna(median_val):
                    # Column is all-null (e.g. aid data entirely absent) — use 0
                    median_val = 0.0
                df[raw_col] = df[raw_col].fillna(median_val)
                self._logger.info(
                    f"  Imputed {n_missing} nulls in '{raw_col}' "
                    f"with national median {median_val:.4f}."
                )

        # ── Min-max normalise ────────────────────────────────────────────────
        df["elec_norm"]        = _safe_normalise(df["elec_access"])
        df["water_norm"]       = _safe_normalise(df["water_access"])
        df["sanitation_norm"]  = _safe_normalise(df["sanit_access"])
        df["aid_norm"]         = _safe_normalise(df["aid_received"])

        # Drop the raw (un-normalised) intermediates — keep only the norm cols
        df.drop(
            columns=["elec_access", "water_access", "sanit_access", "aid_received"],
            inplace=True,
        )

        self._logger.info(
            "[Step 1b] enrich_features() complete. "
            f"New columns: {ENRICHMENT_NORM_COLS}"
        )
        return df

    # ------------------------------------------------------------------
    # Step 2 – calculate_scores()
    # ------------------------------------------------------------------

    def calculate_scores(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute the composite infrastructure-deprivation priority score for
        every school.

        Formula (mirrors notebooks/02_priority_scoring.py)::

            pov_norm  = min-max normalise(mpi_score)           # higher → more deprived
            lit_norm  = 1 - min-max normalise(youth_literacy)  # lower literacy → higher score
            priority_score = (pov_norm * W_POVERTY) + (lit_norm * W_LITERACY)

        Schools are sorted descending by ``priority_score``.
        Results are saved to both:
          - ``data/clean/02_schools_scored.csv``   (intermediate)
          - ``data/schools_priority_ranked.csv``   (consumed by app.py)

        Parameters
        ----------
        df : DataFrame, optional
            Pass a cleaned DataFrame directly; otherwise loads from
            ``data/clean/01_schools_cleaned.csv``.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``pov_norm``, ``lit_norm``, and ``priority_score``
            columns appended, sorted by ``priority_score`` descending.
        """
        self._logger.info("[Step 2] Starting calculate_scores()")

        if df is None:
            if not CSV_CLEAN_STEP1.exists():
                msg = (
                    f"⚠️  Cleaned data not found at {CSV_CLEAN_STEP1}. "
                    "Run clean_data() first, or pass a DataFrame directly."
                )
                self._logger.error(msg)
                raise FileNotFoundError(msg)
            try:
                df = pd.read_csv(CSV_CLEAN_STEP1, encoding="utf-8")
            except Exception as exc:
                self._logger.error(
                    f"Failed to read cleaned CSV: {exc}", exc_info=True
                )
                raise

        # --- Guard: ensure required columns exist ---
        required_cols = SOCIOECONOMIC_COLS + ENRICHMENT_NORM_COLS
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' is missing. "
                    "Ensure clean_data() and enrich_features() have both been "
                    "run before calling calculate_scores()."
                )

        # --- Normalise poverty & literacy ---
        df["pov_norm"] = _safe_normalise(df["mpi_score"].fillna(df["mpi_score"].median()))
        literacy_norm  = _safe_normalise(
            df["youth_literacy_count"].fillna(df["youth_literacy_count"].median())
        )
        # Invert: lower literacy = higher priority score
        df["lit_norm"] = 1 - literacy_norm

        # --- 6-feature weighted composite score ---
        # Infrastructure-deficiency columns (elec, water, sanitation, aid) are
        # INVERTED so that a school with 0% electrification or no aid scores
        # highest (most critical), not lowest.
        df["priority_score"] = (
            df["pov_norm"]                  * W_POVERTY      +   # 0.30
            df["lit_norm"]                  * W_LITERACY     +   # 0.25
            (1 - df["elec_norm"])           * W_ELEC         +   # 0.20  inverted
            (1 - df["water_norm"])          * W_WATER        +   # 0.15  inverted
            (1 - df["sanitation_norm"])     * W_SANITATION   +   # 0.07  inverted
            (1 - df["aid_norm"])            * W_AID              # 0.03  inverted
        )

        # --- Sort ---
        df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

        # --- Attach human-readable tier label ---
        def _tier(score: float) -> str:
            if score > THRESHOLD_CRITICAL:
                return "CRITICAL"
            if score > THRESHOLD_HIGH:
                return "HIGH"
            return "STABLE"

        df["priority_tier"] = df["priority_score"].apply(_tier)

        # --- Standardise column names to lowercase for app.py compatibility ---
        df.columns = df.columns.str.lower().str.strip()

        # --- Save intermediate + final outputs ---
        df.to_csv(CSV_CLEAN_STEP2, index=False)
        df.to_csv(CSV_PRIORITY_RANKED, index=False)

        critical = (df["priority_score"] > THRESHOLD_CRITICAL).sum()
        high     = ((df["priority_score"] > THRESHOLD_HIGH) &
                    (df["priority_score"] <= THRESHOLD_CRITICAL)).sum()
        stable   = (df["priority_score"] <= THRESHOLD_HIGH).sum()

        self._logger.info(f"[Step 2] Done. Scores computed for {len(df)} schools.")
        self._logger.info(f"         CRITICAL (>{THRESHOLD_CRITICAL:.0%}): {critical}")
        self._logger.info(f"         HIGH     (>{THRESHOLD_HIGH:.0%}):  {high}")
        self._logger.info(f"         STABLE:  {stable}")
        self._logger.info(f"         Saved to {CSV_CLEAN_STEP2}")
        self._logger.info(f"         Saved to {CSV_PRIORITY_RANKED} (app-ready)")
        return df

    # ------------------------------------------------------------------
    # GPS helpers  (extracted from link_gps.py / complete_mapping.py)
    # ------------------------------------------------------------------

    def link_coordinates(
        self,
        df: pd.DataFrame,
        geojson_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Attach GPS coordinates from the HOT OSM GeoJSON to a schools DataFrame.

        Tries exact match first, then difflib fuzzy match at the configured
        cutoff (``FUZZY_CUTOFF``).  Mirrors the logic in link_gps.py.

        A hardcoded ``_GPS_OVERRIDES`` dict provides authoritative district-centroid
        coordinates for schools that are persistently unmatched by HOTOSM/Nominatim
        (TVET/vocational institutes with non-standard naming).

        Parameters
        ----------
        df : DataFrame
            Must contain a ``School_Name`` column.
        geojson_path : Path, optional
            Override the default GeoJSON source.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``latitude`` and ``longitude`` columns
            filled where a match was found.
        """
        # Authoritative fallback coordinates for schools persistently unmatched
        # by HOTOSM or Nominatim. Source: GES district capital centroids (GSS 2021).
        _GPS_OVERRIDES: dict[str, tuple[float, float]] = {
            "BUIPE TECH/VOC INST.":               (9.0030,  -1.8215),  # Buipe, Central Gonja
            "WALEWALETECH/ VOC INST.":            (10.3557, -0.3691),  # Walewale, West Mamprusi
            "BIAKOYE COMM. DAY SCHOOL":           (7.2501,   0.4526),  # Nkonya Wurupong, Biakoye
            "FR. DOGLI MEMORIAL VOC.TECH. INST.": (7.3547,   0.3637),  # Jasikan, Oti
        }

        geojson_path = geojson_path or GEOJSON_HDX
        if not geojson_path.exists():
            self._logger.warning(f"GeoJSON not found at {geojson_path}. Skipping GPS linkage.")
            return df

        self._logger.info(f"Loading GeoJSON from {geojson_path}")
        with open(geojson_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Build lookup: UPPER_NAME -> (lat, lon)
        hdx_points: dict[str, tuple[float, float]] = {}
        for feature in data["features"]:
            name = feature["properties"].get("name")
            if name:
                coords = feature["geometry"]["coordinates"]  # [lon, lat]
                hdx_points[name.upper()] = (coords[1], coords[0])

        hdx_names = list(hdx_points.keys())

        if "latitude" not in df.columns:
            df["latitude"] = None
        if "longitude" not in df.columns:
            df["longitude"] = None

        matched = 0
        for idx, row in df.iterrows():
            # Skip if already geocoded
            if pd.notna(row.get("latitude")):
                continue
            school = str(row["School_Name"]).upper()
            if school in hdx_points:
                lat, lon = hdx_points[school]
                matched += 1
            else:
                candidates = get_close_matches(school, hdx_names, n=1, cutoff=FUZZY_CUTOFF)
                if candidates:
                    lat, lon = hdx_points[candidates[0]]
                    matched += 1
                else:
                    continue
            df.at[idx, "latitude"]  = lat
            df.at[idx, "longitude"] = lon

        total = len(df)
        self._logger.info(
            f"GPS linkage: {matched}/{total} matched  "
            f"({total - matched} unresolved)."
        )

        # Apply hardcoded overrides for persistently unmatched schools
        override_count = 0
        for idx, row in df.iterrows():
            if pd.notna(row.get("latitude")):
                continue
            key = str(row.get("School_Name", "")).upper().strip()
            if key in _GPS_OVERRIDES:
                df.at[idx, "latitude"]  = _GPS_OVERRIDES[key][0]
                df.at[idx, "longitude"] = _GPS_OVERRIDES[key][1]
                override_count += 1
        if override_count:
            self._logger.info(f"GPS overrides applied: {override_count} schools.")

        return df

    def geocode_missing(
        self,
        df: pd.DataFrame,
        *,
        delay: float = 2.0,
    ) -> pd.DataFrame:
        """
        Fallback geocoding via Nominatim for schools still missing coordinates.

        Applies TVET abbreviation expansion (``TVET_EXPANSIONS``) to maximise
        hit rate.  Rate-limited to Nominatim's 1 req/s policy with a configurable
        ``delay``.  Mirrors the logic in complete_mapping.py.

        Parameters
        ----------
        df : DataFrame
            Must contain ``School_Name``, ``Region``, ``latitude`` columns.
        delay : float
            Seconds to wait between Nominatim requests (default 2.0).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with additional coordinates filled in.
        """
        try:
            from geopy.geocoders import Nominatim  # type: ignore
            from geopy.extra.rate_limiter import RateLimiter  # type: ignore
        except ImportError:
            self._logger.error(
                "geopy is not installed. Run: pip install geopy"
            )
            return df

        geolocator = Nominatim(user_agent="eduinfra_ghana_pipeline", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=delay)

        missing_mask = df["latitude"].isna()
        self._logger.info(f"Geocoding {missing_mask.sum()} schools via Nominatim…")

        for idx, row in df[missing_mask].iterrows():
            name   = str(row["School_Name"])
            region = str(row.get("Region", ""))
            expanded = _expand_tvet(name)
            queries = [
                f"{name}, {region}, Ghana",
                f"{expanded}, {region}, Ghana",
                f"{name.split(',')[0]}, {region}, Ghana",
            ]
            for query in queries:
                try:
                    self._logger.debug(f"Geocoding: {query}")
                    loc = geocode(query)
                    if loc:
                        df.at[idx, "latitude"]  = loc.latitude
                        df.at[idx, "longitude"] = loc.longitude
                        break
                except Exception:
                    time.sleep(delay)
        return df

    # ------------------------------------------------------------------
    # Orchestrator – run()
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        geocode_fallback: bool = False,
    ) -> pd.DataFrame:
        """
        Execute the full pipeline end-to-end.

        Steps
        -----
        1. ``clean_data()``   — load, merge socioeconomic data, impute
        2. ``calculate_scores()`` — normalise, compute priority score, rank

        Optionally:
        3. ``geocode_missing()`` — Nominatim fallback (slow; opt-in only)

        Intermediate files written to ``data/clean/``:
          - ``01_schools_cleaned.csv``
          - ``02_schools_scored.csv``

        Final output written to:
          - ``data/schools_priority_ranked.csv``  (read by app.py)

        Parameters
        ----------
        geocode_fallback : bool
            If True, run Nominatim geocoding on schools that are still
            missing coordinates after clean_data(). Disabled by default
            because it is slow (~2 s/request).

        Returns
        -------
        pd.DataFrame
            Final ranked DataFrame ready for visualisation.
        """
        self._logger.info("=" * 60)
        self._logger.info("EduInfra Ghana Pipeline — starting full run")
        self._logger.info("=" * 60)

        cleaned = self.clean_data()

        if geocode_fallback:
            cleaned = self.geocode_missing(cleaned)
            # Persist enriched coordinates before scoring
            cleaned.to_csv(CSV_CLEAN_STEP1, index=False)

        # Step 1b: enrich with electrification, water, sanitation, aid
        enriched = self.enrich_features(cleaned)

        ranked = self.calculate_scores(enriched)

        # ── Train & save Random Forest (v2, 6-feature) ───────────────────────
        # Features: pov_norm, lit_norm + four enrichment norms.
        # Target: priority_score (the composite we just computed).
        # Using all available rows (no train/test split here — this is a
        # prioritisation model, not a generalisation model; the score itself
        # IS the ground truth).  We report R² and MAE as a sanity check.
        feature_cols = ["pov_norm", "lit_norm"] + ENRICHMENT_NORM_COLS

        # Encode any remaining categoricals needed (region, category, etc.)
        # For the RF we work only on the six numeric feature columns.
        X = ranked[feature_cols].copy()
        y = ranked["priority_score"].copy()

        # Fill any edge-case NaNs that survived (should be zero after enrich)
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)

        y_pred = rf.predict(X)
        r2  = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(rf, fh)

        print(f"\n═ Random Forest v2 ═")
        print(f"  Features : {feature_cols}")
        print(f"  Samples  : {len(X):,}")
        print(f"  R²       : {r2:.6f}")
        print(f"  MAE      : {mae:.6f}")
        print(f"  Model saved → {MODEL_PATH}\n")

        self._logger.info(f"RF v2 — R²: {r2:.6f}  MAE: {mae:.6f}  → {MODEL_PATH}")
        # ──────────────────────────────────────────────────────────── ───────────────────────────────────────────────────
        # Count how many rows we started with vs ended with, and report
        # on MPI coverage, GPS coverage, and any silent drops.
        _raw_count = 0
        try:
            import pandas as _pd
            _raw_df = _pd.read_csv(self.raw_path, encoding="utf-8")
            _raw_count = len(_raw_df)
        except Exception:
            _raw_count = len(ranked)  # fallback: no drop info available

        _final_count   = len(ranked)
        _gps_count     = int(ranked[["latitude", "longitude"]].notna().all(axis=1).sum())
        _mpi_count     = int(ranked["mpi_score"].notna().sum()) if "mpi_score" in ranked.columns else 0
        _lit_count     = int(ranked["youth_literacy_count"].notna().sum()) if "youth_literacy_count" in ranked.columns else 0
        _lost          = _raw_count - _final_count
        _retention_pct = (_final_count / _raw_count * 100) if _raw_count else 100.0

        self._logger.info("")
        self._logger.info("═" * 60)
        self._logger.info("DATA RETENTION REPORT")
        self._logger.info("═" * 60)
        self._logger.info(f"  Raw schools loaded    : {_raw_count:>6,}")
        self._logger.info(f"  Final ranked output   : {_final_count:>6,}")
        self._logger.info(f"  Schools lost (drops)  : {_lost:>6,}")
        self._logger.info(f"  Retention rate        : {_retention_pct:.2f}%")
        self._logger.info(f"  GPS-resolved schools  : {_gps_count:>6,} / {_final_count}")
        self._logger.info(f"  MPI coverage          : {_mpi_count:>6,} / {_final_count}")
        self._logger.info(f"  Literacy coverage     : {_lit_count:>6,} / {_final_count}")
        if _lost > 0:
            self._logger.warning(
                f"  ⚠️  {_lost} school(s) were lost during processing. "
                "Review clean_data() merge logic or raw CSV for duplicate/empty rows."
            )
        else:
            self._logger.info("  ✅  Zero-loss: all schools accounted for.")
        self._logger.info("═" * 60)
        # ── End Retention Report ─────────────────────────────────────────────────

        self._logger.info("=" * 60)
        self._logger.info(
            f"Pipeline complete. {len(ranked)} schools ranked. "
            f"Output: {CSV_PRIORITY_RANKED}"
        )
        self._logger.info("=" * 60)
        return ranked

    # ------------------------------------------------------------------
    # Error-resilient convenience wrapper
    # ------------------------------------------------------------------

    def safe_run(self, **kwargs) -> Optional[pd.DataFrame]:
        """
        Run the full pipeline with comprehensive error handling.

        Unlike ``run()``, this method never raises — it catches every
        exception, logs it to ``logs/system.log``, prints a friendly
        message, and returns None so callers can check gracefully.

        Returns
        -------
        pd.DataFrame or None
            Ranked DataFrame on success; None on any failure.
        """
        try:
            return self.run(**kwargs)
        except FileNotFoundError as exc:
            self._logger.error(f"Data file missing: {exc}")
            print(
                f"\n⚠️  Raw data missing. Please ensure "
                f"'ghana_schools_master_2025.csv' is in the data folder.\n"
                f"   Detail: {exc}\n"
                f"   Full trace saved to logs/system.log"
            )
            return None
        except ValueError as exc:
            self._logger.error(f"Data validation error: {exc}", exc_info=True)
            print(
                f"\n❌ Data error: {exc}\n"
                f"   Full trace saved to logs/system.log"
            )
            return None
        except Exception as exc:
            self._logger.error(
                f"Unexpected pipeline failure: {exc}", exc_info=True
            )
            print(
                f"\n❌ Unexpected error: {exc}\n"
                f"   Full trace saved to logs/system.log"
            )
            return None
