"""
src/exporter.py
===============
DataExporter — multi-format export layer for the EduInfra Ghana pipeline.

Handles UTF-8 encoding throughout to protect Twi-language school names
(e.g. "Kwame Nkrumah" variants, Akan district names such as "Bosomtwe").

Typical usage
-------------
    from src.exporter import DataExporter

    exporter = DataExporter()
    exporter.export_csv(df)
    exporter.export_json(df)
    exporter.export_summary_report(df, mae=0.043, r_squared=0.91)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.logger import get_logger
from src.config import OUTPUT_DIR, THRESHOLD_CRITICAL

logger = get_logger("DataExporter")


class DataExporter:
    """
    Saves scored school DataFrames to the ``outputs/`` directory in
    multiple professional formats.

    Parameters
    ----------
    output_dir : Path or str, optional
        Destination directory.  Defaults to ``outputs/`` at project root
        (resolved via ``src.config.OUTPUT_DIR``).
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataExporter initialised → {self.output_dir}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_csv(
        self,
        df: pd.DataFrame,
        filename: str = "schools_final.csv",
    ) -> Path:
        """
        Write *df* to a UTF-8 CSV file.

        UTF-8 is explicitly set so Windows systems don't silently re-encode
        Twi-language characters (e.g. Ɔ, ɛ) when opening the file in Excel.
        Instruct users to "Import > UTF-8" in Excel if characters look wrong.

        Parameters
        ----------
        df : pd.DataFrame
            Scored school records.
        filename : str
            Output filename.  Written inside ``self.output_dir``.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        path = self.output_dir / filename
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            # utf-8-sig adds a BOM so Microsoft Excel auto-detects UTF-8
            logger.info(f"✅ CSV exported → {path}  ({len(df):,} rows)")
            return path
        except OSError as exc:
            logger.error(f"❌ CSV export failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def export_json(
        self,
        df: pd.DataFrame,
        filename: str = "schools_final.json",
    ) -> Path:
        """
        Write *df* to a pretty-printed UTF-8 JSON file.

        Float columns (latitude, longitude, priority_score) are kept as
        native JSON numbers; all other non-serialisable types (e.g. numpy
        int64, NaT) are coerced to strings via ``default=str``.

        Parameters
        ----------
        df : pd.DataFrame
            Scored school records.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        path = self.output_dir / filename
        try:
            # Convert to plain Python dicts; ``default=str`` handles numpy types
            records = df.where(pd.notnull(df), None).to_dict(orient="records")
            payload = {
                "meta": {
                    "count": len(records),
                    "source": "EduInfra Ghana Pipeline v2.0",
                    "encoding": "UTF-8",
                },
                "data": records,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            # ensure_ascii=False keeps Twi characters as literal Unicode in the file
            logger.info(f"✅ JSON exported → {path}  ({len(records):,} records)")
            return path
        except OSError as exc:
            logger.error(f"❌ JSON export failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def export_summary_report(
        self,
        df: pd.DataFrame,
        mae: Optional[float] = None,
        r_squared: Optional[float] = None,
        filename: str = "summary_report.json",
    ) -> Path:
        """
        Write a compact JSON report for judges / stakeholders.

        The report contains:
        - Model performance metrics (MAE, R²) if provided.
        - Count of schools at each priority tier.
        - Top-5 most critical schools by name and score.
        - Dataset coverage statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Scored school records (must contain ``priority_score``
            and ``school_name`` or ``School_Name`` columns).
        mae : float, optional
            Mean Absolute Error from model evaluation.
        r_squared : float, optional
            R-squared coefficient from model evaluation.
        filename : str
            Output filename inside ``self.output_dir``.

        Returns
        -------
        Path
            Absolute path of the written report.
        """
        path = self.output_dir / filename

        # --- Normalise column name casing ---
        col_name = "school_name" if "school_name" in df.columns else "School_Name"
        col_score = "priority_score"
        col_region = "region" if "region" in df.columns else "Region"
        col_tier = "priority_tier" if "priority_tier" in df.columns else None

        # --- Tier counts ---
        critical_mask = df[col_score] > THRESHOLD_CRITICAL
        high_mask = (df[col_score] > 0.50) & ~critical_mask
        stable_mask = df[col_score] <= 0.50

        critical_count = int(critical_mask.sum())
        high_count = int(high_mask.sum())
        stable_count = int(stable_mask.sum())

        # --- Top 5 critical schools ---
        top5 = (
            df[critical_mask]
            .nlargest(5, col_score)[[col_name, col_region, col_score]]
            .to_dict(orient="records")
        )
        # Ensure scores are plain floats for JSON
        for rec in top5:
            rec[col_score] = round(float(rec[col_score]), 4)

        # --- GPS coverage ---
        lat_col = "latitude" if "latitude" in df.columns else None
        gps_coverage = None
        if lat_col:
            gps_coverage = round(df[lat_col].notna().mean() * 100, 1)

        report = {
            "report_title": "EduInfra Ghana — Pipeline Summary Report",
            "dataset": {
                "total_schools": len(df),
                "gps_coverage_pct": gps_coverage,
            },
            "model_performance": {
                "mae": round(float(mae), 4) if mae is not None else "N/A",
                "r_squared": round(float(r_squared), 4) if r_squared is not None else "N/A",
                "note": (
                    "MAE and R² reflect the Random-Forest gap-score predictor "
                    "trained on MPI + literacy features."
                ),
            },
            "priority_distribution": {
                "CRITICAL": critical_count,
                "HIGH": high_count,
                "STABLE": stable_count,
            },
            "top_5_critical_schools": top5,
        }

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
            logger.info(
                f"✅ Summary report exported → {path}  "
                f"(CRITICAL={critical_count}, HIGH={high_count}, STABLE={stable_count})"
            )
            return path
        except OSError as exc:
            logger.error(f"❌ Summary report export failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def export_all(
        self,
        df: pd.DataFrame,
        mae: Optional[float] = None,
        r_squared: Optional[float] = None,
    ) -> dict[str, Path]:
        """Run all three exports in one call."""
        return {
            "csv": self.export_csv(df),
            "json": self.export_json(df),
            "summary": self.export_summary_report(df, mae=mae, r_squared=r_squared),
        }
