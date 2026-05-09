"""
src/cli.py
==========
EduInfra Ghana — Professional Terminal Interface (Click CLI)

Entry points
------------
Run from the project root:

    python -m src.cli process --region Northern --tier CRITICAL --format csv
    python -m src.cli report

Or, if installed via setup.py / pyproject.toml:

    eduinfra process --region Northern --tier CRITICAL --format all
    eduinfra report
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import click
import pandas as pd

from src.logger import get_logger
from src.config import (
    CSV_PRIORITY_RANKED,
    CSV_MASTER_COMPLETE,
    CSV_MASTER_MAPPED,
    THRESHOLD_CRITICAL,
    THRESHOLD_HIGH,
    OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Module logger — writes to logs/system.log AND the daily rotating log file
# ---------------------------------------------------------------------------

logger = get_logger("CLI_Interface")

# Add a dedicated system.log handler (append mode, UTF-8)
_sys_log_path = Path(__file__).resolve().parent.parent / "logs" / "system.log"
_sys_log_path.parent.mkdir(parents=True, exist_ok=True)
_sys_handler = logging.FileHandler(_sys_log_path, mode="a", encoding="utf-8")
_sys_handler.setLevel(logging.DEBUG)
_sys_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logger.addHandler(_sys_handler)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_ranked_data() -> pd.DataFrame:
    """
    Load the pipeline-ranked CSV.

    Raises a styled Click error — never a raw exception — if the file is
    missing, so the terminal always receives a human-readable message.
    """
    if not CSV_PRIORITY_RANKED.exists():
        msg = (
            "\n⚠️  Ranked data missing.\n"
            f"   Expected: {CSV_PRIORITY_RANKED}\n\n"
            "   Run the pipeline first:\n"
            "     python -m src.cli process\n"
        )
        logger.error(f"Ranked CSV not found: {CSV_PRIORITY_RANKED}")
        raise click.ClickException(msg)

    try:
        df = pd.read_csv(CSV_PRIORITY_RANKED, encoding="utf-8")
        logger.info(f"Loaded ranked data: {len(df):,} schools from {CSV_PRIORITY_RANKED}")
        return df
    except Exception as exc:
        logger.error(f"Failed to read ranked CSV: {exc}", exc_info=True)
        raise click.ClickException(f"Could not read data file: {exc}") from exc


def _check_raw_data() -> bool:
    """Return True if at least one raw master CSV is present."""
    return CSV_MASTER_COMPLETE.exists() or CSV_MASTER_MAPPED.exists()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="2.0.0", prog_name="EduInfra Ghana CLI")
def cli():
    """
    \b
    ╔══════════════════════════════════════════════════╗
    ║   🇬🇭  EduInfra Ghana — AI Infrastructure CLI   ║
    ║   Mapping school deprivation across all regions  ║
    ╚══════════════════════════════════════════════════╝

    Use --help on any command for details.
    """
    pass


# ---------------------------------------------------------------------------
# Command: process
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--region",
    default="all",
    show_default=True,
    help="Filter results to a specific region (e.g. 'Northern'). Use 'all' for no filter.",
)
@click.option(
    "--tier",
    default=None,
    type=click.Choice(["CRITICAL", "HIGH", "STABLE"], case_sensitive=False),
    help="Filter by priority tier. Omit to include all tiers.",
)
@click.option(
    "--format",
    "out_format",
    default="csv",
    show_default=True,
    type=click.Choice(["csv", "json", "all"], case_sensitive=False),
    help="Export format.",
)
@click.option(
    "--geocode",
    is_flag=True,
    default=False,
    help="Run Nominatim geocoding fallback for schools missing GPS (slow).",
)
@click.option(
    "--top-n",
    default=None,
    type=int,
    help="Limit export to the top N schools by priority score.",
)
def process(region, tier, out_format, geocode, top_n):
    """
    Run the full AI pipeline and export results.

    \b
    Examples:
      python -m src.cli process
      python -m src.cli process --region Northern --tier CRITICAL --format all
      python -m src.cli process --top-n 50 --format json
    """
    from src.pipeline import EduInfraPipeline
    from src.exporter import DataExporter

    click.echo()
    click.secho("🇬🇭  EduInfra Ghana — AI Pipeline", fg="yellow", bold=True)
    click.secho("=" * 48, fg="yellow")

    # --- Guard: raw data must exist ---
    if not _check_raw_data():
        msg = (
            "\n⚠️  Raw data missing. Please ensure "
            "'ghana_schools_master_2025.csv' is in the data folder.\n"
            f"   Looked for: {CSV_MASTER_COMPLETE}\n"
            f"           or: {CSV_MASTER_MAPPED}\n"
        )
        logger.error("Raw master CSV not found. Pipeline aborted.")
        click.secho(msg, fg="red", err=True)
        sys.exit(1)

    # --- Pipeline steps with progress bar ---
    pipeline = EduInfraPipeline()
    steps = [
        ("📥  Loading & cleaning raw school data …", "clean_data"),
        ("🧮  Computing AI priority scores …",       "calculate_scores"),
    ]
    if geocode:
        steps.insert(1, ("🌍  Running GPS geocoding fallback …", "geocode_missing"))

    df = None
    try:
        with click.progressbar(
            steps,
            label="  Running pipeline",
            bar_template="%(label)s  %(bar)s  %(info)s",
            fill_char=click.style("█", fg="green"),
            empty_char="░",
            width=36,
        ) as bar:
            for label, step_name in bar:
                time.sleep(0.1)   # small pause so the bar renders visibly

                if step_name == "clean_data":
                    df = pipeline.clean_data()

                elif step_name == "geocode_missing" and df is not None:
                    df = pipeline.geocode_missing(df)

                elif step_name == "calculate_scores":
                    df = pipeline.calculate_scores(df)

    except FileNotFoundError as exc:
        logger.error(f"Pipeline FileNotFoundError: {exc}", exc_info=True)
        click.echo()
        click.secho(
            f"\n⚠️  Raw data missing. Please ensure "
            f"'ghana_schools_master_2025.csv' is in the data folder.\n"
            f"   Detail: {exc}",
            fg="red", err=True,
        )
        sys.exit(1)

    except Exception as exc:
        logger.error(f"Pipeline failed unexpectedly: {exc}", exc_info=True)
        click.echo()
        click.secho(f"\n❌ Pipeline error: {exc}", fg="red", err=True)
        click.secho(
            "   Full trace written to logs/system.log", fg="yellow", err=True
        )
        sys.exit(1)

    click.echo()

    if df is None or df.empty:
        click.secho("⚠️  Pipeline produced no data. Check logs/system.log.", fg="red")
        sys.exit(1)

    # --- Apply filters ---
    col_region = "region" if "region" in df.columns else "Region"
    col_tier   = "priority_tier" if "priority_tier" in df.columns else None
    col_score  = "priority_score"

    if region.lower() != "all":
        before = len(df)
        df = df[df[col_region].str.lower() == region.lower()]
        click.secho(
            f"  🔍 Region filter '{region}': {before:,} → {len(df):,} schools",
            fg="cyan",
        )

    if tier and col_tier and col_tier in df.columns:
        before = len(df)
        df = df[df[col_tier].str.upper() == tier.upper()]
        click.secho(
            f"  🏷️  Tier filter '{tier}': {before:,} → {len(df):,} schools",
            fg="cyan",
        )

    if top_n:
        df = df.nlargest(top_n, col_score)
        click.secho(f"  🔝 Top-{top_n} schools selected.", fg="cyan")

    if df.empty:
        click.secho(
            "\n⚠️  No schools match the given filters. "
            "Try broadening --region or --tier.",
            fg="yellow",
        )
        sys.exit(0)

    # --- Export ---
    exporter = DataExporter()
    try:
        click.echo()
        click.secho("📦  Exporting results …", fg="blue")

        if out_format == "all":
            paths = exporter.export_all(df)
            for fmt, path in paths.items():
                click.secho(f"   ✅ {fmt.upper():8s} → {path}", fg="green")
        elif out_format == "csv":
            path = exporter.export_csv(df)
            click.secho(f"   ✅ CSV     → {path}", fg="green")
        else:
            path = exporter.export_json(df)
            click.secho(f"   ✅ JSON    → {path}", fg="green")

    except Exception as exc:
        logger.error(f"Export failed: {exc}", exc_info=True)
        click.secho(f"\n❌ Export error: {exc}", fg="red", err=True)
        sys.exit(1)

    # --- Summary footer ---
    critical_n = int((df[col_score] > THRESHOLD_CRITICAL).sum())
    high_n     = int(((df[col_score] > THRESHOLD_HIGH) & (df[col_score] <= THRESHOLD_CRITICAL)).sum())
    stable_n   = int((df[col_score] <= THRESHOLD_HIGH).sum())

    click.echo()
    click.secho("─" * 48, fg="yellow")
    click.secho("  📊 Export summary", bold=True)
    click.secho(f"     Total exported : {len(df):,} schools")
    click.secho(f"     🔴 CRITICAL     : {critical_n:,}")
    click.secho(f"     🟡 HIGH         : {high_n:,}")
    click.secho(f"     🟢 STABLE       : {stable_n:,}")
    click.secho("─" * 48, fg="yellow")
    click.echo()
    logger.info(
        f"process command complete | region={region} | tier={tier} | "
        f"format={out_format} | exported={len(df)}"
    )


# ---------------------------------------------------------------------------
# Command: report
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--region",
    default="all",
    show_default=True,
    help="Scope the report to a specific region.",
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help="Also save the summary report JSON to outputs/.",
)
def report(region, save):
    """
    Print a quick data-health summary to the terminal.

    \b
    Examples:
      python -m src.cli report
      python -m src.cli report --region 'Upper West' --save
    """
    from src.exporter import DataExporter

    click.echo()
    click.secho("🇬🇭  EduInfra Ghana — Data Health Report", fg="yellow", bold=True)
    click.secho("=" * 48, fg="yellow")

    try:
        df = _load_ranked_data()
    except click.ClickException as exc:
        click.secho(str(exc.format_message()), fg="red", err=True)
        sys.exit(1)

    col_region = "region" if "region" in df.columns else "Region"
    col_score  = "priority_score"
    col_name   = "school_name" if "school_name" in df.columns else "School_Name"
    lat_col    = "latitude" if "latitude" in df.columns else None

    # --- Region filter ---
    if region.lower() != "all":
        df = df[df[col_region].str.lower() == region.lower()]
        if df.empty:
            click.secho(
                f"\n⚠️  No data found for region '{region}'.", fg="yellow"
            )
            sys.exit(0)

    # --- Stats ---
    total       = len(df)
    critical_n  = int((df[col_score] > THRESHOLD_CRITICAL).sum())
    high_n      = int(((df[col_score] > THRESHOLD_HIGH) & (df[col_score] <= THRESHOLD_CRITICAL)).sum())
    stable_n    = int((df[col_score] <= THRESHOLD_HIGH).sum())
    avg_score   = df[col_score].mean()
    max_score   = df[col_score].max()

    gps_pct = None
    if lat_col:
        gps_pct = df[lat_col].notna().mean() * 100

    region_label = region if region.lower() != "all" else "All Regions"

    click.echo()
    click.secho(f"  📍 Scope         : {region_label}", bold=True)
    click.secho(f"  🏫 Total schools : {total:,}")
    click.echo()
    click.secho(f"  Priority tiers:")
    click.secho(f"     🔴 CRITICAL (>{THRESHOLD_CRITICAL:.0%}) : {critical_n:,}", fg="red")
    click.secho(f"     🟡 HIGH     (>{THRESHOLD_HIGH:.0%}) : {high_n:,}",   fg="yellow")
    click.secho(f"     🟢 STABLE              : {stable_n:,}",              fg="green")
    click.echo()
    click.secho(f"  📈 Score stats:")
    click.secho(f"     Average score    : {avg_score:.3f}")
    click.secho(f"     Highest score    : {max_score:.3f}")
    if gps_pct is not None:
        color = "green" if gps_pct >= 80 else ("yellow" if gps_pct >= 50 else "red")
        click.secho(f"     GPS coverage     : {gps_pct:.1f}%", fg=color)

    # --- Top 5 critical ---
    top5 = df[df[col_score] > THRESHOLD_CRITICAL].nlargest(5, col_score)
    if not top5.empty:
        click.echo()
        click.secho("  🚨 Top 5 CRITICAL schools:", bold=True, fg="red")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            name   = row[col_name]
            r      = row[col_region]
            score  = row[col_score]
            click.secho(f"     {i}. {name} ({r}) — {score:.3f}", fg="red")

    # --- Optional save ---
    if save:
        click.echo()
        exporter = DataExporter()
        try:
            path = exporter.export_summary_report(df, filename="summary_report.json")
            click.secho(f"  💾 Report saved → {path}", fg="green")
        except Exception as exc:
            logger.error(f"Report save failed: {exc}", exc_info=True)
            click.secho(f"  ❌ Could not save report: {exc}", fg="red")

    click.echo()
    click.secho("─" * 48, fg="yellow")
    click.secho(
        "  Tip: run 'python -m src.cli process' to refresh data.",
        fg="bright_black",
    )
    click.echo()
    logger.info(f"report command complete | region={region} | total={total}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
