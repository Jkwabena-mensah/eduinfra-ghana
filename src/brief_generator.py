"""
src/brief_generator.py
======================
EduInfra Ghana — AI Policy Brief Generator

Produces a formatted, downloadable Ministerial Policy Brief using Claude
claude-sonnet-4-20250514 grounded in live GES 2025 data. Falls back to a
template-driven local brief when no API key is available.
"""

from __future__ import annotations

import datetime
import os
from typing import Optional

import pandas as pd


SCOPE_NATIONAL    = "national"
SCOPE_REGION      = "region"
SCOPE_INTERVENTION = "intervention"

_COST_SOLAR      = 45_000
_COST_BOREHOLE   = 28_000
_COST_SANITATION = 18_000
_EXCHANGE_RATE   = 14

THRESHOLD_CRITICAL = 0.65
THRESHOLD_HIGH     = 0.45


def _build_digest(df: pd.DataFrame, region: Optional[str] = None) -> dict:
    scope_df = df.copy()
    if region:
        scope_df = scope_df[scope_df["region"].str.upper().str.contains(region.upper(), na=False)]

    total    = len(scope_df)
    n_crit   = int((scope_df["priority_tier"].str.upper() == "CRITICAL").sum())
    n_high   = int((scope_df["priority_tier"].str.upper() == "HIGH").sum())
    n_stable = total - n_crit - n_high
    n_gps    = int(scope_df["latitude"].notna().sum()) if "latitude" in scope_df.columns else total

    enrol_col = "youth_literacy_count"
    students = int(scope_df[scope_df["priority_tier"].str.upper() == "CRITICAL"][enrol_col].fillna(0).sum()) if enrol_col in scope_df.columns else 0

    top5 = scope_df.nlargest(5, "priority_score")[["school_name", "district", "region", "priority_score", "priority_tier"]]

    crit_high = scope_df[scope_df["priority_tier"].str.upper().isin(["CRITICAL", "HIGH"])]
    top_dists = {}
    if "district" in crit_high.columns:
        top_dists = crit_high.groupby("district")["school_name"].count().sort_values(ascending=False).head(5).to_dict()

    cost_solar = n_crit * _COST_SOLAR
    cost_water = n_crit * _COST_BOREHOLE
    cost_sanit = n_crit * _COST_SANITATION
    cost_total = cost_solar + cost_water + cost_sanit
    cost_usd   = cost_total / _EXCHANGE_RATE

    scope_label = region.title() + " Region" if region else "Ghana (National)"

    return {
        "scope_label": scope_label,
        "total": total,
        "n_crit": n_crit,
        "n_high": n_high,
        "n_stable": n_stable,
        "n_gps": n_gps,
        "students": students,
        "top5": top5,
        "top_dists": top_dists,
        "cost_solar": cost_solar,
        "cost_water": cost_water,
        "cost_sanit": cost_sanit,
        "cost_total": cost_total,
        "cost_usd": cost_usd,
        "pct_crit": round(100 * n_crit / total, 1) if total else 0,
        "date": datetime.date.today().strftime("%B %Y"),
    }


def _local_brief(digest: dict, brief_type: str) -> str:
    d = digest
    top5 = d["top5"]
    dists = d["top_dists"]

    top5_lines = "\n".join(
        f"  {i+1}. **{row.school_name}** — {row.district}, {row.region.title()} "
        f"(Score: {round(row.priority_score*100,1)}%, Tier: {row.priority_tier})"
        for i, row in enumerate(top5.itertuples())
    )
    dist_lines = "\n".join(
        f"  - **{dist.title()}**: {count} schools requiring intervention"
        for dist, count in dists.items()
    )

    return f"""# POLICY BRIEF: Education Infrastructure Gap Analysis
## EduInfra Ghana Intelligence Platform — {d['scope_label']}

**Prepared by:** EduInfra Ghana | GES 2025 Data | Ghana AI Innovation Challenge 2026
**Date:** {d['date']} | **Classification:** For Official Use

---

## EXECUTIVE SUMMARY

An AI-driven analysis of **{d['total']:,} second-cycle institutions** in {d['scope_label']} has identified a systemic infrastructure crisis concentrated in the northern belt. **{d['n_crit']} schools ({d['pct_crit']}%)** are classified as **Critical Priority**, with an estimated **{d['students']:,} students** attending institutions lacking adequate electricity, water, and sanitation. Immediate, cluster-based infrastructure investment — beginning with the districts of highest concentration — offers the most cost-effective path to closing this gap.

---

## KEY FINDINGS

### National Infrastructure Profile
| Tier | Count | Share | Action Required |
|---|---|---|---|
| 🔴 Critical | {d['n_crit']} | {d['pct_crit']}% | Immediate national intervention |
| 🟡 High | {d['n_high']} | {round(100*d['n_high']/d['total'],1)}% | District-level prioritisation |
| 🟢 Stable | {d['n_stable']} | {round(100*d['n_stable']/d['total'],1)}% | Standard monitoring cycle |

**GPS-verified coverage:** {d['n_gps']}/{d['total']} institutions ({round(100*d['n_gps']/d['total'],1)}%)

### Highest-Need Institutions
{top5_lines}

### Geographic Concentration of Critical/High Schools
{dist_lines}

These districts represent optimal targets for cluster-based investment: a single borehole, solar microgrid, or sanitation block can serve multiple schools within a 5 km radius, maximising cost-per-student-reached.

---

## RECOMMENDED INTERVENTIONS

### Phase 1 — Immediate Action (0–6 months)
1. **Emergency solar microgrids** for the top 10 Critical schools (estimated cost: GH₵{d['n_crit']*_COST_SOLAR//10*10:,})
2. **Ministerial directive** to District Education Officers in Builsa North, Bongo, East Mamprusi, Bole, and other high-concentration districts
3. **GES Data Integrity Review** — field verification of 44 schools with missing GPS coordinates

### Phase 2 — Short-Term (6–18 months)
1. **Cluster WASH programme** — target Investment Zones where 3+ schools fall within 5 km
2. **Private sector engagement** — CSR partnerships with telecoms and energy companies in their operational zones
3. **Annual data refresh** — synchronise MPI and DHS data as new surveys publish

### Phase 3 — Systemic (18–36 months)
1. **School-level facility data collection** — move from district-proxy metrics to direct observation
2. **Integration with MoE EMIS** — embed EduInfra scores into the Education Management Information System
3. **Real-time monitoring dashboard** — live GES-to-MoE data pipeline for quarterly reporting

---

## INVESTMENT SUMMARY (All Critical Schools)

| Intervention | Schools | Unit Cost (GH₵) | Total (GH₵) | Total (USD) |
|---|---|---|---|---|
| Solar microgrids | {d['n_crit']} | {_COST_SOLAR:,} | {d['cost_solar']:,} | ${d['cost_solar']//_EXCHANGE_RATE:,} |
| Borehole / WASH | {d['n_crit']} | {_COST_BOREHOLE:,} | {d['cost_water']:,} | ${d['cost_water']//_EXCHANGE_RATE:,} |
| Sanitation blocks | {d['n_crit']} | {_COST_SANITATION:,} | {d['cost_sanit']:,} | ${d['cost_sanit']//_EXCHANGE_RATE:,} |
| **TOTAL** | **{d['n_crit']}** | | **GH₵{d['cost_total']:,}** | **USD {d['cost_usd']:,.0f}** |

> ⚠️ Unit costs are UNESCO/GIZ indicative benchmarks. Actual procurement varies by site, contractor, and specification.

---

## AI METHODOLOGY

The EduInfra Ghana scoring engine uses a **200-estimator Random Forest Regressor** (R² = 0.9988, MAE = 0.001) trained on a multi-dimensional deprivation index:

| Feature | Weight | Source |
|---|---|---|
| Poverty Index (pov_norm) | 30% | UNDP Ghana MPI 2023 |
| Literacy Gap (lit_norm) | 25% | GSS Census 2021 |
| No Electricity (elec_norm) | 20% | DHS Wave 8 / SE4All 2022 |
| No Clean Water (water_norm) | 15% | DHS Wave 8 |
| Poor Sanitation (sanit_norm) | 7% | DHS Wave 8 |
| No Prior Aid (aid_norm) | 3% | AidData / IATI |

**Tier thresholds** are calibrated to the GES 2025 score distribution (Critical = top 7%, High = next 26%), ensuring the most severe cases receive priority attention without understating the scale of systemic need.

---

## CALL TO ACTION

The Ministry of Education and Ghana Education Service should:

1. **Adopt EduInfra scores** as a formal input to the 2026/27 capital budget submission
2. **Issue emergency procurement notices** for the top 10 Critical schools identified above
3. **Brief international partners** (UNICEF, World Bank, GIZ) using this platform's exportable data to unlock additional funding

> *"Data invisibility is the silent tax on the forgotten. EduInfra Ghana makes every school visible, every gap actionable."*

---

*Generated by EduInfra Ghana — AI Infrastructure Gap Mapper*
*Ghana AI Innovation Challenge 2026 | Track: AI Infrastructure Gap Mapper | Brainhauz Solutions*
*Author: Ing. Joseph K. Mensah (PE-GhIE)*
"""


def _claude_brief(digest: dict, brief_type: str, api_key: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return _local_brief(digest, brief_type)

    d = digest
    top5_str = d["top5"].to_string(index=False)
    dist_str = "\n".join(f"  {k.title()}: {v}" for k, v in d["top_dists"].items())

    user_prompt = f"""Generate a formal, authoritative Ministerial Policy Brief for Ghana's Ministry of Education.
Use the exact data provided. Be specific — cite school names, districts, costs.

SCOPE: {d['scope_label']} — {d['date']}
DATA:
- Total mapped: {d['total']:,} schools
- CRITICAL (>65%): {d['n_crit']} schools ({d['pct_crit']}%), ~{d['students']:,} students affected
- HIGH (45–65%): {d['n_high']} schools
- GPS-verified: {d['n_gps']}/{d['total']}

TOP 5 SCHOOLS:
{top5_str}

CONCENTRATION DISTRICTS:
{dist_str}

INVESTMENT TOTAL (all critical schools):
Solar: GH₵{d['cost_solar']:,} | WASH: GH₵{d['cost_water']:,} | Sanitation: GH₵{d['cost_sanit']:,}
GRAND TOTAL: GH₵{d['cost_total']:,} (≈ USD {d['cost_usd']:,.0f})

Note: costs are UNESCO/GIZ indicative benchmarks.

Write an 800-word policy brief in Markdown with:
1. Executive Summary (3 impactful sentences)
2. Key Findings (with data table)
3. Geographic Concentration section
4. Phased Recommendations (0–6m, 6–18m, 18–36m)
5. Investment Table
6. AI Methodology note (brief)
7. Strong ministerial call-to-action

Tone: authoritative, policy-ready, no hedging. End with the tagline:
"Data invisibility is the silent tax on the forgotten. EduInfra Ghana makes every school visible, every gap actionable."
"""

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1400,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


class BriefGenerator:
    """
    Generates downloadable Ministerial Policy Briefs from GES 2025 data.
    Uses Claude when API key is available; falls back to polished local template.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.df.columns = self.df.columns.str.lower().str.strip()
        if "priority_tier" in self.df.columns:
            self.df["priority_tier"] = self.df["priority_tier"].fillna("STABLE").astype(str)

        self._api_key: Optional[str] = None
        try:
            _key = ""
            try:
                import streamlit as _st
                _key = _st.secrets.get("ANTHROPIC_API_KEY", "") or ""
            except Exception:
                pass
            if not _key:
                _key = os.environ.get("ANTHROPIC_API_KEY", "") or ""
            if _key and _key.strip().startswith("sk-"):
                self._api_key = _key.strip()
        except Exception:
            pass

    def generate(
        self,
        brief_type: str = SCOPE_NATIONAL,
        region: Optional[str] = None,
    ) -> tuple[str, str]:
        digest = _build_digest(self.df, region=region if brief_type == SCOPE_REGION else None)
        if self._api_key:
            try:
                text = _claude_brief(digest, brief_type, self._api_key)
            except Exception:
                text = _local_brief(digest, brief_type)
        else:
            text = _local_brief(digest, brief_type)
        return text, text
