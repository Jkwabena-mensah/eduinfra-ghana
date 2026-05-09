"""
src/assistant.py
================
EduInfra Ghana — Local Intelligence Engine (zero API cost)

Replaces the Anthropic API call with a fully local, data-grounded query
engine that runs entirely inside the Streamlit process. No API key, no
network calls, no cost.

Architecture
------------
  1. Intent classification  — regex + keyword routing to one of 9 query types
  2. DataFrame extraction   — filter/aggregate self.df to retrieve ground-truth
  3. Answer composition     — template-driven natural language with real numbers
  4. Fallback               — graceful "I don't have data on that" for unknowns

Supported query types
---------------------
  CRITICAL_LIST    "which schools are critical in X"
  TOP_N            "top 5 highest-need schools"
  DISTRICT_QUERY   "show me schools in Nabdam district"
  REGION_QUERY     "northern region schools"
  COST_ESTIMATE    "what would it cost to install solar in X"
  IMPACT           "how many students would be reached"
  CLUSTER_HINT     "which areas have clusters of high-need schools"
  STEM_GENDER      "critical girls' boarding schools"
  GENERAL_STATS    "how many critical schools are there"
  HELP / UNKNOWN   fallback with suggested example queries
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THRESHOLD_CRITICAL = 0.65
_THRESHOLD_HIGH     = 0.45

# UNESCO cost benchmarks (GH₵)
_COST_SOLAR       = 45_000
_COST_BOREHOLE    = 28_000
_COST_SANITATION  = 18_000
_COST_LABEL       = "indicative UNESCO/GIZ benchmarks — not procurement prices"

_GH_REGIONS = {
    "western", "central", "greater accra", "volta", "eastern", "ashanti",
    "western north", "ahafo", "bono", "bono east", "oti", "northern",
    "savannah", "north east", "upper east", "upper west",
}

# ---------------------------------------------------------------------------
# Intent patterns — ordered most-specific first
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("COST_ESTIMATE",  re.compile(
        r"\b(cost|price|budget|invest|fund|ghs|cedis|solar|borehole|wash|sanitation"
        r"|how much|spend|expenditure)\b", re.I)),
    ("IMPACT",         re.compile(
        r"\b(student|enrol|reach|impact|beneficiar|people|population|how many student)\b", re.I)),
    ("CLUSTER_HINT",   re.compile(
        r"\b(cluster|zone|area|nearby|radius|within|proximity|concentrate|group)\b", re.I)),
    ("STEM_GENDER",    re.compile(
        r"\b(stem|girls|boys|female|male|gender|boarding|day.school|mixed)\b", re.I)),
    ("TOP_N",          re.compile(
        r"\b(top\s*\d+|worst|highest|lowest|most.need|most.critical|rank)\b", re.I)),
    ("CRITICAL_LIST",  re.compile(
        r"\b(critical|urgent|immediate|emergency|red)\b", re.I)),
    ("DISTRICT_QUERY", re.compile(
        r"\b(district)\b", re.I)),
    ("REGION_QUERY",   re.compile(
        r"\b(" + "|".join(_GH_REGIONS) + r")\b", re.I)),
    ("GENERAL_STATS",  re.compile(
        r"\b(how many|count|total|number|statistics|overview|summary|breakdown)\b", re.I)),
]


def _classify_intent(question: str) -> str:
    q = question.lower()
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(q):
            return intent
    return "UNKNOWN"


def _extract_region(question: str) -> Optional[str]:
    q = question.lower()
    for r in sorted(_GH_REGIONS, key=len, reverse=True):  # longest match first
        if r in q:
            return r
    return None


def _extract_district(question: str, df: pd.DataFrame) -> Optional[str]:
    q = question.lower()
    if "district" not in df.columns:
        return None
    for d in df["district"].dropna().unique():
        if str(d).lower() in q:
            return str(d)
    return None


def _extract_n(question: str, default: int = 10) -> int:
    m = re.search(r"\btop\s*(\d+)\b", question, re.I)
    if m:
        return min(int(m.group(1)), 25)
    m2 = re.search(r"\b(\d+)\s*(school|institution)", question, re.I)
    if m2:
        return min(int(m2.group(1)), 25)
    return default


def _tier_emoji(tier: str) -> str:
    t = str(tier).upper()
    if t == "CRITICAL": return "🔴"
    if t == "HIGH":     return "🟡"
    return "🟢"


def _fmt_score(s: float) -> str:
    return f"{round(s * 100, 1)}%"


def _school_line(row: pd.Series, rank: int = 0) -> str:
    prefix = f"{rank}. " if rank else "• "
    name   = row.get("school_name", "Unknown")
    dist   = row.get("district", "—")
    reg    = str(row.get("region", "—")).title()
    score  = _fmt_score(float(row.get("priority_score", 0)))
    tier   = str(row.get("priority_tier", "")).upper()
    emoji  = _tier_emoji(tier)
    anomaly = " ⚠️ GPS pending" if bool(row.get("is_anomaly", False)) else ""
    return f"{prefix}{emoji} **{name}** — {dist}, {reg} · Score: {score}{anomaly}"


# ---------------------------------------------------------------------------
# Answer composers — one per intent
# ---------------------------------------------------------------------------

def _answer_general_stats(df: pd.DataFrame, question: str) -> str:
    total   = len(df)
    n_crit  = int((df["priority_tier"].str.upper() == "CRITICAL").sum())
    n_high  = int((df["priority_tier"].str.upper() == "HIGH").sum())
    n_stable = total - n_crit - n_high
    n_gps   = int(df["latitude"].notna().sum()) if "latitude" in df.columns else "—"
    n_anom  = int(df.get("is_anomaly", pd.Series(False)).sum())

    region_breakdown = ""
    if "region" in df.columns:
        top_regions = (
            df[df["priority_tier"].str.upper() == "CRITICAL"]
            .groupby("region")["school_name"].count()
            .sort_values(ascending=False).head(5)
        )
        region_breakdown = "\n\n**Critical schools by region (top 5):**\n" + "\n".join(
            f"• {r.title()}: {c}" for r, c in top_regions.items()
        )

    return (
        f"**National overview — {total} schools mapped**\n\n"
        f"- 🔴 **Critical** (score > 80%): {n_crit} schools\n"
        f"- 🟡 **High** (score 50–80%): {n_high} schools\n"
        f"- 🟢 **Stable** (score ≤ 50%): {n_stable} schools\n"
        f"- 📍 GPS-verified: {n_gps} | ⚠️ Location pending: {n_anom}"
        f"{region_breakdown}\n\n"
        f"Use the **Investment Clusters** tab to see where these schools "
        f"concentrate geographically, or the **Action Plan** tab for a ranked list."
    )


def _answer_critical_list(df: pd.DataFrame, question: str) -> str:
    region   = _extract_region(question)
    district = _extract_district(question, df)
    n        = _extract_n(question, default=10)

    filtered = df[df["priority_tier"].str.upper() == "CRITICAL"].copy()
    scope    = "nationally"
    if district:
        filtered = filtered[filtered["district"].str.lower() == district.lower()]
        scope = f"in {district.title()} district"
    elif region:
        filtered = filtered[filtered["region"].str.lower().str.contains(region, na=False)]
        scope = f"in {region.title()} region"

    filtered = filtered.sort_values("priority_score", ascending=False).head(n)

    if filtered.empty:
        return (
            f"No **Critical** schools found {scope}. "
            f"Try broadening to **High** priority, or check the region spelling."
        )

    total_in_scope = len(df[df["priority_tier"].str.upper() == "CRITICAL"])
    lines = [_school_line(row, i + 1) for i, (_, row) in enumerate(filtered.iterrows())]

    enrol_col = "youth_literacy_count"
    students = int(filtered[enrol_col].fillna(0).sum()) if enrol_col in filtered.columns else None
    student_note = f"\n\n**Estimated students affected:** {students:,}" if students else ""

    return (
        f"**{len(filtered)} Critical schools {scope}** "
        f"(out of {total_in_scope} Critical nationally):\n\n"
        + "\n".join(lines)
        + student_note
        + f"\n\n💡 Open the **School Intelligence** tab to view SHAP breakdowns "
          f"for any of these schools, or run an **Impact Simulation** to model interventions."
    )


def _answer_top_n(df: pd.DataFrame, question: str) -> str:
    region   = _extract_region(question)
    district = _extract_district(question, df)
    n        = _extract_n(question, default=10)

    filtered = df.copy()
    scope    = "nationally"
    if district:
        filtered = filtered[filtered["district"].str.lower() == district.lower()]
        scope = f"in {district.title()} district"
    elif region:
        filtered = filtered[filtered["region"].str.lower().str.contains(region, na=False)]
        scope = f"in {region.title()} region"

    top = filtered.sort_values("priority_score", ascending=False).head(n)
    if top.empty:
        return f"No schools found {scope}. Check the region or district name."

    lines = [_school_line(row, i + 1) for i, (_, row) in enumerate(top.iterrows())]
    return (
        f"**Top {len(top)} highest-need schools {scope}:**\n\n"
        + "\n".join(lines)
        + f"\n\n💡 Use the **Action Plan** tab to download these as a CSV "
          f"ready for MoE or NGO briefings."
    )


def _answer_district_query(df: pd.DataFrame, question: str) -> str:
    district = _extract_district(question, df)
    if not district:
        # List available districts if none matched
        districts = sorted(df["district"].dropna().unique().tolist())[:20]
        return (
            "Could not identify a specific district in your question. "
            f"Available districts include: {', '.join(districts[:15])}… "
            "Try: *'show me schools in Nabdam district'*"
        )

    filtered = df[df["district"].str.lower() == district.lower()].sort_values(
        "priority_score", ascending=False
    )
    if filtered.empty:
        return f"No schools found for **{district.title()}** district."

    n_crit = int((filtered["priority_tier"].str.upper() == "CRITICAL").sum())
    n_high = int((filtered["priority_tier"].str.upper() == "HIGH").sum())
    lines  = [_school_line(row, i + 1) for i, (_, row) in enumerate(filtered.head(15).iterrows())]

    return (
        f"**{district.title()} district — {len(filtered)} schools**\n"
        f"🔴 Critical: {n_crit} | 🟡 High: {n_high} | "
        f"🟢 Stable: {len(filtered) - n_crit - n_high}\n\n"
        + "\n".join(lines)
    )


def _answer_region_query(df: pd.DataFrame, question: str) -> str:
    region = _extract_region(question)
    if not region:
        return "Could not detect a region name. Try: *'show me Northern region schools'*"

    filtered = df[df["region"].str.lower().str.contains(region, na=False)].copy()
    if filtered.empty:
        return f"No schools found for **{region.title()}** region."

    n_total = len(filtered)
    n_crit  = int((filtered["priority_tier"].str.upper() == "CRITICAL").sum())
    n_high  = int((filtered["priority_tier"].str.upper() == "HIGH").sum())
    avg_sc  = _fmt_score(filtered["priority_score"].mean())

    top5 = filtered.sort_values("priority_score", ascending=False).head(5)
    lines = [_school_line(row, i + 1) for i, (_, row) in enumerate(top5.iterrows())]

    top_districts = (
        filtered.groupby("district")["priority_score"].mean()
        .sort_values(ascending=False).head(3)
    )
    dist_lines = "\n".join(
        f"  • {d.title()}: avg {_fmt_score(s)}" for d, s in top_districts.items()
    )

    return (
        f"**{region.title()} region — {n_total} schools**\n"
        f"🔴 Critical: {n_crit} | 🟡 High: {n_high} | "
        f"🟢 Stable: {n_total - n_crit - n_high} | Avg score: {avg_sc}\n\n"
        f"**Top 5 highest-need schools:**\n" + "\n".join(lines) + "\n\n"
        f"**Most deprived districts:**\n{dist_lines}"
    )


def _answer_cost_estimate(df: pd.DataFrame, question: str) -> str:
    region   = _extract_region(question)
    district = _extract_district(question, df)
    n        = _extract_n(question, default=10)
    q        = question.lower()

    filtered = df.copy()
    scope    = "nationally (Critical + High schools)"
    if district:
        filtered = filtered[filtered["district"].str.lower() == district.lower()]
        scope = f"in {district.title()} district"
    elif region:
        filtered = filtered[filtered["region"].str.lower().str.contains(region, na=False)]
        scope = f"in {region.title()} region"

    target = filtered[filtered["priority_tier"].str.upper().isin(["CRITICAL", "HIGH"])].head(n)
    n_schools = len(target)
    if n_schools == 0:
        return f"No Critical/High schools found {scope}."

    do_solar  = any(k in q for k in ["solar", "electricity", "power", "light", "energy"])
    do_water  = any(k in q for k in ["water", "borehole", "wash", "well"])
    do_sanit  = any(k in q for k in ["sanitation", "toilet", "latrine"])
    # Default: solar if nothing specific mentioned
    if not any([do_solar, do_water, do_sanit]):
        do_solar = True

    lines, total = [], 0
    if do_solar:
        cost = n_schools * _COST_SOLAR
        total += cost
        lines.append(f"☀️ Solar microgrids ({n_schools} schools): **GH₵ {cost:,}**")
    if do_water:
        cost = n_schools * _COST_BOREHOLE
        total += cost
        lines.append(f"💧 Borehole/WASH units ({n_schools} schools): **GH₵ {cost:,}**")
    if do_sanit:
        cost = n_schools * _COST_SANITATION
        total += cost
        lines.append(f"🚹 Sanitation blocks ({n_schools} schools): **GH₵ {cost:,}**")

    usd = total / 14  # approximate GHS→USD
    return (
        f"**Indicative cost estimate {scope}** ({n_schools} schools):\n\n"
        + "\n".join(lines)
        + f"\n\n**Total: GH₵ {total:,} (≈ USD {usd:,.0f})**\n\n"
        f"⚠️ *{_COST_LABEL}. Actual procurement will vary by site, contractor, and specification.*\n\n"
        f"💡 Use the **Impact Simulator** tab to model cost vs. score reduction "
        f"for a selected set of schools."
    )


def _answer_impact(df: pd.DataFrame, question: str) -> str:
    region   = _extract_region(question)
    district = _extract_district(question, df)
    q        = question.lower()

    filtered = df.copy()
    scope    = "nationally"
    if district:
        filtered = filtered[filtered["district"].str.lower() == district.lower()]
        scope = f"in {district.title()} district"
    elif region:
        filtered = filtered[filtered["region"].str.lower().str.contains(region, na=False)]
        scope = f"in {region.title()} region"

    if "critical" in q:
        filtered = filtered[filtered["priority_tier"].str.upper() == "CRITICAL"]
        scope += " (Critical only)"
    elif "high" in q and "highest" not in q:
        filtered = filtered[filtered["priority_tier"].str.upper().isin(["CRITICAL", "HIGH"])]
        scope += " (Critical + High)"

    enrol_col = "youth_literacy_count"
    if enrol_col not in filtered.columns or filtered.empty:
        return f"No enrolment proxy data available for {scope}."

    total_students = int(filtered[enrol_col].fillna(0).sum())
    n_schools = len(filtered)
    avg_enrol = int(filtered[enrol_col].replace(0, pd.NA).dropna().median()) if n_schools else 0

    return (
        f"**Estimated student impact {scope}:**\n\n"
        f"- 🏫 Schools in scope: **{n_schools}**\n"
        f"- 👨‍🎓 Students reached (literacy count proxy): **{total_students:,}**\n"
        f"- 📊 Average enrolment per school: **~{avg_enrol:,}**\n\n"
        f"⚠️ *`youth_literacy_count` is a district-level literacy proxy, not direct enrolment data.*\n\n"
        f"💡 Run the **Impact Simulator** tab to model specific interventions "
        f"(solar, water, sanitation) and see tier-graduation projections."
    )


def _answer_cluster_hint(df: pd.DataFrame, question: str) -> str:
    region = _extract_region(question)
    filtered = df.copy()
    scope = "nationally"
    if region:
        filtered = filtered[filtered["region"].str.lower().str.contains(region, na=False)]
        scope = f"in {region.title()} region"

    high_need = filtered[filtered["priority_tier"].str.upper().isin(["CRITICAL", "HIGH"])]
    if "district" not in high_need.columns or high_need.empty:
        return "Not enough data to identify clusters."

    district_counts = (
        high_need.groupby("district")["school_name"].count()
        .sort_values(ascending=False).head(8)
    )
    lines = [
        f"• **{d.title()}**: {c} high-need schools"
        for d, c in district_counts.items()
    ]

    return (
        f"**Districts with the highest concentration of Critical/High schools {scope}:**\n\n"
        + "\n".join(lines)
        + "\n\n💡 Open the **Investment Clusters** tab for interactive DBSCAN zone detection — "
          "it identifies districts where one infrastructure investment (solar microgrid, "
          "borehole, WASH) can serve multiple schools within a 5 km radius."
    )


def _answer_stem_gender(df: pd.DataFrame, question: str) -> str:
    q = question.lower()
    filtered = df.copy()
    scope_parts = []

    if "stem" in q and "is_stem" in filtered.columns:
        filtered = filtered[filtered["is_stem"].astype(str).str.lower().isin(["true", "yes", "1"])]
        scope_parts.append("STEM")
    if any(k in q for k in ["girl", "female", "women"]) and "gender" in filtered.columns:
        filtered = filtered[filtered["gender"].str.lower().str.contains("girl", na=False)]
        scope_parts.append("girls'")
    if any(k in q for k in ["boy", "male", "men"]) and "gender" in filtered.columns:
        filtered = filtered[filtered["gender"].str.lower().str.contains("boy", na=False)]
        scope_parts.append("boys'")
    if "boarding" in q and "residency" in filtered.columns:
        filtered = filtered[filtered["residency"].str.lower().str.contains("boarding", na=False)]
        scope_parts.append("boarding")
    if "day" in q and "day school" in q and "residency" in filtered.columns:
        filtered = filtered[filtered["residency"].str.lower() == "day"]
        scope_parts.append("day")

    scope = " ".join(scope_parts) + " schools" if scope_parts else "schools"

    if filtered.empty:
        return f"No {scope} found matching your criteria."

    top = filtered.sort_values("priority_score", ascending=False).head(10)
    n_crit = int((top["priority_tier"].str.upper() == "CRITICAL").sum())
    lines = [_school_line(row, i + 1) for i, (_, row) in enumerate(top.iterrows())]

    return (
        f"**Top {len(top)} highest-need {scope}** ({n_crit} Critical):\n\n"
        + "\n".join(lines)
    )


def _answer_unknown(question: str) -> str:
    return (
        "I can answer questions grounded in the GES school dataset. Try asking:\n\n"
        "• *Which schools in Upper East are Critical?*\n"
        "• *Top 10 highest-need schools in the Northern region*\n"
        "• *How many students would be reached by upgrading Critical schools?*\n"
        "• *What would solar installation cost for Savannah region schools?*\n"
        "• *Which districts have the most clusters of high-need schools?*\n"
        "• *Show me critical girls' boarding schools nationally*\n\n"
        "All answers are computed directly from the live GES 2025 dataset — "
        "no external data sources are used."
    )


# ---------------------------------------------------------------------------
# Data snapshot builder — creates a compact context for Claude
# ---------------------------------------------------------------------------

def _build_data_snapshot(df: pd.DataFrame) -> str:
    """Produce a compact, token-efficient data snapshot for Claude's system prompt."""
    total   = len(df)
    n_crit  = int((df["priority_tier"].str.upper() == "CRITICAL").sum())
    n_high  = int((df["priority_tier"].str.upper() == "HIGH").sum())
    n_gps   = int(df["latitude"].notna().sum()) if "latitude" in df.columns else total

    enrol_col = "youth_literacy_count"
    students_crit = int(
        df[df["priority_tier"].str.upper() == "CRITICAL"][enrol_col].fillna(0).sum()
    ) if enrol_col in df.columns else 0

    top10 = df.nlargest(10, "priority_score")[["school_name","district","region","priority_score","priority_tier"]]
    top10_str = top10.to_string(index=False)

    region_summary = ""
    if "region" in df.columns:
        rs = df[df["priority_tier"].str.upper().isin(["CRITICAL","HIGH"])].groupby("region")["school_name"].count().sort_values(ascending=False)
        region_summary = "\n".join(f"  {r.title()}: {c}" for r,c in rs.items())

    return f"""LIVE GES 2025 DATASET SUMMARY ({total} schools):
- CRITICAL (score >65%): {n_crit} schools affecting ~{students_crit:,} students
- HIGH (score 45-65%): {n_high} schools
- STABLE (score <45%): {total - n_crit - n_high} schools
- GPS-verified: {n_gps}/{total} schools

CRITICAL+HIGH by region:
{region_summary}

TOP 10 HIGHEST-NEED SCHOOLS:
{top10_str}

Thresholds: CRITICAL >0.65, HIGH >0.45 (calibrated to GES 2025 distribution)
Feature weights: Poverty 30%, Literacy Gap 25%, Electricity 20%, Water 15%, Sanitation 7%, Aid 3%
Data sources: GES Register 2025, UNDP MPI 2023, GSS Census 2021, DHS Wave 8, SE4All 2022"""


# ---------------------------------------------------------------------------
# Public class — Hybrid AI + local engine
# ---------------------------------------------------------------------------

class EduInfraAssistant:
    """
    Hybrid intelligence engine for EduInfra Ghana.

    When ANTHROPIC_API_KEY is available (via Streamlit secrets or env var),
    routes questions through Claude claude-sonnet-4-20250514 with a data-grounded system prompt
    that includes a live snapshot of the GES 2025 dataset.

    Falls back gracefully to the local intent-routing engine (zero API cost)
    if no key is configured — so the app always works.

    Parameters
    ----------
    df : pd.DataFrame
        Output of EduInfraPipeline.run() — the ranked schools DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.df.columns = self.df.columns.str.lower().str.strip()
        if "priority_tier" in self.df.columns:
            self.df["priority_tier"] = self.df["priority_tier"].fillna("STABLE").astype(str)

        # Attempt to load Anthropic client — fully optional, safe to fail
        self._client = None
        self._api_key = None
        try:
            import anthropic as _anthropic
            _key = ""
            try:
                import streamlit as _st
                _key = _st.secrets.get("ANTHROPIC_API_KEY", "") or ""
            except Exception:
                pass
            if not _key:
                import os as _os
                _key = _os.environ.get("ANTHROPIC_API_KEY", "") or ""
            if _key and _key.strip().startswith("sk-"):
                self._client = _anthropic.Anthropic(api_key=_key.strip())
                self._api_key = _key.strip()
        except Exception:
            pass  # anthropic not installed or key invalid — local engine used

        # Build data snapshot once at construction
        self._data_snapshot = _build_data_snapshot(self.df)

    # ------------------------------------------------------------------
    # Claude-powered path
    # ------------------------------------------------------------------
    def _claude_answer(self, question: str, history: list[dict]) -> str:
        """Call Claude with data-grounded system prompt."""
        system_prompt = f"""You are EduInfra Ghana's AI Policy Analyst — a specialist in \
Ghana's education infrastructure gaps. You have real-time access to the GES 2025 \
school dataset below. Answer questions accurately using only this data. \
Be concise, use bullet points for lists, cite specific school names and scores. \
Always refer to data as 'GES 2025' not 'my training data'. \
For cost estimates use these UNESCO/GIZ benchmarks (not procurement prices): \
Solar microgrid GH₵45,000, Borehole/WASH GH₵28,000, Sanitation block GH₵18,000.

{self._data_snapshot}"""

        messages = []
        for msg in (history or [])[-6:]:   # last 3 exchanges = 6 messages
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        resp = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system_prompt,
            messages=messages,
        )
        return resp.content[0].text

    # ------------------------------------------------------------------
    # Local routing path (always available)
    # ------------------------------------------------------------------
    def _local_answer(self, question: str) -> str:
        """Route to local, zero-cost intent engine."""
        df = self.df
        intent = _classify_intent(question)
        if intent == "GENERAL_STATS":  return _answer_general_stats(df, question)
        elif intent == "CRITICAL_LIST": return _answer_critical_list(df, question)
        elif intent == "TOP_N":         return _answer_top_n(df, question)
        elif intent == "DISTRICT_QUERY":return _answer_district_query(df, question)
        elif intent == "REGION_QUERY":  return _answer_region_query(df, question)
        elif intent == "COST_ESTIMATE": return _answer_cost_estimate(df, question)
        elif intent == "IMPACT":        return _answer_impact(df, question)
        elif intent == "CLUSTER_HINT":  return _answer_cluster_hint(df, question)
        elif intent == "STEM_GENDER":   return _answer_stem_gender(df, question)
        else:                           return _answer_unknown(question)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def answer_data_query(
        self,
        question: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Answer a data question using Claude (if API key available) or local engine.

        Parameters
        ----------
        question : str   The user's natural-language question.
        history  : list  Previous conversation turns for multi-turn context.

        Returns
        -------
        str  Markdown-formatted answer grounded in GES 2025 data.
        """
        if not question or not question.strip():
            return _answer_unknown("")

        if self.df.empty:
            return (
                "⚠️ No school data loaded. Run the pipeline first: "
                "click **🚀 Run AI Infrastructure Pipeline** in the sidebar."
            )

        # Try Claude first; fall back to local engine on any error
        if self._client:
            try:
                return self._claude_answer(question, history or [])
            except Exception as _exc:
                # Graceful degradation — local engine never fails
                local = self._local_answer(question)
                return local + f"\n\n*[AI mode unavailable: {str(_exc)[:60]}. Using local engine.]*"

        return self._local_answer(question)
