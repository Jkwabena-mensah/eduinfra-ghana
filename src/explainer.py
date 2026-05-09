"""
src/explainer.py
================
SHAP-based explainability for the EduInfra Ghana school priority model.

Provides:
  SchoolExplainer — loads the trained Random Forest, wraps shap.TreeExplainer,
  and exposes per-school explanation dicts + Plotly waterfall charts.

Requires:
  pip install shap
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Graceful SHAP import — raise informative error on import, not at call time
# ---------------------------------------------------------------------------
try:
    import shap  # type: ignore
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


if not _SHAP_AVAILABLE:
    raise ImportError(
        "\n\nshap is required for the School Intelligence tab.\n"
        "Install it with:\n\n"
        "    pip install shap\n\n"
        "Then restart the Streamlit server."
    )


# ---------------------------------------------------------------------------
# Feature label mapping  (internal name → human-readable)
# ---------------------------------------------------------------------------
FEATURE_LABELS: dict[str, str] = {
    "pov_norm":        "Poverty Index",
    "lit_norm":        "Literacy Gap",
    "elec_norm":       "No Electricity",
    "water_norm":      "No Clean Water",
    "sanitation_norm": "Poor Sanitation",
    "aid_norm":        "Prior Aid Received",
}

# ---------------------------------------------------------------------------
# Plain-English explanations  (feature, direction) → template string
# "increases" = positive SHAP  →  raises priority score  →  higher need
# "decreases" = negative SHAP  →  lowers priority score  →  less need
# ---------------------------------------------------------------------------
_PLAIN_ENGLISH: dict[tuple[str, str], str] = {
    ("pov_norm",        "increases"): (
        "High poverty in this area is a primary driver of the high priority score"
    ),
    ("pov_norm",        "decreases"): (
        "Relatively low poverty levels reduce the urgency of this school's need"
    ),
    ("lit_norm",        "increases"): (
        "A significant literacy gap among local youth is raising this school's priority"
    ),
    ("lit_norm",        "decreases"): (
        "Above-average local literacy reduces this school's deprivation score"
    ),
    ("elec_norm",       "increases"): (
        "No electricity access is the biggest driver of this school's critical score"
    ),
    ("elec_norm",       "decreases"): (
        "Existing electricity infrastructure reduces this school's infrastructure deficit"
    ),
    ("water_norm",      "increases"): (
        "Lack of clean water access significantly increases this school's infrastructure need"
    ),
    ("water_norm",      "decreases"): (
        "Relatively good water access lowers this school's deprivation score"
    ),
    ("sanitation_norm", "increases"): (
        "Poor sanitation facilities are contributing to this school's high priority rating"
    ),
    ("sanitation_norm", "decreases"): (
        "Adequate sanitation provision helps lower this school's overall score"
    ),
    ("aid_norm",        "increases"): (
        "This school has received little to no prior aid, increasing its urgency"
    ),
    ("aid_norm",        "decreases"): (
        "Prior aid investment in this district has reduced this school's relative need"
    ),
}


# ---------------------------------------------------------------------------
# SchoolExplainer
# ---------------------------------------------------------------------------

class SchoolExplainer:
    """
    Wraps the EduInfra Ghana Random Forest with shap.TreeExplainer to produce
    per-school SHAP explanations and Plotly waterfall visualisations.

    Parameters
    ----------
    model_path : Path
        Absolute path to a pickled RandomForestRegressor (.pkl).
    feature_names : list[str]
        Ordered feature column names used at training time.
        v2: ["pov_norm","lit_norm","elec_norm","water_norm","sanitation_norm","aid_norm"]
        v1: ["pov_norm","lit_norm"]
    """

    def __init__(self, model_path: Path, feature_names: list[str]) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run EduInfraPipeline().run() to train and save the model."
            )

        with open(model_path, "rb") as fh:
            self._model = pickle.load(fh)

        self._feature_names = feature_names
        # TreeExplainer is exact (not approximated) for tree-based models
        self._explainer = shap.TreeExplainer(self._model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_array(self, school_row: pd.Series) -> np.ndarray:
        """Extract the feature vector as a (1, n_features) float array."""
        return (
            school_row[self._feature_names]
            .values
            .reshape(1, -1)
            .astype(float)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_school(self, school_row: pd.Series) -> dict:
        """
        Compute SHAP values for a single school.

        Returns
        -------
        dict
            shap_values   : list[float]  — one per feature
            feature_names : list[str]
            base_value    : float        — E[f(x)] national average prediction
            prediction    : float        — model output for this school
            top_factors   : list[dict]   — top 3 by |shap_value|, each with:
                              feature, shap_value, direction, label, plain_english
        """
        X        = self._to_array(school_row)
        shap_out = self._explainer(X)

        sv       = shap_out.values[0].tolist()
        base_val = float(shap_out.base_values[0])
        pred     = float(self._model.predict(X)[0])

        # Top-3 factors by magnitude
        ranked = sorted(enumerate(sv), key=lambda t: abs(t[1]), reverse=True)[:3]

        top_factors = []
        for feat_idx, sv_val in ranked:
            feat      = self._feature_names[feat_idx]
            direction = "increases" if sv_val > 0 else "decreases"
            label     = FEATURE_LABELS.get(feat, feat)
            plain     = _PLAIN_ENGLISH.get(
                (feat, direction),
                f"{label} {'raises' if sv_val > 0 else 'lowers'} the priority score",
            )
            top_factors.append({
                "feature":       feat,
                "shap_value":    sv_val,
                "direction":     direction,
                "label":         label,
                "plain_english": plain,
            })

        return {
            "shap_values":   sv,
            "feature_names": self._feature_names,
            "base_value":    base_val,
            "prediction":    pred,
            "top_factors":   top_factors,
        }

    # ------------------------------------------------------------------

    def plot_waterfall(self, school_row: pd.Series) -> go.Figure:
        """
        Plotly horizontal waterfall bar chart of SHAP feature contributions.

        Visual spec
        -----------
        • Red bars   — positive SHAP (increases priority = higher need)
        • Green bars — negative SHAP (decreases priority = lower need)
        • Template: plotly_dark | background: #0e1117 | title: #FFD700
        • Dashed vertical line at base_value (national average prediction)
        • Y-axis labels: human-readable names (FEATURE_LABELS)
        • Each bar annotated with "+X.XXX" or "−X.XXX"
        • Title: "Priority Score Drivers — [school_name]"
        """
        result = self.explain_school(school_row)
        sv     = result["shap_values"]
        base   = result["base_value"]

        school_name = (
            school_row.get("school_name")
            or school_row.get("School_Name")
            or "School"
        )

        labels = [FEATURE_LABELS.get(f, f) for f in self._feature_names]

        # Sort ascending by SHAP value so most negative is at bottom,
        # most positive at top — natural reading order for a bar chart
        order  = sorted(range(len(sv)), key=lambda i: sv[i])
        sv_ord = [sv[i]     for i in order]
        lb_ord = [labels[i] for i in order]

        bar_colors = ["#CF0921" if v > 0 else "#006B3F" for v in sv_ord]
        annotations = [
            (f"+{v:.3f}" if v >= 0 else f"{v:.3f}")
            for v in sv_ord
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=sv_ord,
            y=lb_ord,
            orientation="h",
            marker_color=bar_colors,
            marker_line_width=0,
            text=annotations,
            textposition="outside",
            textfont=dict(size=11, color="#E6EDF3"),
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP contribution: %{x:.4f}<extra></extra>"
            ),
        ))

        # National-average reference line
        fig.add_vline(
            x=base,
            line_dash="dash",
            line_color="#8B949E",
            line_width=1.5,
            annotation_text=f"National avg: {base:.3f}",
            annotation_position="top right",
            annotation_font_color="#8B949E",
            annotation_font_size=10,
        )

        # Zero baseline
        fig.add_vline(x=0, line_color="#30363D", line_width=1)

        x_abs_max = max((abs(v) for v in sv_ord), default=0.05)
        x_pad     = x_abs_max * 1.40

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            title=dict(
                text=f"Priority Score Drivers — {school_name}",
                font=dict(color="#FFD700", size=14, family="Montserrat, sans-serif"),
                x=0,
                xanchor="left",
            ),
            font=dict(family="Inter, sans-serif", color="#E6EDF3", size=12),
            legend=dict(
                font=dict(color="#E6EDF3", size=11),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,215,0,0.15)",
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="#161B22",
                bordercolor="#30363D",
                font=dict(color="#E6EDF3", size=12),
            ),
            xaxis=dict(
                title="SHAP Contribution to Priority Score",
                range=[-x_pad, x_pad],
                gridcolor="#1e242c",
                color="#8B949E",
                zeroline=False,
            ),
            yaxis=dict(color="#E6EDF3", tickfont=dict(size=12), automargin=True),
            margin=dict(l=20, r=90, t=52, b=40),
            height=340,
            showlegend=False,
        )

        return fig
