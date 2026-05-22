# 🇬🇭 EduInfra Ghana — AI Infrastructure Gap Mapper

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-eduinfra--ghana.streamlit.app-FCD116?style=for-the-badge&labelColor=0e1117)](https://eduinfra-ghana.streamlit.app)
[![Ghana AI Challenge 2026](https://img.shields.io/badge/Ghana%20AI%20Innovation%20Challenge-2026-006B3F?style=for-the-badge&labelColor=CF0921)](https://eduinfra-ghana.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-white?style=for-the-badge)](LICENSE)

> **"Data invisibility is the silent tax on the forgotten. EduInfra Ghana makes every school visible, every gap actionable."**

**Ghana AI Innovation Challenge 2026 · Track: AI Infrastructure Gap Mapper · Brainhauz Solutions**

---

## 🚀 Try it live

**[https://eduinfra-ghana.streamlit.app](https://eduinfra-ghana.streamlit.app)**

> Open the app → select a region in the sidebar → explore 721 schools across all 7 intelligence tabs.
> No login required. Works on desktop and mobile.

---

## The Problem

Ghana's 721 second-cycle institutions are not equal. **49 schools (6.8%) are in crisis** — lacking reliable electricity, clean water, and sanitation — yet they are invisible to national planners, NGOs, and donors who could intervene. These aren't just statistics: they represent **607,000+ students** attending school in conditions that suppress learning, drive dropout, and entrench intergenerational poverty.

Traditional tools — annual surveys, siloed spreadsheets, static PDF reports — cannot respond at the speed that policy demands.

---

## The Solution

**EduInfra Ghana** is a full-stack AI decision-support platform that fuses six deprivation datasets into a single, explainable **Multi-Factor Priority Score** for every mapped school in Ghana — then surfaces that intelligence through a ministerial-grade dashboard.

### Live Platform Stats
| Metric | Value |
|---|---|
| 🏫 Schools mapped | 721 (GES 2025 complete register) |
| 🔴 Critical Priority schools | 49 (6.8%) — immediate intervention needed |
| 🟡 High Priority schools | 189 (26.2%) — district-level action required |
| 👨‍🎓 Students in critical schools | 607,082 |
| 📍 GPS-verified coverage | 677/721 (93.9%) |
| 🧠 AI model R² | 0.9988 (Random Forest, 200 estimators) |

---

## Features

| Tab | What it does |
|---|---|
| 🛰️ **Geospatial Intelligence** | Dark-matter map with MarkerCluster, cold-spot heatmap, school scorecards, and fly-to search |
| 💎 **Investment Clusters** | DBSCAN spatial clustering — identifies zones where one investment serves multiple schools |
| 📋 **Action Plan** | Priority-ranked table of all 721 schools with filters, tier badges, and CSV export |
| 🔍 **School Intelligence** | Per-school SHAP waterfall explanations — exactly why a school scored as it did |
| ⚡ **Impact Simulator** | Simulate infrastructure interventions and see tier migration across the dataset |
| 📄 **Policy Brief** | AI-generated ministerial policy brief — downloadable, ready for MoE submission |
| 📊 **Data Story** | Score distribution, methodology transparency, data provenance, model card |

### AI Chat Assistant
The sidebar assistant answers data-grounded questions about the GES 2025 dataset in natural language. When an Anthropic API key is configured it uses Claude Sonnet; otherwise it routes through a local intent engine — so the app always works.

---

## AI Methodology

### Scoring Model
A 200-estimator **Random Forest Regressor** (R² = 0.9988) trained to reproduce a transparent weighted deprivation index:

| Feature | Weight | Source |
|---|---|---|
| Poverty Index (`pov_norm`) | 30% | UNDP Ghana MPI 2023 |
| Literacy Gap (`lit_norm`) | 25% | GSS Population Census 2021 |
| No Electricity (`elec_norm`) | 20% | DHS Wave 8 / SE4All 2022 |
| No Clean Water (`water_norm`) | 15% | DHS Wave 8 |
| Poor Sanitation (`sanitation_norm`) | 7% | DHS Wave 8 |
| No Prior Aid (`aid_norm`) | 3% | AidData / IATI 2023 |

**Tier thresholds** are calibrated to the GES 2025 score distribution:
- 🔴 **CRITICAL**: score > 0.65 (top 6.8% — 49 schools)
- 🟡 **HIGH**: score 0.45–0.65 (next 26.2% — 189 schools)
- 🟢 **STABLE**: score < 0.45 (483 schools)

> Note: `elec_norm`, `water_norm`, and `sanitation_norm` are district-level proxies from DHS surveys — not direct school-level measurements. Future versions will incorporate GES facility inspection data.

### Investment Clusters
DBSCAN (eps=0.08°, min_samples=2) identifies geographic zones where ≥2 schools fall within ~9 km — enabling shared infrastructure investment. Cost benchmarks use UNESCO/GIZ indicative rates.

---

## Data Sources

| Source | Year | Coverage |
|---|---|---|
| Ghana Education Service (GES) Register | 2025 | 721 second-cycle schools |
| UNDP Ghana MPI | 2023 | All 16 regions |
| Ghana Statistical Service Census | 2021 | District youth literacy |
| DHS Ghana Wave 8 | 2022 | WASH & electricity access |
| HOTOSM / OpenStreetMap | 2024 | GPS coordinates (94% coverage) |
| AidData / IATI | 2023 | Donor aid commitments |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/your-repo/eduinfra-ghana.git
cd eduinfra-ghana
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Patch the data (run once — idempotent)
python patch_data.py

# 3. Launch the dashboard
streamlit run app.py
```

### Optional: Enable Claude AI chat
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
Without a key the app still works fully — the chat uses a local data-grounded engine and the Policy Brief uses a polished template.

---

## Project Structure

```
eduinfra-ghana/
├── app.py                          # Main Streamlit dashboard (7 tabs)
├── patch_data.py                   # One-time data calibration script
├── requirements.txt
├── src/
│   ├── config.py                   # Paths, thresholds, colours
│   ├── pipeline.py                 # EduInfraPipeline — RF scorer + DBSCAN
│   ├── assistant.py                # Hybrid Claude API + local intent engine
│   ├── brief_generator.py          # AI Ministerial Policy Brief generator
│   ├── explainer.py                # SHAP waterfall explanations
│   ├── exporter.py                 # CSV / report export
│   └── quality.py                  # Data quality checks
├── data/
│   ├── ghana_schools_master_2025.csv
│   ├── schools_priority_ranked.csv # Pipeline output (auto-patched on startup)
│   ├── clean/                      # Intermediate pipeline stages
│   ├── poverty/                    # UNDP MPI data
│   ├── dhs/                        # DHS Wave 8
│   └── electrification/            # SE4All access data
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## Investment Impact

If all 49 Critical schools received full infrastructure packages (UNESCO/GIZ benchmarks):

| Intervention | Unit Cost | Total (GH₵) | Total (USD) |
|---|---|---|---|
| Solar microgrids | GH₵45,000 | GH₵2,205,000 | ~$157,500 |
| Borehole/WASH | GH₵28,000 | GH₵1,372,000 | ~$98,000 |
| Sanitation blocks | GH₵18,000 | GH₵882,000 | ~$63,000 |
| **Total** | | **GH₵4,459,000** | **~$318,500** |

*These are indicative planning benchmarks, not procurement prices.*

---

## Author

**Ing. Joseph K. Mensah (PE-GhIE)**  
Brainhauz Solutions | Ghana AI Innovation Challenge 2026
