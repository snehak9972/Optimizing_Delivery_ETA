# Delhivery Graph Intelligence Platform

## Quick Start (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure delivery_data.csv is in the same folder as app.py
ls delivery_data.csv   # should exist

# 3. Run the dashboard
streamlit run app.py
```

The app auto-detects `delivery_data.csv` or `delivery_data.csv.gz` in:
- Same folder as app.py  (recommended)
- `/mnt/user-data/uploads/` (Streamlit Cloud / CAC environment)

---

## Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| 📊 Overview | Delay distributions, OSRM vs actual scatter, day×hour heatmap |
| 🕸️ Network Graph | Directed facility graph, betweenness centrality, PageRank |
| 🏭 Bottleneck Hubs | Top-N hubs by composite score, radar profile, revenue impact |
| 🗺️ Corridor Audit | Chronic delay corridors, revenue at risk per corridor |
| 🤖 ML Prediction | Ridge (baseline) vs HistGBR (graph-enhanced) ETA, live predictor |
| 🔀 FTL vs Carting | ML classifier, distance×hour matrix, graph-aware route advisor |
| 🔍 Anomaly Detection | Isolation Forest flagging, anomaly explorer |
| 💡 Strategy Memo | 1-page ops memo, top-5 interventions, downloadable TXT |

---

## Key Technical Choices

- **HistGradientBoostingRegressor** — 10× faster than RandomForest, handles NaN natively, early stopping
- **Betweenness centrality (k=200 sample)** — approximate for speed, reproducible with seed=42
- **PageRank** — identifies relay hubs beyond simple degree
- **Composite bottleneck score** — BTC 30% · PR 15% · SLA 25% · Severe 15% · Revenue 15%
- **Graph advantage** — corridor_risk and network_congestion features from src+dst bottleneck scores

## Files

```
app.py              ← Main dashboard (run this)
requirements.txt    ← Python dependencies
delivery_data.csv   ← Dataset (place here)
README.md           ← This file
```
