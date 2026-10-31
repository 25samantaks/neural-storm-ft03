# 🛠️ Predictive Maintenance Platform (FT-03)

**Neural Storm 2026 · Track: Future Tech · Problem Statement: FT-03 Predictive Maintenance for Industry**

**Team Name:** TEAMDUO

**Members:** Khushi Samanta, Nidhi Agrawal

---

## Problem
Unplanned machine failures cause costly downtime and safety risks. Machines show subtle shifts in vibration, temperature and current long before failing. This platform ingests IoT sensor streams, learns normal behaviour, scores machine health, predicts Remaining Useful Life (RUL) and raises prioritised, low-noise maintenance alerts.

## Approach
1. **Ingest and clean:** sensor streams with missing readings are interpolated per machine.
2. **Feature engineering:** rolling mean, rolling std and slope (window = 15 cycles) for every sensor.
3. **Health score (unsupervised):** an Isolation Forest is trained only on early-life (healthy) data, so it needs no failure labels. This addresses the rarity of labelled failures.
4. **RUL prediction (supervised):** Gradient Boosting regressor on a capped RUL target (125 cycles, the standard C-MAPSS convention).
5. **Alerting:** predictions are smoothed over 5 cycles to cut false alarms, then mapped to a status:
   - **OK:** more than 60 cycles left
   - **WARNING:** 60 cycles or fewer left
   - **CRITICAL:** 30 cycles or fewer left
6. **Dashboard:** Streamlit app with live health, predicted RUL, sensor-drift charts and a prioritised maintenance queue.

## Tech Stack
Python 3.10+, pandas, NumPy, scikit-learn (HistGradientBoosting, IsolationForest), Streamlit, Plotly.

## Setup and Installation
```bash
git clone https://github.com/25samantaks/neural-storm-ft03.git
cd neural-storm-ft03
python -m venv venv
```
Activate the virtual environment:
- **Windows (PowerShell):** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

Then install the dependencies:
```bash
pip install -r requirements.txt
```

## How to Run Locally
```bash
python -m streamlit run app.py
```
Then open http://localhost:8501 in your browser. Models train automatically on first load (about 2 seconds). Use the **"Current cycle"** slider in the sidebar to move a machine from healthy to failing, and the **"Machine"** dropdown to switch machines.

## Project Structure
```
├── .gitignore            
├── README.md
├── app.py            # Streamlit dashboard
├── requirements.txt
├── docs/            # screenshots
└── src/
    ├── __init__.py
    ├── data.py       # synthetic run-to-failure sensor simulator
    └── model.py      # features, health model, RUL model, alert logic
```

## Results (held-out machines, never seen in training)
| Metric | Value |
|--------|-------|
| RUL RMSE | ~21 cycles |
| RUL MAE | ~16 cycles |
| Alert precision (RUL<=60) | ~0.94 |
| Alert recall (RUL<=60) | ~0.84 |

*Measured on simulated sensor data. The same numbers are shown in the app's sidebar.*

## Screenshots / Demo

### Healthy machine
![Healthy machine](docs/dashboard-healthy.png)

### Warning alert
![Warning alert](docs/dashboard-warning.png)

### Critical alert (predicted failure approaching)
![Critical alert](docs/dashboard-critical.png)

### Prioritised maintenance queue
![Maintenance queue](docs/maintenance-queue.png)

## Limitations and Future Work
- Trained on simulated data; next step is benchmarking on NASA C-MAPSS and CWRU bearing data.
- Add MQTT + TimescaleDB for true streaming ingestion.
- Try LSTM / 1D-CNN and uncertainty estimates for RUL.

## AI Tools Used
Claude and ChatGPT for code scaffolding and README drafting, GitHub Copilot for autocomplete. All code was reviewed and tested by the team.
