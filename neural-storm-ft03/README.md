<# Predictive Maintenance (FT-03)🛠️

**Neural Storm 2026 · Track: Future Tech · Problem Statement: FT-03 Predictive Maintenance for Industry**

**Team Name:** TEAMDUO
**Members:** KHUSHI SAMANTA, NIDHI AGRAWAL

## Problem
Unplanned machine failures cause costly downtime and safety risks. Machines show subtle shifts in vibration, temperature and current long before failing. This platform ingests IoT sensor streams, learns normal behaviour, scores machine health, predicts Remaining Useful Life (RUL) and raises prioritised, low-noise maintenance alerts.

## Approach
1. **Ingest and clean:** sensor streams with missing readings are interpolated per machine.
2. **Feature engineering:** rolling mean, rolling std and slope (window = 15 cycles) for every sensor.
3. **Health score (unsupervised):** an Isolation Forest is trained only on early-life (healthy) data, so it needs no failure labels.
4. **RUL prediction (supervised):** Gradient Boosting regressor on a capped RUL target (125 cycles).
5. **Alerting:** predictions are smoothed over 5 cycles to reduce false alarms, then mapped to OK / WARNING (<=60 cycles) / CRITICAL (<=30 cycles).
6. **Dashboard:** Streamlit app with live health, predicted RUL, sensor-drift charts and a prioritised maintenance queue.

## Tech Stack
Python 3.10+, pandas, NumPy, scikit-learn, Streamlit, Plotly.

## Setup and Installation
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## How to Run Locally
```bash
python -m streamlit run app.py
```
Open http://localhost:8501. Use the sidebar slider to move the machine from healthy to failing.

## Project Structure
```
app.py            # Streamlit dashboard
src/data.py       # synthetic run-to-failure sensor simulator
src/model.py      # features, health model, RUL model, alert logic
requirements.txt
```

## Results (held-out machines not seen in training)
| Metric | Value |
|--------|-------|
| RUL RMSE | ~21 cycles |
| RUL MAE | ~16 cycles |
| Alert precision (RUL<=60) | ~0.94 |
| Alert recall (RUL<=60) | ~0.84 |

*Measured on simulated data.*

## Screenshots / Demo
![Healthy machine](docs/dashboard-healthy.png)

![Critical alert](docs/dashboard-critical.png)

![Maintenance queue](docs/maintenance-queue.png)

![Warning alert](docs/dashboard-warning.png)

## Limitations and Future Work
- Trained on simulated data; next step is NASA C-MAPSS and CWRU benchmarks.
- Add MQTT and TimescaleDB for real streaming ingestion.
- Try LSTM / 1D-CNN and RUL uncertainty estimates.

## AI Tools Used
Claude and Chatgpt for code scaffolding and README drafting, GitHub Copilot for autocomplete. All code was reviewed and tested by the team.>
