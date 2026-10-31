import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import SENSORS, simulate_fleet
from src.model import CRIT_RUL, WARN_RUL, score, train

st.set_page_config(page_title="PredictiveMaint AI", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="Training models on simulated fleet...")
def load():
    fleet = simulate_fleet(40, seed=42)
    bundle = train(fleet)
    live = simulate_fleet(12, seed=7)       
    return bundle, live

bundle, live = load()
scored = score(live, bundle)

st.title("Predictive Maintenance 🛠️")
st.caption("Neural Storm 2026 · FT-03 · IoT sensor stream → health score → remaining useful life → alerts")

# ---- Sidebar: simulate time moving forward ----
mid = st.sidebar.selectbox("Machine", sorted(scored.machine_id.unique()))
max_c = int(scored[scored.machine_id == mid].cycle.max())
now = st.sidebar.slider("Current cycle (simulated live time)", 20, max_c, int(max_c * 0.7))
st.sidebar.markdown("**Model quality (unseen machines)**")
for k, v in bundle.metrics.items():
    st.sidebar.write(f"{k}: **{v:.2f}**" if isinstance(v, float) else f"{k}: **{v}**")

# ---- Fleet view at the chosen "now" (only data seen so far) ----
snap = []
for m, g in scored.groupby("machine_id"):
    cut = min(now, g.cycle.max())
    snap.append(g[g.cycle <= cut].iloc[-1])
fleet = pd.DataFrame(snap).sort_values("pred_RUL")

cur = scored[(scored.machine_id == mid) & (scored.cycle <= now)]
last = cur.iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("Health score", f"{last.health:.0f}/100")
c2.metric("Predicted cycles to failure", f"{last.pred_RUL:.0f}")
c3.metric("Status", last.alert)
if last.alert == "CRITICAL":
    st.error(f"Machine {mid}: schedule maintenance immediately (<= {CRIT_RUL} cycles left).")
elif last.alert == "WARNING":
    st.warning(f"Machine {mid}: plan maintenance soon (<= {WARN_RUL} cycles left).")
else:
    st.success(f"Machine {mid}: operating normally.")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    for s in SENSORS[:3]:
        z = (cur[s] - cur[s].iloc[:30].mean()) / (cur[s].iloc[:30].std() + 1e-9)
        fig.add_scatter(x=cur.cycle, y=z, name=s, mode="lines")
    fig.update_layout(title="Sensor drift vs. healthy baseline (z-score)", height=330,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = go.Figure()
    fig.add_scatter(x=cur.cycle, y=cur.pred_RUL, name="Predicted RUL")
    fig.add_scatter(x=cur.cycle, y=cur.health, name="Health score")
    fig.add_hline(y=WARN_RUL, line_dash="dot", line_color="orange")
    fig.add_hline(y=CRIT_RUL, line_dash="dot", line_color="red")
    fig.update_layout(title="Health & remaining useful life", height=330,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Prioritised maintenance queue")
q = fleet[["machine_id", "pred_RUL", "health", "alert"]].rename(
    columns={"machine_id": "Machine", "pred_RUL": "Predicted RUL", "health": "Health", "alert": "Status"})
st.dataframe(q.round(1), use_container_width=True, hide_index=True)
st.caption("Simulated data. Replace `src/data.py` with a NASA C-MAPSS / CWRU loader for real benchmarks.")
