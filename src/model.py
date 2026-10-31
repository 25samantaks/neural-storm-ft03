"""Feature engineering + health score (IsolationForest) + RUL model (GBM)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .data import SENSORS

WINDOW = 15
RUL_CAP = 125          # piece-wise linear RUL target, standard on C-MAPSS
WARN_RUL, CRIT_RUL = 60, 30


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "cycle"]).copy()
    df[SENSORS] = df.groupby("machine_id")[SENSORS].transform(
        lambda s: s.interpolate(limit_direction="both"))     # fill gaps
    g = df.groupby("machine_id")[SENSORS]
    for w_name, fn in [("mean", "mean"), ("std", "std")]:
        r = g.rolling(WINDOW, min_periods=3).agg(fn).reset_index(level=0, drop=True)
        df[[f"{c}_{w_name}" for c in SENSORS]] = r.values
    slope = g.diff(WINDOW // 3) / (WINDOW // 3)
    df[[f"{c}_slope" for c in SENSORS]] = slope.values
    return df.bfill().fillna(0)


def feature_cols(df):
    return [c for c in df.columns if any(c.startswith(s + "_") for s in SENSORS)] + SENSORS


@dataclass
class Bundle:
    rul_model: HistGradientBoostingRegressor
    iso: IsolationForest
    scaler: StandardScaler
    cols: list
    metrics: dict


def train(df: pd.DataFrame, test_frac: float = 0.25, seed: int = 0) -> Bundle:
    feats = add_features(df)
    cols = feature_cols(feats)
    ids = feats.machine_id.unique()
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(ids, int(len(ids) * test_frac), replace=False)
    tr, te = feats[~feats.machine_id.isin(test_ids)], feats[feats.machine_id.isin(test_ids)]

    # 1) RUL regression (capped target)
    rul = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed)
    rul.fit(tr[cols], tr.RUL.clip(upper=RUL_CAP))

    # 2) Unsupervised health model: learn "normal" from the first 40% of life
    life = tr.groupby("machine_id").cycle.transform("max")
    healthy = tr[tr.cycle < 0.4 * life]
    scaler = StandardScaler().fit(healthy[cols])
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(scaler.transform(healthy[cols]))

    # Evaluation on unseen machines
    pred = rul.predict(te[cols])
    true = te.RUL.clip(upper=RUL_CAP)
    near = te.RUL <= WARN_RUL
    flag = pred <= WARN_RUL
    metrics = {
        "RUL RMSE (cycles)": float(np.sqrt(np.mean((pred - true) ** 2))),
        "RUL MAE (cycles)": float(mean_absolute_error(true, pred)),
        f"Alert precision (RUL<={WARN_RUL})": float(precision_score(near, flag)),
        f"Alert recall (RUL<={WARN_RUL})": float(recall_score(near, flag)),
        "Test machines": int(len(test_ids)),
    }
    return Bundle(rul, iso, scaler, cols, metrics)


def score(df: pd.DataFrame, b: Bundle) -> pd.DataFrame:
    """Add predicted RUL, health score (0-100) and alert level per row."""
    f = add_features(df)
    out = f[["machine_id", "cycle"] + SENSORS].copy()
    out["pred_RUL"] = np.clip(b.rul_model.predict(f[b.cols]), 0, RUL_CAP)
    raw = b.iso.score_samples(b.scaler.transform(f[b.cols]))      # higher = normal
    out["health"] = np.clip((raw + 0.75) / (0.35) * 100, 0, 100)  # rough 0-100 scale
    # smooth to cut false alarms
    for c in ["pred_RUL", "health"]:
        out[c] = out.groupby("machine_id")[c].transform(lambda s: s.rolling(5, min_periods=1).mean())
    out["alert"] = np.where(out.pred_RUL <= CRIT_RUL, "CRITICAL",
                    np.where(out.pred_RUL <= WARN_RUL, "WARNING", "OK"))
    if "RUL" in f:
        out["true_RUL"] = f["RUL"].values
    return out