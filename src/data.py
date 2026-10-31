"""Synthetic IoT sensor generator for run-to-failure machines."""
import numpy as np
import pandas as pd

SENSORS = ["vibration", "temperature", "current", "pressure", "rpm"]

def simulate_machine(machine_id: int, rng: np.random.Generator) -> pd.DataFrame:
    life = int(rng.integers(180, 320))
    onset = int(life * rng.uniform(0.55, 0.7))          # degradation start
    t = np.arange(1, life + 1)
    deg = (np.clip(t - onset, 0, None) / (life - onset)) ** 2.2

    base = {
        "vibration": rng.normal(2.0, 0.15),
        "temperature": rng.normal(65, 2),
        "current": rng.normal(12, 0.5),
        "pressure": rng.normal(5.0, 0.2),
        "rpm": rng.normal(1500, 15),
    }
    df = pd.DataFrame({
        "machine_id": machine_id,
        "cycle": t,
        "vibration": base["vibration"] + 3.0 * deg + rng.normal(0, 0.12, life),
        "temperature": base["temperature"] + 22 * deg + rng.normal(0, 1.0, life),
        "current": base["current"] + 4.5 * deg + rng.normal(0, 0.3, life),
        "pressure": base["pressure"] - 1.2 * deg + rng.normal(0, 0.1, life),
        "rpm": base["rpm"] - 90 * deg + rng.normal(0, 8, life),
    })
    df["RUL"] = life - t                                   # true remaining life
    # simulate dropped readings (missing data challenge)
    mask = rng.random((life, len(SENSORS))) < 0.01
    df[SENSORS] = df[SENSORS].mask(mask)
    return df


def simulate_fleet(n_machines: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat(
        [simulate_machine(i, rng) for i in range(1, n_machines + 1)],
        ignore_index=True,
    )