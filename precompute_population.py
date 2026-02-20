"""
precompute_population.py
========================
Run once to generate two static data files used by the Streamlit app:

  population_params.csv  — PL & CS OLS parameters for every athlete in MDLD_speed
  wa_lookup.json         — WA-points vs time lookup tables (per gender × distance)

Usage:
    python precompute_population.py

The MDLD_speed R data file is read from the sibling repository MLM_PL_vs_CS.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("/home/user/MLM_PL_vs_CS/MDLD_speed")
OUT_DIR   = Path(__file__).parent

DISTANCES  = [400, 800, 1500, 3000, 5000, 10000]
TIME_COLS  = ["time_400",  "time_800",  "time_1500",
              "time_3000", "time_5000", "time_10000"]
PTS_COLS   = ["WA_pts_400",  "WA_pts_800",  "WA_pts_1500",
              "WA_pts_3000", "WA_pts_5000", "WA_pts_10000"]

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

print(f"Loading {DATA_PATH} …", flush=True)
if not DATA_PATH.exists():
    sys.exit(f"ERROR: {DATA_PATH} not found. Clone MLM_PL_vs_CS next to this repo.")

result = pyreadr.read_r(str(DATA_PATH))
df = list(result.values())[0]
print(f"  → {len(df):,} athletes, {df.shape[1]} columns", flush=True)

# ---------------------------------------------------------------------------
# 1.  WA scoring lookup tables
#     For each gender × distance keep ~300 representative (time, pts) pairs
#     sampled uniformly across the sorted performance range.
# ---------------------------------------------------------------------------

print("\nBuilding WA scoring lookup tables …", flush=True)

wa_lookup: dict = {}

for gender in ["Male", "Female"]:
    wa_lookup[gender] = {}
    sub_g = df[df["Gender"] == gender]

    for dist, tc, pc in zip(DISTANCES, TIME_COLS, PTS_COLS):
        sub = sub_g[[tc, pc]].dropna()
        sub = sub[sub[pc] > 0].sort_values(tc).drop_duplicates(subset=tc)

        n_sample = min(300, len(sub))
        idx = np.round(np.linspace(0, len(sub) - 1, n_sample)).astype(int)
        sample = sub.iloc[idx]

        wa_lookup[gender][str(dist)] = {
            "times": [round(x, 3) for x in sample[tc].tolist()],
            "pts":   [round(x, 1) for x in sample[pc].tolist()],
        }

        fastest = sub[tc].min()
        slowest = sub[tc].max()
        max_pts = sub[pc].max()
        print(f"  {gender:6} {dist:5}m  n={len(sub):5,}  "
              f"fastest={fastest:.2f}s  slowest={slowest:.2f}s  "
              f"max_pts={max_pts:.0f}", flush=True)

out_wa = OUT_DIR / "wa_lookup.json"
with open(out_wa, "w") as f:
    json.dump(wa_lookup, f, separators=(",", ":"))
print(f"\n  → saved {out_wa}", flush=True)

# ---------------------------------------------------------------------------
# 2.  Population PL & CS parameters
#     For every athlete with ≥ 2 non-missing PBs, fit PL and CS by OLS.
# ---------------------------------------------------------------------------

print("\nFitting PL and CS models for each athlete …", flush=True)


def fit_pl(dists: list, times: list) -> tuple[float, float, float]:
    """Return (S, b, E) from log-log OLS: log(speed) = log(S) + slope·log(t)."""
    d = np.array(dists, dtype=float)
    t = np.array(times, dtype=float)
    v = d / t
    slope, intercept = np.polyfit(np.log(t), np.log(v), 1)
    S = np.exp(intercept)
    b = -slope          # positive decay exponent
    E = 1.0 + slope     # endurance index = 1 - b
    return S, b, E


def fit_cs(dists: list, times: list) -> tuple[float, float]:
    """Return (CS_ms, D_prime) from OLS: distance = CS·time + D'."""
    d = np.array(dists, dtype=float)
    t = np.array(times, dtype=float)
    CS, D_prime = np.polyfit(t, d, 1)
    return CS, D_prime


records = []
skipped = 0

for row in df.itertuples(index=False):
    dists_ok, times_ok = [], []
    for dist, tc in zip(DISTANCES, TIME_COLS):
        t = getattr(row, tc)
        if t is not None and np.isfinite(t) and t > 0:
            dists_ok.append(dist)
            times_ok.append(float(t))

    if len(dists_ok) < 2:
        skipped += 1
        continue

    try:
        S, b, E        = fit_pl(dists_ok, times_ok)
        CS_ms, D_prime = fit_cs(dists_ok, times_ok)
    except Exception:
        skipped += 1
        continue

    # Sanity filter: discard clearly non-physical fits
    if not (np.isfinite(S) and np.isfinite(b) and np.isfinite(CS_ms)
            and np.isfinite(D_prime) and 0 < b < 2 and CS_ms > 0):
        skipped += 1
        continue

    records.append({
        "Gender":     str(row.Gender),
        "Best_event": str(row.Best_event),
        "Max_level":  str(row.Max_level),
        "n_dist":     len(dists_ok),
        "S":          round(S,      6),
        "b":          round(b,      6),
        "E":          round(E,      6),
        "CS_ms":      round(CS_ms,  6),
        "CS_kmh":     round(CS_ms * 3.6, 5),
        "D_prime":    round(D_prime, 3),
    })

pop_df = pd.DataFrame(records)

out_pop = OUT_DIR / "population_params.csv"
pop_df.to_csv(out_pop, index=False)

print(f"\n  → {len(pop_df):,} athletes saved  ({skipped:,} skipped)", flush=True)
print(f"  → saved {out_pop}", flush=True)

print("\nGroup sizes (Gender × Best_event):")
print(pop_df.groupby(["Gender", "Best_event"]).size().to_string())
print("\nDone.")
