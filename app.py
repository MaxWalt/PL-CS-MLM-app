"""
Power Law & Critical Speed Running Model Analyzer
==================================================
Fits the Power Law (PL) and Critical Speed (CS) models to an individual
athlete's personal bests and returns physiological parameters + predictions.

Model equations (Waltenspül et al., 2024-2025):
  PL:  speed = S · t^(-b)   where b = 1 - E
  CS:  speed = CS + D′ / t

Fitting strategy: direct OLS curve-fitting on the individual's PBs.
This differs from the population MLM in the original paper (which pools
information across ~52,000 athletes), but is the natural approach for a
single athlete entering their own race times.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PL & CS Running Analyzer",
    page_icon="🏃",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DISTANCES = [400, 800, 1500, 3000, 5000, 10000]
DIST_LABELS = {
    400: "400 m",
    800: "800 m",
    1500: "1500 m",
    3000: "3000 m",
    5000: "5000 m",
    10000: "10,000 m",
}

# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_time(s: str) -> float:
    """Parse 'h:mm:ss', 'mm:ss.xx', 'mm:ss', or plain seconds → float."""
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 1:
        return float(s)
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Cannot parse: {s!r}")


def fmt_time(t: float) -> str:
    """Format float seconds → 'mm:ss.xx' string."""
    if not np.isfinite(t) or t <= 0:
        return "—"
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:05.2f}"
    if m > 0:
        return f"{m}:{s:05.2f}"
    return f"{s:.2f} s"


def fmt_pace(speed_ms: float) -> str:
    """Format m/s → 'min:ss /km' pace string."""
    if speed_ms <= 0:
        return "—"
    sec_per_km = 1000.0 / speed_ms
    m = int(sec_per_km // 60)
    s = sec_per_km % 60
    return f"{m}:{s:04.1f} /km"


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_pl(distances: np.ndarray, times: np.ndarray) -> dict:
    """
    Fit Power Law in log–log space.

    Model:   log(speed) = log(a) + slope · log(time)
    where    slope = -b  (b > 0)

    Derived parameters:
      a  — scaling coefficient = S (speed capability, m·s^(b-1))
      b  — fatigue exponent  (b = -slope)
      S  — speed parameter   (= a)
      E  — endurance index   (E = 1 + slope = 1 - b)
    """
    speeds = distances / times
    log_t = np.log(times)
    log_s = np.log(speeds)
    slope, intercept = np.polyfit(log_t, log_s, 1)
    a = np.exp(intercept)
    b = -slope
    S = a
    E = 1.0 + slope  # = 1 - b
    return {"a": a, "b": b, "S": S, "E": E}


def pl_predict_time(d: float, a: float, b: float) -> float:
    """
    PL predicted time for distance d.
    From distance = a · t^(1-b)  →  t = (a/d)^(1/(b-1))
    """
    if b == 1.0:
        return float("nan")
    return float((a / d) ** (1.0 / (b - 1.0)))


def pl_speed_curve(a: float, b: float, t: np.ndarray) -> np.ndarray:
    return a * t ** (-b)


def fit_cs(distances: np.ndarray, times: np.ndarray) -> dict:
    """
    Fit Critical Speed model: distance = CS_ms · time + D_prime.

    Derived parameters:
      CS_ms  — critical speed in m/s
      CS_kmh — critical speed in km/h  (= CS_ms × 3.6)
      D_prime — anaerobic reserve in metres
    """
    slope, intercept = np.polyfit(times, distances, 1)
    CS_ms = slope
    D_prime = intercept
    CS_kmh = CS_ms * 3.6
    return {"CS_ms": CS_ms, "CS_kmh": CS_kmh, "D_prime": D_prime}


def cs_predict_time(d: float, CS_ms: float, D_prime: float) -> float:
    """CS predicted time for distance d: t = (d - D') / CS"""
    if CS_ms <= 0:
        return float("nan")
    t = (d - D_prime) / CS_ms
    return float(t) if t > 0 else float("nan")


def cs_speed_curve(CS_ms: float, D_prime: float, t: np.ndarray) -> np.ndarray:
    return CS_ms + D_prime / t


def mae(actual, predicted) -> float:
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


def mare(actual, predicted) -> float:
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    return float(np.mean(np.abs(1.0 - p / a)))


# ─────────────────────────────────────────────────────────────────────────────
# App: header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏃 Power Law & Critical Speed Running Analyzer")
st.markdown(
    "Enter your personal bests for **at least 2 distances**. "
    "The app fits the **Power Law (PL)** and **Critical Speed (CS)** models "
    "to your data and returns individual physiological parameters, "
    "performance predictions for all standard distances, and a "
    "speed–duration profile.\n\n"
    "> Based on the MLM modeling framework from Waltenspül et al. (2024–2025)."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: personal best inputs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Your Personal Bests")
    st.caption(
        "Format: `m:ss` or `m:ss.xx`  \n"
        "Examples: `1:45.50`, `3:32`, `14:06.92`  \n"
        "Leave blank to skip a distance."
    )
    raw_inputs: dict[int, str] = {}
    for d in DISTANCES:
        val = st.text_input(DIST_LABELS[d], key=str(d), placeholder="e.g. 1:45.50")
        if val.strip():
            raw_inputs[d] = val.strip()
    st.divider()
    st.caption("📖 Waltenspül et al. — *CS vs PL paper*, 2024–2025")

# ─────────────────────────────────────────────────────────────────────────────
# Parse & validate
# ─────────────────────────────────────────────────────────────────────────────

inputs: dict[int, float] = {}
for d, raw in raw_inputs.items():
    try:
        t = parse_time(raw)
        if t <= 0:
            st.sidebar.error(f"Time must be positive for {DIST_LABELS[d]}")
        else:
            inputs[d] = t
    except ValueError:
        st.sidebar.error(f"⚠️ Invalid format for {DIST_LABELS[d]}: `{raw}`")

if len(inputs) < 2:
    st.info("👈 Enter at least **2 personal bests** in the sidebar to begin.")
    st.stop()

distances_in = np.array(sorted(inputs.keys()), dtype=float)
times_in = np.array([inputs[int(d)] for d in distances_in], dtype=float)
speeds_in = distances_in / times_in

# ─────────────────────────────────────────────────────────────────────────────
# Fit models
# ─────────────────────────────────────────────────────────────────────────────

try:
    pl = fit_pl(distances_in, times_in)
except Exception as e:
    st.error(f"Power Law fit failed: {e}")
    st.stop()

try:
    cs = fit_cs(distances_in, times_in)
except Exception as e:
    st.error(f"Critical Speed fit failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sanity warnings
# ─────────────────────────────────────────────────────────────────────────────

if cs["D_prime"] < 0:
    st.warning(
        "⚠️ The CS model returned a **negative D′** "
        f"({cs['D_prime']:.1f} m). This can occur when the data range is "
        "narrow or performances are not well-described by the CS model. "
        "Add more distances for a reliable fit."
    )

if not (0 < pl["b"] < 2):
    st.warning(
        f"⚠️ The PL model returned an unusual fatigue exponent "
        f"(**b = {pl['b']:.3f}**). Predictions may be unreliable. "
        "Try adding more distances."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_params, tab_preds, tab_plot = st.tabs(
    ["📊 Parameters", "⏱️ Predictions", "📈 Speed–Duration Profile"]
)

# ── Tab 1: Parameters ─────────────────────────────────────────────────────────
with tab_params:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("⚡ Power Law (PL)")
        m1, m2 = st.columns(2)
        m1.metric("S — Speed parameter", f"{pl['S']:.3f} m/s")
        m2.metric("E — Endurance index", f"{pl['E']:.4f}")
        st.metric("b — Fatigue exponent (1 − E)", f"{pl['b']:.4f}")
        with st.expander("ℹ️ What do PL parameters mean?"):
            st.markdown(
                r"""
                The Power Law describes how speed decays with duration:

                $$\text{speed}(t) = S \cdot t^{-b}$$

                | Parameter | Interpretation |
                |-----------|----------------|
                | **S** | Speed capability — higher = faster athlete at all durations |
                | **E** | Endurance index = 1 − b; closer to 1.0 → less speed decay with time |
                | **b** | Fatigue/decay exponent = 1 − E; higher → speed drops faster with duration |

                *S* and *b* together define your entire speed–duration profile.
                """
            )

    with c2:
        st.subheader("🔵 Critical Speed (CS)")
        m3, m4 = st.columns(2)
        m3.metric("CS", f"{cs['CS_ms']:.3f} m/s")
        m4.metric("CS", f"{cs['CS_kmh']:.2f} km/h")
        st.metric("CS pace", fmt_pace(cs["CS_ms"]))
        st.metric("D′ — Anaerobic reserve", f"{cs['D_prime']:.1f} m")
        with st.expander("ℹ️ What do CS parameters mean?"):
            st.markdown(
                r"""
                The Critical Speed model describes speed as:

                $$\text{speed}(t) = CS + \frac{D'}{t}$$

                | Parameter | Interpretation |
                |-----------|----------------|
                | **CS** | Critical Speed — highest metabolically sustainable speed (anaerobic threshold proxy) |
                | **D′** | Anaerobic work capacity expressed as a distance (m). A larger D′ means more capacity to run above CS |

                *CS* acts as the lower asymptote of the speed–duration curve.
                """
            )

    st.divider()
    st.caption(
        "Parameters are estimated by direct OLS curve-fitting to your personal bests. "
        "This differs from the population MLM in Waltenspül et al., which pools "
        "information across ~52,000 athletes to regularize individual estimates."
    )

# ── Tab 2: Predictions ────────────────────────────────────────────────────────
with tab_preds:
    rows = []
    pl_fits, cs_fits, actuals = [], [], []

    for d in DISTANCES:
        actual = inputs.get(d)
        p_pl = pl_predict_time(float(d), pl["a"], pl["b"])
        p_cs = cs_predict_time(float(d), cs["CS_ms"], cs["D_prime"])

        row = {
            "Distance": DIST_LABELS[d],
            "Your PB": fmt_time(actual) if actual else "—",
            "PL prediction": fmt_time(p_pl),
            "CS prediction": fmt_time(p_cs),
        }
        if actual:
            row["PL error"] = f"{100 * abs(p_pl - actual) / actual:.1f} %"
            row["CS error"] = f"{100 * abs(p_cs - actual) / actual:.1f} %"
            pl_fits.append(p_pl)
            cs_fits.append(p_cs)
            actuals.append(actual)
        else:
            row["PL error"] = "—"
            row["CS error"] = "—"

        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if actuals:
        st.subheader("Overall Fit Quality (on entered PBs)")
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("PL — MAE", f"{mae(actuals, pl_fits):.2f} s")
        ec2.metric("PL — MARE", f"{mare(actuals, pl_fits):.4f}")
        ec3.metric("CS — MAE", f"{mae(actuals, cs_fits):.2f} s")
        ec4.metric("CS — MARE", f"{mare(actuals, cs_fits):.4f}")
        st.caption(
            "**MAE** = Mean Absolute Error (seconds). "
            "**MARE** = Mean Absolute Relative Error (0 = perfect, lower is better)."
        )

# ── Tab 3: Speed–Duration Profile ─────────────────────────────────────────────
with tab_plot:
    t_min = float(max(times_in.min() * 0.3, 20.0))
    t_max = float(times_in.max() * 2.0)
    t_range = np.linspace(t_min, t_max, 600)

    pl_spd = pl_speed_curve(pl["a"], pl["b"], t_range)
    cs_spd = cs_speed_curve(cs["CS_ms"], cs["D_prime"], t_range)

    fig = go.Figure()

    # PL curve
    fig.add_trace(go.Scatter(
        x=t_range, y=pl_spd,
        mode="lines", name="Power Law model",
        line=dict(color="#E74C3C", width=2.5),
    ))

    # CS curve
    fig.add_trace(go.Scatter(
        x=t_range, y=cs_spd,
        mode="lines", name="Critical Speed model",
        line=dict(color="#2980B9", width=2.5),
    ))

    # CS asymptote
    fig.add_hline(
        y=cs["CS_ms"],
        line_dash="dot",
        line_color="#2980B9",
        annotation_text=f"CS = {cs['CS_ms']:.2f} m/s ({cs['CS_kmh']:.1f} km/h)",
        annotation_position="bottom right",
    )

    # Athlete's PBs
    fig.add_trace(go.Scatter(
        x=times_in, y=speeds_in,
        mode="markers+text", name="Your PBs",
        marker=dict(color="black", size=11, symbol="circle"),
        text=[f"{int(d)} m" for d in distances_in],
        textposition="top center",
        textfont=dict(size=12),
    ))

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Speed (m/s)",
        height=500,
        legend=dict(x=0.65, y=0.95),
        margin=dict(t=20, b=50),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Red** = Power Law curve (speed = S · t⁻ᵇ). "
        "**Blue** = Critical Speed curve (speed = CS + D′/t). "
        "The dotted line marks your Critical Speed (CS). "
        "Black dots are your entered personal bests."
    )

    # ── Distance–Time version ──
    st.subheader("Distance–Time View")
    dist_range = np.linspace(t_min, t_max, 600)

    # PL: distance = a · t^(1-b)
    dist_pl = pl["a"] * dist_range ** (1.0 - pl["b"])
    # CS: distance = CS · t + D'
    dist_cs = cs["CS_ms"] * dist_range + cs["D_prime"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dist_range, y=dist_pl,
        mode="lines", name="Power Law",
        line=dict(color="#E74C3C", width=2.5),
    ))
    fig2.add_trace(go.Scatter(
        x=dist_range, y=dist_cs,
        mode="lines", name="Critical Speed",
        line=dict(color="#2980B9", width=2.5),
    ))
    fig2.add_trace(go.Scatter(
        x=times_in, y=distances_in,
        mode="markers+text", name="Your PBs",
        marker=dict(color="black", size=11),
        text=[f"{int(d)} m" for d in distances_in],
        textposition="top center",
        textfont=dict(size=12),
    ))
    fig2.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Distance (m)",
        height=400,
        legend=dict(x=0.05, y=0.95),
        margin=dict(t=20, b=50),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "The CS curve (blue) is linear in the distance–time plane, "
        "reflecting the linear regression used to fit it. "
        "The PL curve (red) is a power function."
    )
