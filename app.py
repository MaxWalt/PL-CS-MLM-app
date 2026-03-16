"""
Power Law & Critical Speed Running Model Analyzer
==================================================
Fits the Power Law (PL) and Critical Speed (CS) models to an individual
athlete's personal bests and returns parameters + predictions.

Model equations (Walt et al., 2025):
  PL:  speed = S · t^(-b)   where b = 1 - E
  CS:  speed = CS + D′ / t

Fitting strategy: direct OLS curve-fitting on the individual's PBs.
This differs from the population MLM in the original paper (which pools
information across ~52,000 athletes), but is the natural approach for a
single athlete entering their own race times.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PL & CS Running Analyzer",
    page_icon="🏃",
    layout="wide",
)

# Keep app alive on Streamlit Community Cloud free tier (10-min heartbeat)
st_autorefresh(interval=600_000, limit=None, key="keepalive")

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

APP_DIR = Path(__file__).parent

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
# WA scoring  (quadratic formula — 2023 World Athletics scoring tables)
#
# Formula:  Points = INT( A·T² + B·T + C )   where T is time in seconds
#
# Coefficients derived from the official WA scoring table PDFs via
# jchen1/iaaf-scoring-tables (github.com/jchen1/iaaf-scoring-tables).
# Verified against MDLD_speed dataset: mean error < 1 pt for all distances.
# ─────────────────────────────────────────────────────────────────────────────

_WA_COEFFS: dict[str, dict[int, tuple]] = {
    "Male": {
        400:   ( 1.0210130425533885,  -161.30922380614925,  6371.289298870831),
        800:   ( 0.1980049254240715,   -72.071360390698,    6558.281603197655),
        1500:  ( 0.04065992530101173,  -31.307736300238858, 6026.662254454998),
        3000:  ( 0.008150049932737697, -13.691983542322477, 5750.592463763857),
        5000:  ( 0.002777997945466534,  -8.000608112339254, 5760.418712472778),
        10000: ( 5.23999442929867e-4,   -3.301192525967682, 5199.371486424821),
    },
    "Female": {
        400:   ( 0.33500597584334024,  -73.69744695937294,  4053.154524419874),
        800:   ( 0.06879989342031843,  -34.39926191656708,  4299.822125128750),
        1500:  ( 0.013399996270512058, -14.471861176585705, 3907.365583602317),
        3000:  ( 0.0025389974609586466, -6.093570428561163, 3656.127933662038),
        5000:  ( 8.079992470755665e-4,  -3.3935897885514335,3563.261678007073),
        10000: ( 1.712000450302986e-4,  -1.5407985033799534,3466.792517297210),
    },
}


def get_wa_points(gender: str, distance: int, time_s: float) -> float | None:
    """
    Return WA points (truncated integer) for a given gender, distance (m),
    and time (seconds) using the 2023 World Athletics quadratic scoring formula.
    Returns None if the event is not in the table.
    """
    coeffs = _WA_COEFFS.get(gender, {}).get(distance)
    if coeffs is None:
        return None
    A, B, C = coeffs
    pts = A * time_s ** 2 + B * time_s + C
    return max(0.0, float(int(pts)))   # truncate, not round


def detect_best_event(gender: str, inputs: dict[int, float]) -> tuple[int, float] | None:
    """
    Return (best_distance, wa_points) — the distance where the athlete scored
    highest WA points.  Returns None if no WA data is available.
    """
    best_d, best_pts = None, -1.0
    for d, t in inputs.items():
        pts = get_wa_points(gender, d, t)
        if pts is not None and pts > best_pts:
            best_pts = pts
            best_d = d
    return (best_d, best_pts) if best_d is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Population data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_population():
    """Load population parameters CSV once per server lifecycle."""
    path = APP_DIR / "population_params.csv"
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# App: header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏃 Power Law & Critical Speed Running Analyzer")
st.markdown(
    "Enter your personal bests for **at least 2 distances**. "
    "The app fits the **Power Law (PL)** and **Critical Speed (CS)** models "
    "to your data and returns individual parameters, "
    "performance predictions for all standard distances, a "
    "speed–duration profile, and your **comparison with peers** "
    "(where you stand among ~52,000 athletes of the same gender and event specialization).\n\n"
    "> Based on the MLM modeling framework from Walt et al. (2025)."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: gender + personal best inputs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Your Profile")

    gender = st.radio(
        "Gender",
        options=["Male", "Female"],
        horizontal=True,
        help="Used to determine your World Athletics points and match you to the correct population group.",
    )

    st.divider()

    st.subheader("Personal Bests")
    st.caption(
        "Format: `ss.xx`, `m:ss`, or `m:ss.xx`  \n"
        "Examples: `46.89`, `3:29.12`, `28:30.30`  \n"
        "Leave blank to skip a distance."
    )
    _PLACEHOLDERS = {
        400: "e.g. 46.89",
        800: "e.g. 1:45.50",
        1500: "e.g. 3:29.12",
        3000: "e.g. 7:52.28",
        5000: "e.g. 13:49.03",
        10000: "e.g. 28:30.30",
    }
    raw_inputs: dict[int, str] = {}
    for d in DISTANCES:
        val = st.text_input(DIST_LABELS[d], key=str(d), placeholder=_PLACEHOLDERS[d])
        if val.strip():
            raw_inputs[d] = val.strip()

    st.divider()
    st.caption("📖 Walt et al. (2025). *Using Multilevel Models to Compare Performance Prediction "
               "and Characterization Abilities Between Power-Law and Critical-Speed Models in Middle- and Long-Distance Running*. "
               "International Journal of Sports Physiology and Performance, 2024–2025")

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
# WA points + best event
# ─────────────────────────────────────────────────────────────────────────────

wa_pts_user: dict[int, float | None] = {
    d: get_wa_points(gender, d, t) for d, t in inputs.items()
}
best_event_result = detect_best_event(gender, inputs)
best_event_dist = best_event_result[0] if best_event_result else None   # int
best_event_pts  = best_event_result[1] if best_event_result else None

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

# CS validated range: 2–20 min (120–1200 s)
_cs_outside = [DIST_LABELS[d] for d, t in inputs.items() if not (120 <= t <= 1200)]
if _cs_outside:
    st.warning(
        f"⚠️ The following performances are outside the range validated for the "
        f"Critical Speed model (2–20 minutes): **{', '.join(_cs_outside)}**. "
        "The CS results may therefore not reflect reality."
    )

# ─────────────────────────────────────────────────────────────────────────────
# WA points summary bar (shown above tabs)
# ─────────────────────────────────────────────────────────────────────────────

if any(v is not None for v in wa_pts_user.values()):
    st.subheader("World Athletics Points & Event Profile")

    wa_cols = st.columns(len(inputs) + 1)
    for i, (d, t) in enumerate(sorted(inputs.items())):
        pts = wa_pts_user.get(d)
        pts_str = f"{pts:.0f}" if pts is not None else "—"
        delta = "⭐ best" if d == best_event_dist else None
        wa_cols[i].metric(
            label=DIST_LABELS[d],
            value=pts_str,
            delta=delta,
            delta_color="off",
        )
        if pts is not None and pts > 1400:
            wa_cols[i].caption("⚠️ > 1400 pts max")

    if best_event_dist:
        wa_cols[-1].metric(
            label="🎯 Best event",
            value=DIST_LABELS[best_event_dist],
            delta=f"{best_event_pts:.0f} pts",
            delta_color="off",
        )

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_params, tab_preds, tab_plot, tab_pop, tab_zones, tab_consult = st.tabs(
    ["📊 Parameters", "⏱️ Predictions", "📈 Speed–Duration Profile",
     "👥 Comparison with Peers", "🏃 Training Zones", "🔬 Want a Deeper Dive?"]
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
                | **S** | Maximal theoretical speed over 1s: higher = faster athlete |
                | **E** | Endurance parameter: closer to 1.0 = more endurant athlete |
                | **b** | Fatigue/decay exponent = 1 − E; higher = speed drops faster with duration |

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
                | **CS** | Critical Speed, highest metabolically sustainable speed (anaerobic threshold proxy) |
                | **D′** | Finite anaerobic work capacity expressed as a distance (m). A larger D′ means more capacity to run above CS |

                *CS* acts as the lower asymptote of the speed–duration curve.
                """
            )

    st.divider()
    st.caption(
        "Parameters are estimated by direct OLS curve-fitting to your personal bests. "
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

    fig.add_trace(go.Scatter(
        x=t_range, y=pl_spd,
        mode="lines", name="Power Law model",
        line=dict(color="#E74C3C", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=t_range, y=cs_spd,
        mode="lines", name="Critical Speed model",
        line=dict(color="#2980B9", width=2.5),
    ))
    fig.add_hline(
        y=cs["CS_ms"],
        line_dash="dot",
        line_color="#2980B9",
        annotation_text=f"CS = {cs['CS_ms']:.2f} m/s ({cs['CS_kmh']:.1f} km/h)",
        annotation_position="bottom right",
    )
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

    st.subheader("Distance–Time View")
    dist_range = np.linspace(t_min, t_max, 600)
    dist_pl = pl["a"] * dist_range ** (1.0 - pl["b"])
    dist_cs = cs["CS_ms"] * dist_range + cs["D_prime"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dist_range, y=dist_pl, mode="lines", name="Power Law",
                              line=dict(color="#E74C3C", width=2.5)))
    fig2.add_trace(go.Scatter(x=dist_range, y=dist_cs, mode="lines", name="Critical Speed",
                              line=dict(color="#2980B9", width=2.5)))
    fig2.add_trace(go.Scatter(x=times_in, y=distances_in,
                              mode="markers+text", name="Your PBs",
                              marker=dict(color="black", size=11),
                              text=[f"{int(d)} m" for d in distances_in],
                              textposition="top center", textfont=dict(size=12)))
    fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Distance (m)",
                       height=400, legend=dict(x=0.05, y=0.95),
                       margin=dict(t=20, b=50), hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "The CS curve (blue) is linear in the distance–time plane. "
        "The PL curve (red) is a power function."
    )

# ── Tab 4: Population Context ─────────────────────────────────────────────────
with tab_pop:
    try:
        pop_df, pop_err = load_population()
    except Exception as e:
        pop_df, pop_err = None, str(e)

    # ── Guard: CSV missing or error ──────────────────────────────────────────
    if pop_df is None:
        st.warning(f"**Population data could not be loaded.**  \n`{pop_err}`")

    # ── Guard: best event unknown ────────────────────────────────────────────
    elif best_event_dist is None:
        st.info("Enter personal bests in the sidebar — WA points will determine your event specialization.")

    # ── Main content ─────────────────────────────────────────────────────────
    else:
        # Filter population to matching gender × best event
        # Best_event is stored as int64 in the CSV (400, 800, 1500, …)
        group = pop_df[
            (pop_df["Gender"] == gender) &
            (pop_df["Best_event"] == best_event_dist)
        ].copy()
        n_group = len(group)

        st.markdown(
            f"Based on your WA points, your best event is the "
            f"**{DIST_LABELS[best_event_dist]}** ({best_event_pts:.0f} pts).  \n"
            f"Comparing you to **{n_group:,}** {gender} "
            f"**{DIST_LABELS[best_event_dist]} specialists** in the Walt et al. (2025) dataset."
        )

        if n_group < 10:
            st.warning(
                f"Only {n_group} athletes in this group — not enough for a meaningful comparison. "
                "Try adding more distances to better identify your event specialization."
            )

        else:
            # Helper: percentile rank
            def pct_rank(arr, val):
                return float(stats.percentileofscore(arr, val, kind="rank"))

            # Helper: histogram with user marker
            def dist_fig(param_col, user_val, xlabel, color, title):
                pop_vals = group[param_col].dropna().values
                pct = pct_rank(pop_vals, user_val)
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pop_vals, nbinsx=40,
                    marker_color=color, opacity=0.75, name="Population",
                ))
                fig.add_vline(
                    x=user_val, line_width=2.5, line_color="black",
                    annotation_text=f"You — {pct:.0f}th percentile",
                    annotation_position="top right", annotation_font_size=12,
                )
                fig.update_layout(
                    title=title, xaxis_title=xlabel,
                    yaxis_title="Number of athletes",
                    height=280, margin=dict(t=40, b=40, l=40, r=20),
                    showlegend=False,
                )
                return fig, pct

            # ── PL distributions ─────────────────────────────────────────────
            st.subheader("Power Law parameters")
            col_s, col_b = st.columns(2)

            with col_s:
                fig_s, pct_s = dist_fig(
                    "S", pl["S"], "S (m/s)", "#E74C3C",
                    f"Speed coefficient S — {gender} {DIST_LABELS[best_event_dist]} specialists",
                )
                st.plotly_chart(fig_s, use_container_width=True)
                st.metric("Your S", f"{pl['S']:.3f} m/s", f"{pct_s:.0f}th percentile")

            with col_b:
                fig_b, pct_b = dist_fig(
                    "E", pl["E"], "E (endurance index)", "#E74C3C",
                    f"Endurance index E — {gender} {DIST_LABELS[best_event_dist]} specialists",
                )
                st.plotly_chart(fig_b, use_container_width=True)
                st.metric(
                    "Your E", f"{pl['E']:.4f}",
                    f"{pct_b:.0f}th percentile",
                )

            # ── CS distributions ─────────────────────────────────────────────
            st.subheader("Critical Speed parameters")
            col_cs, col_dp = st.columns(2)

            with col_cs:
                fig_cs, pct_cs = dist_fig(
                    "CS_ms", cs["CS_ms"], "CS (m/s)", "#2980B9",
                    f"Critical Speed CS — {gender} {DIST_LABELS[best_event_dist]} specialists",
                )
                st.plotly_chart(fig_cs, use_container_width=True)
                st.metric(
                    "Your CS",
                    f"{cs['CS_ms']:.3f} m/s  ({cs['CS_kmh']:.2f} km/h)",
                    f"{pct_cs:.0f}th percentile",
                )

            with col_dp:
                fig_dp, pct_dp = dist_fig(
                    "D_prime", cs["D_prime"], "D′ (m)", "#2980B9",
                    f"Anaerobic reserve D′ — {gender} {DIST_LABELS[best_event_dist]} specialists",
                )
                st.plotly_chart(fig_dp, use_container_width=True)
                st.metric("Your D′", f"{cs['D_prime']:.1f} m", f"{pct_dp:.0f}th percentile")

            # ── Summary table ─────────────────────────────────────────────────
            st.divider()
            st.subheader("Percentile Summary")

            pop_means = group[["S", "E", "CS_ms", "D_prime"]].mean()
            pop_sds   = group[["S", "E", "CS_ms", "D_prime"]].std()

            st.dataframe(pd.DataFrame([
                {
                    "Parameter": "S (m/s)",
                    "Your value": f"{pl['S']:.3f}",
                    "Group mean ± SD": f"{pop_means['S']:.3f} ± {pop_sds['S']:.3f}",
                    "Percentile": f"{pct_s:.0f}",
                    "Explanation": "Higher = higher speed abilities",
                },
                {
                    "Parameter": "E (endurance index)",
                    "Your value": f"{pl['E']:.4f}",
                    "Group mean ± SD": f"{pop_means['E']:.4f} ± {pop_sds['E']:.4f}",
                    "Percentile": f"{pct_b:.0f}",
                    "Explanation": "Higher = better endurance/durability",
                },
                {
                    "Parameter": "CS (m/s)",
                    "Your value": f"{cs['CS_ms']:.3f}",
                    "Group mean ± SD": f"{pop_means['CS_ms']:.3f} ± {pop_sds['CS_ms']:.3f}",
                    "Percentile": f"{pct_cs:.0f}",
                    "Explanation": "Higher = better second threshold/endurance",
                },
                {
                    "Parameter": "D′ (m)",
                    "Your value": f"{cs['D_prime']:.1f}",
                    "Group mean ± SD": f"{pop_means['D_prime']:.1f} ± {pop_sds['D_prime']:.1f}",
                    "Percentile": f"{pct_dp:.0f}",
                    "Explanation": "Higher = better reserve above CS",
                },
            ]), use_container_width=True, hide_index=True)

            st.caption(
                f"Population: {n_group:,} {gender} athletes whose highest WA score was in the "
                f"{DIST_LABELS[best_event_dist]}. "
                "Individual parameters are estimated by OLS (same method as your own fit), "
                "not the full hierarchical MLM of the original paper."
            )

# ── Tab 5: Training Zones ─────────────────────────────────────────────────────
with tab_zones:
    # Need ≥ 2 performances in the 2–20 min validated CS range
    _cs_valid = {d: t for d, t in inputs.items() if 120 <= t <= 1200}

    if len(_cs_valid) < 2:
        _n = len(_cs_valid)
        _missing = 2 - _n
        st.info(
            "⚠️ **Training zones cannot be computed.**\n\n"
            "The Critical Speed model needs **at least 2 performances in the 2–20 minute range** "
            f"(1500 m, 3000 m, or 5000 m typically fall here). "
            f"You currently have **{_n}** such performance{'s' if _n != 1 else ''} entered — "
            f"please add **{_missing} more** to unlock your training zones."
        )
    else:
        CS = cs["CS_kmh"]  # km/h

        # Zone 2 upper boundary varies with CS level (Hunter et al., 2024)
        if CS <= 12.0:
            z2_hi = 0.806
        elif CS <= 14.0:
            z2_hi = 0.832
        else:
            z2_hi = 0.842

        def _pace(speed_kmh: float) -> str:
            """Format km/h → 'mm:ss' pace per km."""
            if speed_kmh <= 0:
                return "—"
            spk = 3600.0 / speed_kmh
            return f"{int(spk // 60)}:{spk % 60:04.1f}"

        # Zone definitions: (label, description, lo_frac, hi_frac | None)
        _zone_defs = [
            ("Zone 1", "Easy / Recovery",  0.0,    0.70),
            ("Zone 2", "Aerobic Base",     0.70,   z2_hi),
            ("Zone 3", "Tempo",            z2_hi,  0.92),
            ("Zone 4", "Threshold",        0.92,   1.05),
            ("Zone 5", "VO₂max",           1.05,   None),
        ]
        _zone_colors = ["#2980B9", "#27AE60", "#D4AC0D", "#E67E22", "#C0392B"]

        _rows = []
        for (zone, desc, lo, hi), color in zip(_zone_defs, _zone_colors):
            lo_spd = CS * lo
            hi_spd = CS * hi if hi is not None else None

            if lo == 0.0:
                spd_str  = f"< {hi_spd:.2f}"
                pace_str = f"> {_pace(hi_spd)}"
                pct_str  = f"< {hi * 100:.1f} %"
            elif hi is None:
                spd_str  = f"> {lo_spd:.2f}"
                pace_str = f"< {_pace(lo_spd)}"
                pct_str  = f"> {lo * 100:.1f} %"
            else:
                spd_str  = f"{lo_spd:.2f} – {hi_spd:.2f}"
                # pace: faster end (hi speed) listed first
                pace_str = f"{_pace(hi_spd)} – {_pace(lo_spd)}"
                pct_str  = f"{lo * 100:.1f} – {hi * 100:.1f} %"

            _rows.append((zone, desc, pct_str, spd_str, pace_str, color))

        # Render as a styled HTML table for per-row color control
        _html = """
        <style>
        .zt { width:100%; border-collapse:collapse; font-family:sans-serif; font-size:14px; }
        .zt th { background:#444444; padding:9px 14px; text-align:left;
                 border-bottom:2px solid #ccc; white-space:nowrap; }
        .zt td { padding:8px 14px; border-bottom:1px solid #ddd; }
        </style>
        <table class="zt">
        <tr>
          <th>Zone</th>
          <th>Description</th>
          <th>% of CS</th>
          <th>Speed (km/h)</th>
          <th>Pace (min:ss /km)</th>
        </tr>
        """
        for zone, desc, pct, spd, pace, color in _rows:
            _html += (
                f'<tr style="background-color:{color}; color:#ffffff;">'
                f"<td><b>{zone}</b></td><td>{desc}</td>"
                f"<td>{pct}</td><td>{spd}</td><td>{pace}</td></tr>"
            )
        _html += "</table>"

        st.markdown(_html, unsafe_allow_html=True)

        st.caption(
            f"CS = **{CS:.2f} km/h** ({cs['CS_ms']:.3f} m/s). "
            "Zone 2 upper boundary adapted by CS level following "
            "**Hunter et al. (2024)**."
        )

# ── Tab 6: Deeper dive / consult booking ─────────────────────────────────────
with tab_consult:
    st.subheader("Want a Deeper Dive into Your Data?")
    st.markdown(
        "Book an introductory one-on-one session to explore how performance "
        "modelling can inform your training and competition approach.")
    st.markdown(        
        "*This session focuses on understanding your context and determining whether a deeper performance analysis would be beneficial.*"
        )
    components.html(
        """
        <!-- Calendly inline widget begin -->
        <div class="calendly-inline-widget"
             data-url="https://calendly.com/maxime-walt/meeting-data"
             style="min-width:320px;height:700px;"></div>
        <script type="text/javascript"
                src="https://assets.calendly.com/assets/external/widget.js"
                async></script>
        <!-- Calendly inline widget end -->
        """,
        height=720,
    )
