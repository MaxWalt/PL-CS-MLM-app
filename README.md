# PL & CS Running Model Analyzer

A Streamlit web app that fits the **Power Law (PL)** and **Critical Speed (CS)** models
to an individual athlete's personal bests (400 m – 10,000 m) and returns:

- Individual physiological parameters (S, E, CS, D′)
- Predicted race times for all standard middle- and long-distance events
- Speed–duration and distance–time profile charts
- Fit quality metrics (MAE, MARE)

## Background

Based on the multi-level modeling (MLM) framework from:

> Waltenspül et al. (2024–2025). *Using MLM to compare Power Law vs Critical Speed.*

The two models:

| Model | Equation | Parameters |
|-------|----------|------------|
| **Power Law (PL)** | speed = S · t⁻ᵇ | S (speed capability), E = 1−b (endurance index) |
| **Critical Speed (CS)** | speed = CS + D′/t | CS (critical speed, m/s or km/h), D′ (anaerobic reserve, m) |

> **Note on fitting approach:** Parameters are estimated by direct OLS curve-fitting
> to the user's personal bests. The original paper uses a population MLM on ~52,000
> athletes to regularize individual estimates via partial pooling. Both approaches
> yield the same parameter definitions — they differ in regularization.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Usage

1. Enter personal bests in the sidebar (at least 2 distances, up to 6).
2. Accepted format: `m:ss`, `m:ss.xx`, or plain seconds (e.g. `1:45.50`, `3:32`, `910`).
3. Results update automatically.
