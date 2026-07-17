# ForecastAgent Feature Summary

ForecastAgent is an enterprise-grade Time Series Forecasting SaaS solution designed to demonstrate the power of foundation models in time series prediction. It combines dynamic, domain-aware synthetic time-series generation with zero-shot forecasting capabilities powered by **TiRex-2**.

---

## 🎯 Core Features

### 1. Interactive Forecasting Dashboard
*   **Modern Theme**: Out-of-the-box native dark mode design system matching enterprise aesthetics (dark slate background, vibrant violet controls, and clear typography).
*   **Plotly Visualizations**: High-fidelity charts with fluid zooming, panning, and legend toggling.
*   **Dual-Interval Fan Charts**: Displays both **50%** (inner dark purple) and **90%** (outer light purple) confidence bands to illustrate statistical model uncertainty over the forecasting horizon.
*   **Seamless Visual Continuity**: Features a visual continuity connection from the tail of historical actuals to the point forecast and confidence bounds.

### 2. Multi-Domain Industry Scenarios
Supports 6 distinct industry use cases with specialized baseline values, seasonality equations, noise thresholds, and action recommendations:

| Industry Domain | Target Metric | Future Covariate | Business Recommendation Focus |
| :--- | :--- | :--- | :--- |
| **Infrastructure & Asset Management** | Pothole Volumetric Expansion (Liters) | Heavy Rainfall (mm) | Dispatch road patching crews before predicted precipitation washout |
| **Transportation & Traffic** | Commuter Traffic Flow (Vehicles/hour) | Public Holiday (0/1) | Schedule extra transit shuttles and align road work during low-traffic holidays |
| **Business, Retail & E-commerce** | Daily Sales Volume (Units) | Active Ad Campaign (0/1) | Alert inventory planners to increase buffer stocks ahead of campaign spikes |
| **Education** | Student Course Enrollment Counts | Intake Window (0/1) | Coordinate Teaching Assistant contracts and room sizes for online intakes |
| **Weather & Environment** | Natural Disaster Risk Index (0-100) | Extreme Wind Warning (0/1) | Preemptively mobilize emergency response fleets in risk zones |
| **Disease & Macro Trends** | Active Case Infections | Lockdown/Health Policy (0/1) | Pre-allocate hospital ward staffing 7-10 days ahead of transmission peaks |

### 3. Covariate & Event Overlay
*   **Overlay Toggle**: Allows users to turn on/off future-known event overlays on the Plotly chart.
*   **Event Representation**: Displays binary events (such as holidays or intakes) as transparent orange vertical bars, and continuous covariates (such as rainfall) as secondary-axis dashed trend lines to prevent visual scaling distortion.

---

## 🛠️ Technical Architecture

### 1. Dual-Engine Inference System
ForecastAgent implements a resilient dual-engine architecture:
*   **Production Engine (Real Model)**: Imports `tirex2`, loads the weights from Hugging Face Hub, and performs CPU-optimized zero-shot inference.
*   **Simulated Engine (Fallback)**: If dependencies (such as CUDA/Triton) are missing locally, or Hugging Face authentication is unavailable, the application gracefully degrades to a domain-aware mock engine. This engine simulates realistic forecasts and overlays the validation ground truth directly, outputting standard uncertainty quantiles.

### 2. Resource Management & Scale
*   **RAM & CPU Allocation**: Configured with **4 GiB of RAM** and **2 vCPUs** in production to prevent out-of-memory (OOM) pressure during model initialization and ensure lightning-fast page loading and inference times.

---

## 🚀 DevOps & Deployment

*   **Google Cloud Run Serverless**: Packaged as a lightweight container using a `Dockerfile` based on `python:3.11-slim`.
*   **Robust Build Pipeline**: Utilizes a custom build-time download helper script (`download_model.py`) that handles gated model repository access errors gracefully. This allows container compilation to succeed even in restricted build environments, completing authentication checks at runtime using `HF_TOKEN`.
*   **Streamlit Configuration**: Forces native dark mode startup globally using `.streamlit/config.toml` (setting `base = "dark"`).
