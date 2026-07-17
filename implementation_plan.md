# Implementation Plan - ForecastAgent SaaS Portal

This plan outlines the design and implementation of **ForecastAgent**, a Time Series Forecasting SaaS interactive portal powered by the zero-shot foundation model, **TiRex-2**.

## Technical Analysis & Design Decisions

1. **Python Version Compatibility**: 
   - The user requested a `Dockerfile` using `python:3.10-slim`. However, the local `tirex-2` repository's `pyproject.toml` explicitly requires Python `>=3.11,<3.14`.
   - **Decision**: We will use `python:3.11-slim` to prevent pip installation failures, while keeping the container as slim and optimized as possible.
2. **Graceful Model Fallback (Developer Experience)**:
   - The actual `tirex-2` library depends on `triton` via `flashrnn`. Triton is not supported on Windows, meaning running the model directly on a Windows development host will fail with a `ModuleNotFoundError` for `triton`.
   - **Decision**: We will implement a dual-mode initialization in `app.py`. If the real `tirex-2` model or its dependencies fail to load, the application will degrade gracefully to a **Simulation Engine**. The Simulation Engine will mirror the `ForecastModel` API exactly, generating statistically sound forecasts with confidence intervals based on the input series. A subtle, professional indicator in the UI will inform the user of the current mode (Real Model vs. Simulation Engine).
3. **Synthetic Data Realism**:
   - The `DataSimulator` will produce complex multivariate series with multiple seasonality components (e.g., daily + weekly), long-term trends, random noise, and dynamic covariate injections (e.g., weather events, public holidays, scheduled maintenance).
   - Future covariates will span the history + prediction window.

---

## Proposed Changes

### 1. New Project Files

#### [NEW] [data_simulator.py](file:///c:/Users/wei.liu/Documents/ForecastAgent/data_simulator.py)
A module containing the `DataSimulator` class that generates synthetic time-series data with covariates for the following 8 domains:
*   **Infrastructure & Asset Management** (Pavement defect deterioration with rainfall events)
*   **Transportation & Traffic** (Regional road network flow with holiday dropouts/spikes)
*   **Business & Retail** (Marketing campaign ROI with campaign spikes/ad spend)
*   **Sensor-Based Manufacturing** (Vibration/heat degradation with scheduled maintenance resets)
*   **Financial Markets** (Stock price trends, risk volatility index, interest rate announcement events)
*   **Education** (Cyclical academic enrollments and online vs in-person modalities)
*   **Weather & Environment** (Temperature climate tracking and natural disaster probability indices)
*   **Disease & Macro Trends** (Epidemiological transmission waves and macro GDP policy change dates)

#### [NEW] [app.py](file:///c:/Users/wei.liu/Documents/ForecastAgent/app.py)
The core Streamlit application.
*   **Theme & Design**: Custom dark-mode style CSS with high-end typography, glassmorphic metrics cards, and smooth transitions.
*   **Sidebar**: Selection of domain, use case, parameters (`context_length`, `prediction_length`), and model settings.
*   **Main Panel**: Branding header, top metrics (peak demand, uncertainty bounds, trend direction, model status), interactive Plotly chart with point forecast and shaded quantiles (representing confidence intervals), and covariate overlay toggles.
*   **Robust Imports**: Graceful fallback to `MockForecastModel` if `tirex2` throws `ImportError` or other errors.

#### [NEW] [requirements.txt](file:///c:/Users/wei.liu/Documents/ForecastAgent/requirements.txt)
Defines project dependencies.
*   Specifies CPU-only PyTorch to minimize image size: `--extra-index-url https://download.pytorch.org/whl/cpu`
*   Lists `streamlit`, `plotly`, `pandas`, `numpy`, `einops`, `xlstm`, `huggingface_hub`, `PyYAML`, and `./tirex-2`.

#### [NEW] [Dockerfile](file:///c:/Users/wei.liu/Documents/ForecastAgent/Dockerfile)
Optimized for Google Cloud Run deployment.
*   Base image: `python:3.11-slim`
*   Installs build-essential and git (required for some packages).
*   Copies code and installs dependencies using `requirements.txt`.
*   Includes a `RUN` step to pre-download `NX-AI/TiRex-2` weights into the Hugging Face cache folder during the image build to prevent runtime cold-start issues.
*   Binds to port `8080`.

---

## Verification Plan

### Automated Verification
*   We will run `streamlit` locally on the Windows host and verify that:
    1. The app initializes successfully in simulation mode without crashing on missing `triton` library.
    2. All 8 domains and their respective use cases load correctly.
    3. The Plotly charts render beautifully and support zooming/toggling of covariates.
    4. Modifying parameters recalculates and updates the charts dynamically.

### Manual Verification
*   Verify the Dockerfile compiles and is ready for Cloud Run by running a local Docker build test (if Docker is available).
