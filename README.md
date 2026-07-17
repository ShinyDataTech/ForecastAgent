# ForecastAgent 1.0

ForecastAgent 1.0 is a professional, production-ready Python SDK and Serving API for Zero-Shot Time Series Forecasting, powered by an xLSTM-based time series foundation model.

With ForecastAgent 1.0, you can run accurate, probabilistic time series forecasting out of the box with zero training, or deploy a self-hosted SaaS API for scalable production workloads.

---

## Key Features
*   **Zero-Shot Generalization**: Perform high-quality forecasting on new time series datasets without fine-tuning.
*   **Probabilistic Quantiles**: Predicts 9 distinct quantiles (from 10th to 90th percentile) to model uncertainty.
*   **Covariate Support**: Supports both past (historical) and future (known ahead of time) covariates.
*   **Windows & Linux Native**: Includes automated monkeypatches to support execution on Windows hosts bypassing Triton and MSVC compiler dependencies.
*   **FastAPI Serve**: Built-in CLI command to spin up a high-performance serving backend.

---

## Installation

### From PyPI
```bash
pip install forecast-agent-sdk
```

### From Source
Clone this repository and install the package in editable mode:
```bash
git clone https://github.com/shinydatatech/ForecastAgent.git
cd ForecastAgent
pip install -e .
```

### Dependencies
The package requires:
*   `torch>=2.0.0`
*   `numpy`
*   `xlstm>=2.0.0`
*   `fastapi`, `uvicorn`, `pydantic`
*   `huggingface_hub`

---

## Quickstart

### 1. Python SDK Usage
Load the model (either from Hugging Face or a local directory) and run zero-shot forecasting:

```python
from forecastagent import ForecastAgent

# Load the model (automatically downloads weights from Hugging Face if not cached)
agent = ForecastAgent.from_pretrained("shinydatatech/forecastagent-v1.0")

# Input target history (e.g., hourly electricity consumption)
history = [10.2, 11.5, 12.1, 11.8, 13.0, 14.5, 15.2, 14.8, 13.9, 13.1]

# Predict 3 steps forward
results = agent.predict(
    target=history,
    prediction_length=3,
    freq="h"
)

print("Median Forecast (50th percentile):", results["median"])
print("Lower Bound (10th percentile):", results["lower"])
print("Upper Bound (90th percentile):", results["upper"])
print("All 9 Quantiles:", results["quantiles"])
```

### 2. Launch Serving API Server
You can launch the FastAPI server via the command-line interface:

```bash
# Launch server locally on port 8000
forecastagent-api --model-path shinydatatech/forecastagent-v1.0 --port 8000
```

#### API Endpoints
*   `GET /`: Health check and model metadata.
*   `POST /v1/predict`: Forecast endpoint.
    ```json
    {
      "instances": [
        {
          "target": [10.2, 11.5, 12.1, 11.8, 13.0],
          "start": "2026-07-01T00:00:00",
          "freq": "h"
        }
      ],
      "prediction_length": 3
    }
    ```


## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
