# ForecastAgent SDK Usage Guide

This guide details how to use the `forecast-agent-sdk` Python library to perform zero-shot and fine-tuned time series forecasting.

---

## 1. Importing the SDK

To start using the SDK, import the main `ForecastAgent` wrapper and the `TimeseriesType` dataclass:

```python
from forecastagent import ForecastAgent, TimeseriesType
```

---

## 2. Initializing the Agent

You can load model weights either automatically from Hugging Face or from a local folder:

### Load from Hugging Face
The model weights will be downloaded to your local Hugging Face snapshot cache folder and loaded automatically:
```python
agent = ForecastAgent.from_pretrained("shinydatatech/forecastagent-v1.0", device="cpu")
```

### Load from Local Directory
If you have downloaded the standalone model folder (containing `model.ckpt` and `model-config.yaml`) locally:
```python
agent = ForecastAgent.from_pretrained("./path/to/forecastagent-v1-standalone", device="cpu")
```

*Note: Change `device="cpu"` to `device="cuda"` to run inference on an Nvidia GPU (requires CUDA drivers and a CUDA-enabled PyTorch build).*

---

## 3. Running Zero-Shot Forecasts

The `predict` method is the core function of the SDK. It takes historical series and predicts future steps.

### Method Signature
```python
def predict(
    self,
    target: Union[List[float], np.ndarray, torch.Tensor],
    prediction_length: int = 24,
    past_covariates: Optional[Union[List[List[float]], np.ndarray, torch.Tensor]] = None,
    future_covariates: Optional[Union[List[List[float]], np.ndarray, torch.Tensor]] = None,
    freq: str = "h"
) -> Dict[str, Union[List[float], List[List[float]]]]:
```

### Basic Example (No Covariates)
```python
# Hourly data for the past 12 hours
history = [22.4, 23.1, 23.5, 24.0, 23.8, 23.2, 22.9, 22.1, 21.8, 21.0, 20.8, 20.9]

# Predict next 4 hours
results = agent.predict(
    target=history,
    prediction_length=4,
    freq="h"
)

print("Median Forecast (50th percentile):", results["median"])
print("Lower Bound (10th percentile):", results["lower"])
print("Upper Bound (90th percentile):", results["upper"])
print("All 9 Quantiles:", results["quantiles"])
```

### Advanced Example (With Covariates)
If your model requires covariates (features associated with the time series), you can pass them as 2D lists or arrays (shape: `features x time_steps`).

*   **Past Covariates**: Features known only for the historical period (length equal to history length).
*   **Future Covariates**: Features known for both the historical and future periods (length equal to history length + prediction length).

```python
import numpy as np

# History length = 24 steps, Prediction length = 6 steps
history = np.random.rand(24)

# 2 past covariates (e.g. ambient temperature and pressure)
# Shape: (2, 24)
past_covs = np.random.rand(2, 24).tolist()

# 1 future covariate (e.g. holiday indicator, scheduled maintenance)
# Length: 24 (history) + 6 (prediction) = 30 steps
# Shape: (1, 30)
future_covs = np.random.rand(1, 30).tolist()

results = agent.predict(
    target=history.tolist(),
    prediction_length=6,
    past_covariates=past_covs,
    future_covariates=future_covs,
    freq="h"
)
```

---

## 4. Returned Output Format

The returned dictionary contains:
*   `median`: List of float values representing the 50th percentile forecast.
*   `lower`: List of float values representing the 10th percentile uncertainty bound.
*   `upper`: List of float values representing the 90th percentile uncertainty bound.
*   `quantiles`: A 2D list of shape `(9, prediction_length)` containing the complete forecast for the 9 default quantiles `[10%, 20%, 30%, 40%, 50%, 60%, 70%, 88%, 90%]`.
