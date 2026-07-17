# ForecastAgent serving API Deployment Guide

The `forecast-agent-sdk` packages a production-ready FastAPI serving application that allows you to deploy ForecastAgent 1.0 as a microservice in container environments (like Docker, Kubernetes, or Google Cloud Run).

---

## 1. Running the Server

You can launch the serving server using the built-in CLI entry point:

```bash
forecastagent-api --model-path shinydatatech/forecastagent-v1.0 --port 8000
```

### CLI Command Options
*   `--model-path`, `-m`: The local directory path containing checkpoints or a Hugging Face repository ID. Defaults to downloading `shinydatatech/forecastagent-v1.0` from Hugging Face if no local checkpoint is found.
*   `--host`: Network interface bind address. Defaults to `127.0.0.1`.
*   `--port`, `-p`: Listening port. Defaults to `8000`.
*   `--device`, `-d`: Execution device (`cpu` or `cuda`). Defaults to `cpu`.
*   `--api-key`, `-k`: Enforce bearer authentication. You can also configure this by setting the `FORECASTAGENT_API_KEY` environment variable.

---

## 2. API Authentication

If you provide an API Key, all POST requests must include an `Authorization` header containing the API key as a bearer token:

```
Authorization: Bearer <your_api_key>
```

---

## 3. Endpoints

### 3.1 Health Check
*   **Method**: `GET`
*   **Path**: `/`
*   **Response**:
    ```json
    {
      "status": "active",
      "model": "ForecastAgent 1.0",
      "device": "cpu",
      "auth_enabled": false
    }
    ```

### 3.2 Forecasting
*   **Method**: `POST`
*   **Path**: `/v1/predict`
*   **Headers**: `Content-Type: application/json`
*   **Request Schema**:
    *   `prediction_length` (integer): Forecast horizon length.
    *   `instances` (array of objects): Batch of series to forecast:
        *   `target` (array of floats): History values.
        *   `start` (string): Start timestamp.
        *   `freq` (string): Time frequency (e.g. `h` or `d`).
        *   `past_covariates` (optional, array of float arrays): Past covariates.
        *   `future_covariates` (optional, array of float arrays): Future covariates.

#### Sample Request
```json
{
  "prediction_length": 3,
  "instances": [
    {
      "target": [12.0, 14.5, 15.0, 16.2, 15.5],
      "start": "2026-07-01T00:00:00",
      "freq": "h"
    }
  ]
}
```

#### Sample Response
```json
{
  "predictions": [
    {
      "median": [15.2, 14.9, 14.6],
      "lower": [13.1, 12.8, 12.5],
      "upper": [17.3, 17.0, 16.7],
      "quantiles": [
        [13.1, 12.8, 12.5],
        [13.6, 13.3, 13.0],
        [14.1, 13.8, 13.5],
        [14.7, 14.4, 14.1],
        [15.2, 14.9, 14.6],
        [15.7, 15.4, 15.1],
        [16.2, 15.9, 15.6],
        [16.8, 16.5, 16.2],
        [17.3, 17.0, 16.7]
      ]
    }
  ]
}
```

---

## 4. Querying the Server (Code Example)

You can query the API server programmatically using Python's `requests` library:

```python
import requests

url = "http://127.0.0.1:8000/v1/predict"
headers = {
    "Authorization": "Bearer my-secret-api-key",
    "Content-Type": "application/json"
}

payload = {
    "prediction_length": 24,
    "instances": [
        {
            "target": [10.5, 11.2, 11.8, 12.0, 11.5],
            "freq": "h"
        }
    ]
}

response = requests.post(url, json=payload, headers=headers)
predictions = response.json()["predictions"]
print("Median Forecast:", predictions[0]["median"])
```
