"""
ForecastAgent SaaS API Server
=============================
A FastAPI-based serving backend that exposes the pre-trained zero-shot TiRex-2 
time series forecasting model.

Endpoints:
- POST /v1/predict: Submits history and returns point/quantile forecasts.
"""

import os
import logging
from typing import List, Optional

# Enforce eager serving mode to bypass C++ compiler compilation checks on Windows
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import numpy as np
from fastapi import FastAPI, Security, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ForecastAgentAPI")

# Load TiRex-2 model (Check for standalone model first, then fallback to base)
try:
    from tirex2 import load_model, TimeseriesType
    device = "cpu"
    standalone_path = "./forecastagent-v1-standalone"
    
    if os.path.exists(standalone_path):
        logger.info(f"Loading standalone ForecastAgent 1.0 model weights from '{standalone_path}' on {device}...")
        model = load_model(standalone_path, device=device)
        logger.info("Standalone ForecastAgent 1.0 model loaded successfully.")
    else:
        logger.info(f"Loading pre-trained base TiRex-2 model weights on {device}...")
        base_model = load_model("NX-AI/TiRex-2", device=device)
        
        # Load and merge the fine-tuned LoRA adapters if present
        adapter_path = "./forecastagent-v1-lora-joint"
        if os.getenv("FORECASTAGENT_LORA_PATH"):
            adapter_path = os.getenv("FORECASTAGENT_LORA_PATH")
            
        if os.path.exists(adapter_path):
            logger.info(f"Loading fine-tuned LoRA adapter from '{adapter_path}'...")
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(base_model.model, adapter_path)
            base_model.model = peft_model.merge_and_unload()
            logger.info("Successfully merged fine-tuned LoRA adapter weights into backbone.")
        else:
            logger.info("No fine-tuned joint LoRA adapter found. Running zero-shot base model.")
            
        model = base_model
        
    logger.info(f"ForecastAgent model successfully loaded on {device}.")
except Exception as e:
    logger.exception(f"Fatal error: Could not initialize model: {e}")
    model = None

# FastAPI Application
app = FastAPI(
    title="ForecastAgent SaaS API",
    description="Zero-Shot Time Series Forecasting Server powered by TiRex-2",
    version="1.0.0"
)

# API Authentication Security
security = HTTPBearer()
API_KEY = os.getenv("FORECASTAGENT_API_KEY")

if API_KEY:
    logger.info("Authentication enabled: API key verification active.")
else:
    logger.warning("FORECASTAGENT_API_KEY environment variable is not set. Running in open access mode.")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validates the Bearer token if FORECASTAGENT_API_KEY is configured.
    """
    if API_KEY and credentials.credentials != API_KEY:
        logger.warning("Authentication failed: Invalid API token submitted.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized API key credentials."
        )
    return credentials.credentials

# Pydantic Schemas
class Instance(BaseModel):
    target: List[float] = Field(..., description="1D historical target series values.")
    start: str = Field("2026-07-01T00:00:00", description="Start datetime of the series.")
    freq: str = Field("h", description="Frequency string ('h' or 'd').")
    past_covariates: Optional[List[List[float]]] = Field(None, description="2D past covariates.")
    future_covariates: Optional[List[List[float]]] = Field(None, description="2D future-known covariates.")

class PredictRequest(BaseModel):
    instances: List[Instance] = Field(..., description="Batch of timeseries instances to forecast.")
    prediction_length: int = Field(24, description="Horizon steps to forecast forward.")

class PredictResponseItem(BaseModel):
    median: List[float] = Field(..., description="50th percentile (median) point forecast.")
    lower: List[float] = Field(..., description="10th percentile uncertainty bound.")
    upper: List[float] = Field(..., description="90th percentile uncertainty bound.")
    quantiles: List[List[float]] = Field(..., description="All 9 output quantiles [10th to 90th].")

class PredictResponse(BaseModel):
    predictions: List[PredictResponseItem]

# Endpoints
@app.get("/")
def read_root():
    return {
        "status": "active",
        "model": "ForecastAgent",
        "device": "cpu",
        "auth_enabled": bool(API_KEY)
    }

@app.post("/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest, token: str = Depends(verify_token)):
    """
    Accepts time series data, executes zero-shot forecasting, and returns quantiles.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TiRex-2 forecasting engine failed to load at startup."
        )
        
    predictions_output = []
    
    for idx, instance in enumerate(request.instances):
        logger.info(
            f"Processing instance {idx+1}/{len(request.instances)}: "
            f"length={len(instance.target)}, freq={instance.freq}"
        )
        try:
            # 1. Build PyTorch tensors
            target_tensor = torch.tensor(instance.target, dtype=torch.float32).unsqueeze(0)
            
            past_cov = None
            if instance.past_covariates:
                past_cov = torch.tensor(instance.past_covariates, dtype=torch.float32)
                
            fut_cov = None
            if instance.future_covariates:
                fut_cov = torch.tensor(instance.future_covariates, dtype=torch.float32)
                
            # 2. Build TiRex-2 TimeseriesType object
            ts = TimeseriesType(
                target=target_tensor,
                past_covariates=past_cov,
                future_covariates=fut_cov
            )
            
            # 3. Execute zero-shot forecast
            # Return array has shape: (n_targets=1, n_quantiles=9, prediction_length)
            forecast = model.forecast(
                [ts], 
                prediction_length=request.prediction_length, 
                output_type="numpy"
            )[0]
            
            # 4. Extract quantiles
            # Index 0 is 10th percentile, 4 is 50th percentile (median), 8 is 90th percentile
            lower = forecast[0, 0, :].tolist()
            median = forecast[0, 4, :].tolist()
            upper = forecast[0, 8, :].tolist()
            quantiles = forecast[0].tolist()
            
            predictions_output.append(PredictResponseItem(
                median=median,
                lower=lower,
                upper=upper,
                quantiles=quantiles
            ))
            
        except Exception as e:
            logger.error(f"Error serving forecast for instance {idx+1}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed on instance {idx+1}: {str(e)}"
            )
            
    return PredictResponse(predictions=predictions_output)

if __name__ == "__main__":
    import uvicorn
    # Allow running the script directly
    uvicorn.run(app, host="127.0.0.1", port=8000)
