import os
import logging
from typing import List, Optional

from fastapi import FastAPI, Security, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from forecastagent.agent import ForecastAgent

logger = logging.getLogger("ForecastAgentAPI")

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

def create_app(
    model_path_or_repo: str = "shinydatatech/forecastagent-v1.0",
    device: str = "cpu",
    api_key: Optional[str] = None
) -> FastAPI:
    """
    Creates and returns the FastAPI application with the ForecastAgent model pre-loaded.
    """
    app = FastAPI(
        title="ForecastAgent SaaS API",
        description="Zero-Shot Time Series Forecasting Server powered by an xLSTM foundation model",
        version="1.0.0"
    )

    # Initialize model
    logger.info(f"Initializing ForecastAgent model from '{model_path_or_repo}' on {device}...")
    try:
        agent = ForecastAgent.from_pretrained(model_path_or_repo, device=device)
        logger.info("ForecastAgent model loaded successfully.")
    except Exception as e:
        logger.exception(f"Fatal: Could not load model: {e}")
        agent = None

    # Setup API Key authentication
    security = HTTPBearer()

    def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
        if api_key and credentials.credentials != api_key:
            logger.warning("Authentication failed: Invalid API token submitted.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or unauthorized API key credentials."
            )
        return credentials.credentials

    @app.get("/")
    def read_root():
        return {
            "status": "active" if agent is not None else "error_loading_model",
            "model": "ForecastAgent 1.0",
            "device": device,
            "auth_enabled": api_key is not None
        }

    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(request: PredictRequest, token: str = Depends(verify_token) if api_key else None):
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Forecasting agent is not initialized."
            )

        predictions_output = []
        for idx, instance in enumerate(request.instances):
            logger.info(
                f"Processing instance {idx+1}/{len(request.instances)}: "
                f"length={len(instance.target)}, freq={instance.freq}"
            )
            try:
                res = agent.predict(
                    target=instance.target,
                    prediction_length=request.prediction_length,
                    past_covariates=instance.past_covariates,
                    future_covariates=instance.future_covariates,
                    freq=instance.freq
                )
                predictions_output.append(PredictResponseItem(
                    median=res["median"],
                    lower=res["lower"],
                    upper=res["upper"],
                    quantiles=res["quantiles"]
                ))
            except Exception as e:
                logger.error(f"Error serving forecast for instance {idx+1}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference failed on instance {idx+1}: {str(e)}"
                )

        return PredictResponse(predictions=predictions_output)

    return app
