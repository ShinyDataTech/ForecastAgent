import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch

from forecastagent.modeling.base import load_model
from forecastagent.modeling.model.types import TimeseriesType

class ForecastAgent:
    """
    ForecastAgent 1.0: Zero-shot time series forecasting agent SDK.
    Provides a simple Pythonic API to load models (locally or from Hugging Face)
    and perform inference using the ForecastAgent backbone.
    """
    def __init__(self, model_wrapper: Any, device: str = "cpu"):
        self.model = model_wrapper
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: Union[str, Path] = "shinydatatech/forecastagent-v1.0",
        device: str = "cpu",
        **hf_kwargs
    ) -> "ForecastAgent":
        """
        Loads the ForecastAgent model from Hugging Face Hub or a local directory.

        Args:
            repo_id_or_path: Hugging Face repo ID (e.g. 'shinydatatech/forecastagent-v1.0') or local directory path.
            device: Runtime device ('cpu' or 'cuda').
            hf_kwargs: Additional arguments passed to huggingface_hub.snapshot_download.
        """
        if not torch.cuda.is_available() and device == "cuda":
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            
        model_wrapper = load_model(repo_id_or_path, device=device, hf_kwargs=hf_kwargs)
        return cls(model_wrapper=model_wrapper, device=device)

    def predict(
        self,
        target: Union[List[float], np.ndarray, torch.Tensor],
        prediction_length: int = 24,
        past_covariates: Optional[Union[List[List[float]], np.ndarray, torch.Tensor]] = None,
        future_covariates: Optional[Union[List[List[float]], np.ndarray, torch.Tensor]] = None,
        freq: str = "h"
    ) -> Dict[str, Union[List[float], List[List[float]]]]:
        """
        Performs zero-shot probabilistic forecasting for a single time series.

        Args:
            target: 1D historical values of the time series.
            prediction_length: Horizon steps to forecast forward.
            past_covariates: Optional 2D past covariates (shape: covariates x historical_steps).
            future_covariates: Optional 2D future-known covariates (shape: covariates x (historical_steps + prediction_steps)).
            freq: Frequency string ('h', 'd', etc.).

        Returns:
            Dict containing:
                - 'median': 50th percentile forecast.
                - 'lower': 10th percentile uncertainty bound.
                - 'upper': 90th percentile uncertainty bound.
                - 'quantiles': List of forecasts for all 9 quantiles (10% to 90%).
        """
        # Ensure target is 1D tensor of shape (1, sequence_length)
        if isinstance(target, list):
            target_tensor = torch.tensor(target, dtype=torch.float32)
        elif isinstance(target, np.ndarray):
            target_tensor = torch.from_numpy(target).float()
        elif isinstance(target, torch.Tensor):
            target_tensor = target.float()
        else:
            raise TypeError("target must be a list, numpy array, or PyTorch tensor")

        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)
        elif target_tensor.dim() == 2 and target_tensor.size(0) == 1:
            pass
        else:
            raise ValueError("target must be a 1D sequence or have shape (1, sequence_length)")

        # Handle past covariates (should be 2D: num_features x seq_len)
        past_cov_tensor = None
        if past_covariates is not None:
            if isinstance(past_covariates, list):
                past_cov_tensor = torch.tensor(past_covariates, dtype=torch.float32)
            elif isinstance(past_covariates, np.ndarray):
                past_cov_tensor = torch.from_numpy(past_covariates).float()
            else:
                past_cov_tensor = past_covariates.float()

        # Handle future covariates (should be 2D: num_features x (seq_len + pred_len))
        fut_cov_tensor = None
        if future_covariates is not None:
            if isinstance(future_covariates, list):
                fut_cov_tensor = torch.tensor(future_covariates, dtype=torch.float32)
            elif isinstance(future_covariates, np.ndarray):
                fut_cov_tensor = torch.from_numpy(future_covariates).float()
            else:
                fut_cov_tensor = future_covariates.float()

        # Construct TimeseriesType wrapper
        ts = TimeseriesType(
            target=target_tensor,
            past_covariates=past_cov_tensor,
            future_covariates=fut_cov_tensor
        )

        # Execute forecasting on the ForecastAgent backbone
        # Returns shape: (n_targets=1, n_quantiles=9, prediction_length)
        forecast = self.model.forecast(
            [ts],
            prediction_length=prediction_length,
            output_type="numpy"
        )[0]

        # Extract specific quantiles (index 0 is 10th percentile, 4 is 50th/median, 8 is 90th percentile)
        lower = forecast[0, 0, :].tolist()
        median = forecast[0, 4, :].tolist()
        upper = forecast[0, 8, :].tolist()
        quantiles = forecast[0].tolist()

        return {
            "median": median,
            "lower": lower,
            "upper": upper,
            "quantiles": quantiles
        }
