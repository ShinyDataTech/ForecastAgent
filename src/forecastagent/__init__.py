import os
import sys

# Enforce eager serving mode to bypass Triton / C++ compiler check on Windows
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Apply monkeypatches for Windows compatibility to the internal vendored modeling code
try:
    import forecastagent.modeling.model.component.mlstm_block as mlstm_block
    import forecastagent.modeling.model.component.flashrnn_slstm as flashrnn_slstm
    from xlstm.xlstm_large.model import mLSTMBackendConfig

    def custom_mlstm_backend_config(config, device):
        if device == "cpu":
            return mlstm_block._mlstm_backend_config_orig(config, "cpu")
        else:
            return mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--native_custbw",
                sequence_kernel="native_sequence__native",
                step_kernel="native",
                mode=config.mode,
                chunk_size=config.chunk_size,
                return_last_states=config.return_last_states,
                autocast_kernel_dtype="float32",
                eps=config.eps,
                inference_state_dtype="float32",
            )

    if not hasattr(mlstm_block, "_mlstm_backend_config_orig"):
        mlstm_block._mlstm_backend_config_orig = mlstm_block._mlstm_backend_config
        mlstm_block._mlstm_backend_config = custom_mlstm_backend_config

    def custom_flashrnn_backend(device):
        return "vanilla"

    flashrnn_slstm._flashrnn_backend = custom_flashrnn_backend
except Exception as e:
    pass

# Expose public SDK interfaces
from forecastagent.agent import ForecastAgent
from forecastagent.modeling.model.types import TimeseriesType

__version__ = "1.0.0"
__all__ = ["ForecastAgent", "TimeseriesType"]
