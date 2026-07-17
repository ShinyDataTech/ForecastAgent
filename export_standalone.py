import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import torch
import shutil
import yaml
import tirex2.model.component.mlstm_block as mlstm_block
import tirex2.model.component.flashrnn_slstm as flashrnn_slstm
from xlstm.xlstm_large.model import mLSTMBackendConfig

# Monkeypatches for local load stability
def custom_mlstm_backend_config(config, device):
    if device == 'cpu': return mlstm_block._mlstm_backend_config_orig(config, 'cpu')
    return mLSTMBackendConfig(
        chunkwise_kernel='chunkwise--native_custbw',
        sequence_kernel='native_sequence__native',
        step_kernel='native',
        mode=config.mode,
        chunk_size=config.chunk_size,
        return_last_states=config.return_last_states,
        autocast_kernel_dtype='float32',
        eps=config.eps,
        inference_state_dtype='float32',
    )
if not hasattr(mlstm_block, "_mlstm_backend_config_orig"):
    mlstm_block._mlstm_backend_config_orig = mlstm_block._mlstm_backend_config
    mlstm_block._mlstm_backend_config = custom_mlstm_backend_config

flashrnn_slstm._flashrnn_backend = lambda device: 'vanilla'

from tirex2 import load_model
from peft import PeftModel
from tirex2.base import _resolve_ckpt_dir, CONFIG_FILENAME, CKPT_FILENAME

print("Loading base model...")
base_model = load_model("NX-AI/TiRex-2", device="cpu")

print("Loading joint LoRA adapter weights...")
peft_model = PeftModel.from_pretrained(base_model.model, "./forecastagent-v1-lora-joint")

print("Merging LoRA weights into base backbone...")
merged_inner_model = peft_model.merge_and_unload()

# Create standalone directory
standalone_dir = "./forecastagent-v1-standalone"
os.makedirs(standalone_dir, exist_ok=True)

# Save merged state dict
print("Saving merged state_dict to standalone directory...")
checkpoint_path = os.path.join(standalone_dir, "model.ckpt")
torch.save(merged_inner_model.state_dict(), checkpoint_path)

# Copy config file
print("Copying model configuration file...")
ckpt_dir = _resolve_ckpt_dir("NX-AI/TiRex-2")
shutil.copy(ckpt_dir / CONFIG_FILENAME, os.path.join(standalone_dir, CONFIG_FILENAME))

print(f"SUCCESS: Standalone time series foundation model exported to '{standalone_dir}'!")
