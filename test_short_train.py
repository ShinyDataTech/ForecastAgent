import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import yaml
import torch
import tirex2.model.component.mlstm_block as mlstm_block
import tirex2.model.component.flashrnn_slstm as flashrnn_slstm
from xlstm.xlstm_large.model import mLSTMBackendConfig

# Monkeypatches
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

from tirex2 import load_model, TimeseriesType
from train import configure_lora, pinball_loss

print("Loading model on CUDA...")
model = load_model('NX-AI/TiRex-2', device='cuda')
peft_inner_model = configure_lora(model.model)
model.model = peft_inner_model
model.model.train()

# Dummy batch
context = torch.sin(torch.arange(168).float() / 8).unsqueeze(0).to('cuda')
target = torch.sin(torch.arange(168, 168+24).float() / 8).unsqueeze(0).to('cuda')
ts_batch = [TimeseriesType(target=context, past_covariates=None, future_covariates=None)]

print("Transforming input...")
# 1. Transform input
batch, args, kwargs = model.model.postprocessor.transform_input(
    [ts.target for ts in ts_batch],
    24,
    past_covariates=[None]*len(ts_batch),
    past_future_covariates=[None]*len(ts_batch),
    tta_diff=False,
)

# Move batch to GPU
batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
print("Batch x shape:", batch['x'].shape)

# 2. Forward pass with no 2368 padding
pred = model.model(batch)
print("Forward pass completed. Slicing prediction...")

# 3. Slice the last 24 steps
predictions = pred[:, :, -target.size(-1) :].unsqueeze(1)
print("Predictions shape:", predictions.shape)
print("Target shape:", target.unsqueeze(1).shape)

# 4. Loss
loss = pinball_loss(predictions, target.unsqueeze(1))
print(f"Loss value: {loss.item()}")

# 5. Backward
print("Running backward pass...")
loss.backward()
print("SUCCESS: Backward pass completed successfully on CUDA!")
