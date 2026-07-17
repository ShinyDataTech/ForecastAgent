import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Apply monkeypatches to bypass Triton and MSVC (cl.exe) compiler dependencies on Windows GPU
try:
    import tirex2.model.component.mlstm_block as mlstm_block
    import tirex2.model.component.flashrnn_slstm as flashrnn_slstm
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
except Exception:
    pass

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from dataset_builder import TimeSeriesWindowDataset
from data_collector import load_and_split_data
from tirex2 import load_model, TimeseriesType
from peft import LoraConfig, get_peft_model

def configure_lora(model):
    """
    Injects LoRA layers into the temporal/variate mixing attention layers of TiRex-2.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Target the query and value projection layers in the xLSTM time mixers and attention variate mixers
        target_modules=["q", "v", "WQ", "WV"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["output_patch_embedding"] # Fine-tune the output projection block
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model

def pinball_loss(y_pred, y_true):
    """
    Computes the pinball loss over 9 target quantiles.
    y_pred shape: (batch_size, n_targets, 9, prediction_len)
    y_true shape: (batch_size, n_targets, prediction_len)
    """
    quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], device=y_pred.device)
    loss = 0.0
    
    # Iterate over each quantile and calculate standard pinball formulation
    for i, q in enumerate(quantiles):
        diff = y_true - y_pred[:, :, i, :]
        loss_q = torch.max(q * diff, (q - 1) * diff)
        loss += loss_q.mean()
        
    return loss / len(quantiles)

def train_epoch(model, dataloader, optimizer, device):
    from tqdm import tqdm
    model.model.train() # Set the inner PyTorch module to training mode
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in progress_bar:
        # Move tensors to device
        context = batch["context"].to(device)
        target = batch["target"].to(device)
        
        optimizer.zero_grad()
        
        # Instantiate TiRex-2 TimeseriesType batch
        ts_batch = [
            TimeseriesType(target=context[i], past_covariates=None, future_covariates=None)
            for i in range(context.size(0))
        ]
        
        # 1. Transform inputs (list of target tensors) using postprocessor
        context_tensors = [ts.target for ts in ts_batch]
        prep_batch, args, kwargs = model.model.postprocessor.transform_input(
            context_tensors,
            target.size(-1),
            past_covariates=[None] * len(ts_batch),
            past_future_covariates=[None] * len(ts_batch),
            tta_diff=False,
        )
        
        # Move prep_batch tensors to device
        prep_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prep_batch.items()}
        
        # 2. Execute the forward pass directly on the inner model (keeps gradient tracking active!)
        pred = model.model(prep_batch)
        
        # 3. Slice prediction back to prediction_length (last prediction_length steps of sequence)
        pred = pred[:, :, -target.size(-1) :]
        
        # 4. Unsqueeze the target variates dimension (shape: batch_size, 1, 9, prediction_length)
        predictions = pred.unsqueeze(1)
        
        loss = pinball_loss(predictions, target)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        progress_bar.set_postfix(loss=f"{loss_val:.5f}")
        
    return total_loss / len(dataloader)

if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "electricity"
    if dataset_name not in ["electricity", "retail", "bike", "all", "joint"]:
        print(f"Unknown dataset '{dataset_name}'. Available: electricity, retail, bike, all, joint.")
        sys.exit(1)
        
    # Automatically use local GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting ForecastAgent 1.0 fine-tuning on dataset '{dataset_name}' on device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU. GPU execution is highly recommended for speed.")
        
    # Load base model (in eager mode)
    base_model = load_model("NX-AI/TiRex-2", device=device)
    
    # Inject LoRA parameters into the inner PyTorch model (base_model.model)
    peft_inner_model = configure_lora(base_model.model)
    peft_inner_model.to(device)
    
    # Delegate the inner model back to the ForecastModel wrapper
    base_model.model = peft_inner_model
    
    # Load dataset(s)
    if dataset_name in ["all", "joint"]:
        from torch.utils.data import ConcatDataset
        datasets = []
        for name in ["electricity", "retail", "bike"]:
            train_data, _ = load_and_split_data(name)
            datasets.append(TimeSeriesWindowDataset(train_data, context_len=168, prediction_len=24))
        dataset = ConcatDataset(datasets)
        print(f"Loaded joint dataset with {len(dataset)} total window samples.")
    else:
        train_data, val_data = load_and_split_data(dataset_name)
        dataset = TimeSeriesWindowDataset(train_data, context_len=168, prediction_len=24)
        print(f"Loaded dataset '{dataset_name}' with {len(dataset)} window samples.")
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Configure optimizer to target only the trainable LoRA parameters
    optimizer = optim.AdamW(peft_inner_model.parameters(), lr=5e-5)
    
    # Execute epochs
    for epoch in range(1, 6):
        avg_loss = train_epoch(base_model, dataloader, optimizer, device)
        print(f"Epoch {epoch}/5 | Train Pinball Loss: {avg_loss:.5f}")
        
    # Save adapter checkpoints
    save_name = "joint" if dataset_name in ["all", "joint"] else dataset_name
    save_path = f"./forecastagent-v1-lora-{save_name}"
    os.makedirs(save_path, exist_ok=True)
    peft_inner_model.save_pretrained(save_path)
    print(f"Fine-tuning completed. LoRA adapter saved to '{save_path}'.")
