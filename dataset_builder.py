import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesWindowDataset(Dataset):
    """
    Dataset to generate sliding window views of a univariate/multivariate time series.
    """
    def __init__(self, values: np.ndarray, context_len: int, prediction_len: int):
        self.values = values.astype(np.float32)
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.total_window = context_len + prediction_len
        
    def __len__(self):
        return len(self.values) - self.total_window + 1

    def __getitem__(self, idx):
        window = self.values[idx : idx + self.total_window]
        
        # Split into context history and future ground-truth target
        context = window[:self.context_len]  # Shape: (context_len,) or (context_len, n_targets)
        target = window[self.context_len:]   # Shape: (prediction_len,) or (prediction_len, n_targets)
        
        # Format shapes to (n_targets, steps)
        if len(context.shape) == 1:
            context = np.expand_dims(context, axis=0)
            target = np.expand_dims(target, axis=0)
        else:
            context = context.T
            target = target.T
            
        # Instance Normalization (normalize input context, preserve scaling stats)
        mean = np.mean(context, axis=-1, keepdims=True)
        std = np.std(context, axis=-1, keepdims=True) + 1e-5
        
        norm_context = (context - mean) / std
        norm_target = (target - mean) / std
        
        return {
            "context": torch.tensor(norm_context, dtype=torch.float32),
            "target": torch.tensor(norm_target, dtype=torch.float32),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32)
        }
