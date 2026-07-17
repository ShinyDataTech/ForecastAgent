import os
import urllib.request
import pandas as pd
import numpy as np

DATASET_URLS = {
    "electricity": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "retail": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/simulated_items_sales.csv",
    "bike": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/bike_sharing_dataset_clean.csv"
}

DATASET_TARGETS = {
    "electricity": "OT",
    "retail": "item_1",
    "bike": "users"
}

def load_and_split_data(dataset_name: str, cache_dir: str = "./data", train_split: float = 0.85):
    """
    Downloads raw time series data, extracts the target series, and splits into train/val arrays.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"{dataset_name}.csv")
    
    # 1. Download raw CSV if not cached
    if not os.path.exists(file_path):
        url = DATASET_URLS[dataset_name]
        print(f"Downloading {dataset_name} dataset from {url}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
        
    # 2. Parse target series values
    df = pd.read_csv(file_path)
    target_col = DATASET_TARGETS[dataset_name]
    values = df[target_col].values
    
    # 3. Split into Train & Validation arrays (temporal split to prevent leakage)
    split_idx = int(len(values) * train_split)
    train_values = values[:split_idx]
    val_values = values[split_idx:]
    
    print(f"Loaded '{dataset_name}' target values. Total length: {len(values)}")
    print(f"Split: Train={len(train_values)} | Validation={len(val_values)}")
    return train_values, val_values
