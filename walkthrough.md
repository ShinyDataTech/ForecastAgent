# Walkthrough - ForecastAgent API Development & Full Real Model Execution

This walkthrough documents the creation of the ForecastAgent backend API, the final, successful execution of the benchmarking pipeline using **100% real model weights** across all models, the implementation of dynamic LoRA fine-tuning, and the final packaging of the codebase.

## 1. Developed ForecastAgent API Server
We created [api_server.py](file:///c:/Users/wei.liu/Documents/ForecastAgent/api_server.py) using FastAPI to wrap the local `tirex-2` model:
- **Framework**: FastAPI + Uvicorn server.
- **Port**: Serves locally on `127.0.0.1:8000`.
- **Eager mode CPU serving**: Automatically disables PyTorch dynamic inductor compilers (`TORCHDYNAMO_DISABLE=1` and `TORCH_COMPILE_DISABLE=1`) to avoid MSVC compiling dependencies (`cl.exe`) on Windows development hosts.
- **Endpoint**: Serves a `POST /v1/predict` endpoint that validates historical target values, converts them to PyTorch tensors, runs zero-shot inference, and extracts point predictions and quantiles matching our pipeline expectations.

---

## 2. Updated dependencies ([requirements.txt](file:///c:/Users/wei.liu/Documents/ForecastAgent/requirements.txt))
We appended the serving and compiler compatibility dependencies:
```
fastapi
uvicorn[standard]
pydantic
torchvision
peft
accelerate
```
This ensures that running `pip install -r requirements.txt` on any fresh environment will configure the environment correctly for local modeling, API hosting, and fine-tuning.

---

## 3. GPU-Accelerated LoRA Fine-Tuning & Multi-Task Joint Training
We implemented and executed an optimized, GPU-accelerated LoRA fine-tuning flow:
- **Windows CUDA Monkeypatches**: Applied runtime monkeypatches to override the model's sequence kernels.
  - mLSTM was routed to the Windows-compatible `native_custbw` backend.
  - sLSTM was routed to the PyTorch native `vanilla` backend.
  This bypassed all Triton and MSVC compiler (`cl.exe`) JIT compilation requirements, enabling GPU-accelerated training directly on Windows CUDA out of the box.
- **Sequence Length Optimization**: Bypassed the default zero-shot padding (which padded every sequence to `max_ts_len = 2,368` steps). Training at the native dataset layout of `168 context + 24 prediction = 192` steps reduced the sLSTM recurrent loop step count by **12x** (from 74 to 6 steps), dropping iteration latency to sub-second speeds.
- **Joint Multi-Task Fine-Tuning**: Enabled joint training across all three benchmark datasets (`electricity`, `retail`, and `bike`) concurrently using PyTorch's `ConcatDataset`.
- **Autograd Preservation**: Bypassed the model's hardcoded `.detach().cpu()` statement inside `transform_output` by executing the loss calculation directly on the sliced, normalized predictions (`predictions = pred.unsqueeze(1)`).

The joint fine-tuning completed successfully on your Blackwell GPU (NVIDIA RTX PRO 2000) for 5 epochs:
```powershell
python train.py joint
```
```
Starting ForecastAgent 1.0 fine-tuning on dataset 'joint' on device: cuda
trainable params: 2,353,728 || all params: 84,855,472 || trainable%: 2.7738
Loaded joint dataset with 30078 total window samples.
Epoch 1/5 | Train Pinball Loss: 0.30461
Epoch 2/5 | Train Pinball Loss: 0.24747
Epoch 3/5 | Train Pinball Loss: 0.23510
Epoch 4/5 | Train Pinball Loss: 0.22669
Epoch 5/5 | Train Pinball Loss: 0.22017
Fine-tuning completed. LoRA adapter saved to './forecastagent-v1-lora-joint'.
```

---

## 4. API Server Adapter Serving
We updated [api_server.py](file:///c:/Users/wei.liu/Documents/ForecastAgent/api_server.py) to automatically check for `./forecastagent-v1-lora-joint` on startup. If present, it loads the adapter and uses PEFT's `.merge_and_unload()` to fold the LoRA weights directly into the base backbone layers. This provides the custom forecasting accuracy of our fine-tuned weights with **zero runtime serving latency**!

---

## 5. Final Verification & Real Leaderboard
With the local FastAPI server serving TiRex-2 in the background, we ran the benchmarking pipeline:
```
================================================================================
                      FORECASTING BENCHMARK LEADERBOARD
================================================================================
    Dataset         Model       MAPE     sMAPE      RMSE  Inference_Time_Sec
electricity ForecastAgent   7.310531  7.074499  0.752644            0.191910
electricity       TimesFM   9.821286  9.312607  0.990773            0.235276
electricity       Chronos   4.896761  4.784621  0.538022            0.440809
electricity      Baseline  10.957523 10.616339  1.201099            0.000172
     retail ForecastAgent  28.167138 22.182655  4.850062            0.182079
     retail       TimesFM  26.914614 21.228129  4.683194            0.252628
     retail       Chronos  20.078426 16.129705  3.701195            0.135083
     retail      Baseline  25.091369 20.837681  4.233487            0.000140
       bike ForecastAgent  72.010400 55.578984 66.538088            0.200734
       bike       TimesFM 100.342944 50.372610 61.086746            0.305079
       bike       Chronos  59.048243 59.712459 72.121329            0.381063
       bike      Baseline  99.954450 58.925591 64.006185            0.000138
================================================================================
```
The outputs [metrics_summary.csv](file:///c:/Users/wei.liu/Documents/ForecastAgent/metrics_summary.csv) and [timeseries_results.json](file:///c:/Users/wei.liu/Documents/ForecastAgent/timeseries_results.json) feed directly into the Vite React dashboard portal.

---

## 6. Packaging for Deployment
We packaged the repository files into a zip file named [ForecastAgent_Deployment.zip](file:///c:/Users/wei.liu/Documents/ForecastAgent/ForecastAgent_Deployment.zip).
To optimize file size and exclude platform-specific files, the packaging tool automatically excluded:
- `.git/` folder.
- `node_modules/` folder (which should be reinstalled on the destination machine via `npm install`).
- `__pycache__` python compilation caches.
