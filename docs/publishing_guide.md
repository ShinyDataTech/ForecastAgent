# ForecastAgent 1.0 Implementation & Publishing Guide

This guide provides step-by-step instructions for:
1. **Publishing the ForecastAgent 1.0 Model** (weights and configurations) to Hugging Face.
2. **Developing, Building, and Publishing the `forecast-agent-sdk` Python Package** to PyPI.
3. **Setting up CI/CD workflows** to automate these processes.

---

## Part 1: Publishing ForecastAgent 1.0 to Hugging Face

ForecastAgent 1.0 is built on top of the TiRex-2 architecture, fine-tuned on target datasets using LoRA adapters, and exported into a single merged checkpoint. 

### 1.1 Prerequisites
You need the Hugging Face Hub library installed and an active Hugging Face account:
```bash
pip install huggingface_hub
```

### 1.2 Generate an Access Token
1. Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens).
2. Click **New token**.
3. Set the name (e.g., `ForecastAgent-Publish`) and select the **Write** role.
4. Copy the generated token.

### 1.3 Authenticate via Command Line
Run the login command in your terminal and enter your token when prompted:
```bash
huggingface-cli login
```

### 1.4 Uploading via CLI or Python Script
Your merged model checkpoints are stored locally in the directory:
`./forecastagent-v1-standalone`
It contains two crucial files:
- `model.ckpt`: The merged PyTorch weight dictionary (~330MB).
- `model-config.yaml`: The configuration file.

#### Option A: Command Line Upload
Create the repository and upload the folder contents:
```bash
# Create the model repository on Hugging Face
huggingface-cli repo create shinydatatech/forecastagent-v1.0 --type model

# Upload the standalone directory contents to the repository
huggingface-cli upload shinydatatech/forecastagent-v1.0 ./forecastagent-v1-standalone/ .
```

#### Option B: Python Script Upload (Recommended)
You can automate the repository creation and folder upload with a Python script:
```python
from huggingface_hub import HfApi, create_repo

repo_id = "shinydatatech/forecastagent-v1.0"
local_folder = "./forecastagent-v1-standalone"

# 1. Create remote repository on Hugging Face
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Hugging Face repository '{repo_id}' created or already exists.")
except Exception as e:
    print(f"Error creating repository: {e}")

# 2. Upload files
api = HfApi()
print("Uploading model weights to Hugging Face...")
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model"
)
print("SUCCESS: ForecastAgent 1.0 weights successfully uploaded to Hugging Face!")
```

### 1.5 Create a Model Card (`README.md`)
Add a metadata header to the top of a `README.md` file in the Hugging Face repository to document it properly:
```yaml
---
license: apache-2.0
library_name: PyTorch
tags:
- time-series
- zero-shot-forecasting
- xlstm
- forecasting
pipeline_tag: time-series-forecasting
---
# ForecastAgent 1.0

ForecastAgent 1.0 is a zero-shot and fine-tuned time series forecasting foundation model powered by the xLSTM backbone architecture.

## Usage
You can load and use this model directly using the `forecastagent` Python SDK:

```python
from forecastagent import ForecastAgent

# Download and initialize from Hugging Face
agent = ForecastAgent.from_pretrained("shinydatatech/forecastagent-v1.0")

# Perform zero-shot prediction
results = agent.predict(
    target=[12.5, 14.2, 13.9, 15.1, 16.0, 15.5],
    prediction_length=3,
    freq="h"
)
print("Median forecast:", results["median"])
print("Uncertainty bounds (10th/90th):", results["lower"], results["upper"])
\```
```

---

## Part 2: Developing and Publishing the Python Package

We package the SDK code as a modern Python distribution using `pyproject.toml`.

### 2.1 Package Layout
The codebase is structured as follows:
```
forecast-agent-sdk/
├── pyproject.toml              # Build & dependency metadata
├── README.md                   # Core documentation
├── LICENSE                     # Software license (Apache-2.0)
├── src/
│   └── forecastagent/
│       ├── __init__.py         # Windows patches & imports
│       ├── agent.py            # High-level SDK class
│       ├── api.py              # FastAPI application server code
│       ├── cli.py              # Command-line entry points
│       └── modeling/           # Vendored TiRex-2 core architecture code
│           ├── __init__.py
│           ├── base.py
│           ├── api_adapter/
│           └── model/
├── tests/
│   └── test_agent.py           # Basic test coverage
└── scripts/
    └── create_github_repo.py   # GitHub creation script
```

### 2.2 Local Installation & Testing
To install the package locally in editable mode for testing:
```bash
pip install -e .
```
Verify the installation by running the CLI interface:
```bash
forecastagent-api --help
```

### 2.3 Build the Distribution Archives
Install the build frontend:
```bash
pip install build twine
```
Build the source distribution (`.tar.gz`) and wheel (`.whl`):
```bash
python -m build
```
This generates build artifacts under the `dist/` directory.

### 2.4 Publish to PyPI (Python Package Index)
Upload the built packages using Twine:
```bash
# Verify the package description and syntax
twine check dist/*

# Upload to PyPI (will prompt for your PyPI API token)
twine upload dist/*
```
> **Tip:** You can register at [PyPI](https://pypi.org/) and create an API token under **Account Settings** for secure uploads.

---

## Part 3: GitHub CI/CD Automation

To automatically build and publish your Python package when a new release tag is pushed to GitHub, add a GitHub workflow at `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI and Hugging Face

on:
  release:
    types: [published]

jobs:
  publish-pypi:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-size: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build Package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```
Ensure you add the secret `PYPI_API_TOKEN` under your GitHub Repository Settings > Secrets and Variables > Actions.
