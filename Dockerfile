# Use Python 3.11-slim as a base image, since tirex-2 requires Python >= 3.11
FROM python:3.11-slim

# Set environment variables for Python and Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install system dependencies (git is required for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local dependency source first to leverage Docker layer caching
COPY tirex-2 /app/tirex-2
COPY requirements.txt /app/requirements.txt

# Install PyTorch (CPU-only) and application dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download model weights from Hugging Face during build (preventing cold-start latency)
COPY download_model.py /app/download_model.py
RUN python /app/download_model.py && rm /app/download_model.py

# Copy theme configuration and application files
COPY .streamlit /app/.streamlit
COPY app.py data_simulator.py /app/

# Expose Streamlit's default port on Cloud Run
EXPOSE 8080

# Run Streamlit upon container startup
ENTRYPOINT ["streamlit", "run", "app.py"]
