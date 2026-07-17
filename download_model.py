try:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='NX-AI/TiRex-2', allow_patterns=['model-config.yaml', 'model.ckpt'])
    print('Model weights pre-downloaded successfully.')
except Exception as e:
    print(f'Warning: Could not pre-download model weights: {e}. Build will continue.')
