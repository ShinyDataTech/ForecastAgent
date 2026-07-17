import os
import argparse
import logging
import uvicorn

from forecastagent.api import create_app

def main():
    parser = argparse.ArgumentParser(description="ForecastAgent 1.0 SaaS API Server CLI")
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default="./forecastagent-v1-standalone" if os.path.exists("./forecastagent-v1-standalone") else "shinydatatech/forecastagent-v1.0",
        help="Local directory containing model checkpoints or Hugging Face repo ID"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Network interface to bind the server to"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port number to listen on"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        help="Inference execution device ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=os.getenv("FORECASTAGENT_API_KEY"),
        help="API Key for bearer token authentication. Defaults to FORECASTAGENT_API_KEY env var."
    )

    args = parser.parse_args()

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    # Build and launch application
    app = create_app(
        model_path_or_repo=args.model_path,
        device=args.device,
        api_key=args.api_key
    )

    print(f"Starting ForecastAgent API server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
