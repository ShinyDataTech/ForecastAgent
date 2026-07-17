import os
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient

import forecastagent
from forecastagent.agent import ForecastAgent
from forecastagent.api import create_app

class TestForecastAgent(unittest.TestCase):
    def setUp(self):
        self.local_model_path = "./forecastagent-v1-standalone"
        self.has_local_model = os.path.exists(self.local_model_path)

    def test_version_and_imports(self):
        """Verify the package exports the expected interfaces and has a version."""
        self.assertIsNotNone(forecastagent.__version__)
        self.assertTrue(hasattr(forecastagent, "ForecastAgent"))
        self.assertTrue(hasattr(forecastagent, "TimeseriesType"))

    def test_agent_prediction_local(self):
        """If local model exists, verify loading and inference works."""
        if not self.has_local_model:
            self.skipTest("Local standalone model weights not found. Skipping inference test.")

        print("\nLoading local standalone model for testing...")
        agent = ForecastAgent.from_pretrained(self.local_model_path, device="cpu")
        self.assertIsNotNone(agent.model)

        # Test prediction with simple target
        target = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        prediction_length = 3
        
        print("Running prediction...")
        res = agent.predict(
            target=target,
            prediction_length=prediction_length,
            freq="h"
        )
        
        # Verify response structure
        self.assertIn("median", res)
        self.assertIn("lower", res)
        self.assertIn("upper", res)
        self.assertIn("quantiles", res)

        self.assertEqual(len(res["median"]), prediction_length)
        self.assertEqual(len(res["lower"]), prediction_length)
        self.assertEqual(len(res["upper"]), prediction_length)
        self.assertEqual(len(res["quantiles"]), 9) # 9 quantiles
        self.assertEqual(len(res["quantiles"][0]), prediction_length)

    def test_api_server_endpoints(self):
        """Test API router endpoints with client mock."""
        # Create an app in dry-run/simulation mode or using base mock
        # We can bypass model initialization by loading a dummy app or skipping if model doesn't exist
        if not self.has_local_model:
            self.skipTest("Local standalone model weights not found. Skipping API server test.")
            
        app = create_app(model_path_or_repo=self.local_model_path, device="cpu", api_key="test-key")
        client = TestClient(app)
        
        # Test health check root
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "ForecastAgent 1.0")

        # Test auth failure
        response = client.post(
            "/v1/predict",
            json={"instances": [{"target": [1.0, 2.0], "start": "2026-07-01", "freq": "h"}], "prediction_length": 3}
        )
        self.assertEqual(response.status_code, 401)

        # Test predictions with API key auth
        headers = {"Authorization": "Bearer test-key"}
        response = client.post(
            "/v1/predict",
            headers=headers,
            json={
                "instances": [
                    {
                        "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                        "start": "2026-07-01T00:00:00",
                        "freq": "h"
                    }
                ],
                "prediction_length": 3
            }
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertIn("predictions", res_data)
        self.assertEqual(len(res_data["predictions"]), 1)
        self.assertEqual(len(res_data["predictions"][0]["median"]), 3)

if __name__ == "__main__":
    unittest.main()
