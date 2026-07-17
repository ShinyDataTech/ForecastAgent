import numpy as np
import pandas as pd
import torch
import datetime
from typing import Dict, Any, Tuple

class DataSimulator:
    """
    DataSimulator generates highly realistic synthetic multivariate time-series data
    for 8 specific industry domains, incorporating trends, seasonality, noise, 
    and dynamic future-known covariates.
    """
    
    DOMAINS = {
        "infrastructure": {
            "name": "Infrastructure & Asset Management",
            "use_case": "Pothole Volumetric Expansion & Washout",
            "target_name": "Total Pothole Volume (Liters)",
            "covariate_name": "Heavy Rainfall (mm)",
            "desc": "Simulates pothole expansion. Water pooling in asphalt cracks washes out aggregate under traffic load, causing sharp volumetric jumps during heavy rain. Helps road crews preemptively schedule patching crews.",
            "baseline": 5.0,
            "trend": 0.05,  # Slow upward degradation
            "noise": 0.2
        },
        "transportation": {
            "name": "Transportation & Traffic",
            "use_case": "Regional Commuter Traffic Flow",
            "target_name": "Vehicles per Hour",
            "covariate_name": "Public Holiday (0/1)",
            "desc": "Simulates traffic patterns on regional road networks. Strong daily/weekly seasonality. Traffic drops significantly on public holidays.",
            "baseline": 1200.0,
            "trend": 0.2,   # Slight growth in volume
            "noise": 50.0
        },
        "retail": {
            "name": "Business, Retail & E-commerce",
            "use_case": "E-commerce Demand & Marketing ROI",
            "target_name": "Sales Volume (units)",
            "covariate_name": "Active Ad Campaign (0/1)",
            "desc": "Simulates daily e-commerce order volumes. Shows weekly cycles, seasonal spikes, and huge sales spikes during active marketing campaigns.",
            "baseline": 450.0,
            "trend": 0.15,
            "noise": 25.0
        },
        "education": {
            "name": "Education",
            "use_case": "Cyclical Enrollment Patterns & Learning Modalities",
            "target_name": "Active Enrollment Count",
            "covariate_name": "Semester Intake Window (0/1)",
            "desc": "Simulates cyclical university enrollments. In-person and online course modalities show distinct patterns during the semester start window.",
            "baseline": 8500.0,
            "trend": 1.5,
            "noise": 100.0
        },
        "weather": {
            "name": "Weather & Environment",
            "use_case": "Climate Tracking & Disaster Probability Index",
            "target_name": "Disaster Risk Index (0-100)",
            "covariate_name": "Extreme Wind Event Warning (0/1)",
            "desc": "Simulates seasonal climate fluctuations and extreme weather events. Strong annual seasonality with wind warnings acting as immediate risk spikes.",
            "baseline": 10.0,
            "trend": 0.02,
            "noise": 2.0
        },
        "epidemiology": {
            "name": "Disease & Macro Trends",
            "use_case": "Epidemiological Transmission Rates (R-value)",
            "target_name": "Active Infection Count",
            "covariate_name": "Lockdown / Health Policy (0/1)",
            "desc": "Simulates seasonal virus transmission waves. Dynamic health policies or lockdown announcements serve to suppress transmission and flatten the curve.",
            "baseline": 150.0,
            "trend": 0.5,
            "noise": 15.0
        }
    }

    @classmethod
    def get_domains(cls) -> Dict[str, Any]:
        return cls.DOMAINS

    def generate(
        self, 
        domain_id: str, 
        context_length: int = 200, 
        prediction_length: int = 48
    ) -> Dict[str, Any]:
        """
        Generates history and future time series with covariates.
        
        Parameters:
        -----------
        domain_id: str
            Key for the domain.
        context_length: int
            Number of historical steps.
        prediction_length: int
            Number of future forecasting steps.
            
        Returns:
        --------
        Dict containing DataFrames and pre-formatted torch tensors for TiRex-2 inference.
        """
        if domain_id not in self.DOMAINS:
            raise ValueError(f"Unknown domain: {domain_id}")
            
        cfg = self.DOMAINS[domain_id]
        total_length = context_length + prediction_length
        
        # Start date
        start_date = datetime.datetime(2026, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(total_length)]
        
        # Base components
        t = np.arange(total_length)
        noise = np.random.normal(0, cfg["noise"], total_length)
        trend = t * cfg["trend"]
        
        # Initialize target and covariates
        target = np.zeros(total_length)
        covariate = np.zeros(total_length)
        
        # 1. Infrastructure
        if domain_id == "infrastructure":
            # Slow degradation, seasonal weathering (yearly)
            seasonality = 1.0 * np.sin(2 * np.pi * t / 365)
            # Covariate: Rainfall events. Random heavy rain (e.g. 5-15% chance)
            np.random.seed(42)
            rain_days = np.random.choice([0, 1], size=total_length, p=[0.92, 0.08])
            rain_amount = rain_days * np.random.uniform(10.0, 55.0, size=total_length)
            covariate = rain_amount
            
            # Target is pothole volume. Rain events trigger step increases in deterioration due to washout
            current_defect = cfg["baseline"]
            for i in range(total_length):
                # Traffic wear + washout step increases
                washout = np.random.uniform(2.0, 6.0) * (rain_amount[i] / 10.0) if rain_amount[i] > 0 else 0.0
                current_defect += cfg["trend"] + washout
                # Seasonal fluctuation
                target[i] = current_defect + seasonality[i] + noise[i]
                
        # 2. Transportation
        elif domain_id == "transportation":
            # Traffic hourly/daily representation
            # Let's say t represents days, so we have weekly seasonality
            weekly_season = 300 * np.sin(2 * np.pi * t / 7)
            # Add weekend drop (e.g. step drop for Saturday/Sunday)
            weekend_mask = np.array([(d.weekday() >= 5) for d in dates], dtype=float)
            weekly_season -= weekend_mask * 400
            
            # Covariate: Public holidays
            holiday_days = np.zeros(total_length)
            # Place holidays approximately every 30 days
            for h in range(15, total_length, 30):
                holiday_days[h] = 1.0
            covariate = holiday_days
            
            # Traffic flow target
            for i in range(total_length):
                val = cfg["baseline"] + trend[i] + weekly_season[i] + noise[i]
                if covariate[i] == 1.0:
                    val *= 0.5  # Holiday traffic drops by 50%
                target[i] = max(100.0, val)

        # 3. Retail
        elif domain_id == "retail":
            # Weekly seasonality
            weekly_season = 50 * np.sin(2 * np.pi * t / 7)
            # Annual shopping spikes (around index 120 and 280)
            annual_season = 150 * np.exp(-((t % 365 - 120) / 15) ** 2) + 250 * np.exp(-((t % 365 - 320) / 10) ** 2)
            
            # Covariate: Ad campaigns (active for 5-day intervals)
            np.random.seed(123)
            campaigns = np.zeros(total_length)
            campaign_starts = np.random.choice(range(total_length - 5), size=max(1, int(total_length / 45)), replace=False)
            for start in campaign_starts:
                campaigns[start:start+5] = 1.0
            covariate = campaigns
            
            # Sales target
            for i in range(total_length):
                val = cfg["baseline"] + trend[i] + weekly_season[i] + annual_season[i] + noise[i]
                if covariate[i] == 1.0:
                    val += np.random.uniform(150.0, 300.0)  # Huge campaign spike
                target[i] = max(10.0, val)

        # 6. Education
        elif domain_id == "education":
            # Semester intake starts spike target
            intakes = np.zeros(total_length)
            # Semesters start every 120 days (approx)
            for s in range(10, total_length, 120):
                intakes[s:s+15] = 1.0  # 15 days enrollment window
            covariate = intakes
            
            # Base enrollment + intake spikes
            for i in range(total_length):
                spike = 1500.0 * np.sin(np.pi * (i % 120) / 15) if (i % 120) < 15 else 0.0
                target[i] = cfg["baseline"] + trend[i] + spike + noise[i]

        # 7. Weather
        elif domain_id == "weather":
            # Extreme wind alerts
            wind_alerts = np.zeros(total_length)
            np.random.seed(99)
            alert_days = np.random.choice(range(total_length), size=max(1, int(total_length / 25)), replace=False)
            wind_alerts[alert_days] = 1.0
            covariate = wind_alerts
            
            # Risk Index. Seasonal high in winter/summer, spikes during wind alerts
            seasonality = 15.0 * np.sin(2 * np.pi * t / 365)
            for i in range(total_length):
                val = cfg["baseline"] + seasonality[i] + noise[i]
                if wind_alerts[i] == 1.0:
                    val += np.random.uniform(30.0, 55.0)  # Huge risk spike
                target[i] = clip(val, 0, 100)

        # 8. Epidemiology
        elif domain_id == "epidemiology":
            # Epidemic waves (100 day cycle)
            wave = 200 * np.sin(2 * np.pi * t / 100)
            
            # Covariates: Lockdown / Health Policies
            lockdown = np.zeros(total_length)
            # Active lockdown between 40-70, 140-170, etc.
            for start in range(40, total_length, 100):
                end = min(start + 30, total_length)
                lockdown[start:end] = 1.0
            covariate = lockdown
            
            # Infections spike, lockdowns suppress it
            curr_cases = cfg["baseline"]
            for i in range(total_length):
                growth = 3.0 * np.sin(2 * np.pi * i / 100)
                if lockdown[i] == 1.0:
                    growth -= 6.0  # Lockdowns compress/flatten transmission
                curr_cases = max(10.0, curr_cases + growth + np.random.normal(0, 2.0))
                target[i] = curr_cases + noise[i]

        # Slice data into history and future
        history_dates = dates[:context_length]
        future_dates = dates[context_length:]
        
        history_target = target[:context_length]
        future_target = target[context_length:]
        
        history_cov = covariate[:context_length]
        future_cov = covariate[context_length:]
        
        # Prepare DataFrames
        df_history = pd.DataFrame({
            "target": history_target,
            "covariate": history_cov
        }, index=history_dates)
        df_history.index.name = "date"
        
        df_future = pd.DataFrame({
            "target": future_target,
            "covariate": future_cov
        }, index=future_dates)
        df_future.index.name = "date"
        
        df_all = pd.concat([df_history, df_future])
        
        # Pre-format torch tensors for TiRex-2 integration
        # target tensor shape: [V_t, T] -> [1, context_length]
        target_tensor = torch.tensor(history_target, dtype=torch.float32).unsqueeze(0)
        
        # past_covariates: [V_p, T] -> [1, context_length] (if we treat the covariate as past as well)
        past_cov_tensor = torch.tensor(history_cov, dtype=torch.float32).unsqueeze(0)
        
        # future_covariates: [V_f, T + H] -> [1, total_length]
        # In TiRex-2, future covariates include BOTH history and future forecast steps.
        # This is a total size of T + H.
        future_cov_tensor = torch.tensor(covariate, dtype=torch.float32).unsqueeze(0)
        
        return {
            "df_history": df_history,
            "df_future": df_future,
            "df_all": df_all,
            "target_tensor": target_tensor,
            "past_cov_tensor": past_cov_tensor,
            "future_cov_tensor": future_cov_tensor,
            "meta": cfg
        }

def clip(val, min_val, max_val):
    return max(min_val, min(val, max_val))
