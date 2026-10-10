import pandas as pd
import numpy as np
import random

random.seed(42); np.random.seed(42)

#Reference dataset: larger sample of the baseline distribution, used as the fixed comparison point for drift tests against each production month.
reference_data = []
for i in range(200):
    reference_data.append({
        "rouge_l":    round(np.random.normal(0.42, 0.05), 4),
        "latency_ms": round(np.random.normal(800, 100), 0),
        "token_count": random.randint(150, 300),
    })

#Month 1: baseline (reference) distribution - no drift.
month_1_data = []
for i in range(200):
    month_1_data .append({
        "rouge_l":    round(np.random.normal(0.42, 0.05), 4),
        "latency_ms": round(np.random.normal(800, 100), 0),
        "token_count": random.randint(150, 300),
    })

#Month 2: mild drift - quality/latency start shifting from baseline.
month_2_data = []
for i in range(200):
    month_2_data.append({
        "rouge_l":    round(np.random.normal(0.37, 0.06), 4),
        "latency_ms": round(np.random.normal(950, 150), 0),
        "token_count": random.randint(180, 380),
    })

#Month 3: severe drift - quality has degraded further and latency/token_count are well outside baseline range.
month_3_data = []
for i in range(200):
    month_3_data.append({
        "rouge_l":    round(np.random.normal(0.29, 0.09), 4),
        "latency_ms": round(np.random.normal(1300, 250), 0),
        "token_count": random.randint(250, 500),
    })

reference_data = pd.DataFrame(reference_data)
production_data = {"month_1": pd.DataFrame(month_1_data),
                    "month_2": pd.DataFrame(month_2_data),
                    "month_3": pd.DataFrame(month_3_data)}
