# =========================================================
# IMPORT LIBRARIES
# =========================================================
#Python Modules
import pandas as pd, json, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tabulate

#Local Modules
from modules.cost_model import calculate_cost, USD_TO_KSH
from modules.cost_tracker import InferenceCost, CostTracker

# =========================================================
# GET CWD
# =========================================================
BASE_DIR = Path(__file__).resolve().parent



# =========================================================
# GET CWD
# =========================================================
FEATURES = {
    "triage_advisor":     {"daily_requests": 8000,  "avg_prompt_toks": 300, "avg_completion_toks": 180},
    "epidemic_alerts":    {"daily_requests": 25000, "avg_prompt_toks": 250, "avg_completion_toks": 80},
    "clinical_summaries": {"daily_requests": 15000, "avg_prompt_toks": 200, "avg_completion_toks": 120},
}

tracker = CostTracker(BASE_DIR/"outputs"/"costs.jsonl")
base = datetime(2026, 6, 1)

for day in range(7):
    for feature, cfg in FEATURES.items():
        for model, share in [("gpt-4o-mini", 0.75), ("gpt-4o", 0.25)]:
            n = int(cfg["daily_requests"] * share * 0.02)  # 2% sample
            for _ in range(n):
                p = max(50, int(np.random.normal(cfg["avg_prompt_toks"], 30)))
                c = max(20, int(np.random.normal(cfg["avg_completion_toks"], 20)))
                cost = calculate_cost(model, p, c)
                tracker.record(InferenceCost(
                    request_id=str(np.random.randint(1000, 9999)),
                    timestamp=(base + timedelta(days=day)).isoformat(),
                    model=model, feature=feature,
                    prompt_tokens=p, completion_tokens=c, total_tokens=p+c,
                    cost_usd=cost, cost_ksh=round(cost * USD_TO_KSH, 4),
                    latency_ms=round(np.random.normal(700 if "mini" in model else 1800, 120), 1),
                    success=np.random.random() > 0.01,
                ))

# Scale 2% sample to full production (SCALE = 50)
SCALE = 50

# Read the recorded costs and summarize by model
with open(f"{BASE_DIR}/outputs/costs.jsonl") as f:
    records = [json.loads(l) for l in f]
df = pd.DataFrame(records)

# Summarize costs by model and calculate projected costs for 7 days.
summary = df.groupby(["model", "feature"]).agg(
    Requests_Audited=("request_id", "count"),
    Sample_Cost_USD =("cost_usd",   "sum"),
).assign(Projected_7d_USD=lambda x: (x.Sample_Cost_USD * SCALE).round(2))

# Calculate projected costs for 30 days based on the 7-day projection.
summary["Projected_30d_USD"] = (summary["Projected_7d_USD"] * 30 / 7).round(2)
print(summary.to_string())

#print the summary in a tabular format for better readability
print(tabulate.tabulate(summary.reset_index(), headers="keys", tablefmt="grid"))

#Save the summary to a CSV file for further analysis or reporting.
summary.to_csv(f"{BASE_DIR}/outputs/cost_summary.csv", index=True)






# =========================================================
# GET CWD
# =========================================================
FEATURES_PROMPTS = {
    "triage_advisor_direct": {"daily_requests": 8000, "avg_prompt_toks": 300, "avg_completion_toks": 180,},
    "epidemic_alerts_direct": {"daily_requests": 25000, "avg_prompt_toks": 250, "avg_completion_toks": 80,},
    "clinical_summaries_direct": {"daily_requests": 15000, "avg_prompt_toks": 200, "avg_completion_toks": 120,},
    "triage_advisor_cot": {"daily_requests": 8000, "avg_prompt_toks": 500, "avg_completion_toks": 350,},
    "epidemic_alerts_cot": {"daily_requests": 25000, "avg_prompt_toks": 420, "avg_completion_toks": 180,},
    "clinical_summaries_cot": {"daily_requests": 15000, "avg_prompt_toks": 380, "avg_completion_toks": 240,}  
}

prompts_tracker = CostTracker(BASE_DIR/"outputs"/"costs_prompt_type.jsonl")

for day in range(7):
    for feature, cfg in FEATURES_PROMPTS.items():
        for model, share in [("gpt-4o-mini", 0.75), ("gpt-4o", 0.25)]:
            n = int(cfg["daily_requests"] * share * 0.02)  # 2% sample
            for _ in range(n):
                p = max(50, int(np.random.normal(cfg["avg_prompt_toks"], 30)))
                c = max(20, int(np.random.normal(cfg["avg_completion_toks"], 20)))
                cost = calculate_cost(model, p, c)
                prompts_tracker.record(InferenceCost(
                    request_id=str(np.random.randint(1000, 9999)),
                    timestamp=(base + timedelta(days=day)).isoformat(),
                    model=model, feature=feature,
                    prompt_tokens=p, completion_tokens=c, total_tokens=p+c,
                    cost_usd=cost, cost_ksh=round(cost * USD_TO_KSH, 4),
                    latency_ms=round(np.random.normal(700 if "mini" in model else 1800, 120), 1),
                    success=np.random.random() > 0.01,
                ))

# Scale 2% sample to full production (SCALE = 50)
SCALE = 50

# Read the recorded costs and summarize by model
with open(f"{BASE_DIR}/outputs/costs_prompt_type.jsonl") as f:
    records = [json.loads(l) for l in f]
df = pd.DataFrame(records)

# Summarize costs by model and calculate projected costs for 7 days.
summary = df.groupby(["model", "feature"]).agg(
    Requests_Audited=("request_id", "count"),
    Sample_Cost_USD =("cost_usd",   "sum"),
).assign(Projected_7d_USD=lambda x: (x.Sample_Cost_USD * SCALE).round(2))

# Calculate projected costs for 30 days based on the 7-day projection.
summary["Projected_30d_USD"] = (summary["Projected_7d_USD"] * 30 / 7).round(2)
print(summary.to_string())

#print the summary in a tabular format for better readability
print(tabulate.tabulate(summary.reset_index(), headers="keys", tablefmt="grid"))

#Save the summary to a CSV file for further analysis or reporting.
summary.to_csv(f"{BASE_DIR}/outputs/cost_summary_prompt_type.csv", index=True)

# =========================================================
# LOWER COST MODEL EVALUATION
# =========================================================
FEATURES_LC_MODEL = {
    "triage_advisor":     {"daily_requests": 8000,  "avg_prompt_toks": 300, "avg_completion_toks": 180},
    "epidemic_alerts":    {"daily_requests": 25000, "avg_prompt_toks": 250, "avg_completion_toks": 80},
    "clinical_summaries": {"daily_requests": 15000, "avg_prompt_toks": 200, "avg_completion_toks": 120},
}

lc_model_tracker = CostTracker(BASE_DIR/"outputs"/"costs_lower_cost_model.jsonl")

for day in range(7):
    for feature, cfg in FEATURES_LC_MODEL.items():
        for model, share in [("gpt-4o-mini", 1)]:
            n = int(cfg["daily_requests"] * share * 0.02)  # 2% sample
            for _ in range(n):
                p = max(50, int(np.random.normal(cfg["avg_prompt_toks"], 30)))
                c = max(20, int(np.random.normal(cfg["avg_completion_toks"], 20)))
                cost = calculate_cost(model, p, c)
                lc_model_tracker.record(InferenceCost(
                    request_id=str(np.random.randint(1000, 9999)),
                    timestamp=(base + timedelta(days=day)).isoformat(),
                    model=model, feature=feature,
                    prompt_tokens=p, completion_tokens=c, total_tokens=p+c,
                    cost_usd=cost, cost_ksh=round(cost * USD_TO_KSH, 4),
                    latency_ms=round(np.random.normal(700 if "mini" in model else 1800, 120), 1),
                    success=np.random.random() > 0.01,
                ))

# Scale 2% sample to full production (SCALE = 50)
SCALE = 50

# Read the recorded costs and summarize by model
with open(f"{BASE_DIR}/outputs/costs_lower_cost_model.jsonl") as f:
    records = [json.loads(l) for l in f]
df = pd.DataFrame(records)

# Summarize costs by model and calculate projected costs for 7 days.
summary = df.groupby(["model", "feature"]).agg(
    Requests_Audited=("request_id", "count"),
    Sample_Cost_USD =("cost_usd",   "sum"),
).assign(Projected_7d_USD=lambda x: (x.Sample_Cost_USD * SCALE).round(2))

# Calculate projected costs for 30 days based on the 7-day projection.
summary["Projected_30d_USD"] = (summary["Projected_7d_USD"] * 30 / 7).round(2)
print(summary.to_string())

#print the summary in a tabular format for better readability
print(tabulate.tabulate(summary.reset_index(), headers="keys", tablefmt="grid"))

#Save the summary to a CSV file for further analysis or reporting.
summary.to_csv(f"{BASE_DIR}/outputs/cost_summary_lower_cost_model.csv", index=True)