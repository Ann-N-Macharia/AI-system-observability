# =========================================================
# IMPORT LIBRARIES
# =========================================================
#Python Modules
from dataclasses import dataclass
from typing import List
import pandas as pd
import tabulate


#create dataclass for model candidates and task requirements
@dataclass
class ModelCandidate:
    name: str
    quality_score: float
    avg_latency_ms: float
    cost_per_1m_input_req: float
    cost_per_1m_output_req: float
    hallucination_rate: float

@dataclass
class TaskRequirements:
    task_name: str
    min_quality_score: float
    max_latency_ms: float
    max_cost_per_1m_input_req: float
    max_cost_per_1m_output_req: float
    max_hallucination_rate: float
    risk_level: str
    daily_volume: int

def evaluate_candidates(candidates: List[ModelCandidate], req: TaskRequirements) -> pd.DataFrame:
    requirements = []
    for m in candidates:
        meets_quality = m.quality_score      >= req.min_quality_score
        meets_latency = m.avg_latency_ms     <= req.max_latency_ms
        meets_input_cost    = m.cost_per_1m_input_req    <= req.max_cost_per_1m_input_req
        meets_output_cost   = m.cost_per_1m_output_req   <= req.max_cost_per_1m_output_req
        meets_halluc  = m.hallucination_rate <= req.max_hallucination_rate
        monthly_input      = (m.cost_per_1m_input_req / 1000000) * req.daily_volume * 30
        monthly_output     = (m.cost_per_1m_output_req / 1000000) * req.daily_volume * 30

        passes = all([meets_quality, meets_latency, meets_input_cost, meets_output_cost, meets_halluc])

        requierements = []   
        data = {
            "model": m.name,

            #quality
            "quality_actual": m.quality_score,
            "quality_threshold": f"Above {req.min_quality_score}",
            "quality_status": "Pass" if meets_quality else "Fail",

            #latency
            "latency_actual": m.avg_latency_ms,
            "latency_threshold": f"Below {req.max_latency_ms} ms",
            "latency_status": "Pass" if meets_latency else "Fail",

            #hallucination
            "hallucination_actual": m.hallucination_rate,
            "hallucination_threshold": f"Below {req.max_hallucination_rate}",   
            "hallucination_status": "Pass" if meets_halluc else "Fail",

            #input cost
            "input_cost_actual": m.cost_per_1m_input_req,
            "input_cost_threshold": f"Below ${req.max_cost_per_1m_input_req} per 1M input req",
            "input_cost_status": "Pass" if meets_input_cost else "Fail",

            #output cost
            "output_cost_actual": m.cost_per_1m_output_req,
            "output_cost_threshold": f"Below ${req.max_cost_per_1m_output_req} per 1M output req",
            "output_cost_status": "Pass" if meets_output_cost else "Fail",

            # "monthly_usd": f"${monthly_input + monthly_output:.2f}",
            # "eligible":    "ELIGIBLE" if passes else "FAILS"
        }

        dimensions = ["Quality", "Latency", "Hallucination", "Input_Cost", "Output_Cost"]

        for dim in dimensions:
            requirements.append({
            "model": data["model"],
            "dimension": dim.lower(),
            "actual_score": data[f"{dim.lower()}_actual"] if f"{dim.lower()}_actual" in data else data[f"{dim.lower()}_actual"],
            "requirement_score": data[f"{dim.lower()}_threshold"],
            "status": data[f"{dim.lower()}_status"].lower(),
        })

    return pd.DataFrame(requirements)


#Run the evaluation with candidates and requirements
#Candidates.
candidates = [
    ModelCandidate("gpt-4o",      0.95, 500, 2.5, 10.0, 0.04),
    ModelCandidate("gpt-4o-mini", 0.88, 600, 0.15, 0.6, 0.07),
]

#Model Requirements.
model_reqs = TaskRequirements(task_name="FAQ_Bot",
                              min_quality_score=0.8,
                               max_latency_ms=700,
                                max_cost_per_1m_input_req=3.0,
                                 max_cost_per_1m_output_req=7.0,
                                  max_hallucination_rate=0.08,
                                   risk_level='low',
                                    daily_volume=2000)

df = evaluate_candidates(candidates, model_reqs)
print(tabulate.tabulate(df, headers="keys", tablefmt="grid"))

#save to csv
df.to_csv("3.cost/outputs/model_requirements.csv", index=False)