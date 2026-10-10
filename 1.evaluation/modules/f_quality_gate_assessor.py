# =========================
# IMPORT LIBRARIES
# =========================

#Python Modules:
from dataclasses import dataclass
import pandas as pd

#Create class which will be used to create an instance of scores.
@dataclass
class ModelRequirements:
    bleu_4: float
    rouge_l: float
    token_f1: float
    judge_overall: float


# =================================
# CREATE QUALITY GATE ASSESSOR
# =================================
# def assess_quality_gate(df: pd.DataFrame, requirements: ModelRequirements) -> list:
#     """Flags each model as eligible only if it meets every threshold in requirements."""
#     results = []
#     for _, row in df.iterrows():
#         eligible = all([
#             row["bleu_4"] >= requirements.bleu_4,
#             row["rouge_l"] >= requirements.rouge_l,
#             row["token_f1"] >= requirements.token_f1,
#             row["judge_overall"] >= requirements.judge_overall,
#         ])
#         results.append({
#             "model": row["model"],
#             "eligible": eligible,
#         })
#     return results


def assess_quality_gate(df: pd.DataFrame, requirements: ModelRequirements) -> list:
    """Flags each model as eligible only if it meets every threshold in requirements."""
    results = []
    for _, row in df.iterrows():
        checks = {
            "model": row["model"],

            #bleu4
            "bleu_4_score": row["bleu_4"],
            "bleu_4_threshold": requirements.bleu_4,
            "bleu_4_status": "Pass" if row["bleu_4"] >= requirements.bleu_4 else "Fail",

            #rouge_l
            "rouge_l_score": row["rouge_l"],
            "rouge_l_threshold": requirements.rouge_l,
            "rouge_l_status": "Pass" if row["rouge_l"] >= requirements.rouge_l else "Fail",

            #token_f1
            "token_f1_score": row["token_f1"],  
            "token_f1_threshold": requirements.token_f1,
            "token_f1_status": "Pass" if row["token_f1"] >= requirements.token_f1 else "Fail",

            #judge_overall
            "judge_overall_score": row["judge_overall"],
            "judge_overall_threshold": requirements.judge_overall,
            "judge_overall_status": "Pass" if row["judge_overall"] >= requirements.judge_overall else "Fail",
        }

        dimensions = ["bleu_4", "rouge_l", "token_f1", "judge_overall"]
        
        for dim in dimensions:
            results.append({
            "model": checks["model"],
            "dimension": dim.lower(),
            "actual_score": checks[f"{dim.lower()}_score"],
            "requirement_score": checks[f"{dim.lower()}_threshold"],
            "status": checks[f"{dim.lower()}_status"].lower(),
        })
    return results

