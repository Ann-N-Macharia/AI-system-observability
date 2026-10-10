# =========================================================
# IMPORT LIBRARIES
# =========================================================

#Python Modules:
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv
import openpyxl

#Local Modules (Variables):
from modules.v_evaluation_questions import EVAL_DATASET

#Local Modules (Functions):
from modules.f_query_model import query_model
from modules.f_lexical_metrcs_evaluator import evaluate_response
from modules.f_llm_judge_evaluator import llm_judge
from modules.f_quality_gate_assessor import ModelRequirements, assess_quality_gate

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

# =========================================================
# LIST MODELS TO COMPARE
# =========================================================
MODELS = ["gpt-4o-mini", "gpt-4o"]

# =========================================================
# QUALITY GATE THRESHOLDS
# =========================================================
Requirements = ModelRequirements(
    bleu_4=0.13,
    rouge_l=0.30,
    token_f1=0.40,
    judge_overall=4.5,
)

# =========================================================
# RUN EVALUATION FUNCTION
# =========================================================
def run_full_evaluation():
    results = []
    count = 0
    for example in EVAL_DATASET:
        for model in MODELS:
            hypothesis   = query_model(model, example["question"])
            auto_scores  = evaluate_response(example["reference"], hypothesis)
            judge_scores = llm_judge(example["question"], example["reference"], hypothesis, model)
            results.append({
                "id": example["id"], "model": model,
                **auto_scores,
                "correctness":   judge_scores.get("correctness", 0),
                "groundedness":  judge_scores.get("groundedness", 0),
                "relevance":     judge_scores.get("relevance", 0), 
                "helpfulness":   judge_scores.get("helpfulness", 0),
                "judge_overall": judge_scores.get("overall", 0),
                "reasoning":     judge_scores.get("reasoning", ""),
            })
            count+=1
            print(count)
    
    #Full Output:
    df = pd.DataFrame(results)
    df.to_csv("1.evaluation/outputs/full_evaluation_results.csv", index=False)

    #Summary Output:
    summary = df.groupby("model")[
            ["bleu_4","rouge_l","token_f1","correctness","groundedness","helpfulness","judge_overall"]
        ].mean().round(3).reset_index()
    summary.to_csv("1.evaluation/outputs/summary_evaluation_results.csv", index=False)

    # Quality Gate Assessment
    gate_results = assess_quality_gate(summary.reset_index(), Requirements)
    gate_df = pd.DataFrame(gate_results)
    gate_df.to_csv("1.evaluation/outputs/quality_gate_results.csv", index=False)
    print(tabulate(gate_df, headers='keys', tablefmt='grid', showindex=False))

    return gate_df

if __name__ == "__main__":

    #Run Evaluation
    eval_summary = run_full_evaluation()
    print(tabulate(eval_summary, headers='keys', tablefmt='grid', floatfmt=".3f"))
