# =========================================================
# IMPORT LIBRARIES
# =========================================================
# Python Modules
import tabulate
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from pathlib import Path
import matplotlib.pyplot as plt

#Local Modules
from modules.v_simulate_production_data import reference_data, production_data
from modules.drift_analysis import run_drift_analysis


# =========================================================
# GET CWD
# =========================================================
BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# SUMMARISE DATA STATS
# =========================================================
#Create empty dataframe to store drift trend summary
drift_trend_table = pd.DataFrame()


#loop through each month and compute mean metrics for rouge_l, latency_ms, and token_count
for month, data in production_data.items():
    data["month"] = month
    summary = data.groupby("month")[
            ["rouge_l", "latency_ms", "token_count"]
        ].mean().round(3)
    drift_trend_table = pd.concat([drift_trend_table, summary], axis=0)

# Save drift trend summary to CSV
drift_trend_table = (
    drift_trend_table.T.rename_axis("measure").reset_index()
)

drift_trend_table.to_csv(f"{BASE_DIR}/outputs/drift_trend_table.csv", index=True)

#Trend plot:

months = ["month_1", "month_2", "month_3"]

for _, row in drift_trend_table.iterrows():
    plt.figure(figsize=(6, 4))

    plt.plot(
        months,
        row[months].astype(float),
        marker="o",
        linewidth=2
    )

    plt.title(f"{row['measure']} Trend")
    plt.xlabel("Month")
    plt.ylabel(row["measure"])
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"{BASE_DIR}/outputs/drift_trend.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()  # Closes the figure to free memory

print(f"Saved plots to: {BASE_DIR}/outputs/")


# =========================================================
# RUN DRIFT ANALYSIS
# =========================================================
for month, data in production_data.items():
    report, alert_log = run_drift_analysis(
        metrics=[
            DatasetDriftMetric(),
            DataDriftPreset(),
            ColumnDriftMetric(column_name="rouge_l"),
            ColumnDriftMetric(column_name="latency_ms"),
            ColumnDriftMetric(column_name="token_count")
        ],
        baseline_df=reference_data,
        production_df=data,
        report_name=month
    )  

    report.save_html(f"{BASE_DIR}/outputs/{month}-drift_report.html")
    print(f"Drift report for {month} saved as {month}-drift_report.html")

    print(f"Alert Summary for {month}:")
    print(tabulate.tabulate([alert_log], headers="keys", tablefmt="grid"))