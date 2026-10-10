# =========================================================
# IMPORT LIBRARIES
# =========================================================
#Python Modules
from evidently.report import Report
import pandas as pd
from datetime import datetime



def run_drift_analysis(metrics:list, baseline_df:pd.DataFrame, production_df:pd.DataFrame, report_name:str):
    #running_report:
    report = Report(metrics=metrics)
    report.run(reference_data=baseline_df, current_data=production_df)
    print(f"Saving drift report for {report_name}...")

    #Collecting alerts
    raw = report.as_dict()
    alerts = []

    for metric in raw["metrics"]:
        r = metric.get("result", {})
        if r.get("drift_detected"):
            alerts.append(
                {
                "column": r.get("column_name"), 
                "p_value": r.get("p_value")
                 }
                )
            
    monthly_alert_log =  {
        "month": report_name,
        "timestamp":    datetime.now().isoformat(),
        "drift_alerts": alerts,
        "alert_count":  len(alerts),
        "needs_action": len(alerts) > 0,
    }

    return report, monthly_alert_log


# =========================================================
#TESTING
# =========================================================
if __name__ == "__main__":
    #testing python modules
    import tabulate
    import pandas as pd
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric


    #testing Local Modules
    from v_simulate_production_data import reference_data, production_data


    for month, data in production_data.items():
        report, alerts = run_drift_analysis(
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

        print(report)
        print(alerts)



# # Targeted column-level: pinpoints specific degrading metrics
# targeted = Report(metrics=[
#     ColumnDriftMetric(column_name="input_length"),
#     ColumnDriftMetric(column_name="rouge_l"),
#     ColumnDriftMetric(column_name="latency_ms"),
#     ColumnDriftMetric(column_name="token_count"),
# ])
# targeted.run(reference_data=reference_df, current_data=production_df)
# targeted.save_html("drift_report_targeted.html")

# # Machine-readable results for alerting
# results = targeted.as_dict()
# for m in results["metrics"]:
#     r = m.get("result", {})
#     if "drift_detected" in r:
#         col     = r.get("column_name", "unknown")
#         drifted = r["drift_detected"]
#         pval    = r.get("p_value", "N/A")
#         status  = "DRIFT DETECTED" if drifted else "No drift"
#         print(f"  {col:<20}: {status}  (p={pval})")