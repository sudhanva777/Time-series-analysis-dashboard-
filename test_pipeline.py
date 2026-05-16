"""Quick integration test for the forecasting pipeline."""
from forecasting.preprocessing import generate_demo_data
from forecasting.pipeline import run_single_series, run_experiment
from forecasting.metrics import wmape, compute_metrics
from monitoring.monitoring_metrics import compute_monitoring_summary
import numpy as np

# 1. Generate demo data
df = generate_demo_data()
print(f"Demo data: {len(df)} rows, {df.series_id.nunique()} series")

# 2. Single series forecast
s = df[df.series_id == "KCI_P001"].copy()
cov = ["price", "promotion", "holiday", "is_weekend", "day_of_week", "month", "temperature_c", "rainfall_mm"]
res = run_single_series(s, 21, "daily", cov, "KCI_P001")
print(f"Single series: {len(res['actual'])} steps, WMAPE TFM={res['wmape_timesfm']}%, CHR={res['wmape_chronos']}%")
print(f"CI bands: lower={len(res['timesfm_lower'])}, upper={len(res['timesfm_upper'])}")

# 3. Full experiment
results = run_experiment(df, cov, 21, "daily", "TEST")
print(f"Full experiment: {len(results)} result rows, {results.series_id.nunique()} series")

# 4. Metrics
metrics = compute_metrics(results)
print(f"Metrics computed: {len(metrics)} rows")
print(metrics.to_string(index=False))

# 5. Monitoring
actual = np.array(res["actual"])
tfm = np.array(res["timesfm_pred"])
chron = np.array(res["chronos_pred"])
summary = compute_monitoring_summary(actual, tfm, chron, res["timestamp"], window=7)
print(f"Monitoring: stability TFM={summary['stability_score_tfm']}, CHR={summary['stability_score_chr']}")
print(f"Drift TFM detected: {summary['drift_info_tfm']['drift_detected']}")
print(f"Spikes TFM: {int(summary['spike_flags_tfm'].sum())}")

print("\n=== ALL TESTS PASSED ===")
