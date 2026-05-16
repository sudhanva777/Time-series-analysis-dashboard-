"""
Time Series Forecasting Dashboard
Streamlit frontend | TimesFM + Chronos backend | WMAPE / MAPE metrics
Enhanced: validation, filtering, CI bands, exports, monitoring alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, io, datetime

from forecasting.preprocessing import (
    smart_load, aggregate_to_series, validate_dataframe,
    generate_demo_data, get_dataset_summary,
    COVARIATE_COLS_DAILY, COVARIATE_COLS_HOURLY,
)
from forecasting.pipeline import run_single_series, run_experiment
from forecasting.metrics  import compute_metrics, compute_metrics_extended, wmape, mape

from monitoring.drift             import detect_drift, rolling_wmape, detect_spikes
from monitoring.alerts            import generate_alerts, alerts_to_streamlit
from monitoring.monitoring_metrics import compute_monitoring_summary, monitoring_summary_to_df

from visualizations.plots import (
    plot_forecast_interactive, plot_rolling_wmape,
    plot_error_trend, plot_historical_overview,
)
from visualizations.comparison_plots import (
    plot_metrics_bar, plot_error_distribution,
    plot_series_ranking, plot_leaderboard, plot_model_winner_pie,
)
from visualizations.monitoring_plots import (
    plot_drift_visualization, plot_spike_overlay, plot_stability_gauge,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TS Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card { background:#1e2130; border-radius:8px; padding:14px 18px; }
    .stMetric { background:#1e2130; border-radius:8px; padding:10px; }
    section[data-testid="stSidebar"] { background:#131621; }
    .block-container { padding-top:1.2rem; }
    div[data-testid="stMetricValue"] { font-size:1.1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 6px; padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

PRODUCT_NAMES = {
    "P001": "Tender Coconut", "P002": "Mature Coconut",
    "P003": "Coconut Oil 1L", "P004": "Coconut Water Bottle",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"), exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 TS Forecast")
    st.caption("TimesFM + Chronos | Multi-Product Forecasting")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Forecast", "📊 Model Comparison", "🔍 Monitoring"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload dataset", type=["xlsx", "csv"])
    freq = st.radio("Frequency", ["Daily", "Hourly"], horizontal=True)
    freq_key   = "D" if freq == "Daily" else "h"
    freq_token = "daily" if freq_key == "D" else "hourly"
    cov_cols   = COVARIATE_COLS_DAILY if freq_key == "D" else COVARIATE_COLS_HOURLY

    st.markdown("---")
    st.markdown("### ⚙️ Models")
    use_timesfm = st.checkbox("TimesFM", value=True)
    use_chronos  = st.checkbox("Chronos",  value=True)
    horizon = st.slider(
        "Forecast horizon" + (" (days)" if freq_key == "D" else " (hours)"),
        min_value=7, max_value=30 if freq_key == "D" else 168,
        value=21 if freq_key == "D" else 168,
    )

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
data_loaded = False
df_working  = None
series_ids  = []
validation  = None

if uploaded:
    try:
        sheet = "Daily_Orders" if freq_key == "D" else "Hourly_Orders"
        df_raw     = smart_load(uploaded, freq_key, sheet)
        validation = validate_dataframe(df_raw, freq_key)

        if validation["valid"]:
            df_working = aggregate_to_series(df_raw, freq_key)
            series_ids = sorted(df_working["series_id"].unique().tolist())
            data_loaded = True
            st.sidebar.success(f"✅ {len(series_ids)} series loaded")
        else:
            for e in validation["errors"]:
                st.sidebar.error(e)
    except Exception as e:
        st.sidebar.error(f"Load error: {e}")

if not data_loaded:
    df_working = generate_demo_data()
    series_ids = sorted(df_working["series_id"].unique().tolist())
    validation = validate_dataframe(df_working, "D")
    if not uploaded:
        st.sidebar.info("Demo mode — upload data for real forecasting.")

# ── Dynamic filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    companies = sorted(df_working["company_code"].unique().tolist()) if "company_code" in df_working.columns else []
    products  = sorted(df_working["product_id"].unique().tolist())   if "product_id"   in df_working.columns else []
    regions   = sorted(df_working["region"].unique().tolist())       if "region"       in df_working.columns else []
    cities    = sorted(df_working["city"].unique().tolist())         if "city"         in df_working.columns else []

    sel_companies = st.multiselect("Company", companies, default=companies) if companies else companies
    sel_products  = st.multiselect("Product", products,  default=products)  if products  else products
    sel_regions   = st.multiselect("Region",  regions,   default=regions)   if regions   else regions
    sel_cities    = st.multiselect("City",    cities,    default=cities)    if cities    else cities

# Apply filters
if sel_companies and "company_code" in df_working.columns:
    df_working = df_working[df_working["company_code"].isin(sel_companies)]
if sel_products and "product_id" in df_working.columns:
    df_working = df_working[df_working["product_id"].isin(sel_products)]
if sel_regions and "region" in df_working.columns:
    df_working = df_working[df_working["region"].isin(sel_regions)]
if sel_cities and "city" in df_working.columns:
    df_working = df_working[df_working["city"].isin(sel_cities)]
series_ids = sorted(df_working["series_id"].unique().tolist()) if len(df_working) else []


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.header("📈 Forecasting Dashboard")
    mode_label = "Real data" if data_loaded else "Demo mode"
    st.caption(f"{mode_label} · {freq} · {len(series_ids)} series · Horizon {horizon}")

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    info = validation["info"] if validation else {}
    c1.metric("Series", len(series_ids))
    c2.metric("Frequency", freq)
    c3.metric("Horizon", f"{horizon}{'d' if freq_key=='D' else 'h'}")
    c4.metric("Models", ("TFM" if use_timesfm else "") + ("+" if use_timesfm and use_chronos else "") + ("Chr" if use_chronos else ""))
    c5.metric("Total Orders", f"{info.get('total_orders', 0):,}")
    c6.metric("Covariates", len(cov_cols))

    st.markdown("---")

    # Historical trend
    st.subheader("📊 Historical Order Quantities")
    fig_hist = plot_historical_overview(df_working, series_ids, n_hist=90)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Dataset info
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📋 Dataset Info")
        if info:
            info_data = {
                "Metric": ["Rows", "Date Range", "Companies", "Products", "Missing %", "Duplicates"],
                "Value": [
                    f"{info.get('n_rows', '?'):,}",
                    f"{info.get('date_min', '?')} → {info.get('date_max', '?')}",
                    ", ".join(info.get("companies", [])),
                    ", ".join(info.get("products", [])),
                    f"{info.get('missing_pct', 0)}%",
                    str(info.get("duplicate_rows", 0)),
                ],
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("📈 Series Overview")
        cards = st.columns(min(4, len(series_ids)))
        for i, sid in enumerate(series_ids[:8]):
            s = df_working[df_working["series_id"] == sid]["order_qty"]
            with cards[i % len(cards)]:
                avg = int(s.mean()) if len(s) else 0
                std = int(s.std())  if len(s) > 1 else 0
                trend = "↑" if len(s) > 7 and s.iloc[-1] > s.iloc[-7] else "↓"
                st.metric(sid, f"{avg} avg", f"{trend} ±{std}")

    # Validation warnings
    if validation and validation.get("warnings"):
        with st.expander("⚠️ Data Warnings", expanded=False):
            for w in validation["warnings"]:
                st.warning(w)

    # Raw data preview
    with st.expander("🔎 Raw Data Preview", expanded=False):
        st.dataframe(df_working.head(100), use_container_width=True, height=300)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.header("🔮 Generate Forecast")

    if not series_ids:
        st.warning("No series available. Upload data or adjust filters.")
        st.stop()

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_sid = st.selectbox("Series", series_ids)
        run_btn      = st.button("▶ Generate Forecast", type="primary", use_container_width=True)
        show_history = st.number_input("History to display", 14, 90, 30)
        show_ci      = st.checkbox("Show confidence bands", value=True)

    # Run or use cached results
    if run_btn:
        try:
            with st.spinner(f"Running forecasts for {selected_sid}…"):
                series_df = df_working[df_working["series_id"] == selected_sid].copy()
                results = run_single_series(series_df, horizon, freq_token, cov_cols, selected_sid)
                st.session_state["fc_results"] = results
                st.session_state["fc_sid"]     = selected_sid
        except Exception as e:
            st.error(f"Forecast error: {e}")
            st.stop()

    if "fc_results" not in st.session_state:
        with col2:
            st.info("Select a series and click **Generate Forecast** to begin.")
        st.stop()

    results = st.session_state["fc_results"]
    sid     = st.session_state["fc_sid"]
    actual  = np.array(results["actual"])
    tfm_p   = np.array(results["timesfm_pred"])
    chr_p   = np.array(results["chronos_pred"])
    timestamps = results["timestamp"]

    # Metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    w_tfm, w_chr = results["wmape_timesfm"], results["wmape_chronos"]
    m_tfm, m_chr = round(mape(actual, tfm_p), 1), round(mape(actual, chr_p), 1)

    if use_timesfm:
        mc1.metric("TimesFM WMAPE", f"{w_tfm}%")
        mc2.metric("TimesFM MAPE",  f"{m_tfm}%")
    if use_chronos:
        mc3.metric("Chronos WMAPE", f"{w_chr}%")
        mc4.metric("Chronos MAPE",  f"{m_chr}%")

    # Trend + winner badge
    best = "TimesFM" if w_tfm <= w_chr else "Chronos"
    trend_dir = "↑ Rising" if actual[-1] > actual[0] else "↓ Falling" if actual[-1] < actual[0] else "→ Flat"
    bc1, bc2, bc3 = st.columns(3)
    bc1.success(f"🏆 Best model: **{best}**")
    bc2.info(f"📈 Trend: **{trend_dir}**")
    bc3.info(f"📏 Horizon: **{len(actual)} steps**")

    # Main forecast chart
    series_df = df_working[df_working["series_id"] == sid] if sid in df_working["series_id"].values else df_working.head(0)
    fig = plot_forecast_interactive(series_df, results, sid, n_hist=int(show_history), show_tfm=use_timesfm, show_chr=use_chronos, show_ci=show_ci)
    st.plotly_chart(fig, use_container_width=True)

    # Tabs
    tab_data, tab_err, tab_export = st.tabs(["📋 Predictions Table", "📉 Error Analysis", "💾 Export"])

    with tab_data:
        pred_df = pd.DataFrame({
            "timestamp": timestamps, "actual": actual,
            **({} if not use_timesfm else {"timesfm_pred": tfm_p}),
            **({} if not use_chronos else {"chronos_pred": chr_p}),
        })
        st.dataframe(pred_df, use_container_width=True, height=300)

    with tab_err:
        err_t = np.abs(actual - tfm_p)
        err_c = np.abs(actual - chr_p) if use_chronos else None
        fig_e = plot_error_trend(timestamps, err_t, err_c)
        st.plotly_chart(fig_e, use_container_width=True)

    with tab_export:
        ec1, ec2, ec3 = st.columns(3)
        # Forecast CSV
        export_df = pd.DataFrame({
            "timestamp": timestamps, "actual": actual,
            "timesfm_pred": tfm_p, "chronos_pred": chr_p,
            "timesfm_lower": results["timesfm_lower"], "timesfm_upper": results["timesfm_upper"],
            "chronos_lower": results["chronos_lower"], "chronos_upper": results["chronos_upper"],
        })
        ec1.download_button("📥 Forecast CSV", export_df.to_csv(index=False), f"forecast_{sid}.csv", "text/csv")

        # Metrics CSV
        metrics_rows = [
            {"model": "TimesFM", "WMAPE_%": w_tfm, "MAPE_%": m_tfm},
            {"model": "Chronos", "WMAPE_%": w_chr, "MAPE_%": m_chr},
        ]
        met_csv = pd.DataFrame(metrics_rows).to_csv(index=False)
        ec2.download_button("📥 Metrics CSV", met_csv, f"metrics_{sid}.csv", "text/csv")

        # Simple report
        report = f"""Forecast Report — {sid}
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Frequency: {freq} | Horizon: {horizon} steps
Models: TimesFM={'ON' if use_timesfm else 'OFF'}, Chronos={'ON' if use_chronos else 'OFF'}

Results:
  TimesFM WMAPE: {w_tfm}%  |  MAPE: {m_tfm}%
  Chronos WMAPE: {w_chr}%  |  MAPE: {m_chr}%
  Best Model: {best}
  Trend Direction: {trend_dir}
  Forecast Steps: {len(actual)}
"""
        ec3.download_button("📥 Report TXT", report, f"report_{sid}.txt", "text/plain")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.header("📊 Model Comparison")

    if not series_ids:
        st.warning("No series available.")
        st.stop()

    metric_choice = st.radio("Metric", ["WMAPE %", "MAPE %"], horizontal=True)
    metric_key    = "WMAPE_%" if "WMAPE" in metric_choice else "MAPE_%"

    run_comp = st.button("▶ Run Comparison (all series)", type="primary")

    if run_comp:
        prog = st.progress(0, text="Evaluating series…")
        def _cb(cur, tot, sid):
            if tot > 0:
                prog.progress(min(cur / tot, 1.0), text=f"Series {cur}/{tot}: {sid}")

        try:
            results_all = run_experiment(df_working, cov_cols, horizon, freq_token, "ALL", progress_cb=_cb)
            st.session_state["cmp_results"] = results_all
            prog.empty()
        except Exception as e:
            st.error(f"Comparison error: {e}")
            st.stop()

    if "cmp_results" not in st.session_state:
        st.info("Click **Run Comparison** to evaluate all series with both models.")
        st.stop()

    results_all = st.session_state["cmp_results"]
    if results_all.empty:
        st.warning("No results — all series may be too short for the selected horizon.")
        st.stop()

    metrics_df = compute_metrics_extended(results_all)
    overall    = metrics_df[metrics_df["series_id"] == "OVERALL"]
    per_ser    = metrics_df[metrics_df["series_id"] != "OVERALL"]

    # Overall KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    for col, model in zip([kc1, kc2], ["timesfm", "chronos"]):
        row = overall[overall["model"] == model]
        if not row.empty:
            col.metric(f"{model.upper()} WMAPE", f"{row['WMAPE_%'].values[0]:.1f}%")
    for col, model in zip([kc3, kc4], ["timesfm", "chronos"]):
        row = overall[overall["model"] == model]
        if not row.empty:
            col.metric(f"{model.upper()} MAPE", f"{row['MAPE_%'].values[0]:.1f}%")

    st.markdown("---")

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_bar = plot_metrics_bar(per_ser, metric_key)
        st.plotly_chart(fig_bar, use_container_width=True)
    with ch2:
        fig_pie = plot_model_winner_pie(per_ser, metric_key)
        st.plotly_chart(fig_pie, use_container_width=True)

    tab_rank, tab_dist, tab_tbl, tab_exp = st.tabs(["🏅 Ranking", "📊 Error Distribution", "📋 Full Table", "💾 Export"])

    with tab_rank:
        fig_rank = plot_series_ranking(per_ser, metric_key)
        st.plotly_chart(fig_rank, use_container_width=True)

    with tab_dist:
        fig_dist = plot_error_distribution(results_all)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab_tbl:
        fig_lb = plot_leaderboard(metrics_df)
        st.plotly_chart(fig_lb, use_container_width=True)
        st.dataframe(metrics_df, use_container_width=True, height=300)

    with tab_exp:
        st.download_button("📥 Comparison CSV", metrics_df.to_csv(index=False), "model_comparison.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MONITORING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Monitoring":
    st.header("🔍 Forecast Monitoring")

    if not series_ids:
        st.warning("No series available.")
        st.stop()

    mc1, mc2 = st.columns([2, 1])
    with mc1:
        mon_sid = st.selectbox("Series to monitor", series_ids)
    with mc2:
        roll_w  = st.slider("Rolling window", 3, 14, 7)

    run_mon = st.button("▶ Run Monitoring", type="primary")

    if run_mon:
        try:
            with st.spinner(f"Computing monitoring for {mon_sid}…"):
                s = df_working[df_working["series_id"] == mon_sid].copy()
                mon_res = run_single_series(s, horizon, freq_token, cov_cols, mon_sid)
                st.session_state["mon_res"] = mon_res
                st.session_state["mon_sid"] = mon_sid
        except Exception as e:
            st.error(f"Monitoring error: {e}")
            st.stop()

    if "mon_res" not in st.session_state:
        st.info("Select a series and click **Run Monitoring**.")
        st.stop()

    mon_res = st.session_state["mon_res"]
    mon_sid = st.session_state["mon_sid"]
    actual    = np.array(mon_res["actual"])
    tfm_preds = np.array(mon_res["timesfm_pred"])
    chr_preds = np.array(mon_res["chronos_pred"])
    timestamps = mon_res["timestamp"]

    # Full monitoring summary
    summary = compute_monitoring_summary(actual, tfm_preds, chr_preds, timestamps, window=roll_w)

    # Alerts
    st.subheader("🚨 Alert Panel")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**TimesFM**")
        alerts_to_streamlit(summary["alerts_tfm"], st)
    with ac2:
        st.markdown("**Chronos**")
        alerts_to_streamlit(summary["alerts_chr"], st)

    # Stability gauges
    gc1, gc2 = st.columns(2)
    with gc1:
        fig_g1 = plot_stability_gauge(summary["stability_score_tfm"], "TimesFM")
        st.plotly_chart(fig_g1, use_container_width=True)
    with gc2:
        fig_g2 = plot_stability_gauge(summary["stability_score_chr"], "Chronos")
        st.plotly_chart(fig_g2, use_container_width=True)

    # Monitoring metrics row
    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("WMAPE TFM", f"{summary['overall_wmape_tfm']}%")
    mm2.metric("WMAPE Chr", f"{summary['overall_wmape_chr']}%")
    mm3.metric("Spikes TFM", int(summary["spike_flags_tfm"].sum()))
    mm4.metric("Spikes Chr", int(summary["spike_flags_chr"].sum()))

    st.markdown("---")

    # Tabs
    tab_rw, tab_err, tab_drift, tab_spike, tab_exp = st.tabs(
        ["📈 Rolling WMAPE", "📉 Error Trend", "🔄 Drift", "⚡ Spikes", "💾 Export"]
    )

    with tab_rw:
        fig_rw = plot_rolling_wmape(summary["rolling_timestamps"], summary["rolling_wmape_tfm"], summary["rolling_wmape_chr"])
        st.plotly_chart(fig_rw, use_container_width=True)

    with tab_err:
        fig_e = plot_error_trend(timestamps, summary["errors_tfm"], summary["errors_chr"])
        st.plotly_chart(fig_e, use_container_width=True)

    with tab_drift:
        fig_d = plot_drift_visualization(timestamps, actual, tfm_preds, chr_preds, summary["drift_info_tfm"])
        st.plotly_chart(fig_d, use_container_width=True)

    with tab_spike:
        sc1, sc2 = st.columns(2)
        with sc1:
            fig_s1 = plot_spike_overlay(timestamps, summary["errors_tfm"], summary["spike_flags_tfm"], "TimesFM")
            st.plotly_chart(fig_s1, use_container_width=True)
        with sc2:
            fig_s2 = plot_spike_overlay(timestamps, summary["errors_chr"], summary["spike_flags_chr"], "Chronos", color="#7ee787")
            st.plotly_chart(fig_s2, use_container_width=True)

    with tab_exp:
        mon_df = monitoring_summary_to_df(summary, mon_sid)
        st.download_button("📥 Monitoring CSV", mon_df.to_csv(index=False), f"monitoring_{mon_sid}.csv", "text/csv")
        st.dataframe(mon_df, use_container_width=True)
