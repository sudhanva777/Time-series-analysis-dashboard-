"""
monitoring/alerts.py
─────────────────────
Alert generation from monitoring signals.
Returns structured alert dicts for Streamlit display.
"""

import numpy as np


SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING  = "warning"
SEVERITY_INFO     = "info"
SEVERITY_OK       = "ok"


def generate_alerts(
    drift_info:         dict,
    spike_flags:        np.ndarray,
    rolling_wmape_vals: np.ndarray,
    wmape_threshold:    float = 30.0,
) -> list:
    """
    Generate a list of alert dicts from monitoring signals.

    Each alert dict has:
        severity : "critical" | "warning" | "info" | "ok"
        title    : short alert title
        message  : detailed message

    Parameters
    ----------
    drift_info         : output from detect_drift()
    spike_flags        : bool array from detect_spikes()
    rolling_wmape_vals : float array from rolling_wmape()
    wmape_threshold    : WMAPE % above which to raise a warning
    """
    alerts = []

    # ── Drift alert ───────────────────────────────────────────────────────────
    if drift_info.get("drift_detected"):
        alerts.append({
            "severity": SEVERITY_CRITICAL,
            "title":    "Model Drift Detected",
            "message":  (
                f"WMAPE increased from {drift_info['early_wmape']:.1f}% "
                f"to {drift_info['late_wmape']:.1f}% "
                f"({drift_info['ratio']:.2f}× ratio). "
                "The model may need retraining or recalibration."
            ),
        })
    else:
        alerts.append({
            "severity": SEVERITY_OK,
            "title":    "No Drift Detected",
            "message":  (
                f"WMAPE is stable: {drift_info['early_wmape']:.1f}% → "
                f"{drift_info['late_wmape']:.1f}%"
            ),
        })

    # ── Spike alert ───────────────────────────────────────────────────────────
    n_spikes = int(np.sum(spike_flags)) if len(spike_flags) > 0 else 0
    pct_spikes = n_spikes / len(spike_flags) * 100 if len(spike_flags) > 0 else 0

    if n_spikes == 0:
        alerts.append({
            "severity": SEVERITY_OK,
            "title":    "No Forecast Spikes",
            "message":  "All forecast errors are within the expected range.",
        })
    elif pct_spikes > 20:
        alerts.append({
            "severity": SEVERITY_CRITICAL,
            "title":    f"High Spike Rate — {n_spikes} spikes ({pct_spikes:.0f}%)",
            "message":  (
                f"{n_spikes} of {len(spike_flags)} steps have unusually large errors. "
                "The forecast is highly unstable for this series."
            ),
        })
    else:
        alerts.append({
            "severity": SEVERITY_WARNING,
            "title":    f"{n_spikes} Forecast Spike(s) Detected",
            "message":  (
                f"{n_spikes} step(s) have errors significantly above the mean. "
                "Check the spike detection chart for exact positions."
            ),
        })

    # ── High WMAPE alert ──────────────────────────────────────────────────────
    valid_wmape = rolling_wmape_vals[~np.isnan(rolling_wmape_vals)]
    if len(valid_wmape) > 0:
        recent_wmape = float(np.mean(valid_wmape[-3:]))  # average of last 3 windows
        if recent_wmape > wmape_threshold * 1.5:
            alerts.append({
                "severity": SEVERITY_CRITICAL,
                "title":    f"Very High Recent WMAPE — {recent_wmape:.1f}%",
                "message":  (
                    f"Recent rolling WMAPE ({recent_wmape:.1f}%) far exceeds the "
                    f"threshold ({wmape_threshold:.0f}%). Forecast quality is poor."
                ),
            })
        elif recent_wmape > wmape_threshold:
            alerts.append({
                "severity": SEVERITY_WARNING,
                "title":    f"Elevated Recent WMAPE — {recent_wmape:.1f}%",
                "message":  (
                    f"Recent rolling WMAPE ({recent_wmape:.1f}%) exceeds the "
                    f"threshold ({wmape_threshold:.0f}%)."
                ),
            })

    return alerts


def alerts_to_streamlit(alerts: list, st) -> None:
    """
    Render alerts in Streamlit using appropriate status widgets.

    Parameters
    ----------
    alerts : list of alert dicts from generate_alerts()
    st     : the streamlit module
    """
    for alert in alerts:
        sev = alert["severity"]
        msg = f"**{alert['title']}** — {alert['message']}"
        if sev == SEVERITY_CRITICAL:
            st.error(msg)
        elif sev == SEVERITY_WARNING:
            st.warning(msg)
        elif sev == SEVERITY_OK:
            st.success(msg)
        else:
            st.info(msg)
