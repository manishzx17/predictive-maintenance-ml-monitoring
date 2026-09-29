"""
Streamlit Dashboard for Machine Failure Prediction and Drift Monitoring.
Connected to the FastAPI backend (Streamlit -> FastAPI -> src/).
"""

import os
import io
import json
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from src.client import MonitorApiClient
from src.features import RAW_NUMERICAL_COLS, CATEGORICAL_COLS
from src.api import app as fastapi_app
from fastapi.testclient import TestClient

# Page configuration
st.set_page_config(
    page_title="Machine Failure Drift Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for modern industrial UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        background-color: #F8FAFC;
    }
    .outcome-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .outcome-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .outcome-alert {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .outcome-inspection {
        background-color: #F3E8FF;
        color: #6B21A8;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        border: 1px solid #D8B4FE;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_api_client(base_url: str, use_in_process: bool) -> MonitorApiClient:
    """Initialize cached API client (either live HTTP or in-memory TestClient)."""
    if use_in_process:
        return MonitorApiClient(client=TestClient(fastapi_app))
    return MonitorApiClient(base_url=base_url)


# --- Sidebar: Connection & Diagnostics ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/engine.png", width=64)
    st.title("System Diagnostics")

    api_url = st.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")
    connection_mode = st.radio(
        "Connection Adapter",
        ["In-Process TestClient (Zero-Latency)", "Live HTTP Server"],
        help="In-process TestClient directly communicates with FastAPI in memory without needing a separate uvicorn terminal process."
    )

    use_in_process = connection_mode.startswith("In-Process")
    client = get_api_client(api_url, use_in_process)

    try:
        health_data = client.get_health()
        st.success("● FastAPI Backend Online")
        st.metric("Pipeline Status", health_data["status"].upper())
        st.caption(f"**Version**: {health_data['pipeline_version']}")
        st.caption(f"**Operational Threshold ($t^*$)**: `{health_data['decision_threshold']}`")
        st.caption(f"**Conformal Quantile ($\\hat{{q}}_{{0.10}}$)**: `{health_data['conformal_q_hat']}`")
    except Exception as e:
        st.error("● Backend Offline or Unreachable")
        st.caption(f"Error: {e}")

    st.divider()
    st.markdown("### Governance Guardrails")
    st.info("🔒 **Autonomous Retraining**: Strictly Prohibited\n\n🛡️ **Final Test Set**: 100% Quarantined")

# --- Main Application Header ---
st.markdown('<div class="main-header">Predictive Maintenance & Automated Drift Surveillance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production Telemetry Interface connecting Streamlit ➔ FastAPI ➔ Model Engine</div>', unsafe_allow_html=True)

# Tabs
tab_inference, tab_monitoring, tab_governance = st.tabs([
    "🔍 Real-Time Inference (/predict)",
    "📊 Batch Drift Surveillance (/monitor)",
    "🛡️ Escalation Governance"
])


# =====================================================================
# TAB 1: Single-Machine Telemetry Inference
# =====================================================================
with tab_inference:
    st.subheader("Machine Telemetry Inference Form")
    st.markdown("Submit live machine sensor readings to evaluate failure probability, optimal dispatch decision ($t^* = 0.15$), and conformal prediction sets.")

    # Presets
    preset_choice = st.selectbox(
        "Load Sensor Telemetry Preset",
        ["Custom Manual Entry", "Nominal Healthy Machine", "High Tool Wear Stress", "Thermal Overheating", "Mechanical Torque Surge"]
    )

    default_type = "M"
    default_air_temp = 298.1
    default_proc_temp = 308.6
    default_rpm = 1551.0
    default_torque = 42.8
    default_wear = 0.0

    if preset_choice == "Nominal Healthy Machine":
        default_type, default_air_temp, default_proc_temp, default_rpm, default_torque, default_wear = "M", 298.1, 308.6, 1551.0, 42.8, 10.0
    elif preset_choice == "High Tool Wear Stress":
        default_type, default_air_temp, default_proc_temp, default_rpm, default_torque, default_wear = "L", 299.5, 309.5, 1420.0, 52.0, 215.0
    elif preset_choice == "Thermal Overheating":
        default_type, default_air_temp, default_proc_temp, default_rpm, default_torque, default_wear = "H", 304.5, 313.8, 1380.0, 48.5, 95.0
    elif preset_choice == "Mechanical Torque Surge":
        default_type, default_air_temp, default_proc_temp, default_rpm, default_torque, default_wear = "M", 298.5, 308.8, 1250.0, 68.0, 180.0

    with st.form("inference_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            input_type = st.selectbox("Product Quality Type", ["L", "M", "H"], index=["L", "M", "H"].index(default_type))
            air_temp = st.number_input("Air Temperature [K]", min_value=250.0, max_value=350.0, value=float(default_air_temp), step=0.1)
            proc_temp = st.number_input("Process Temperature [K]", min_value=250.0, max_value=350.0, value=float(default_proc_temp), step=0.1)

        with col2:
            rot_speed = st.number_input("Rotational Speed [rpm]", min_value=100.0, max_value=5000.0, value=float(default_rpm), step=10.0)
            torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=200.0, value=float(default_torque), step=0.5)
            tool_wear = st.number_input("Tool Wear [min]", min_value=0.0, max_value=500.0, value=float(default_wear), step=1.0)

        with col3:
            st.markdown("#### Derived Feature Previews")
            temp_diff_prev = proc_temp - air_temp
            power_prev = torque * rot_speed * (2 * 3.14159 / 60)
            wear_load_prev = tool_wear * torque
            st.metric("Temp_Diff [K]", f"{temp_diff_prev:.2f} K")
            st.metric("Mechanical Power [W]", f"{power_prev:,.1f} W")
            st.metric("Wear_Load [min·Nm]", f"{wear_load_prev:,.1f}")

        submit_inference = st.form_submit_button("⚡ Submit Telemetry to FastAPI (/predict)", use_container_width=True)

    if submit_inference:
        telemetry_payload = {
            "Type": input_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }

        try:
            pred_response = client.predict_failure(telemetry_payload)

            st.markdown("### Inference Results")
            res_col1, res_col2, res_col3 = st.columns(3)

            p_fail = pred_response["failure_probability"]
            decision = pred_response["prediction"]
            threshold = pred_response["threshold_used"]
            prediction_set = pred_response["prediction_set"]
            is_ambiguous = pred_response["is_ambiguous"]

            with res_col1:
                st.metric("Calibrated Failure Probability", f"{p_fail:.2%}")
                st.progress(min(1.0, p_fail))

            with res_col2:
                if decision == 1:
                    st.error("🚨 **MAINTENANCE DISPATCH REQUIRED (1)**")
                    st.caption(f"Failure probability ({p_fail:.3f}) $\\ge$ cost-optimal threshold ($t^* = {threshold}$)")
                else:
                    st.success("✅ **NOMINAL OPERATION (0)**")
                    st.caption(f"Failure probability ({p_fail:.3f}) < cost-optimal threshold ($t^* = {threshold}$)")

            with res_col3:
                st.markdown(r"**Conformal Prediction Set** ($90\%$ Coverage):")
                st.code(str(prediction_set))
                if is_ambiguous:
                    st.warning("⚠️ **Ambiguous Set** $\\rightarrow$ High uncertainty. Routed to Human Technician Inspection.")
                else:
                    st.info("🎯 **Singleton Set** $\\rightarrow$ High confidence decision.")

        except Exception as err:
            st.error(f"Inference request failed: {err}")


# =====================================================================
# TAB 2: Batch Distribution & Calibration Surveillance
# =====================================================================
with tab_monitoring:
    st.subheader("Batch Drift & Calibration Surveillance (/monitor)")
    st.markdown("Transmit incoming sensor telemetry batches to the FastAPI monitoring engine to evaluate input drift, prediction drift, and calibration reliability.")

    batch_source = st.radio(
        "Select Telemetry Batch Source",
        ["Phase 10 Synthetic Drift Scenarios", "Upload Custom CSV Batch"],
        horizontal=True
    )

    batch_df = None
    batch_name = ""

    if batch_source == "Phase 10 Synthetic Drift Scenarios":
        scenario_map = {
            "Reference Baseline (Nominal Validation Split)": "data/processed/reference_baseline.csv",
            "Scenario A (Thermal Shift: +3K Air Temp)": "data/processed/drift_scenario_a_temperature.csv",
            "Scenario B (Torque Shift: +10Nm Torque)": "data/processed/drift_scenario_b_torque.csv",
            "Scenario C (Variance Expansion: 2.5x Dispersion)": "data/processed/drift_scenario_c_variance.csv",
            "Scenario D (Combined Drift: Thermal + Mechanical)": "data/processed/drift_scenario_d_combined.csv",
        }
        selected_scenario = st.selectbox("Select Scenario", list(scenario_map.keys()))
        csv_path = scenario_map[selected_scenario]
        if os.path.exists(csv_path):
            batch_df = pd.read_csv(csv_path)
            batch_name = selected_scenario
        else:
            st.error(f"File not found: {csv_path}")

    else:
        uploaded_file = st.file_uploader("Upload Batch CSV (Must contain sensor telemetry columns)", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                batch_name = uploaded_file.name
                st.success(f"Loaded {len(batch_df):,} rows from {batch_name}")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")

    if batch_df is not None:
        st.write(f"**Batch Preview**: {len(batch_df):,} records")
        st.dataframe(batch_df.head(5), use_container_width=True)

        include_labels = False
        if "Machine failure" in batch_df.columns:
            include_labels = st.checkbox(
                "Include ground-truth failure labels for supervised calibration evaluation (ECE & Brier Score)",
                value=True
            )

        if st.button("🚀 Run Batch Surveillance via FastAPI (/monitor)", type="primary", use_container_width=True):
            records = batch_df.to_dict(orient="records")
            labels = batch_df["Machine failure"].tolist() if (include_labels and "Machine failure" in batch_df.columns) else None

            with st.spinner("Evaluating telemetry across input, prediction, calibration, and conformal layers..."):
                try:
                    report = client.monitor_batch(records=records, labels=labels)

                    st.divider()
                    st.markdown("## Consolidated Operational Verdict")

                    outcome = report["decision"]["outcome"]
                    justification = report["decision"]["justification"]

                    # Outcome banner
                    if outcome == "NORMAL":
                        st.markdown(f'<div class="outcome-normal">STATUS: NORMAL</div>', unsafe_allow_html=True)
                    elif outcome == "WARNING":
                        st.markdown(f'<div class="outcome-warning">STATUS: WARNING</div>', unsafe_allow_html=True)
                    elif outcome == "DRIFT_ALERT":
                        st.markdown(f'<div class="outcome-alert">STATUS: DRIFT_ALERT</div>', unsafe_allow_html=True)
                    elif outcome == "HUMAN_INSPECTION":
                        st.markdown(f'<div class="outcome-inspection">STATUS: HUMAN_INSPECTION</div>', unsafe_allow_html=True)

                    st.markdown(f"**Operational Justification**: {justification}")

                    if report["decision"]["escalation_required"]:
                        st.error("⚠️ **Escalation Required**: Human reliability engineers must review this batch according to the 5-stage protocol.")

                    st.divider()
                    st.markdown("### Surveillance Telemetry Metrics")
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

                    with m_col1:
                        max_psi = report["input_drift"]["max_feature_psi"]
                        top_feat = report["input_drift"]["top_drifting_feature"]
                        st.metric("Max Feature PSI", f"{max_psi:.4f}", help=f"Top Drifting Feature: {top_feat}")
                        st.caption(f"Top Drift: **{top_feat}**")

                    with m_col2:
                        w_dist = report["prediction_drift"]["wasserstein_distance"]
                        ks_p = report["prediction_drift"]["ks_p_value"]
                        st.metric("Prediction Wasserstein (W)", f"{w_dist:.5f}")
                        st.caption(f"KS p-value: {ks_p:.2e}")

                    with m_col3:
                        ambig_rate = report["conformal_uncertainty"]["ambiguous_rate"]
                        st.metric("Conformal Ambiguity Rate", f"{ambig_rate:.2f}%")
                        cov = report["conformal_uncertainty"].get("empirical_coverage")
                        if cov is not None:
                            st.caption(f"Empirical Coverage: **{cov:.2f}%**")

                    with m_col4:
                        if report["is_supervised"]:
                            ece = report["reliability"]["10_bin_ece"]
                            brier = report["reliability"]["brier_score"]
                            st.metric("10-Bin ECE", f"{ece:.5f}")
                            st.caption(f"Brier Score: {brier:.5f}")
                        else:
                            st.metric("10-Bin ECE", "N/A (Unsupervised)")
                            st.caption("No ground-truth labels supplied")

                    # Input Drift Breakdown Table
                    st.markdown("### Input Feature Drift Breakdown")
                    feat_df = pd.DataFrame(report["input_drift"]["feature_table"])
                    st.dataframe(
                        feat_df.style.format({
                            "KS_Statistic": "{:.4f}",
                            "KS_P_Value": "{:.2e}",
                            "PSI": "{:.4f}",
                        }),
                        use_container_width=True
                    )

                except Exception as err:
                    st.error(f"Monitoring evaluation failed: {err}")


# =====================================================================
# TAB 3: Human-in-the-Loop Escalation Governance
# =====================================================================
with tab_governance:
    st.subheader("Human-in-the-Loop Operational Escalation Protocol")
    st.markdown("""
    In high-consequence industrial manufacturing environments, automated machine-directed dispatch requires strict operational safeguards:
    - **No Autonomous Retraining**: Models must **never** automatically retrain and deploy without human validation.
    - **Root Cause Determination**: True mechanical regime changes must be separated from transducer/sensor calibration faults.
    """)

    st.markdown("""
    ```mermaid
    flowchart LR
        A[Stage 1: Drift Flag\nAutomated Engine] --> B[Stage 2: Root Cause Review\nPlant Engineers]
        B --> C[Stage 3: Retraining Sandbox\nOffline Environment]
        C --> D[Stage 4: Validation Gate\nECE <= 0.02, Cost <= Champion]
        D --> E[Stage 5: Human Approval\nFormal Sign-off]
        E --> F[Deployment\nBlue-Green Rollout]
    ```
    """)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Operational Escalation Stages")
        st.markdown("""
        1. **Stage 1 (Detection)**: The engine detects `DRIFT_ALERT` (PSI > 0.25 / W >= 0.05) or `HUMAN_INSPECTION` (ECE > 0.10 / Ambiguity >= 95%).
        2. **Stage 2 (Root Cause Audit)**: Reliability engineers verify physical machinery and sensor calibration logs.
        3. **Stage 3 (Candidate Retraining)**: If confirmed as an operational process change, candidate models are trained in an isolated sandbox.
        """)

    with g2:
        st.markdown("#### Validation & Deployment Gates")
        st.markdown(r"""
        4. **Stage 4 (Validation Gate)**: Candidate models must pass discrimination, ECE calibration ($\le 0.02$), and cost-optimal threshold evaluation.
        5. **Stage 5 (Human Approval)**: Plant reliability and operations teams sign off before production deployment.
        """)
