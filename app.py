import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.pipeline import (
    ROOT,
    DATA_PATH,
    CACHE_PATH,
    get_dataset_metadata,
    load_and_aggregate,
    get_downsampled_telemetry,
)
from src.dashboard_components import (
    CUSTOM_CSS,
    SENSOR_INFO,
    render_sidebar,
    render_page_header,
    create_telemetry_chart,
    create_model_comparison_bar,
    create_tradeoff_chart,
    create_confusion_matrix_chart,
    create_score_gauge,
    render_prediction_status_card,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Early Failure Prediction AI — Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_inference_model():
    """Load the trained machine learning pipeline artifact."""
    model_file = ROOT / "models" / "best_model.joblib"
    if model_file.exists():
        try:
            return joblib.load(model_file)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_artifacts_and_reports():
    """Load saved evaluation reports, model metadata, and metrics."""
    report_path = ROOT / "reports" / "model_comparison.json"
    info_path = ROOT / "models" / "model_info.json"
    csv_path = ROOT / "reports" / "model_comparison.csv"

    report_data = None
    info_data = {}
    df_csv = pd.DataFrame()

    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if info_path.exists():
        try:
            info_data = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if csv_path.exists():
        try:
            df_csv = pd.read_csv(csv_path)
        except Exception:
            pass

    trained = (report_data is not None) and (info_path.exists())
    return report_data, info_data, df_csv, trained


@st.cache_data(show_spinner="Loading aggregated sensor data...")
def get_sensor_dataset():
    """Load cached 5-minute aggregated sensor windows and proxy labels."""
    return load_and_aggregate()


@st.cache_data(show_spinner=False)
def get_downsampled_trends():
    """Load fast multi-sensor trend series for the Overview dashboard."""
    return get_downsampled_telemetry(max_points=1200)


# ---------------------------------------------------------
# Initialize State and Metadata
# ---------------------------------------------------------
dataset_meta = get_dataset_metadata(DATA_PATH)
report_data, report_info, df_comparison, is_trained = load_artifacts_and_reports()
page = render_sidebar(dataset_meta, is_trained, report_info)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if page == "Overview":
    render_page_header(
        title="Early Failure Prediction AI",
        subtitle="Sensor-based predictive maintenance analytics for industrial air compressors. Evaluates 5 machine learning classifiers across multivariate telemetry windows.",
        badge_text="METROPT-3 COMPRESSOR TELEMETRY",
    )

    # Key Performance Indicator (KPI) Metric Cards
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Sensor Windows</div>
                <div class="kpi-value">50,782</div>
                <div class="kpi-sub">5-min aggregated intervals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        feature_count = report_info.get("feature_count", 64)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Extracted Features</div>
                <div class="kpi-value">{feature_count}</div>
                <div class="kpi-sub">Mean, std, min, max (16 ch)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Classifiers</div>
                <div class="kpi-value">5 Models</div>
                <div class="kpi-sub">KNN, SVM, GNB, LogReg, DT</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        best_name = report_info.get("best_model", "Logistic Regression")
        best_f1_str = "0.9632"
        if report_data and "results" in report_data:
            for r in report_data["results"]:
                if r["model"] == best_name:
                    best_f1_str = f"{r.get('validation_f1', 0.9632):.4f}"
                    break
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Top Validation Model</div>
                <div class="kpi-value" style="font-size:1.15rem; color:#a78bfa;">{best_name}</div>
                <div class="kpi-sub">Validation F1: {best_f1_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k5:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Proxy Anomaly Ratio</div>
                <div class="kpi-value" style="color:#fcd34d;">2.00%</div>
                <div class="kpi-sub">1,016 / 50,782 flagged windows</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section: Sensor Telemetry Analytics
    st.markdown("### 📈 Sensor Telemetry Trends")
    st.markdown(
        '<p style="color:#94a3b8; font-size:0.88rem; margin-top:-6px; margin-bottom:12px;">'
        'Continuous physical readings captured from MetroPT-3 compressor telemetry over 8 months (Feb–Sep 2020), downsampled for interactive exploration.'
        '</p>',
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns([3, 1])
    with t_col2:
        available_channels = [
            "Oil_temperature",
            "Motor_current",
            "Reservoirs",
            "DV_pressure",
            "TP2",
            "TP3",
            "H1",
        ]
        selected_channels = st.multiselect(
            "Select Sensor Channels",
            options=available_channels,
            default=["Oil_temperature", "Motor_current", "Reservoirs"],
            help="Choose physical signals to superimpose on the timeline.",
        )
        st.markdown(
            """
            <div style="background:#0f172a; border:1px solid rgba(148,163,184,0.12); border-radius:8px; padding:12px; font-size:0.78rem; color:#94a3b8; margin-top:12px;">
                <div style="font-weight:600; color:#cbd5e1; margin-bottom:6px;">Channel Legend</div>
                • <b>Oil Temp</b>: Compressor lubrication thermal load (°C)<br>
                • <b>Motor Current</b>: Drive motor electrical demand (A)<br>
                • <b>Reservoirs</b>: Output air delivery pressure (bar)<br>
                • <b>DV Pressure</b>: Drain valve discharge pressure (bar)
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col1:
        if selected_channels:
            df_trends = get_downsampled_trends()
            if not df_trends.empty:
                fig_telemetry = create_telemetry_chart(df_trends, selected_channels)
                st.plotly_chart(fig_telemetry, use_container_width=True)
            else:
                st.info("Telemetry data is being initialized...")
        else:
            st.warning("Please select at least one sensor channel to display the telemetry chart.")

    # Section: Model Performance Summary & Anomaly Methodology
    st.markdown("---")
    m_col1, m_col2 = st.columns([3, 2])

    with m_col1:
        st.markdown("### 🧪 Classifier Benchmark Overview")
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.86rem; margin-top:-6px;">'
            'Comparison of the 5 implemented classifiers on held-out test data (F1-score, Precision, Recall, and PR-AUC).'
            '</p>',
            unsafe_allow_html=True,
        )
        if not df_comparison.empty:
            metric_choice = st.selectbox(
                "Benchmark Metric for Comparison",
                options=["f1", "pr_auc", "precision", "recall", "balanced_accuracy", "accuracy"],
                format_func=lambda x: {
                    "f1": "F1-Score (Harmonic Mean)",
                    "pr_auc": "PR-AUC (Precision-Recall Area)",
                    "precision": "Precision (PPV)",
                    "recall": "Recall (Sensitivity)",
                    "balanced_accuracy": "Balanced Accuracy",
                    "accuracy": "Overall Accuracy",
                }.get(x, x),
                index=0,
            )
            fig_overview_models = create_model_comparison_bar(df_comparison, metric=metric_choice)
            st.plotly_chart(fig_overview_models, use_container_width=True)
        else:
            st.warning("Model comparison reports not found. Please verify `reports/model_comparison.json`.")

    with m_col2:
        st.markdown("### ⚠️ Methodology & Limitations")
        st.markdown(
            """
            <div class="content-card" style="margin-top:4px;">
                <div style="font-size:0.84rem; line-height:1.55; color:#cbd5e1;">
                    <div style="display:flex; align-items:center; gap:6px; margin-bottom:8px;">
                        <span class="badge badge-purple">ANOMALY TAXONOMY</span>
                    </div>
                    <b>1. Physical Sensor Readings</b><br>
                    <span style="color:#94a3b8; font-size:0.8rem;">
                    10-second industrial telemetry from Portuguese Metro trains (UCI ML Repository #791).
                    </span>
                    <br><br>
                    <b>2. Statistical Anomaly Proxies</b><br>
                    <span style="color:#94a3b8; font-size:0.8rem;">
                    Because verified machine teardown failure logs are unavailable in continuous telemetry, labels are computed via multivariate Median Absolute Deviation (top 2% cutoff).
                    </span>
                    <br><br>
                    <b>3. Confirmed Machine Failures</b><br>
                    <span style="color:#94a3b8; font-size:0.8rem;">
                    Proxy flags identify abnormal operating regimes; they are <b>not</b> certified mechanical breakdowns or safety alarms.
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 2: MODEL LAB
# =========================================================
elif page == "Model Lab":
    render_page_header(
        title="Model Lab — Five Classifier Benchmark",
        subtitle="Comprehensive comparative evaluation of KNN, SVM, Gaussian Naive Bayes, Logistic Regression, and Decision Tree on held-out test data.",
        badge_text="VALIDATION & TEST BENCHMARK",
    )

    if not is_trained or df_comparison.empty or not report_data:
        st.error(
            "Evaluation artifacts not detected in `reports/model_comparison.json`. "
            "Please ensure the training script has been executed: `python src/train_model.py`."
        )
    else:
        results = report_data.get("results", [])

        # Leaderboard Table
        st.markdown("### 🏆 Classifier Ranking Leaderboard")
        df_display = df_comparison.copy()
        df_display = df_display.sort_values("f1", ascending=False).reset_index(drop=True)
        df_display.index += 1
        df_display.index.name = "Rank"

        # Format numbers for clean academic presentation
        formatted_df = df_display.copy()
        numeric_cols = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "pr_auc", "threshold", "validation_f1"]
        for c in numeric_cols:
            if c in formatted_df.columns:
                formatted_df[c] = formatted_df[c].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v)

        formatted_df = formatted_df.rename(
            columns={
                "model": "Classifier",
                "f1": "F1 (Test)",
                "validation_f1": "F1 (Val)",
                "pr_auc": "PR-AUC",
                "precision": "Precision",
                "recall": "Recall",
                "balanced_accuracy": "Balanced Acc",
                "accuracy": "Accuracy",
                "threshold": "Decision Threshold",
            }
        )

        st.dataframe(formatted_df, use_container_width=True)

        st.markdown("---")

        # Visual Comparisons: Bar Comparison + Precision-Recall Tradeoff
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            st.markdown("### 📊 Metric Comparison")
            sel_metric = st.selectbox(
                "Select Primary Comparison Metric",
                options=["f1", "pr_auc", "precision", "recall", "balanced_accuracy", "accuracy"],
                format_func=lambda x: {
                    "f1": "F1-Score (Harmonic Mean)",
                    "pr_auc": "PR-AUC (Precision-Recall Area)",
                    "precision": "Precision (PPV)",
                    "recall": "Recall (Sensitivity)",
                    "balanced_accuracy": "Balanced Accuracy",
                    "accuracy": "Overall Accuracy",
                }.get(x, x),
                key="model_lab_metric",
            )
            fig_bar = create_model_comparison_bar(df_comparison, metric=sel_metric)
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_chart2:
            st.markdown("### ⚖️ Precision vs Recall Tradeoff")
            st.caption("Displays the precision-recall frontier across the five evaluated classifiers.")
            fig_tradeoff = create_tradeoff_chart(df_comparison)
            st.plotly_chart(fig_tradeoff, use_container_width=True)

        st.markdown("---")

        # Confusion Matrix Deep Dive
        st.markdown("### 🔍 Confusion Matrix & Error Analysis")
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.86rem; margin-top:-6px;">'
            'Examine exact True Positive, False Positive, True Negative, and False Negative counts on the 10,157 held-out test windows.'
            '</p>',
            unsafe_allow_html=True,
        )

        cm_col1, cm_col2 = st.columns([1, 2])

        model_names = [r["model"] for r in results]
        with cm_col1:
            selected_model_name = st.selectbox(
                "Select Classifier for Matrix Inspection",
                options=model_names,
                index=model_names.index(report_info.get("best_model", "Logistic Regression"))
                if report_info.get("best_model", "Logistic Regression") in model_names
                else 0,
            )

            # Retrieve selected model result row
            sel_res = next(r for r in results if r["model"] == selected_model_name)
            cm = np.array(sel_res["confusion_matrix"])
            tn, fp = cm[0]
            fn, tp = cm[1]
            total_test = tn + fp + fn + tp
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            st.markdown(
                f"""
                <div class="content-card" style="margin-top:10px;">
                    <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
                        <div style="font-weight:700; color:#f8fafc; margin-bottom:8px;">{selected_model_name} Diagnostics</div>
                        • <b>Tested Windows</b>: {total_test:,}<br>
                        • <b>True Anomaly Detected (TP)</b>: <span style="color:#34d399; font-weight:600;">{tp}</span> / {tp+fn}<br>
                        • <b>False Alarms (FP)</b>: <span style="color:#fbbf24; font-weight:600;">{fp}</span><br>
                        • <b>Missed Anomalies (FN)</b>: <span style="color:#f87171; font-weight:600;">{fn}</span><br>
                        • <b>Specificity (TNR)</b>: {specificity*100:.2f}%<br>
                        • <b>Decision Threshold</b>: <span style="color:#a78bfa; font-weight:600;">{sel_res.get('threshold', 0.5):.4f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with cm_col2:
            fig_cm = create_confusion_matrix_chart(cm, selected_model_name)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(
            "Note: Decision thresholds were tuned exclusively on the validation partition to optimize F1-score; "
            "evaluation metrics shown reflect performance on the completely untouched test set."
        )


# =========================================================
# PAGE 3: PREDICT & INSPECT
# =========================================================
elif page == "Predict & Inspect":
    render_page_header(
        title="Predict & Inspect — Sensor Window Inference",
        subtitle="Interactive analysis of 5-minute historical sensor windows. Inspect feature values, evaluate model scores against tuned decision thresholds, and verify anomaly proxies.",
        badge_text="TELEMETRY INFERENCE ENGINE",
    )

    try:
        X, y, timestamps = get_sensor_dataset()
        model = load_inference_model()

        if model is None:
            st.warning("Trained model artifact `models/best_model.joblib` not found. Please train models first.")
        else:
            total_windows = len(X)
            threshold = float(report_info.get("threshold", 0.948061))
            best_model_name = report_info.get("best_model", "Logistic Regression")

            # Initialize session state for window index
            if "selected_window_idx" not in st.session_state:
                st.session_state.selected_window_idx = min(100, total_windows - 1)

            # Quick Navigation Presets
            st.markdown("#### ⏱️ Historical Window Selector")
            q1, q2, q3, q4, q5 = st.columns(5)

            # Pre-compute a known proxy anomaly index for demo purposes
            anomaly_indices = np.where(y.values == 1)[0]
            normal_indices = np.where(y.values == 0)[0]
            demo_anomaly_idx = int(anomaly_indices[15]) if len(anomaly_indices) > 15 else 0
            demo_normal_idx = int(normal_indices[15]) if len(normal_indices) > 15 else 0

            with q1:
                if st.button("⚡ Known Anomaly Window", use_container_width=True, help="Jump to a verified proxy anomaly window (y=1)"):
                    st.session_state.selected_window_idx = demo_anomaly_idx
                    st.rerun()

            with q2:
                if st.button("🟢 Known Normal Window", use_container_width=True, help="Jump to a typical normal operational window (y=0)"):
                    st.session_state.selected_window_idx = demo_normal_idx
                    st.rerun()

            with q3:
                if st.button("🎲 Random Window", use_container_width=True, help="Select a random 5-minute window from the timeline"):
                    st.session_state.selected_window_idx = int(np.random.randint(0, total_windows))
                    st.rerun()

            with q4:
                if st.button("⏮️ Earliest Window", use_container_width=True):
                    st.session_state.selected_window_idx = 0
                    st.rerun()

            with q5:
                if st.button("⏭️ Latest Window", use_container_width=True):
                    st.session_state.selected_window_idx = total_windows - 1
                    st.rerun()

            # Window Index Selector Slider & Input
            sl_col1, sl_col2 = st.columns([3, 1])
            with sl_col1:
                selected_idx = st.slider(
                    "Historical Window Index",
                    min_value=0,
                    max_value=total_windows - 1,
                    value=st.session_state.selected_window_idx,
                    help="Slide to explore any 5-minute window across the 8-month compressor timeline.",
                    key="slider_window_idx",
                )
                st.session_state.selected_window_idx = selected_idx

            with sl_col2:
                st.session_state.selected_window_idx = st.number_input(
                    "Exact Index",
                    min_value=0,
                    max_value=total_windows - 1,
                    value=st.session_state.selected_window_idx,
                    step=1,
                    key="number_window_idx",
                )

            current_idx = st.session_state.selected_window_idx
            sample_features = X.iloc[[current_idx]]
            current_time = timestamps.iloc[current_idx] if hasattr(timestamps, "iloc") else timestamps[current_idx]
            actual_proxy_label = int(y.iloc[current_idx]) if hasattr(y, "iloc") else int(y[current_idx])

            # Inference Computation
            if hasattr(model, "predict_proba"):
                raw_score = float(model.predict_proba(sample_features)[0, 1])
                score_type = "Predictive Anomaly Score"
            else:
                raw_score = float(model.decision_function(sample_features)[0])
                score_type = "Decision Boundary Score"

            is_flagged = raw_score >= threshold

            st.markdown("---")

            # Prediction Output Card & Score Gauge
            st.markdown("### 📊 Model Inference & Threshold Comparison")

            i_col1, i_col2, i_col3, i_col4 = st.columns(4)
            with i_col1:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Selected Model</div>
                        <div class="kpi-value" style="font-size:1.15rem; color:#a78bfa;">{best_model_name}</div>
                        <div class="kpi-sub">Pipeline with Imputer & Scaler</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with i_col2:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Raw Model Score</div>
                        <div class="kpi-value" style="color:{'#ef4444' if is_flagged else '#10b981'}; font-size:1.25rem;">
                            {raw_score:.6f}
                        </div>
                        <div class="kpi-sub">{score_type}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with i_col3:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Decision Threshold</div>
                        <div class="kpi-value" style="color:#fcd34d; font-size:1.25rem;">
                            {threshold:.6f}
                        </div>
                        <div class="kpi-sub">Validation-Optimized Cutoff</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with i_col4:
                proxy_str = (
                    '<span style="color:#ef4444; font-weight:700;">1 (Proxy Anomaly)</span>'
                    if actual_proxy_label == 1
                    else '<span style="color:#10b981; font-weight:700;">0 (Normal Regime)</span>'
                )
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Telemetry Timestamp</div>
                        <div class="kpi-value" style="font-size:0.95rem; color:#e2e8f0;">
                            {str(current_time)[:19]}
                        </div>
                        <div class="kpi-sub">Ground Truth Proxy: {proxy_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Render Status Banner
            render_prediction_status_card(is_flagged, raw_score, threshold, best_model_name)

            # Score Gauge Chart
            st.markdown("#### Score vs Threshold Gauge")
            fig_gauge = create_score_gauge(raw_score, threshold, is_flagged)
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("---")

            # Section: Sensor Features Inspection Table
            st.markdown("### 🔬 Sensor Feature Breakdown for Selected Window")
            st.markdown(
                '<p style="color:#94a3b8; font-size:0.86rem; margin-top:-6px;">'
                'Detailed breakdown of aggregated 5-minute statistical features (Mean, Std Dev, Min, Max). '
                'Values deviating significantly (|Z| &gt; 2.5) from dataset medians are flagged for engineering inspection.'
                '</p>',
                unsafe_allow_html=True,
            )

            # Group features by physical sensor channel
            feature_series = sample_features.iloc[0]
            feature_names = list(X.columns)

            # Extract distinct sensor names
            sensor_channels = []
            for col in feature_names:
                base = col.rsplit("_", 1)[0]
                if base not in sensor_channels and base != "Unnamed: 0":
                    sensor_channels.append(base)

            # Compute median baselines across dataset
            @st.cache_data(show_spinner=False)
            def compute_sensor_baselines(_sample_df):
                medians = _sample_df.median()
                mads = _sample_df.sub(medians).abs().median().replace(0, np.nan).fillna(_sample_df.std().replace(0, 1))
                return medians, mads

            medians, mads = compute_sensor_baselines(X)

            rows = []
            for s in sensor_channels:
                mean_col = f"{s}_mean"
                std_col = f"{s}_std"
                min_col = f"{s}_min"
                max_col = f"{s}_max"

                mean_val = feature_series.get(mean_col, np.nan)
                std_val = feature_series.get(std_col, np.nan)
                min_val = feature_series.get(min_col, np.nan)
                max_val = feature_series.get(max_col, np.nan)

                # Compute Z-score deviation against median
                med = medians.get(mean_col, 0)
                mad = mads.get(mean_col, 1)
                z_score = abs(mean_val - med) / mad if mad > 0 else 0.0

                s_info = SENSOR_INFO.get(s, {"name": s, "unit": "—", "category": "General"})

                deviation_badge = "Normal"
                if z_score > 3.0:
                    deviation_badge = "High Deviation (|Z|>3)"
                elif z_score > 2.0:
                    deviation_badge = "Elevated (|Z|>2)"

                rows.append({
                    "Channel": s,
                    "Description": s_info["name"],
                    "Category": s_info["category"],
                    "Unit": s_info["unit"],
                    "Mean": round(mean_val, 4),
                    "Std Dev": round(std_val, 4),
                    "Min": round(min_val, 4),
                    "Max": round(max_val, 4),
                    "Baseline Status": deviation_badge,
                })

            df_features_summary = pd.DataFrame(rows)

            # Category filter
            cat_filter = st.selectbox(
                "Filter Sensor Category",
                options=["All Categories", "Pneumatic", "Thermal & Electrical", "Valves & Controls"],
            )

            if cat_filter != "All Categories":
                df_filtered = df_features_summary[df_features_summary["Category"] == cat_filter]
            else:
                df_filtered = df_features_summary

            st.dataframe(df_filtered, use_container_width=True, hide_index=True)

            st.caption(
                "Note: Baseline status indicates univariate deviation from the dataset-wide median. "
                "The machine learning model combines all 64 multivariate dimensions simultaneously to assign the final anomaly score."
            )

    except Exception as exc:
        st.error(f"Error during Predict & Inspect: {exc}")


# =========================================================
# PAGE 4: PROJECT GUIDE
# =========================================================
elif page == "Project Guide":
    render_page_header(
        title="Project Guide — Technical & Academic Reference",
        subtitle="Comprehensive documentation of system architecture, mathematical methodologies, classifier benchmarks, and viva defense reference for Early Failure Prediction AI.",
        badge_text="ACADEMIC & SYSTEM SPECIFICATION",
    )

    g_tab1, g_tab2, g_tab3, g_tab4 = st.tabs([
        "System Architecture",
        "Pipeline & Methodology",
        "The 5 Classifiers",
        "Viva & Academic Defense",
    ])

    with g_tab1:
        st.markdown("### 🏗️ Application & Engineering Architecture")
        st.markdown(
            """
            The **Early Failure Prediction AI** system is designed for high-throughput industrial telemetry analysis, 
            transforming 1.5 million raw compressor sensor measurements into actionable predictive maintenance intelligence.
            """
        )

        st.markdown(
            """
            <div class="content-card">
                <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.7;">
                    <div style="font-weight:700; color:#f8fafc; font-size:1rem; margin-bottom:8px;">End-to-End Dataflow</div>
                    1. <b>Data Ingestion</b>: 208 MB CSV file containing 1,516,948 high-frequency records across 16 sensor channels.<br>
                    2. <b>Temporal Aggregation</b>: Continuous 5-minute resampling into 50,782 discrete feature windows.<br>
                    3. <b>Feature Extraction</b>: Computation of 4 statistical moments (Mean, Standard Deviation, Minimum, Maximum) per channel yielding 64 input dimensions.<br>
                    4. <b>Proxy Anomaly Labeling</b>: Multivariate Median Absolute Deviation (MAD) identifying extreme operational regimes at the 98th percentile.<br>
                    5. <b>Split & Scaling</b>: Stratified 80/20 train/test partition, with internal validation split for threshold tuning and median imputation + standard scaling.<br>
                    6. <b>Comparative Benchmark</b>: Training and evaluating KNN, SVM, Gaussian Naive Bayes, Logistic Regression, and Decision Tree.<br>
                    7. <b>Interactive Dashboard</b>: Streamlit frontend powered by fast Parquet caching and Plotly dark telemetry visualizations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Technology Stack")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Frontend / UI", "Streamlit 1.65")
        s2.metric("ML Engine", "Scikit-Learn 1.7")
        s3.metric("Data Engine", "Pandas & PyArrow")
        s4.metric("Visualization", "Plotly 7.1")

    with g_tab2:
        st.markdown("### ⚙️ Feature Pipeline & Proxy Labeling")

        st.markdown(
            """
            #### 1. Why 5-Minute Window Aggregation?
            Raw sensor logs are sampled every 10 seconds, generating high-frequency fluctuations, mechanical vibration noise, 
            and massive memory overhead (~1.5 million rows). Aggregating into 5-minute temporal windows:
            - Suppresses transient sensor noise while preserving macroscopic thermodynamic and electrical trends.
            - Extracts distribution dynamics: **Mean** (central operational level), **Std Dev** (instability/vibration), **Min/Max** (peak excursions).
            - Compresses 1.5 million raw timestamps into **50,782** robust, analytically tractable feature vectors.
            """
        )

        st.markdown(
            """
            #### 2. Multivariate Proxy Anomaly Formulation
            In real-world field telemetry, component teardown failure logs are rarely recorded alongside raw SCADA streams. 
            To benchmark classifiers rigorously without fabricating ground truth:
            - For each sensor feature $X_i$, compute the robust deviation score:
            $$Z_i = \\frac{|X_i - \\text{median}(X_i)|}{\\text{MAD}(X_i)}$$
            - Aggregate across all multivariate dimensions: $\\text{Score} = \\frac{1}{N}\\sum Z_i$
            - Label the top 2% most deviant operational regimes ($Q_{0.98}$) as anomalous proxy windows ($y = 1$), leaving 98% as nominal ($y = 0$).
            """
        )

        st.markdown(
            """
            #### 3. Threshold Tuning Without Data Leakage
            Standard classification applies a default threshold of 0.5. In severe class imbalance (2% anomaly ratio), 0.5 leads to suboptimal F1.
            - We split training data (80%) into fit (64%) and validation (16%) sets.
            - Decision thresholds are calibrated **strictly on the validation set** to maximize $F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$.
            - The selected threshold is then evaluated on the untouched 20% test partition, preventing data leakage.
            """
        )

    with g_tab3:
        st.markdown("### 🧠 The Five Machine Learning Classifiers")

        st.markdown(
            """
            The project benchmarks five foundational machine learning classifiers specified for academic evaluation:
            """
        )

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(
                """
                <div class="content-card">
                    <div style="font-weight:700; color:#c4b5fd; margin-bottom:4px;">1. K-Nearest Neighbors (KNN)</div>
                    <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.5;">
                        • Non-parametric instance-based learner.<br>
                        • Configured with <i>k = 7</i> and distance-weighted voting.<br>
                        • Sensitive to feature scale; relies on median imputation and standard scaling.<br>
                        • Test F1: <b>0.9418</b> | PR-AUC: <b>0.9787</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="content-card">
                    <div style="font-weight:700; color:#c4b5fd; margin-bottom:4px;">2. Support Vector Machine (SVM)</div>
                    <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.5;">
                        • Maximum-margin separating hyperplane with Radial Basis Function (RBF) kernel.<br>
                        • Balanced class weights to counteract 98:2 imbalance.<br>
                        • Effective at capturing non-linear boundary manifolds in 64-dimensional space.<br>
                        • Test F1: <b>0.9505</b> | PR-AUC: <b>0.9826</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="content-card">
                    <div style="font-weight:700; color:#c4b5fd; margin-bottom:4px;">3. Gaussian Naive Bayes</div>
                    <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.5;">
                        • Probabilistic classifier assuming conditional feature independence.<br>
                        • Extremely fast, but the conditional independence assumption is violated by correlated compressor sensors.<br>
                        • High recall (0.9951) but low precision (0.4612) due to false alarms.<br>
                        • Test F1: <b>0.6303</b> | PR-AUC: <b>0.4590</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                """
                <div class="content-card" style="border-color:rgba(139,92,246,0.5);">
                    <div style="font-weight:700; color:#a78bfa; margin-bottom:4px;">4. Logistic Regression (Top Performer)</div>
                    <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.5;">
                        • Regularized linear log-odds classifier with balanced class weighting.<br>
                        • Combined with validation threshold tuning (t = 0.9481).<br>
                        • Provides optimal balance between precision (0.9466) and recall (0.9606).<br>
                        • Highest validation F1 (<b>0.9632</b>) and test F1 (<b>0.9535</b>) | PR-AUC: <b>0.9798</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="content-card">
                    <div style="font-weight:700; color:#c4b5fd; margin-bottom:4px;">5. Decision Tree</div>
                    <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.5;">
                        • Non-parametric hierarchical rule-based partitioner.<br>
                        • Regularized with max depth = 6 and min samples per leaf = 10 to mitigate overfitting.<br>
                        • Highly interpretable split rules, but susceptible to variance on continuous sensor data.<br>
                        • Test F1: <b>0.9183</b> | PR-AUC: <b>0.8791</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div style="background:#0f172a; border:1px solid rgba(148,163,184,0.14); border-radius:8px; padding:12px; font-size:0.8rem; color:#94a3b8;">
                    <b style="color:#f8fafc;">Why Random Forest is Excluded</b><br>
                    Per academic project requirements and syllabus guidelines, Random Forest was intentionally excluded to focus rigorously on comparing these five foundational classifiers.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with g_tab4:
        st.markdown("### 🎓 Academic Viva & Presentation Defense")

        st.markdown(
            """
            Key technical questions and concise defenses for academic viva voce examinations:
            """
        )

        qa_items = [
            (
                "Q1: What is the core problem addressed by predictive maintenance in compressors?",
                "Unexpected mechanical failure in metro air compressors disrupts braking, suspension, and door pneumatics, causing severe service delays. Predictive maintenance uses continuous multi-sensor telemetry to flag abnormal operational degradation before catastrophic failure occurs.",
            ),
            (
                "Q2: Why is raw accuracy a flawed metric for this dataset?",
                "The dataset exhibits severe class imbalance (98% normal windows vs 2% anomaly windows). A trivial dummy classifier predicting 'Normal' 100% of the time achieves 98% accuracy while failing to detect a single anomaly. Therefore, F1-Score, Balanced Accuracy, and PR-AUC are required.",
            ),
            (
                "Q3: Why are these labels called 'Proxy Anomaly Labels' rather than verified failures?",
                "The UCI MetroPT-3 dataset contains continuous physical sensor logs but does not include certified ground-truth component teardown logs. Describing statistical deviation as proven real-world mechanical failure would be scientifically inaccurate. Proxy labels provide a rigorous, transparent benchmark.",
            ),
            (
                "Q4: Why was a stratified split chosen instead of a chronological split in this implementation?",
                "Because anomalous states constitute only 2% of total operational time, a naive chronological split could place almost all anomaly windows into one segment, leaving another segment with zero or near-zero positive examples. Stratified random splitting preserves class proportions across train, validation, and test sets.",
            ),
            (
                "Q5: How does decision threshold tuning improve classifier performance?",
                "Standard classification defaults to a 0.5 decision boundary. In highly skewed distributions, 0.5 often yields high false negatives or false alarms. By optimizing the decision threshold on a validation set to maximize F1, the model balances precision and recall without contaminating test data.",
            ),
            (
                "Q6: Why did Logistic Regression outperform more complex models like Decision Tree and Naive Bayes?",
                "After robust multivariate standardization, sensor deviations exhibit strong linear log-odds separability. Decision trees suffered from axis-aligned partitioning variance, and Gaussian Naive Bayes failed due to correlated sensor pairs (e.g. pressure and motor current). Logistic Regression with balanced weighting and tuned threshold achieved the most stable generalization.",
            ),
        ]

        for question, answer in qa_items:
            with st.expander(f"📌 {question}"):
                st.markdown(f'<p style="color:#cbd5e1; font-size:0.88rem; line-height:1.6;">{answer}</p>', unsafe_allow_html=True)
