from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Custom Dark Industrial Analytics CSS
CUSTOM_CSS = """
<style>
/* Base Theme & Background */
.stApp {
    background-color: #090d16;
    background-image: 
        radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.05) 0px, transparent 50%);
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Container Spacing */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1480px;
}

/* Headings */
h1 {
    color: #f8fafc;
    font-weight: 700;
    font-size: 1.85rem;
    letter-spacing: -0.025em;
    margin-bottom: 0.35rem;
}
h2, h3, h4 {
    color: #f1f5f9;
    font-weight: 600;
    letter-spacing: -0.015em;
}

/* Section Header Container */
.page-header-card {
    background: #0f172a;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.4);
}
.page-title {
    color: #f8fafc;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.page-subtitle {
    color: #94a3b8;
    font-size: 0.93rem;
    margin-top: 5px;
    margin-bottom: 0;
    line-height: 1.45;
}

/* KPI Card */
.kpi-card {
    background: #0f172a;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover {
    border-color: rgba(139, 92, 246, 0.38);
    transform: translateY(-2px);
}
.kpi-label {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.kpi-value {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 4px;
}
.kpi-sub {
    color: #64748b;
    font-size: 0.76rem;
}

/* Badges & Pills */
.badge {
    display: inline-block;
    padding: 3px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 6px;
    letter-spacing: 0.03em;
}
.badge-purple {
    background: rgba(139, 92, 246, 0.18);
    color: #c4b5fd;
    border: 1px solid rgba(139, 92, 246, 0.35);
}
.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #67e8f9;
    border: 1px solid rgba(6, 182, 212, 0.32);
}
.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.32);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.32);
}
.badge-red {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.32);
}

/* Status Cards for Prediction Results */
.status-card-normal {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 10px;
    margin-bottom: 15px;
}
.status-card-flagged {
    background: rgba(239, 68, 68, 0.09);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 10px;
    margin-bottom: 15px;
}
.status-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.status-desc {
    font-size: 0.85rem;
    color: #cbd5e1;
    margin: 0;
    line-height: 1.4;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0b1120;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}
.sidebar-brand-card {
    padding: 8px 4px 16px 4px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    margin-bottom: 16px;
}
.sidebar-title {
    color: #f8fafc;
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 0;
}
.sidebar-sub {
    color: #94a3b8;
    font-size: 0.74rem;
    margin-top: 3px;
    margin-bottom: 8px;
}
.sidebar-info-card {
    background: #0f172a;
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 8px;
    padding: 12px;
    margin-top: 16px;
    margin-bottom: 14px;
}
.sidebar-info-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.77rem;
    padding: 4px 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.06);
}
.sidebar-info-item:last-child {
    border-bottom: none;
}
.sidebar-info-label {
    color: #94a3b8;
}
.sidebar-info-val {
    color: #e2e8f0;
    font-weight: 600;
}
.sidebar-footer {
    padding-top: 14px;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    font-size: 0.72rem;
    color: #64748b;
    line-height: 1.4;
}

/* Standard Content Card */
.content-card {
    background: #0f172a;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 1.25rem;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid rgba(167, 139, 250, 0.4);
    border-radius: 8px;
    padding: 0.45rem 1.1rem;
    transition: all 0.15s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    border-color: rgba(167, 139, 250, 0.7);
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35);
}

/* Table and Dataframe container */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 8px;
    background: #0f172a;
}

/* Metrics in columns */
[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.4rem !important;
    font-weight: 700;
}
</style>
"""

# Sensor Channel Descriptions and Units
SENSOR_INFO = {
    "TP2": {"name": "Compressor Temp 2", "unit": "bar / °C", "category": "Pneumatic"},
    "TP3": {"name": "Compressor Temp 3", "unit": "bar / °C", "category": "Pneumatic"},
    "H1": {"name": "High-Pressure Discharge", "unit": "°C", "category": "Thermal & Electrical"},
    "DV_pressure": {"name": "Drain Valve Pressure", "unit": "bar", "category": "Pneumatic"},
    "Reservoirs": {"name": "Air Reservoirs Pressure", "unit": "bar", "category": "Pneumatic"},
    "Oil_temperature": {"name": "Compressor Oil Temp", "unit": "°C", "category": "Thermal & Electrical"},
    "Motor_current": {"name": "Motor Electric Current", "unit": "A", "category": "Thermal & Electrical"},
    "COMP": {"name": "Compressor Active State", "unit": "binary", "category": "Valves & Controls"},
    "DV_eletric": {"name": "Electric Drain Valve State", "unit": "binary", "category": "Valves & Controls"},
    "Towers": {"name": "Air Dryer Towers Active", "unit": "binary", "category": "Valves & Controls"},
    "MPG": {"name": "Motor Protection Group", "unit": "signal", "category": "Valves & Controls"},
    "LPS": {"name": "Low Pressure Switch", "unit": "binary", "category": "Valves & Controls"},
    "Pressure_switch": {"name": "Pressure Switch Trigger", "unit": "binary", "category": "Valves & Controls"},
    "Oil_level": {"name": "Oil Level Normal Status", "unit": "binary", "category": "Valves & Controls"},
    "Caudal_impulses": {"name": "Flow Impulse Counter", "unit": "pulses", "category": "Valves & Controls"},
}


def render_sidebar(dataset_meta, trained, report_info):
    """Render the redesigned sidebar with branding, navigation, and system status."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand-card">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.45rem;">⚙️</span>
                <span class="sidebar-title">EARLY FAILURE AI</span>
            </div>
            <div class="sidebar-sub">Predictive Maintenance Intelligence</div>
            <span class="badge badge-purple">RESEARCH PROTOTYPE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Model Lab", "Predict & Inspect", "Project Guide"],
        index=0,
        label_visibility="collapsed",
    )

    # Telemetry dataset status check
    ds_status = (
        f'<span class="badge badge-green">Ready ({dataset_meta["size_mb"]} MB)</span>'
        if dataset_meta["exists"]
        else '<span class="badge badge-amber">Missing</span>'
    )

    model_status = (
        f'<span class="badge badge-purple">{report_info.get("best_model", "Trained")}</span>'
        if trained
        else '<span class="badge badge-amber">Pending</span>'
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-info-card">
            <div style="font-size:0.75rem; font-weight:700; color:#cbd5e1; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.04em;">System Status</div>
            <div class="sidebar-info-item">
                <span class="sidebar-info-label">Dataset</span>
                <span class="sidebar-info-val">MetroPT-3</span>
            </div>
            <div class="sidebar-info-item">
                <span class="sidebar-info-label">Raw Telemetry</span>
                {ds_status}
            </div>
            <div class="sidebar-info-item">
                <span class="sidebar-info-label">Classifiers</span>
                <span class="sidebar-info-val">5 Models</span>
            </div>
            <div class="sidebar-info-item">
                <span class="sidebar-info-label">Model Pipeline</span>
                {model_status}
            </div>
            <div class="sidebar-info-item">
                <span class="sidebar-info-label">Validation</span>
                <span class="sidebar-info-val">Stratified (80/20)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            <div style="font-weight:600; color:#94a3b8; margin-bottom:3px;">Version 2.4.0 (Industrial AI)</div>
            UCI MetroPT-3 compressor research. Multivariate statistical proxy anomaly detection. Not for certified physical shutdown.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return page


def render_page_header(title, subtitle, badge_text="METROPT-3 COMPRESSOR"):
    """Render a clean, high-contrast industrial section header."""
    st.markdown(
        f"""
        <div class="page-header-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                <div>
                    <h2 class="page-title">{title}</h2>
                    <p class="page-subtitle">{subtitle}</p>
                </div>
                <div>
                    <span class="badge badge-cyan">{badge_text}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_telemetry_chart(df_telemetry, selected_channels):
    """
    Build an interactive multi-channel time-series visualization using Plotly.
    Optimized with dark theme styling and clean layout.
    """
    fig = go.Figure()

    colors = ["#a78bfa", "#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#818cf8"]

    for i, col in enumerate(selected_channels):
        if col in df_telemetry.columns:
            color = colors[i % len(colors)]
            info = SENSOR_INFO.get(col, {"name": col, "unit": ""})
            unit_str = f" ({info['unit']})" if info["unit"] else ""
            fig.add_trace(
                go.Scatter(
                    x=df_telemetry.index,
                    y=df_telemetry[col],
                    mode="lines",
                    name=f"{info['name']}{unit_str}",
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"<b>{info['name']}</b>: %{{y:.2f}}{unit_str}<br>Time: %{{x}}<extra></extra>",
                )
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11, color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.6)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.08)",
            color="#94a3b8",
            title="Timestamp",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.08)",
            color="#94a3b8",
            title="Sensor Reading",
        ),
        hovermode="x unified",
    )
    return fig


def create_model_comparison_bar(df_models, metric="f1"):
    """Render a sorted comparative bar chart across the five classifiers."""
    df_sorted = df_models.sort_values(metric, ascending=True)

    metric_labels = {
        "f1": "F1-Score (Test)",
        "pr_auc": "PR-AUC / Average Precision",
        "precision": "Precision",
        "recall": "Recall",
        "balanced_accuracy": "Balanced Accuracy",
        "accuracy": "Overall Accuracy",
    }
    label = metric_labels.get(metric, metric.replace("_", " ").title())

    # Muted industrial palette with purple highlight for top model
    top_model = df_sorted.iloc[-1]["model"]
    bar_colors = [
        "#8b5cf6" if m == top_model else "#334155" for m in df_sorted["model"]
    ]

    fig = go.Figure(
        go.Bar(
            y=df_sorted["model"],
            x=df_sorted[metric],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="#a78bfa", width=1),
            ),
            text=[f"{v:.4f}" for v in df_sorted[metric]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>" + label + ": %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        height=320,
        margin=dict(l=110, r=45, t=20, b=40),
        xaxis=dict(
            range=[0, min(1.05, max(df_sorted[metric].max() * 1.15, 0.1))],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.08)",
            color="#94a3b8",
            title=label,
        ),
        yaxis=dict(
            color="#e2e8f0",
            autorange=True,
        ),
    )
    return fig


def create_tradeoff_chart(df_models):
    """Render an interactive Precision vs Recall scatter plot."""
    fig = px.scatter(
        df_models,
        x="recall",
        y="precision",
        text="model",
        size=[18] * len(df_models),
        color="f1",
        color_continuous_scale="Purples",
        template="plotly_dark",
        hover_data={"balanced_accuracy": ":.3f", "pr_auc": ":.3f"},
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1.5, color="#cbd5e1")),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        height=340,
        margin=dict(l=40, r=40, t=25, b=40),
        xaxis=dict(
            title="Recall (Sensitivity)",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.08)",
            color="#94a3b8",
            range=[min(0.4, df_models["recall"].min() - 0.05), 1.05],
        ),
        yaxis=dict(
            title="Precision (PPV)",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.08)",
            color="#94a3b8",
            range=[min(0.4, df_models["precision"].min() - 0.05), 1.05],
        ),
        coloraxis_colorbar=dict(
            title="F1",
            thickness=12,
            len=0.7,
        ),
    )
    return fig


def create_confusion_matrix_chart(cm, model_name):
    """Render a confusion matrix heatmap with counts and percentage rates."""
    total = np.sum(cm)
    tn, fp = cm[0]
    fn, tp = cm[1]

    annotations = [
        [f"<b>TN: {tn:,}</b><br>({tn/total*100:.1f}%)", f"<b>FP: {fp:,}</b><br>({fp/total*100:.2f}%)"],
        [f"<b>FN: {fn:,}</b><br>({fn/total*100:.2f}%)", f"<b>TP: {tp:,}</b><br>({tp/total*100:.1f}%)"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Anomaly"],
            text=annotations,
            texttemplate="%{text}",
            colorscale=[[0, "#0f172a"], [0.2, "#1e293b"], [0.6, "#6d28d9"], [1, "#8b5cf6"]],
            showscale=False,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        height=300,
        margin=dict(l=80, r=20, t=30, b=40),
        xaxis=dict(color="#cbd5e1", side="bottom"),
        yaxis=dict(color="#cbd5e1", autorange="reversed"),
    )
    return fig


def create_score_gauge(score, threshold, is_flagged):
    """
    Render a horizontal bullet-style gauge comparing the model score against the threshold.
    Correctly handles very small scores without misleading visual distortion.
    """
    fig = go.Figure()

    # Normal zone [0 to threshold]
    fig.add_trace(
        go.Bar(
            y=["Score"],
            x=[threshold],
            orientation="h",
            marker=dict(color="rgba(16, 185, 129, 0.22)"),
            name="Normal Operational Range",
            hoverinfo="skip",
        )
    )

    # Anomaly zone [threshold to 1.0]
    fig.add_trace(
        go.Bar(
            y=["Score"],
            x=[max(0.0, 1.0 - threshold)],
            base=[threshold],
            orientation="h",
            marker=dict(color="rgba(239, 68, 68, 0.25)"),
            name="Anomaly Flag Zone",
            hoverinfo="skip",
        )
    )

    # Indicator bar for current score
    bar_color = "#ef4444" if is_flagged else "#10b981"
    fig.add_trace(
        go.Bar(
            y=["Score"],
            x=[min(max(score, 0.0), 1.0)],
            orientation="h",
            width=0.35,
            marker=dict(color=bar_color, line=dict(color="#ffffff", width=1)),
            name=f"Current Window Score ({score:.6f})",
            hovertemplate="Model Raw Score: %{x:.6f}<extra></extra>",
        )
    )

    # Decision Threshold vertical marker line
    fig.add_vline(
        x=threshold,
        line_width=2.5,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Threshold ({threshold:.4f})",
        annotation_position="top right",
        annotation_font=dict(color="#fcd34d", size=11),
    )

    fig.update_layout(
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=140,
        margin=dict(l=10, r=20, t=30, b=30),
        xaxis=dict(
            range=[0, 1.02],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.1)",
            tickvals=[0, 0.25, 0.5, 0.75, round(threshold, 2), 1.0],
            color="#94a3b8",
            title="Score Scale [0.0 - 1.0]",
        ),
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )
    return fig


def render_prediction_status_card(is_flagged, score, threshold, model_name):
    """Render a restrained, professional industrial status card for prediction outcome."""
    if is_flagged:
        st.markdown(
            f"""
            <div class="status-card-flagged">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="status-title" style="color:#fca5a5;">⚠️ FLAGGED FOR MAINTENANCE INSPECTION</span>
                    <span class="badge badge-red">ANOMALY PROXY</span>
                </div>
                <p class="status-desc">
                    The raw score (<b>{score:.6f}</b>) exceeds the validated decision threshold (<b>{threshold:.4f}</b>) for <b>{model_name}</b>. 
                    Multivariate sensor features exhibit significant statistical deviation from normal operating baselines.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-card-normal">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="status-title" style="color:#6ee7b7;">✅ NORMAL OPERATIONAL REGIME</span>
                    <span class="badge badge-green">HEALTHY STATE</span>
                </div>
                <p class="status-desc">
                    The raw score (<b>{score:.6f}</b>) is safely below the validated decision threshold (<b>{threshold:.4f}</b>) for <b>{model_name}</b>. 
                    Compressor sensor telemetry aligns with typical nominal operational envelopes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
