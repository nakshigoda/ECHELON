"""
ECHELON — AI-Powered Predictive Maintenance Dashboard
Stage 2.5: Multi-Page Refinement

Pages:
1. Overview: Live Machine Monitoring (Simulated Sensor Feed from AI4I 2020)
2. Analyze Machine: Manual Input & On-Demand Inference
3. What-If Simulator: Scenario Comparison & Sensitivity Analysis
4. Model Insights: Technical ML Metrics, Confusion Matrix, ROC, Feature Importance
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import time

from src.prediction import predict, load_model

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="ECHELON | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — Light Industrial Aesthetic
# ============================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #FAF8F5;
    --bg-card: #FFFFFF;
    --bg-sidebar: #F0ECE7;
    --text-primary: #1A1A1A;
    --text-secondary: #5F5F5F;
    --text-tertiary: #8C8C8C;
    --accent: #C4662B;
    --accent-light: #F5E6D8;
    --accent-hover: #A8551F;
    --border: #E5E0DA;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.05);
    --green: #2D7D46;
    --green-bg: #EBF5EE;
    --amber: #C4880F;
    --amber-bg: #FDF4E4;
    --red: #B83A2E;
    --red-bg: #FCEAE8;
    --radius: 8px;
}

/* ===== GLOBAL ===== */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp > header { background-color: transparent !important; }

/* Hide default Streamlit elements */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', sans-serif !important;
}

/* Radio nav items */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    padding: 0.6rem 0.875rem !important;
    border-radius: 6px !important;
    transition: background-color 0.15s ease !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background-color: rgba(196, 102, 43, 0.07) !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div > p {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.005em !important;
}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}
h1 { letter-spacing: -0.03em !important; }
h2 { letter-spacing: -0.025em !important; }
h3 { letter-spacing: -0.02em !important; }

/* ===== INPUTS ===== */
.stNumberInput input, .stTextInput input {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-card) !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

.stSelectbox [data-baseweb="select"] > div {
    border-color: var(--border) !important;
    border-radius: 6px !important;
    background-color: var(--bg-card) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.15rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
    border: 1px solid var(--border) !important;
    background-color: #FFFFFF !important;
    color: var(--text-primary) !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stButton > button[kind="primary"], div.row-widget.stButton > button[kind="primary"] {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover, div.row-widget.stButton > button[kind="primary"]:hover {
    background-color: var(--accent-hover) !important;
    color: #FFFFFF !important;
}

/* ===== SLIDER ===== */
.stSlider label p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
}

/* ===== DIVIDERS ===== */
hr {
    border-color: var(--border) !important;
    opacity: 0.6 !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: var(--text-secondary) !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
"""


def inject_styles():
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# ============================================================
# Data & Model Artifact Caching
# ============================================================

@st.cache_resource
def get_model_artifacts():
    """Load trained model and preprocessing artifacts once."""
    return load_model()


@st.cache_data
def get_evaluation_results():
    """Load pre-computed evaluation results."""
    with open("models/evaluation_results.json", "r") as f:
        return json.load(f)


@st.cache_data
def get_dataset():
    """Load AI4I 2020 dataset for simulated sensor streaming."""
    return pd.read_csv("data/ai4i2020.csv")


# ============================================================
# Helpers & Business Logic
# ============================================================

def health_score(failure_prob):
    return round((1.0 - failure_prob) * 100, 1)


def risk_level(failure_prob):
    if failure_prob < 0.10:
        return "LOW", "var(--green)", "var(--green-bg)"
    elif failure_prob < 0.40:
        return "MODERATE", "var(--amber)", "var(--amber-bg)"
    elif failure_prob < 0.70:
        return "HIGH", "var(--red)", "var(--red-bg)"
    else:
        return "CRITICAL", "var(--red)", "var(--red-bg)"


def machine_status_label(prediction):
    if prediction == 0:
        return "Healthy", "var(--green)", "var(--green-bg)"
    return "Failure Predicted", "var(--red)", "var(--red-bg)"


def format_pct(value):
    if value < 0.0001 and value > 0:
        return "<0.01%"
    elif value == 0:
        return "0.0%"
    else:
        return f"{value * 100:.1f}%"


def maintenance_recommendations(input_data, failure_prob):
    """Rule-based maintenance recommendations (not ML model)."""
    recs = []
    tw = input_data.get("Tool wear [min]", input_data.get("Tool wear (min)", 0))
    torque = input_data.get("Torque [Nm]", input_data.get("Torque (Nm)", 0))
    air_t = input_data.get("Air temperature [K]", input_data.get("Air temperature (K)", 0))
    proc_t = input_data.get("Process temperature [K]", input_data.get("Process temperature (K)", 0))
    rpm = input_data.get("Rotational speed [rpm]", input_data.get("Rotational speed (rpm)", 0))

    if failure_prob >= 0.70:
        recs.append(("critical", "CRITICAL RISK: Immediate equipment inspection and maintenance intervention required to prevent failure."))
    elif failure_prob >= 0.40:
        recs.append(("warning", "ELEVATED RISK: Operating telemetry indicates heightened stress. Schedule preemptive service."))

    if tw >= 200:
        recs.append(("critical", f"Tool replacement advised: Tool wear ({tw} min) has reached critical wear boundary."))
    elif tw >= 140:
        recs.append(("warning", f"Tool inspection recommended: Tool wear ({tw} min) is approaching replacement threshold."))

    if torque >= 60:
        recs.append(("warning", f"Drive / Spindle check recommended: High torque ({torque:.1f} Nm) suggests heavy mechanical resistance."))
    elif torque <= 15 and rpm >= 2200:
        recs.append(("warning", f"Spindle overspeed warning: High RPM ({rpm} rpm) with low torque ({torque:.1f} Nm) indicates potential power transmission issue."))

    temp_diff = proc_t - air_t
    if temp_diff < 8.6:
        recs.append(("warning", f"Thermal dissipation anomaly: Temperature difference ({temp_diff:.1f} K) is below expected process delta."))
    elif air_t >= 303 or proc_t >= 313:
        recs.append(("warning", f"Thermal condition elevated: Temperature ({air_t:.1f} K air / {proc_t:.1f} K process) is above standard limits."))

    if not recs or (failure_prob < 0.10 and not [r for r in recs if r[0] != "healthy"]):
        recs = [("healthy", "All operating parameters within nominal limits. Routine monitoring schedule sufficient.")]

    return recs


INPUT_RANGES = {
    "Air temperature [K]": (295.3, 304.5),
    "Process temperature [K]": (305.7, 313.8),
    "Rotational speed [rpm]": (1168, 2886),
    "Torque [Nm]": (3.8, 76.6),
    "Tool wear [min]": (0, 253),
}


def validate_input(key, value):
    """Check if value is within physical range."""
    if key in INPUT_RANGES:
        lo, hi = INPUT_RANGES[key]
        if value < lo or value > hi:
            return False, f"Typical range: {lo}–{hi}"
    return True, ""


# ============================================================
# UI Component Helpers
# ============================================================

def render_header():
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center;
                padding:0.75rem 0 1rem 0; margin-bottom:0.25rem;
                border-bottom:1px solid #E5E0DA;">
        <div>
            <div style="font-size:1.65rem; font-weight:800; letter-spacing:-0.04em;
                        color:#1A1A1A; line-height:1;">ECHELON</div>
            <div style="font-size:0.8rem; color:#5F5F5F; margin-top:3px;
                        letter-spacing:0.03em;">Predict before failure.</div>
        </div>
        <div style="display:inline-flex; align-items:center; gap:6px;
                    background:#EBF5EE; padding:5px 12px; border-radius:4px;">
            <span style="width:7px;height:7px;background:#2D7D46;border-radius:50%;
                         display:inline-block;"></span>
            <span style="font-size:0.65rem;font-weight:700;color:#2D7D46;
                         letter-spacing:0.08em;">SYSTEM ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, sublabel="", accent=False):
    border_left = "border-left:3px solid var(--accent);" if accent else ""
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:8px;
                padding:1.15rem 1.25rem; {border_left}
                box-shadow:0 1px 2px rgba(0,0,0,0.03);">
        <div style="font-size:0.68rem; font-weight:600; color:#8C8C8C;
                    letter-spacing:0.07em; text-transform:uppercase;">{label}</div>
        <div style="font-size:1.7rem; font-weight:700; color:#1A1A1A;
                    letter-spacing:-0.02em; margin-top:4px; line-height:1.1;">{value}</div>
        <div style="font-size:0.72rem; color:#8C8C8C; margin-top:3px;">{sublabel}</div>
    </div>
    """, unsafe_allow_html=True)


def sensor_card(label, value, unit):
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:6px;
                padding:0.85rem 1rem; box-shadow:0 1px 2px rgba(0,0,0,0.02); height:100%;">
        <div style="font-size:0.65rem; font-weight:600; color:#8C8C8C;
                    letter-spacing:0.06em; text-transform:uppercase;">{label}</div>
        <div style="margin-top:4px;">
            <span style="font-size:1.3rem; font-weight:700; color:#1A1A1A;
                         letter-spacing:-0.01em;">{value}</span>
            <span style="font-size:0.75rem; color:#8C8C8C; margin-left:3px;">{unit}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle=""):
    sub = f'<div style="font-size:0.85rem; color:#5F5F5F; margin-top:2px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-top:1.75rem; margin-bottom:0.75rem;">
        <div style="font-size:1.15rem; font-weight:700; color:#1A1A1A;
                    letter-spacing:-0.015em;">{title}</div>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def rec_card(level, text):
    colors = {
        "critical": ("#B83A2E", "#FCEAE8", "⚠"),
        "warning": ("#C4880F", "#FDF4E4", "△"),
        "healthy": ("#2D7D46", "#EBF5EE", "✓"),
    }
    color, bg, icon = colors.get(level, colors["warning"])
    st.markdown(f"""
    <div style="background:{bg}; border-left:3px solid {color}; border-radius:0 6px 6px 0;
                padding:0.8rem 1rem; margin-bottom:0.5rem;">
        <span style="font-weight:600; color:{color}; margin-right:6px;">{icon}</span>
        <span style="font-size:0.84rem; color:#1A1A1A;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", color="#1A1A1A", size=12),
    plot_bgcolor="#FAF8F5",
    paper_bgcolor="#FFFFFF",
    margin=dict(l=50, r=30, t=40, b=40),
)


def styled_figure(fig, height=360):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#F0ECE7", zerolinecolor="#E5E0DA")
    fig.update_yaxes(gridcolor="#F0ECE7", zerolinecolor="#E5E0DA")
    return fig


# ============================================================
# PAGE 1 — Overview: Live Machine Monitoring (Simulated Sensor Feed)
# ============================================================

def page_overview():
    artifacts = get_model_artifacts()
    df = get_dataset()

    # Initialize feed state
    if "feed_idx" not in st.session_state:
        st.session_state.feed_idx = 0
    if "is_monitoring" not in st.session_state:
        st.session_state.is_monitoring = False
    if "feed_history" not in st.session_state:
        st.session_state.feed_history = []

    # Title & Simulated Sensor Feed Badge
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin:1.25rem 0 0.5rem 0; flex-wrap:wrap; gap:10px;">
        <div>
            <h1 style="font-size:1.9rem; font-weight:800; margin:0; line-height:1.15;
                       letter-spacing:-0.035em;">Live Machine Monitoring</h1>
            <p style="font-size:0.95rem; color:#5F5F5F; margin-top:4px; max-width:650px;">
                Real-time condition monitoring and automated failure risk detection.
            </p>
        </div>
        <div style="display:inline-flex; align-items:center; gap:7px;
                    background:#FDF4E4; border:1px solid #E5E0DA; padding:6px 14px; border-radius:6px;">
            <span style="width:8px;height:8px;background:#C4880F;border-radius:50%;
                         display:inline-block; animation: pulse 2s infinite;"></span>
            <span style="font-size:0.7rem;font-weight:700;color:#C4880F;
                         letter-spacing:0.07em;">● SIMULATED SENSOR FEED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prototype Disclaimer Note
    st.markdown("""
    <div style="background:#FAF8F5; border:1px solid #E5E0DA; border-radius:6px;
                padding:0.65rem 1rem; margin-bottom:1rem; font-size:0.8rem; color:#5F5F5F;">
        <b>Prototype Telemetry Stream:</b> Sensor telemetry is streamed row-by-row from the verified 
        <b>AI4I 2020 Predictive Maintenance Dataset</b> (10,000 recorded machine runs) to simulate an industrial IIoT environment.
    </div>
    """, unsafe_allow_html=True)

    # Feed Navigation Controls
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1.2, 1, 1, 1, 2.5])
    
    with ctrl_col1:
        if st.session_state.is_monitoring:
            if st.button("⏸  Pause", use_container_width=True):
                st.session_state.is_monitoring = False
                st.rerun()
        else:
            if st.button("▶  Start Monitoring", type="primary", use_container_width=True):
                st.session_state.is_monitoring = True
                st.rerun()

    with ctrl_col2:
        if st.button("⏭  Next Reading", use_container_width=True):
            st.session_state.feed_idx = (st.session_state.feed_idx + 1) % len(df)
            st.rerun()

    with ctrl_col3:
        if st.button("⏮  Prev Reading", use_container_width=True):
            st.session_state.feed_idx = max(0, st.session_state.feed_idx - 1)
            st.rerun()

    with ctrl_col4:
        if st.button("Reset Feed", use_container_width=True):
            st.session_state.feed_idx = 0
            st.session_state.feed_history = []
            st.session_state.is_monitoring = False
            st.rerun()

    with ctrl_col5:
        anomaly_options = {
            "Select Machine / Scenario...": st.session_state.feed_idx,
            "Row #1 — Normal (Product M14860)": 0,
            "Row #51 — Power Failure (Product L47230)": 50,
            "Row #70 — Overstrain Failure (Product L47249)": 69,
            "Row #78 — Tool Wear Failure (Product L47257)": 77,
            "Row #161 — Torque Overload (Product L47340)": 160,
            "Row #208 — Heat Dissipation Anomaly (Product M15067)": 207,
        }
        selected_anomaly = st.selectbox(
            "Quick Select Sample",
            options=list(anomaly_options.keys()),
            index=0,
            label_visibility="collapsed"
        )
        if selected_anomaly != "Select Machine / Scenario...":
            new_idx = anomaly_options[selected_anomaly]
            if new_idx != st.session_state.feed_idx:
                st.session_state.feed_idx = new_idx
                st.rerun()

    # Extract current row data
    curr_row = df.iloc[st.session_state.feed_idx]
    machine_udi = int(curr_row.get("UDI", st.session_state.feed_idx + 1))
    product_id = str(curr_row.get("Product ID", f"M{machine_udi:05d}"))
    type_code = str(curr_row["Type"])
    type_name_map = {"L": "Low Quality Variant (L)", "M": "Medium Quality Variant (M)", "H": "High Quality Variant (H)"}
    type_full = type_name_map.get(type_code, f"Variant {type_code}")

    input_data = {
        "Type": type_code,
        "Air temperature [K]": float(curr_row["Air temperature [K]"]),
        "Process temperature [K]": float(curr_row["Process temperature [K]"]),
        "Rotational speed [rpm]": int(curr_row["Rotational speed [rpm]"]),
        "Torque [Nm]": float(curr_row["Torque [Nm]"]),
        "Tool wear [min]": int(curr_row["Tool wear [min]"]),
    }

    # Run prediction
    result = predict(input_data, model_artifacts=artifacts)
    fp = result["failure_probability"]
    nfp = result["no_failure_probability"]
    hs = health_score(fp)
    rl, rl_color, rl_bg = risk_level(fp)
    ms, ms_color, ms_bg = machine_status_label(result["prediction"])

    # Update history in session_state
    current_entry = {
        "step": len(st.session_state.feed_history) + 1,
        "row_idx": st.session_state.feed_idx + 1,
        "product_id": product_id,
        "prob": fp * 100,
        "health": hs,
        "status": ms,
        "risk": rl,
    }

    # Avoid duplicate consecutive entries if index hasn't changed
    if not st.session_state.feed_history or st.session_state.feed_history[-1]["row_idx"] != current_entry["row_idx"]:
        st.session_state.feed_history.append(current_entry)
        if len(st.session_state.feed_history) > 40:
            st.session_state.feed_history.pop(0)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Machine Header Info
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:8px; padding:0.9rem 1.25rem;
                display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:1rem;">
        <div>
            <span style="font-size:0.68rem; font-weight:600; color:#8C8C8C; letter-spacing:0.06em; text-transform:uppercase;">MACHINE IDENTIFIER</span>
            <div style="font-size:1.25rem; font-weight:700; color:#1A1A1A; letter-spacing:-0.02em; margin-top:2px;">
                {product_id} <span style="font-size:0.85rem; font-weight:500; color:#8C8C8C;">(UDI #{machine_udi:,})</span>
            </div>
        </div>
        <div style="text-align:right;">
            <span style="font-size:0.68rem; font-weight:600; color:#8C8C8C; letter-spacing:0.06em; text-transform:uppercase;">FEED TELEMETRY POSITION</span>
            <div style="font-size:0.95rem; font-weight:600; color:#1A1A1A; margin-top:2px;">
                Row {st.session_state.feed_idx + 1:,} <span style="color:#8C8C8C;">/ {len(df):,}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Central Machine Health Card & Risk Assessment
    col_health, col_sensors = st.columns([2, 3])

    with col_health:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hs,
            number=dict(suffix="", font=dict(size=38, color="#1A1A1A", family="Inter")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#E5E0DA",
                          tickfont=dict(size=10, color="#8C8C8C")),
                bar=dict(color="#2D7D46" if hs >= 60 else "#C4880F" if hs >= 30 else "#B83A2E",
                         thickness=0.25),
                bgcolor="#F0ECE7",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30], color="#FCEAE8"),
                    dict(range=[30, 60], color="#FDF4E4"),
                    dict(range=[60, 100], color="#EBF5EE"),
                ],
            ),
            title=dict(text="Machine Health Score", font=dict(size=13, color="#5F5F5F", family="Inter")),
        ))
        fig_gauge.update_layout(
            height=210, margin=dict(l=20, r=20, t=35, b=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:0.62rem;font-weight:600;color:#8C8C8C;letter-spacing:0.06em;text-transform:uppercase;">STATUS</div>
                <div style="margin-top:3px;">
                    <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;background:{ms_bg};color:{ms_color};">{ms}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:0.62rem;font-weight:600;color:#8C8C8C;letter-spacing:0.06em;text-transform:uppercase;">FAILURE RISK</div>
                <div style="font-size:1.15rem;font-weight:800;color:#1A1A1A;margin-top:2px;">{format_pct(fp)}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:0.62rem;font-weight:600;color:#8C8C8C;letter-spacing:0.06em;text-transform:uppercase;">RISK LEVEL</div>
                <div style="margin-top:3px;">
                    <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;background:{rl_bg};color:{rl_color};">{rl}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_sensors:
        st.markdown("""
        <div style="font-size:0.7rem;font-weight:600;color:#8C8C8C;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;">
            LIVE SENSOR READINGS
        </div>
        """, unsafe_allow_html=True)
        
        s1, s2, s3 = st.columns(3)
        with s1:
            sensor_card("Product Quality", type_code, type_full)
        with s2:
            sensor_card("Air Temp", f"{input_data['Air temperature [K]']:.1f}", "K")
        with s3:
            sensor_card("Process Temp", f"{input_data['Process temperature [K]']:.1f}", "K")
        
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        s4, s5, s6 = st.columns(3)
        with s4:
            sensor_card("Rotational Speed", f"{input_data['Rotational speed [rpm]']}", "rpm")
        with s5:
            sensor_card("Torque", f"{input_data['Torque [Nm]']:.1f}", "Nm")
        with s6:
            sensor_card("Tool Wear", f"{input_data['Tool wear [min]']}", "min")

    # Maintenance Advisory for Current Reading
    section_header("Maintenance Advisory", "Rule-based operational guidance derived from active telemetry.")
    recs = maintenance_recommendations(input_data, fp)
    for level, text in recs:
        rec_card(level, text)

    # Telemetry Stream History Chart
    section_header("Failure Risk & Health Trend", "Timeline of model inference across the telemetry sequence.")

    if len(st.session_state.feed_history) > 1:
        hist_df = pd.DataFrame(st.session_state.feed_history)
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Scatter(
            x=[f"#{r['row_idx']} ({r['product_id']})" for _, r in hist_df.iterrows()],
            y=hist_df["prob"],
            mode="lines+markers",
            name="Failure Risk (%)",
            line=dict(color="#C4662B", width=2.5),
            marker=dict(size=6, color="#C4662B"),
            fill="tozeroy",
            fillcolor="rgba(196,102,43,0.08)"
        ))

        # Threshold lines
        fig_hist.add_hline(y=40, line_dash="dash", line_color="#C4880F", opacity=0.6,
                           annotation_text="Moderate Risk (40%)", annotation_position="top left",
                           annotation_font=dict(size=10, color="#C4880F"))
        fig_hist.add_hline(y=70, line_dash="dash", line_color="#B83A2E", opacity=0.6,
                           annotation_text="Critical Threshold (70%)", annotation_position="top left",
                           annotation_font=dict(size=10, color="#B83A2E"))

        fig_hist.update_layout(
            yaxis=dict(title="Failure Risk (%)", range=[0, 105]),
            xaxis=dict(title="Stream Sequence (Row # & Product ID)"),
            showlegend=False,
            height=300
        )
        fig_hist = styled_figure(fig_hist, height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Advance the stream (click 'Next Reading' or 'Start Monitoring') to generate trend history.")

    #Visual Pipeline Architecture
    section_header("Inference Architecture")
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center; gap:0;
                flex-wrap:wrap; padding:1rem 0;">
        <div style="text-align:center; padding:0.9rem 1.25rem; background:#FFFFFF;
                    border:1px solid #E5E0DA; border-radius:8px; min-width:130px;">
            <div style="font-size:1.3rem; margin-bottom:2px;"></div>
            <div style="font-size:0.75rem; font-weight:600; color:#1A1A1A;">Sensor Data</div>
            <div style="font-size:0.65rem; color:#8C8C8C;">6 parameters</div>
        </div>
        <div style="font-size:1.1rem; color:#C4662B; padding:0 0.6rem; font-weight:300;">→</div>
        <div style="text-align:center; padding:0.9rem 1.25rem; background:#FFFFFF;
                    border:1px solid #E5E0DA; border-radius:8px; min-width:130px;">
            <div style="font-size:1.3rem; margin-bottom:2px;"></div>
            <div style="font-size:0.75rem; font-weight:600; color:#1A1A1A;">Preprocessing</div>
            <div style="font-size:0.65rem; color:#8C8C8C;">StandardScaler & Encoder</div>
        </div>
        <div style="font-size:1.1rem; color:#C4662B; padding:0 0.6rem; font-weight:300;">→</div>
        <div style="text-align:center; padding:0.9rem 1.25rem; background:#FFFFFF;
                    border:1px solid #E5E0DA; border-radius:8px; min-width:130px;">
            <div style="font-size:1.3rem; margin-bottom:2px;"></div>
            <div style="font-size:0.75rem; font-weight:600; color:#1A1A1A;">XGBoost Model</div>
            <div style="font-size:0.65rem; color:#8C8C8C;">Binary Classifier</div>
        </div>
        <div style="font-size:1.1rem; color:#C4662B; padding:0 0.6rem; font-weight:300;">→</div>
        <div style="text-align:center; padding:0.9rem 1.25rem; background:#FFFFFF;
                    border:1px solid #E5E0DA; border-radius:8px; min-width:130px;">
            <div style="font-size:1.3rem; margin-bottom:2px;"></div>
            <div style="font-size:0.75rem; font-weight:600; color:#1A1A1A;">Failure Risk</div>
            <div style="font-size:0.65rem; color:#8C8C8C;">Calibrated Probability</div>
        </div>
        <div style="font-size:1.1rem; color:#C4662B; padding:0 0.6rem; font-weight:300;">→</div>
        <div style="text-align:center; padding:0.9rem 1.25rem; background:#FFFFFF;
                    border:1px solid #E5E0DA; border-radius:8px; min-width:130px;">
            <div style="font-size:1.3rem; margin-bottom:2px;"></div>
            <div style="font-size:0.75rem; font-weight:600; color:#1A1A1A;">Maintenance Advisory</div>
            <div style="font-size:0.65rem; color:#8C8C8C;">Rule-Based Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # If continuous monitoring is active, auto-advance with short delay
    if st.session_state.is_monitoring:
        time.sleep(1.2)
        st.session_state.feed_idx = (st.session_state.feed_idx + 1) % len(df)
        st.rerun()


# ============================================================
# PAGE 2 — Analyze Machine: Manual Input & Diagnostics
# ============================================================

def page_analyze():
    artifacts = get_model_artifacts()

    st.markdown("""
    <div style="margin:1.25rem 0 0.5rem 0;">
        <h1 style="font-size:1.9rem; font-weight:800; margin:0; letter-spacing:-0.03em;">
            Analyze Machine</h1>
        <p style="font-size:0.95rem; color:#5F5F5F; margin-top:4px;">
            Manual operating parameter input for on-demand failure risk prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for manual inputs and predictions
    if "manual_type" not in st.session_state:
        st.session_state.manual_type = "M — Medium"
    if "manual_air_temp" not in st.session_state:
        st.session_state.manual_air_temp = 298.1
    if "manual_proc_temp" not in st.session_state:
        st.session_state.manual_proc_temp = 308.6
    if "manual_rpm" not in st.session_state:
        st.session_state.manual_rpm = 1551
    if "manual_torque" not in st.session_state:
        st.session_state.manual_torque = 42.8
    if "manual_tool_wear" not in st.session_state:
        st.session_state.manual_tool_wear = 0
    if "manual_result" not in st.session_state:
        st.session_state.manual_result = None
    if "manual_input_used" not in st.session_state:
        st.session_state.manual_input_used = None

    # Sample Presets Bar
    p_col1, p_col2, p_col3, p_col4 = st.columns([1.5, 1.5, 1.5, 2])
    with p_col1:
        if st.button("Load Sample (Nominal)", use_container_width=True):
            st.session_state.manual_type = "M — Medium"
            st.session_state.manual_air_temp = 298.1
            st.session_state.manual_proc_temp = 308.6
            st.session_state.manual_rpm = 1551
            st.session_state.manual_torque = 42.8
            st.session_state.manual_tool_wear = 0
            st.session_state.manual_result = None
            st.rerun()

    with p_col2:
        if st.button("Load Sample (High Torque)", use_container_width=True):
            st.session_state.manual_type = "L — Low"
            st.session_state.manual_air_temp = 302.5
            st.session_state.manual_proc_temp = 311.2
            st.session_state.manual_rpm = 1320
            st.session_state.manual_torque = 68.4
            st.session_state.manual_tool_wear = 195
            st.session_state.manual_result = None
            st.rerun()

    with p_col3:
        if st.button("Load Sample (Tool Wear)", use_container_width=True):
            st.session_state.manual_type = "H — High"
            st.session_state.manual_air_temp = 299.0
            st.session_state.manual_proc_temp = 309.5
            st.session_state.manual_rpm = 1450
            st.session_state.manual_torque = 52.0
            st.session_state.manual_tool_wear = 225
            st.session_state.manual_result = None
            st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Input Form
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:8px;
                padding:1.25rem 1.5rem; margin-bottom:1rem; box-shadow:0 1px 2px rgba(0,0,0,0.02);">
        <div style="font-size:0.75rem; font-weight:600; color:#8C8C8C;
                    letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.75rem;">
            MANUAL OPERATING PARAMETERS</div>
    </div>
    """, unsafe_allow_html=True)

    type_options = ["L — Low", "M — Medium", "H — High"]
    type_idx = type_options.index(st.session_state.manual_type) if st.session_state.manual_type in type_options else 1

    c1, c2, c3 = st.columns(3)
    with c1:
        product_type = st.selectbox(
            "Product Quality Variant",
            options=type_options,
            index=type_idx,
            help="Product quality variant: L (Low), M (Medium), H (High)"
        )
        type_val = product_type[0]

        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=290.0, max_value=310.0,
            value=float(st.session_state.manual_air_temp),
            step=0.1, format="%.1f",
            help="Typical range: 295.3–304.5 K"
        )

    with c2:
        proc_temp = st.number_input(
            "Process Temperature [K]",
            min_value=300.0, max_value=320.0,
            value=float(st.session_state.manual_proc_temp),
            step=0.1, format="%.1f",
            help="Typical range: 305.7–313.8 K"
        )
        rpm = st.number_input(
            "Rotational Speed [rpm]",
            min_value=500, max_value=3500,
            value=int(st.session_state.manual_rpm),
            step=1, help="Typical range: 1168–2886 rpm"
        )

    with c3:
        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0, max_value=100.0,
            value=float(st.session_state.manual_torque),
            step=0.1, format="%.1f",
            help="Typical range: 3.8–76.6 Nm"
        )
        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0, max_value=300,
            value=int(st.session_state.manual_tool_wear),
            step=1, help="Typical range: 0–253 min"
        )

    # Range validation warnings
    input_data = {
        "Type": type_val,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    warnings = []
    for key in INPUT_RANGES:
        valid, msg = validate_input(key, input_data[key])
        if not valid:
            warnings.append(f"**{key}**: {msg}")
    if warnings:
        st.warning("Sensor value outside nominal operational boundaries:\n" + "\n".join(f"- {w}" for w in warnings))

    # Action Button
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("Analyze Machine", type="primary", use_container_width=False)

    if analyze_clicked:
        with st.spinner("Executing XGBoost inference pipeline..."):
            res = predict(input_data, model_artifacts=artifacts)
            st.session_state.manual_result = res
            st.session_state.manual_input_used = input_data.copy()

    # Display Inference Results
    if st.session_state.manual_result:
        res = st.session_state.manual_result
        input_used = st.session_state.manual_input_used
        fp = res["failure_probability"]
        nfp = res["no_failure_probability"]
        hs = health_score(fp)
        rl, rl_color, rl_bg = risk_level(fp)
        ms, ms_color, ms_bg = machine_status_label(res["prediction"])

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

        # Central Result Card
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:10px;
                    padding:2.25rem; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.04);
                    margin-bottom:1.25rem;">
            <div style="font-size:0.7rem; font-weight:600; color:#8C8C8C;
                        letter-spacing:0.08em; text-transform:uppercase;">MACHINE DIAGNOSTIC STATUS</div>
            <div style="font-size:1.4rem; font-weight:700; margin-top:8px;">
                <span style="display:inline-block; padding:6px 24px; border-radius:6px;
                             background:{ms_bg}; color:{ms_color};">{ms}</span>
            </div>
            <div style="display:flex; justify-content:center; gap:3.5rem; margin-top:2rem;
                        flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.68rem; font-weight:600; color:#8C8C8C;
                                letter-spacing:0.06em; text-transform:uppercase;">FAILURE PROBABILITY</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#1A1A1A;
                                letter-spacing:-0.02em; margin-top:4px;">{format_pct(fp)}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem; font-weight:600; color:#8C8C8C;
                                letter-spacing:0.06em; text-transform:uppercase;">HEALTH SCORE</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#1A1A1A;
                                letter-spacing:-0.02em; margin-top:4px;">
                        {hs} <span style="font-size:0.95rem; font-weight:500; color:#8C8C8C;">/ 100</span>
                    </div>
                </div>
                <div>
                    <div style="font-size:0.68rem; font-weight:600; color:#8C8C8C;
                                letter-spacing:0.06em; text-transform:uppercase;">RISK CLASSIFICATION</div>
                    <div style="margin-top:8px;">
                        <span style="display:inline-block; padding:6px 18px; border-radius:4px;
                                     font-size:0.85rem; font-weight:700; letter-spacing:0.06em;
                                     background:{rl_bg}; color:{rl_color};">{rl}</span>
                    </div>
                </div>
                <div>
                    <div style="font-size:0.68rem; font-weight:600; color:#8C8C8C;
                                letter-spacing:0.06em; text-transform:uppercase;">NO-FAILURE PROBABILITY</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#1A1A1A;
                                letter-spacing:-0.02em; margin-top:4px;">{format_pct(nfp)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Transparent Rule-Based Maintenance Advisory
        section_header("Maintenance Advisory",
                       "Transparent rule-based advisory layer aligned with predicted machine risk.")
        recs = maintenance_recommendations(input_used, fp)
        for level, text in recs:
            rec_card(level, text)


# ============================================================
# PAGE 3 — What-If Simulator: Scenario Comparison
# ============================================================

def page_simulator():
    artifacts = get_model_artifacts()

    st.markdown("""
    <div style="margin:1.25rem 0 0.5rem 0;">
        <h1 style="font-size:1.9rem; font-weight:800; margin:0; letter-spacing:-0.03em;">
            What-If Simulator</h1>
        <p style="font-size:0.95rem; color:#5F5F5F; margin-top:4px;">
            Evaluate how adjusting operational parameters impacts predicted failure probability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "sim_curr_res" not in st.session_state:
        st.session_state.sim_curr_res = None
    if "sim_sim_res" not in st.session_state:
        st.session_state.sim_sim_res = None

    # Shared Product Quality Variant
    type_option = st.selectbox(
        "Product Quality Variant (Common)",
        options=["L — Low", "M — Medium", "H — High"],
        index=1, key="sim_type_select",
    )
    type_val = type_option[0]

    # Two-Column Comparative Setup
    col_curr, col_spacer, col_sim = st.columns([5, 0.4, 5])

    with col_curr:
        st.markdown("""
        <div style="font-size:0.75rem; font-weight:700; color:#1A1A1A;
                    letter-spacing:0.06em; text-transform:uppercase;
                    padding-bottom:0.5rem; border-bottom:2px solid #E5E0DA;
                    margin-bottom:0.75rem;">CURRENT OPERATING CONDITION</div>
        """, unsafe_allow_html=True)
        curr_air = st.slider("Air Temperature [K]", 295.0, 305.0, 298.5, 0.1, key="c_air")
        curr_proc = st.slider("Process Temperature [K]", 305.0, 315.0, 309.0, 0.1, key="c_proc")
        curr_rpm = st.slider("Rotational Speed [rpm]", 1000, 3000, 1500, 10, key="c_rpm")
        curr_torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5, key="c_torque")
        curr_wear = st.slider("Tool Wear [min]", 0, 253, 50, 1, key="c_wear")

    with col_sim:
        st.markdown("""
        <div style="font-size:0.75rem; font-weight:700; color:#C4662B;
                    letter-spacing:0.06em; text-transform:uppercase;
                    padding-bottom:0.5rem; border-bottom:2px solid #C4662B;
                    margin-bottom:0.75rem;">SIMULATED OPERATING CONDITION</div>
        """, unsafe_allow_html=True)
        sim_air = st.slider("Air Temperature [K]", 295.0, 305.0, 299.5, 0.1, key="s_air")
        sim_proc = st.slider("Process Temperature [K]", 305.0, 315.0, 310.5, 0.1, key="s_proc")
        sim_rpm = st.slider("Rotational Speed [rpm]", 1000, 3000, 1350, 10, key="s_rpm")
        sim_torque = st.slider("Torque [Nm]", 3.0, 80.0, 62.0, 0.5, key="s_torque")
        sim_wear = st.slider("Tool Wear [min]", 0, 253, 185, 1, key="s_wear")

    curr_input = {
        "Type": type_val,
        "Air temperature [K]": curr_air,
        "Process temperature [K]": curr_proc,
        "Rotational speed [rpm]": curr_rpm,
        "Torque [Nm]": curr_torque,
        "Tool wear [min]": curr_wear,
    }
    sim_input = {
        "Type": type_val,
        "Air temperature [K]": sim_air,
        "Process temperature [K]": sim_proc,
        "Rotational speed [rpm]": sim_rpm,
        "Torque [Nm]": sim_torque,
        "Tool wear [min]": sim_wear,
    }

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("⚡  Compare Operating Scenarios", type="primary"):
        st.session_state.sim_curr_res = predict(curr_input, model_artifacts=artifacts)
        st.session_state.sim_sim_res = predict(sim_input, model_artifacts=artifacts)

    # Auto-run initial comparison if not present
    if st.session_state.sim_curr_res is None:
        st.session_state.sim_curr_res = predict(curr_input, model_artifacts=artifacts)
        st.session_state.sim_sim_res = predict(sim_input, model_artifacts=artifacts)

    curr_res = st.session_state.sim_curr_res
    sim_res = st.session_state.sim_sim_res

    if curr_res and sim_res:
        curr_fp = curr_res["failure_probability"]
        sim_fp = sim_res["failure_probability"]
        curr_rl, curr_rl_col, curr_rl_bg = risk_level(curr_fp)
        sim_rl, sim_rl_col, sim_rl_bg = risk_level(sim_fp)
        
        delta = (sim_fp - curr_fp) * 100
        delta_sign = "+" if delta >= 0 else ""
        delta_color = "var(--red)" if delta > 0.5 else "var(--green)" if delta < -0.5 else "var(--text-secondary)"

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # Comparative Metric Cards
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:8px;
                        padding:1.5rem; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:0.68rem; font-weight:600; color:#8C8C8C;
                            letter-spacing:0.07em; text-transform:uppercase;">CURRENT FAILURE RISK</div>
                <div style="font-size:2.25rem; font-weight:800; color:#1A1A1A;
                            letter-spacing:-0.02em; margin-top:6px;">{format_pct(curr_fp)}</div>
                <div style="margin-top:6px;">
                    <span style="display:inline-block; padding:3px 12px; border-radius:4px;
                                 font-size:0.75rem; font-weight:700; background:{curr_rl_bg}; color:{curr_rl_col};">
                        {curr_rl} RISK (Health: {health_score(curr_fp)}/100)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #C4662B; border-radius:8px;
                        padding:1.5rem; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:0.68rem; font-weight:600; color:#C4662B;
                            letter-spacing:0.07em; text-transform:uppercase;">SIMULATED FAILURE RISK</div>
                <div style="font-size:2.25rem; font-weight:800; color:#1A1A1A;
                            letter-spacing:-0.02em; margin-top:6px;">{format_pct(sim_fp)}</div>
                <div style="margin-top:6px;">
                    <span style="display:inline-block; padding:3px 12px; border-radius:4px;
                                 font-size:0.75rem; font-weight:700; background:{sim_rl_bg}; color:{sim_rl_col};">
                        {sim_rl} RISK (Health: {health_score(sim_fp)}/100)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:8px;
                        padding:1.5rem; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:0.68rem; font-weight:600; color:#8C8C8C;
                            letter-spacing:0.07em; text-transform:uppercase;">ABSOLUTE RISK DELTA</div>
                <div style="font-size:2.25rem; font-weight:800; color:{delta_color};
                            letter-spacing:-0.02em; margin-top:6px;">
                    {delta_sign}{delta:.1f}<span style="font-size:1.1rem; font-weight:600;"> pp</span></div>
                <div style="font-size:0.75rem; color:#8C8C8C; margin-top:6px;">percentage points</div>
            </div>
            """, unsafe_allow_html=True)

        # Parametric Sensitivity Analysis
        section_header("Parametric Sensitivity Curve",
                       "Simulate non-linear failure risk changes across a continuous range of any single variable.")

        sweep_var = st.selectbox(
            "Select Variable for Sensitivity Curve",
            ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
             "Air temperature [K]", "Process temperature [K]"],
            key="sim_sweep_select",
        )

        sweep_ranges = {
            "Rotational speed [rpm]": np.linspace(1000, 3000, 45).astype(int),
            "Torque [Nm]": np.linspace(3, 80, 45),
            "Tool wear [min]": np.linspace(0, 253, 45).astype(int),
            "Air temperature [K]": np.linspace(295, 305, 45),
            "Process temperature [K]": np.linspace(305, 315, 45),
        }

        with st.spinner("Computing parameter sensitivity..."):
            sweep_vals = sweep_ranges[sweep_var]
            probs = []
            base = curr_input.copy()
            for v in sweep_vals:
                test = base.copy()
                test[sweep_var] = float(v)
                r = predict(test, model_artifacts=artifacts)
                probs.append(r["failure_probability"] * 100 if r else 0)

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=sweep_vals, y=probs,
                mode="lines", name="Predicted Risk Curve",
                line=dict(color="#C4662B", width=2.5),
                fill="tozeroy", fillcolor="rgba(196,102,43,0.08)",
            ))
            
            # Current Point
            c_val = curr_input[sweep_var]
            c_prob = curr_fp * 100
            fig_sens.add_trace(go.Scatter(
                x=[c_val], y=[c_prob],
                mode="markers+text", name="Current",
                marker=dict(color="#1A1A1A", size=9, symbol="circle"),
                text=[f" Current ({c_prob:.1f}%)"], textposition="top right",
                textfont=dict(size=11, color="#1A1A1A"),
            ))

            # Simulated Point
            s_val = sim_input[sweep_var]
            s_prob = sim_fp * 100
            if s_val != c_val:
                fig_sens.add_trace(go.Scatter(
                    x=[s_val], y=[s_prob],
                    mode="markers+text", name="Simulated",
                    marker=dict(color="#C4662B", size=9, symbol="diamond"),
                    text=[f" Simulated ({s_prob:.1f}%)"], textposition="top right",
                    textfont=dict(size=11, color="#C4662B"),
                ))

            fig_sens.update_layout(
                xaxis_title=sweep_var,
                yaxis_title="Failure Probability (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                showlegend=True,
            )
            fig_sens = styled_figure(fig_sens, height=380)
            st.plotly_chart(fig_sens, use_container_width=True)


# ============================================================
# PAGE 4 — Model Insights: ML Evaluation & Transparency
# ============================================================

def page_insights():
    results = get_evaluation_results()
    metrics = results["metrics"]
    cm = results["confusion_matrix"]
    roc = results["roc_curve"]
    fi = results["feature_importance"]
    ds = results["dataset_info"]

    st.markdown("""
    <div style="margin:1.25rem 0 0.5rem 0;">
        <h1 style="font-size:1.9rem; font-weight:800; margin:0; letter-spacing:-0.03em;">
            Model Insights</h1>
        <p style="font-size:0.95rem; color:#5F5F5F; margin-top:4px;">
            Stage 1 XGBoost model performance metrics, confusion matrix, ROC-AUC, and feature contribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Core Metrics Summary
    section_header("Stage 1 Model Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Accuracy", f"{metrics['accuracy']*100:.2f}%", "Overall test accuracy")
    with c2:
        metric_card("Precision", f"{metrics['precision']*100:.2f}%", "Positive predictive value")
    with c3:
        metric_card("Recall", f"{metrics['recall']*100:.2f}%", "Minority failure detection", accent=True)
    with c4:
        metric_card("F1 Score", f"{metrics['f1_score']*100:.2f}%", "Harmonic mean (P/R)", accent=True)
    with c5:
        metric_card("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%", "Area under ROC curve")

    # Evaluation Visualizations
    section_header("Evaluation Charts & Diagnostic Curves")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "XGBoost Feature Importance"])

    with tab1:
        labels_x = ["Predicted: No Failure (0)", "Predicted: Failure (1)"]
        labels_y = ["Actual: No Failure (0)", "Actual: Failure (1)"]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels_x,
            y=labels_y,
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=20, color="#1A1A1A", family="Inter"),
            colorscale=[[0, "#FAF8F5"], [1, "#C4662B"]],
            showscale=False,
            hovertemplate="<b>%{y}</b> → %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            title=dict(text="Confusion Matrix — Stratified Test Split (n=2,000)", font=dict(size=14)),
            xaxis=dict(title="", side="bottom"),
            yaxis=dict(title="", autorange="reversed"),
        )
        fig_cm = styled_figure(fig_cm, height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown(f"""
        <div style="font-size:0.82rem; color:#5F5F5F; line-height:1.6; max-width:800px; margin-top:0.5rem;">
            <b>Test Breakdown:</b> True Negatives: <b>{cm[0][0]:,}</b> (correct non-failures) · 
            False Positives: <b>{cm[0][1]}</b> (false alarms) · 
            False Negatives: <b>{cm[1][0]}</b> (missed failures) · 
            True Positives: <b>{cm[1][1]}</b> (successfully detected failures).
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        fpr = roc["fpr"]
        tpr = roc["tpr"]

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"XGBoost Classifier (AUC = {metrics['roc_auc']:.4f})",
            line=dict(color="#C4662B", width=2.5),
            fill="tozeroy", fillcolor="rgba(196,102,43,0.06)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Guessing Reference",
            line=dict(color="#CCCCCC", dash="dash", width=1.5),
        ))
        fig_roc.update_layout(
            title=dict(text="Receiver Operating Characteristic (ROC) Curve", font=dict(size=14)),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig_roc = styled_figure(fig_roc, height=400)
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        features = fi["features"]
        importances = fi["importances"]

        display_names = [
            f.replace("(K)", "[K]").replace("(rpm)", "[rpm]")
             .replace("(Nm)", "[Nm]").replace("(min)", "[min]")
            for f in features
        ]

        sorted_pairs = sorted(zip(importances, display_names), key=lambda x: x[0])
        sorted_imp, sorted_names = zip(*sorted_pairs)

        fig_fi = go.Figure(go.Bar(
            x=list(sorted_imp),
            y=list(sorted_names),
            orientation="h",
            marker_color="#C4662B",
            text=[f"{v:.3f}" for v in sorted_imp],
            textposition="outside",
            textfont=dict(size=11, color="#5F5F5F"),
        ))
        fig_fi.update_layout(
            title=dict(text="XGBoost Feature Importance (Gain / Split Contribution)", font=dict(size=14)),
            xaxis_title="Relative Feature Importance Score",
            yaxis_title="",
        )
        fig_fi = styled_figure(fig_fi, height=380)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        <div style="background:#FAF8F5; border:1px solid #E5E0DA; border-radius:6px;
                    padding:1rem 1.25rem; margin-top:0.5rem;">
            <div style="font-size:0.82rem; color:#5F5F5F; line-height:1.6;">
                The model identifies <b>rotational speed</b>, <b>torque</b>, and <b>tool wear</b>
                as the strongest predictive features in the trained dataset. 
                <br><em>Note: Feature importance scores reflect statistical split contributions across the gradient-boosted decision trees, not physical causality.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model Specification Panel
    section_header("Model Architecture & Dataset Specifications")

    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E5E0DA; border-radius:8px;
                padding:1.5rem; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
        <table style="width:100%; border-collapse:collapse; font-family:'Inter',sans-serif;">
            <tr style="border-bottom:1px solid #F0ECE7;">
                <td style="padding:8px 12px; font-size:0.8rem; font-weight:600; color:#8C8C8C; width:220px;">Model Algorithm</td>
                <td style="padding:8px 12px; font-size:0.85rem; color:#1A1A1A;">XGBoost Classifier (<code>n_estimators=200, max_depth=6, lr=0.1</code>)</td>
            </tr>
            <tr style="border-bottom:1px solid #F0ECE7;">
                <td style="padding:8px 12px; font-size:0.8rem; font-weight:600; color:#8C8C8C;">Dataset</td>
                <td style="padding:8px 12px; font-size:0.85rem; color:#1A1A1A;">AI4I 2020 Predictive Maintenance Dataset</td>
            </tr>
            <tr style="border-bottom:1px solid #F0ECE7;">
                <td style="padding:8px 12px; font-size:0.8rem; font-weight:600; color:#8C8C8C;">Total Dataset Size</td>
                <td style="padding:8px 12px; font-size:0.85rem; color:#1A1A1A;">{ds['total_samples']:,} samples</td>
            </tr>
            <tr style="border-bottom:1px solid #F0ECE7;">
                <td style="padding:8px 12px; font-size:0.8rem; font-weight:600; color:#8C8C8C;">Class Distribution</td>
                <td style="padding:8px 12px; font-size:0.85rem; color:#1A1A1A;">{ds['failure_cases']} Failures ({ds['failure_rate']}%) · {ds['total_samples'] - ds['failure_cases']:,} Healthy</td>
            </tr>
            <tr style="border-bottom:1px solid #F0ECE7;">
                <td style="padding:8px 12px; font-size:0.8rem; font-weight:600; color:#8C8C8C;">Train / Test Partition</td>
                <td style="padding:8px 12px; font-size:0.85rem; color:#1A1A1A;">{ds['train_samples']:,} Training / {ds['test_samples']:,} Testing (Stratified 80/20)</td>
            </tr>
            <tr>
                <td style="padding:8px 12px; font-size:0.8rem; font-weight:600; color:#8C8C8C;">Class Imbalance Handling</td>
                <td style="padding:8px 12px; font-size:0.85rem; color:#1A1A1A;">XGBoost <code>scale_pos_weight = 28.52</code> (ratio of negative to positive training samples)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Class Imbalance Callout
    st.markdown("""
    <div style="background:#FDF4E4; border-left:3px solid #C4880F; border-radius:0 6px 6px 0;
                padding:0.9rem 1.1rem; margin-top:1rem;">
        <div style="font-size:0.78rem; font-weight:600; color:#C4880F; margin-bottom:4px;
                    letter-spacing:0.03em;">Why Accuracy Alone is Insufficient</div>
        <div style="font-size:0.82rem; color:#1A1A1A; line-height:1.55;">
            In industrial predictive maintenance datasets where failures comprise only ~3.4% of total instances, 
            a trivial model predicting "No Failure" for every machine would trivially yield <b>96.6% accuracy</b> 
            while completely failing to predict any downtime. Therefore, <b>Recall (77.94%)</b>, <b>F1-Score (74.13%)</b>, 
            and <b>ROC-AUC (96.72%)</b> serve as the authentic evaluation benchmarks.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Navigation & Entry Point
# ============================================================

def render_sidebar():
    st.markdown("""
    <div style="padding:0.5rem 0 1rem 0;">
        <div style="font-size:1.15rem; font-weight:800; letter-spacing:-0.03em;
                    color:#1A1A1A; line-height:1;">ECHELON</div>
        <div style="font-size:0.7rem; color:#5F5F5F; margin-top:3px;
                    letter-spacing:0.02em;">Predictive Maintenance</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊  Overview", "🔍  Analyze Machine", "🔄  What-If Simulator", "📈  Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.68rem; color:#8C8C8C; line-height:1.5; padding-top:0.5rem;">
        <div style="font-weight:600; letter-spacing:0.04em;">ECHELON v1.0</div>
        <div style="margin-top:2px;">XGBoost Classifier</div>
        <div>AI4I 2020 Telemetry</div>
    </div>
    """, unsafe_allow_html=True)

    return page


def main():
    inject_styles()
    render_header()

    with st.sidebar:
        page = render_sidebar()

    page_map = {
        "📊  Overview": page_overview,
        "🔍  Analyze Machine": page_analyze,
        "🔄  What-If Simulator": page_simulator,
        "📈  Model Insights": page_insights,
    }

    page_fn = page_map.get(page, page_overview)
    page_fn()


if __name__ == "__main__":
    main()
