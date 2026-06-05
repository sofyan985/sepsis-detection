import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ── Page config
st.set_page_config(
    page_title="Sepsis Watch — ICU Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }

.stApp {
    background-color: #0a0e1a;
    color: #e0e6f0;
}

section[data-testid="stSidebar"] {
    background-color: #0d1224;
    border-right: 1px solid #1e2d4a;
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

.metric-card {
    background: linear-gradient(135deg, #0d1224 0%, #111827 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
}

.metric-card.warning { border-color: #f59e0b; box-shadow: 0 0 12px rgba(245,158,11,0.15); }
.metric-card.danger  { border-color: #ef4444; box-shadow: 0 0 12px rgba(239,68,68,0.2); }
.metric-card.safe    { border-color: #10b981; box-shadow: 0 0 12px rgba(16,185,129,0.15); }

.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1.1;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.risk-banner {
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
    margin: 16px 0;
    font-family: 'IBM Plex Mono', monospace;
}

.risk-high    { background: rgba(239,68,68,0.12);  border: 2px solid #ef4444; }
.risk-medium  { background: rgba(245,158,11,0.12); border: 2px solid #f59e0b; }
.risk-low     { background: rgba(16,185,129,0.12); border: 2px solid #10b981; }

.history-row {
    background: #0d1224;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

.tag-high   { color: #ef4444; font-weight: 600; }
.tag-medium { color: #f59e0b; font-weight: 600; }
.tag-low    { color: #10b981; font-weight: 600; }

.stSlider > div > div > div { background: #1e2d4a !important; }
div[data-testid="stMetric"] { background: #0d1224; border-radius: 8px; padding: 12px; border: 1px solid #1e2d4a; }
</style>
""", unsafe_allow_html=True)


## ── Load models
@st.cache_resource
def load_models():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer

        xgb_model    = joblib.load('xgb_sepsis_model.pkl')
        rf_model     = joblib.load('rf_sepsis_model.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        with open('ensemble_config.json') as f:
            config = json.load(f)

        # ── Rebuild imputer fresh instead of loading pkl
        imputer = SimpleImputer(strategy='median')
        dummy   = pd.DataFrame(np.zeros((2, len(feature_cols))), columns=feature_cols)
        imputer.fit(dummy)

        return imputer, xgb_model, rf_model, feature_cols, config, True
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None, None, False

# ← call AFTER definition
imputer, xgb_model, rf_model, feature_cols, config, models_loaded = load_models()

# ── Session state for patient history
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []
if 'patient_counter' not in st.session_state:
    st.session_state.patient_counter = 1


# ── Prediction function
def predict(age, hr, temp, resp, sbp, o2sat, lactate, wbc, creatinine, icu_hours):
    demo = {col: np.nan for col in feature_cols}
    demo['Age']                  = age
    demo['HR_mean']              = hr
    demo['HR_max']               = hr + 10
    demo['HR_min']               = hr - 8
    demo['HR_last']              = hr
    demo['HR_std']               = 5.0
    demo['HR_trend']             = 0.0
    demo['HR_missing_pct']       = 0.0
    demo['HR_was_measured']      = 1
    demo['Temp_mean']            = temp
    demo['Temp_last']            = temp
    demo['Temp_max']             = temp + 0.2
    demo['Temp_min']             = temp - 0.2
    demo['Temp_missing_pct']     = 0.0
    demo['Temp_was_measured']    = 1
    demo['Resp_mean']            = resp
    demo['Resp_last']            = resp
    demo['Resp_max']             = resp + 2
    demo['Resp_missing_pct']     = 0.0
    demo['Resp_was_measured']    = 1
    demo['SBP_mean']             = sbp
    demo['SBP_last']             = sbp
    demo['SBP_min']              = sbp - 5
    demo['SBP_missing_pct']      = 0.0
    demo['SBP_was_measured']     = 1
    demo['MAP_mean']             = sbp * 0.7
    demo['MAP_last']             = sbp * 0.7
    demo['MAP_missing_pct']      = 0.0
    demo['MAP_was_measured']     = 1
    demo['O2Sat_mean']           = o2sat
    demo['O2Sat_last']           = o2sat
    demo['O2Sat_min']            = o2sat - 1
    demo['O2Sat_missing_pct']    = 0.0
    demo['O2Sat_was_measured']   = 1
    demo['Lactate_mean']         = lactate
    demo['Lactate_last']         = lactate
    demo['Lactate_max']          = lactate + 0.2
    demo['Lactate_missing_pct']  = 0.0
    demo['Lactate_was_measured'] = 1
    demo['WBC_mean']             = wbc
    demo['WBC_last']             = wbc
    demo['WBC_missing_pct']      = 0.0
    demo['WBC_was_measured']     = 1
    demo['Creatinine_mean']      = creatinine
    demo['Creatinine_last']      = creatinine
    demo['Creatinine_missing_pct'] = 0.0
    demo['Creatinine_was_measured'] = 1
    demo['ICULOS_max']           = icu_hours
    demo['qSOFA_resp_high']      = int(resp >= 22)
    demo['qSOFA_sbp_low']        = int(sbp <= 100)
    demo['qSOFA_score']          = int(resp >= 22) + int(sbp <= 100)
    demo['shock_index_mean']     = hr / sbp if sbp > 0 else 1.0
    demo['shock_index_max']      = (hr + 10) / max(sbp - 5, 1)
    demo['lactate_ever_high']    = int(lactate > 2.0)
    demo['creatinine_ever_high'] = int(creatinine > 1.2)
    demo['wbc_ever_abnormal']    = int(wbc > 12.0 or wbc < 4.0)
    demo['temp_ever_abnormal']   = int(temp >= 38.0 or temp <= 36.0)

    X     = pd.DataFrame([demo])[feature_cols]
    X_imp = imputer.transform(X)

    xgb_prob = xgb_model.predict_proba(X_imp)[0, 1]
    rf_prob  = rf_model.predict_proba(X_imp)[0, 1]
    prob     = config['w_xgb'] * xgb_prob + config['w_rf'] * rf_prob
    return prob, xgb_prob, rf_prob


def risk_label(p):
    if p >= 0.6:  return "HIGH RISK",   "risk-high",   "🔴"
    if p >= 0.35: return "MODERATE",    "risk-medium",  "🟡"
    return             "LOW RISK",    "risk-low",    "🟢"


def is_abnormal(name, val):
    ranges = {
        'hr':         (60, 100),
        'temp':       (36.0, 37.9),
        'resp':       (12, 20),
        'sbp':        (90, 140),
        'o2sat':      (95, 100),
        'lactate':    (0, 2.0),
        'wbc':        (4.0, 11.0),
        'creatinine': (0.6, 1.2),
    }
    lo, hi = ranges.get(name, (None, None))
    if lo is None: return False
    return val < lo or val > hi


# ════════════════════════════════════════════
# SIDEBAR — Patient Input
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 SEPSIS WATCH")
    st.markdown("<p style='color:#64748b;font-size:0.8rem;margin-top:-12px'>ICU Early Warning System</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 👤 Patient Info")
    patient_name = st.text_input("Patient ID / Name", value=f"PT-{st.session_state.patient_counter:03d}")
    age  = st.slider("Age (years)", 18, 100, 65)
    icu_hours = st.slider("Hours in ICU", 1, 72, 12)

    st.markdown("### 💓 Vital Signs")
    hr     = st.slider("Heart Rate (bpm)",        40,  180, 88)
    temp   = st.slider("Temperature (°C)",        35.0, 42.0, 37.2, step=0.1)
    resp   = st.slider("Respiratory Rate (/min)", 8,   40,  16)
    sbp    = st.slider("Systolic BP (mmHg)",      60,  200, 122)
    o2sat  = st.slider("O₂ Saturation (%)",       70,  100, 98)

    st.markdown("### 🧪 Lab Values")
    lactate    = st.slider("Lactate (mmol/L)",   0.5, 10.0, 1.1, step=0.1)
    wbc        = st.slider("WBC (×10³/µL)",      1.0, 30.0, 7.5, step=0.5)
    creatinine = st.slider("Creatinine (mg/dL)", 0.4, 8.0,  0.9, step=0.1)

    st.divider()
    predict_btn = st.button("🔍 RUN PREDICTION", use_container_width=True, type="primary")
    clear_btn   = st.button("🗑️ Clear History",  use_container_width=True)

    if clear_btn:
        st.session_state.patient_history = []
        st.session_state.patient_counter = 1
        st.rerun()

    st.markdown("---")
    st.markdown("**Quick Presets**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚠️ High Risk", use_container_width=True):
            st.session_state['preset'] = 'high'
    with col2:
        if st.button("✅ Low Risk", use_container_width=True):
            st.session_state['preset'] = 'low'

# Apply presets
if st.session_state.get('preset') == 'high':
    hr, temp, resp, sbp, o2sat = 118, 38.7, 24, 88, 93
    lactate, wbc, creatinine   = 3.2, 15.0, 1.8
    st.session_state.pop('preset')
elif st.session_state.get('preset') == 'low':
    hr, temp, resp, sbp, o2sat = 72, 36.8, 15, 128, 99
    lactate, wbc, creatinine   = 0.9, 7.0, 0.8
    st.session_state.pop('preset')


# ════════════════════════════════════════════
# MAIN DASHBOARD
# ════════════════════════════════════════════
st.markdown("# 🏥 SEPSIS WATCH")
st.markdown("<p style='color:#64748b;margin-top:-16px'>Early Warning System — PhysioNet ICU Dataset · XGBoost + Random Forest Ensemble</p>", unsafe_allow_html=True)
st.divider()

if not models_loaded:
    st.error("⚠️ Model files not found. Make sure these files are in the same folder as app.py: imputer.pkl, xgb_sepsis_model.pkl, rf_sepsis_model.pkl, feature_cols.pkl, ensemble_config.json")
    st.stop()

# ── Run prediction
prob, xgb_prob, rf_prob = predict(age, hr, temp, resp, sbp, o2sat, lactate, wbc, creatinine, icu_hours)
label, css_class, emoji = risk_label(prob)

if predict_btn:
    entry = {
        'time':       datetime.now().strftime("%H:%M:%S"),
        'patient':    patient_name,
        'prob':       prob,
        'label':      label,
        'hr':         hr,
        'temp':       temp,
        'sbp':        sbp,
        'lactate':    lactate,
    }
    st.session_state.patient_history.insert(0, entry)
    st.session_state.patient_counter += 1

# ── TOP ROW — Risk Banner + Gauge
col_banner, col_gauge = st.columns([1.4, 1])

with col_banner:
    st.markdown(f"""
    <div class='risk-banner {css_class}'>
        <div style='font-size:3rem'>{emoji}</div>
        <div style='font-size:2.2rem;font-weight:700;letter-spacing:0.05em'>{label}</div>
        <div style='font-size:3.5rem;font-weight:700;margin:8px 0'>{prob:.1%}</div>
        <div style='color:#94a3b8;font-size:0.85rem'>Sepsis Probability · Threshold: {config['best_threshold']:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Model breakdown
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#60a5fa'>{xgb_prob:.1%}</div>
            <div class='metric-label'>XGBoost</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#a78bfa'>{rf_prob:.1%}</div>
            <div class='metric-label'>Random Forest</div>
        </div>""", unsafe_allow_html=True)

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 36, 'color': '#e0e6f0', 'family': 'IBM Plex Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'tickfont': {'color': '#64748b'}},
            'bar':  {'color': '#ef4444' if prob >= 0.6 else '#f59e0b' if prob >= 0.35 else '#10b981', 'thickness': 0.25},
            'bgcolor': '#0d1224',
            'bordercolor': '#1e2d4a',
            'steps': [
                {'range': [0,  35], 'color': 'rgba(16,185,129,0.08)'},
                {'range': [35, 60], 'color': 'rgba(245,158,11,0.08)'},
                {'range': [60, 100],'color': 'rgba(239,68,68,0.08)'},
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 2},
                'thickness': 0.75,
                'value': config['best_threshold'] * 100
            }
        }
    ))
    fig_gauge.update_layout(
        height=250, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e0e6f0'}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ── VITALS ROW
st.markdown("### 📊 Current Vitals")

vitals = [
    ("HR",          hr,         "bpm",    'hr',         60,  100),
    ("Temp",        temp,       "°C",     'temp',       36,  38),
    ("Resp Rate",   resp,       "/min",   'resp',       12,  20),
    ("Sys BP",      sbp,        "mmHg",   'sbp',        90,  140),
    ("O₂ Sat",      o2sat,      "%",      'o2sat',      95,  100),
    ("Lactate",     lactate,    "mmol/L", 'lactate',    0,   2.0),
    ("WBC",         wbc,        "k/µL",   'wbc',        4,   11),
    ("Creatinine",  creatinine, "mg/dL",  'creatinine', 0.6, 1.2),
]

cols = st.columns(8)
for i, (name, val, unit, key, lo, hi) in enumerate(vitals):
    abnormal = val < lo or val > hi
    color    = "#ef4444" if abnormal else "#10b981"
    card_cls = "danger" if abnormal else "safe"
    with cols[i]:
        st.markdown(f"""<div class='metric-card {card_cls}'>
            <div class='metric-val' style='color:{color};font-size:1.4rem'>{val}</div>
            <div class='metric-label'>{name}</div>
            <div style='font-size:0.7rem;color:#475569'>{unit}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── BOTTOM ROW — Radar + History
col_radar, col_hist = st.columns([1, 1.2])

with col_radar:
    st.markdown("### 🕸️ Risk Factor Radar")

    def normalize(val, lo, hi):
        return min(max((val - lo) / (hi - lo + 1e-6), 0), 1)

    categories = ['Heart Rate', 'Temperature', 'Resp Rate', 'Low BP', 'Low O₂', 'Lactate', 'WBC', 'Creatinine']
    values     = [
        normalize(hr,         60,  100),
        normalize(temp,       36,  38.5),
        normalize(resp,       12,  22),
        normalize(140 - sbp,  0,   80),
        normalize(100 - o2sat,0,   10),
        normalize(lactate,    0,   4),
        normalize(wbc,        4,   15),
        normalize(creatinine, 0.6, 3),
    ]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill='toself',
        fillcolor='rgba(239,68,68,0.15)' if prob >= 0.6 else 'rgba(245,158,11,0.15)' if prob >= 0.35 else 'rgba(16,185,129,0.15)',
        line=dict(color='#ef4444' if prob >= 0.6 else '#f59e0b' if prob >= 0.35 else '#10b981', width=2),
        name='Patient'
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], tickfont={'color':'#64748b'}, gridcolor='#1e2d4a'),
            angularaxis=dict(tickfont={'color':'#94a3b8'}, gridcolor='#1e2d4a')
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=320, margin=dict(t=20, b=20, l=40, r=40),
        font={'color':'#e0e6f0', 'family':'IBM Plex Sans'},
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_hist:
    st.markdown("### 📋 Patient History")

    if not st.session_state.patient_history:
        st.markdown("<p style='color:#64748b;text-align:center;padding:40px'>No predictions yet.<br>Run a prediction to see history.</p>", unsafe_allow_html=True)
    else:
        # Probability trend chart
        if len(st.session_state.patient_history) > 1:
            hist_df = pd.DataFrame(st.session_state.patient_history[::-1])
            fig_trend = go.Figure(go.Scatter(
                x=hist_df['time'], y=hist_df['prob']*100,
                mode='lines+markers',
                line=dict(color='#60a5fa', width=2),
                marker=dict(
                    size=8,
                    color=['#ef4444' if p >= 0.6 else '#f59e0b' if p >= 0.35 else '#10b981'
                           for p in hist_df['prob']],
                    line=dict(color='#0a0e1a', width=1)
                ),
                fill='tozeroy',
                fillcolor='rgba(96,165,250,0.06)'
            ))
            fig_trend.add_hline(y=config['best_threshold']*100, line_dash='dash',
                                line_color='#ef4444', opacity=0.5,
                                annotation_text='Threshold', annotation_font_color='#ef4444')
            fig_trend.update_layout(
                height=160, margin=dict(t=10, b=30, l=40, r=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, tickfont={'color':'#64748b','size':10}),
                yaxis=dict(gridcolor='#1e2d4a', tickfont={'color':'#64748b','size':10},
                           range=[0, 100], title='Risk %', title_font={'color':'#64748b','size':10}),
                font={'color':'#e0e6f0'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # History table
        for entry in st.session_state.patient_history[:8]:
            tag_cls = 'tag-high' if entry['label'] == 'HIGH RISK' else \
                      'tag-medium' if entry['label'] == 'MODERATE' else 'tag-low'
            st.markdown(f"""
            <div class='history-row'>
                <span style='color:#94a3b8'>{entry['time']}</span>
                <span style='color:#e0e6f0'>{entry['patient']}</span>
                <span class='{tag_cls}'>{entry['prob']:.1%}</span>
                <span class='{tag_cls}'>{entry['label']}</span>
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align:center;color:#334155;font-size:0.75rem'>Sepsis Watch · PhysioNet 2019 · XGBoost + RF Ensemble · For educational demonstration only</p>", unsafe_allow_html=True)
