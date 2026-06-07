import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="SepsisWatch ICU",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

* { 
    font-family: 'DM Sans', sans-serif; 
}
code, .mono { 
    font-family: 'DM Mono', monospace; 
}

.stApp { 
    background-color: #f8f9fc; 
    color: #1e2937; 
}

section[data-testid="stSidebar"] {
    background-color: #f1f3f8;
    border-right: 1px solid #e2e8f0;
}

/* === SECTION HEADERS (Vitals, CBC, etc.) === */
.sidebar-section {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #334155;
    margin: 20px 0 10px 0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
}

/* Streamlit Input Labels */
label[data-testid="stWidgetLabel"] {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    margin-bottom: 4px !important;
}

/* Risk Banners */
.risk-banner {
    border-radius: 6px;
    padding: 28px 36px;
    text-align: center;
    margin: 0 0 20px 0;
    font-family: 'DM Mono', monospace;
    border-left: 6px solid;
}
.risk-high   { background: #fef2f2; border-color: #e11d48; }
.risk-medium { background: #fffbeb; border-color: #f59e0b; }
.risk-low    { background: #ecfdf5; border-color: #10b981; }

.risk-high .risk-pct   { color: #e11d48; }
.risk-medium .risk-pct { color: #f59e0b; }
.risk-low .risk-pct    { color: #10b981; }

.risk-label { 
    font-size: 0.78rem; 
    letter-spacing: 0.12em; 
    text-transform: uppercase; 
    color: #64748b; 
    margin-bottom: 4px; 
}
.risk-pct   { font-size: 3.9rem; font-weight: 500; line-height: 1; }
.risk-name  { font-size: 1.05rem; font-weight: 500; margin-top: 6px; color: #1e2937; }

/* VCards */
.vcard {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 14px 12px;
    text-align: center;
    margin-bottom: 8px;
    transition: all 0.2s;
}
.vcard:hover { 
    box-shadow: 0 4px 12px rgba(0,0,0,0.06); 
    transform: translateY(-1px);
}
.vcard.abn   { border-left: 3px solid #e11d48; background: #fef2f2; }
.vcard.norm  { border-left: 3px solid #10b981; }

.vcard-val   { 
    font-family: 'DM Mono', monospace; 
    font-size: 1.38rem; 
    font-weight: 500; 
    color: #1e2937; 
}
.vcard-val.abn-val { color: #e11d48; }
.vcard-name  { font-size: 0.71rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-top: 2px; }
.vcard-unit  { font-size: 0.68rem; color: #94a3b8; }

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.97rem !important;
    letter-spacing: 0.02em !important;
    padding: 12px 20px !important;
    height: auto !important;
    min-height: 48px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

.stButton > button[kind="primary"] {
    background-color: #14b8a6 !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #0f766e !important;
    box-shadow: 0 4px 12px rgba(20, 184, 166, 0.35) !important;
}

.stButton > button[kind="secondary"] {
    background-color: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
}

/* === PATIENT HISTORY - Restored to Original Style === */
.hist-row {
    background: #ffffff;
    border: 1px solid #ddd9d0;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 3px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}

.tag-h { color: #c0392b; font-weight: 500; }
.tag-m { color: #d68910; font-weight: 500; }
.tag-l { color: #1e8449; font-weight: 500; }

.section-header {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin: 22px 0 12px 0;
    font-weight: 600;
}

.model-pill {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 14px 16px;
    text-align: center;
}
.model-pill-val  { font-size: 1.65rem; font-weight: 500; }
.model-pill-name { font-size: 0.71rem; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-top: 3px; }

.stSlider > div > div > div { background: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


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

        imputer = SimpleImputer(strategy='median')
        dummy   = pd.DataFrame(np.zeros((2, len(feature_cols))), columns=feature_cols)
        imputer.fit(dummy)

        return imputer, xgb_model, rf_model, feature_cols, config, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, None, None, False


imputer, xgb_model, rf_model, feature_cols, config, models_loaded = load_models()

if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []
if 'patient_counter' not in st.session_state:
    st.session_state.patient_counter = 1


def predict(inputs):
    age = inputs['age']
    hr = inputs['hr']; temp = inputs['temp']; resp = inputs['resp']
    sbp = inputs['sbp']; dbp = inputs['dbp']; map_val = inputs['map_val']
    o2sat = inputs['o2sat']; etco2 = inputs['etco2']
    lactate = inputs['lactate']; wbc = inputs['wbc']; creatinine = inputs['creatinine']
    glucose = inputs['glucose']; bun = inputs['bun']; hgb = inputs['hgb']
    hct = inputs['hct']; platelets = inputs['platelets']; potassium = inputs['potassium']
    sodium = inputs['sodium']; ph = inputs['ph']; paco2 = inputs['paco2']
    hco3 = inputs['hco3']; base_excess = inputs['base_excess']; fio2 = inputs['fio2']
    bilirubin_total = inputs['bilirubin_total']; ast = inputs['ast']
    troponin = inputs['troponin']; ptt = inputs['ptt']; fibrinogen = inputs['fibrinogen']
    magnesium = inputs['magnesium']; phosphate = inputs['phosphate']
    calcium = inputs['calcium']; chloride = inputs['chloride']
    icu_hours = inputs['icu_hours']; gender = inputs['gender']

    demo = {col: np.nan for col in feature_cols}

    # Vitals
    demo['Age'] = age
    demo['HR_mean'] = hr; demo['HR_max'] = hr+10; demo['HR_min'] = hr-8
    demo['HR_last'] = hr; demo['HR_std'] = 5.0; demo['HR_trend'] = 0.0
    demo['HR_missing_pct'] = 0.0; demo['HR_was_measured'] = 1
    demo['Temp_mean'] = temp; demo['Temp_last'] = temp
    demo['Temp_max'] = temp+0.2; demo['Temp_min'] = temp-0.2
    demo['Temp_missing_pct'] = 0.0; demo['Temp_was_measured'] = 1
    demo['Resp_mean'] = resp; demo['Resp_last'] = resp; demo['Resp_max'] = resp+2
    demo['Resp_missing_pct'] = 0.0; demo['Resp_was_measured'] = 1
    demo['SBP_mean'] = sbp; demo['SBP_last'] = sbp; demo['SBP_min'] = sbp-5
    demo['SBP_missing_pct'] = 0.0; demo['SBP_was_measured'] = 1
    demo['DBP_mean'] = dbp; demo['DBP_last'] = dbp
    demo['DBP_missing_pct'] = 0.0; demo['DBP_was_measured'] = 1
    demo['MAP_mean'] = map_val; demo['MAP_last'] = map_val
    demo['MAP_missing_pct'] = 0.0; demo['MAP_was_measured'] = 1
    demo['O2Sat_mean'] = o2sat; demo['O2Sat_last'] = o2sat; demo['O2Sat_min'] = o2sat-1
    demo['O2Sat_missing_pct'] = 0.0; demo['O2Sat_was_measured'] = 1
    demo['EtCO2_mean'] = etco2; demo['EtCO2_last'] = etco2
    demo['EtCO2_missing_pct'] = 0.0; demo['EtCO2_was_measured'] = 1

    # Labs
    demo['Lactate_mean'] = lactate; demo['Lactate_last'] = lactate; demo['Lactate_max'] = lactate+0.2
    demo['Lactate_missing_pct'] = 0.0; demo['Lactate_was_measured'] = 1
    demo['WBC_mean'] = wbc; demo['WBC_last'] = wbc
    demo['WBC_missing_pct'] = 0.0; demo['WBC_was_measured'] = 1
    demo['Creatinine_mean'] = creatinine; demo['Creatinine_last'] = creatinine
    demo['Creatinine_missing_pct'] = 0.0; demo['Creatinine_was_measured'] = 1
    demo['Glucose_mean'] = glucose; demo['Glucose_last'] = glucose
    demo['Glucose_missing_pct'] = 0.0; demo['Glucose_was_measured'] = 1
    demo['BUN_mean'] = bun; demo['BUN_last'] = bun
    demo['BUN_missing_pct'] = 0.0; demo['BUN_was_measured'] = 1
    demo['Hgb_mean'] = hgb; demo['Hgb_last'] = hgb
    demo['Hgb_missing_pct'] = 0.0; demo['Hgb_was_measured'] = 1
    demo['Hct_mean'] = hct; demo['Hct_last'] = hct
    demo['Hct_missing_pct'] = 0.0; demo['Hct_was_measured'] = 1
    demo['Platelets_mean'] = platelets; demo['Platelets_last'] = platelets
    demo['Platelets_missing_pct'] = 0.0; demo['Platelets_was_measured'] = 1
    demo['Potassium_mean'] = potassium; demo['Potassium_last'] = potassium
    demo['Potassium_missing_pct'] = 0.0; demo['Potassium_was_measured'] = 1
    demo['pH_mean'] = ph; demo['pH_last'] = ph
    demo['pH_missing_pct'] = 0.0; demo['pH_was_measured'] = 1
    demo['PaCO2_mean'] = paco2; demo['PaCO2_last'] = paco2
    demo['PaCO2_missing_pct'] = 0.0; demo['PaCO2_was_measured'] = 1
    demo['HCO3_mean'] = hco3; demo['HCO3_last'] = hco3
    demo['HCO3_missing_pct'] = 0.0; demo['HCO3_was_measured'] = 1
    demo['BaseExcess_mean'] = base_excess; demo['BaseExcess_last'] = base_excess
    demo['BaseExcess_missing_pct'] = 0.0; demo['BaseExcess_was_measured'] = 1
    demo['FiO2_mean'] = fio2; demo['FiO2_last'] = fio2
    demo['FiO2_missing_pct'] = 0.0; demo['FiO2_was_measured'] = 1
    demo['Bilirubin_total_mean'] = bilirubin_total; demo['Bilirubin_total_last'] = bilirubin_total
    demo['Bilirubin_total_missing_pct'] = 0.0; demo['Bilirubin_total_was_measured'] = 1
    demo['AST_mean'] = ast; demo['AST_last'] = ast
    demo['AST_missing_pct'] = 0.0; demo['AST_was_measured'] = 1
    demo['TroponinI_mean'] = troponin; demo['TroponinI_last'] = troponin
    demo['TroponinI_missing_pct'] = 0.0; demo['TroponinI_was_measured'] = 1
    demo['PTT_mean'] = ptt; demo['PTT_last'] = ptt
    demo['PTT_missing_pct'] = 0.0; demo['PTT_was_measured'] = 1
    demo['Fibrinogen_mean'] = fibrinogen; demo['Fibrinogen_last'] = fibrinogen
    demo['Fibrinogen_missing_pct'] = 0.0; demo['Fibrinogen_was_measured'] = 1
    demo['Magnesium_mean'] = magnesium; demo['Magnesium_last'] = magnesium
    demo['Magnesium_missing_pct'] = 0.0; demo['Magnesium_was_measured'] = 1
    demo['Phosphate_mean'] = phosphate; demo['Phosphate_last'] = phosphate
    demo['Phosphate_missing_pct'] = 0.0; demo['Phosphate_was_measured'] = 1
    demo['Calcium_mean'] = calcium; demo['Calcium_last'] = calcium
    demo['Calcium_missing_pct'] = 0.0; demo['Calcium_was_measured'] = 1
    demo['Chloride_mean'] = chloride; demo['Chloride_last'] = chloride
    demo['Chloride_missing_pct'] = 0.0; demo['Chloride_was_measured'] = 1

    # Derived
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

    X = pd.DataFrame([demo])[feature_cols]
    X_imp = imputer.transform(X)
    xgb_prob = xgb_model.predict_proba(X_imp)[0, 1]
    rf_prob  = rf_model.predict_proba(X_imp)[0, 1]
    prob     = config['w_xgb'] * xgb_prob + config['w_rf'] * rf_prob
    return prob, xgb_prob, rf_prob


def risk_info(p):
    if p >= 0.6:  return "HIGH RISK",   "risk-high",   "tag-h"
    if p >= 0.35: return "MODERATE",    "risk-medium",  "tag-m"
    return             "LOW RISK",    "risk-low",    "tag-l"


RANGES = {
    'hr':           (60, 100),
    'temp':         (36.0, 37.9),
    'resp':         (12, 20),
    'sbp':          (90, 140),
    'dbp':          (60, 90),
    'map_val':      (70, 100),
    'o2sat':        (95, 100),
    'etco2':        (35, 45),
    'lactate':      (0, 2.0),
    'wbc':          (4.0, 11.0),
    'creatinine':   (0.6, 1.2),
    'glucose':      (70, 140),
    'bun':          (7, 25),
    'hgb':          (12, 17),
    'hct':          (36, 50),
    'platelets':    (150, 400),
    'potassium':    (3.5, 5.1),
    'ph':           (7.35, 7.45),
    'paco2':        (35, 45),
    'hco3':         (22, 29),
    'base_excess':  (-2, 2),
    'fio2':         (0.21, 0.6),
    'bilirubin_total': (0.2, 1.2),
    'ast':          (10, 40),
    'troponin':     (0, 0.04),
    'ptt':          (25, 35),
    'fibrinogen':   (200, 400),
    'magnesium':    (1.7, 2.2),
    'phosphate':    (2.5, 4.5),
    'calcium':      (8.5, 10.5),
    'chloride':     (98, 106),
}


# ─── SIDEBAR ─────────────────────────────────
with st.sidebar:
    st.markdown("### SEPSIS WATCH")
    st.markdown("<p style='color:#95a5a6;font-size:0.78rem;margin-top:-10px'>ICU Early Warning — PhysioNet 2019</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='sidebar-section'>Patient</div>", unsafe_allow_html=True)
    patient_name = st.text_input("Patient ID", value=f"PT-{st.session_state.patient_counter:03d}", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1: age = st.number_input("Age", 18, 100, 65, label_visibility="visible")
    with c2: gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    icu_hours = st.slider("ICU Hours", 1, 72, 12)

    st.markdown("<div class='sidebar-section'>Vitals</div>", unsafe_allow_html=True)
    hr     = st.slider("Heart Rate (bpm)",        40, 180, 88)
    temp   = st.slider("Temperature (C)",         35.0, 42.0, 37.2, step=0.1)
    resp   = st.slider("Respiratory Rate /min",   8, 40, 16)
    sbp    = st.slider("Systolic BP mmHg",        60, 200, 122)
    dbp    = st.slider("Diastolic BP mmHg",       30, 130, 78)
    map_val = st.slider("MAP mmHg",               40, 160, 93)
    o2sat  = st.slider("O2 Saturation %",         70, 100, 98)
    etco2  = st.slider("EtCO2 mmHg",              10, 70, 38)

    st.markdown("<div class='sidebar-section'>Blood Gases / Acid-Base</div>", unsafe_allow_html=True)
    ph     = st.slider("pH",                     7.0, 7.7, 7.40, step=0.01)
    paco2  = st.slider("PaCO2 mmHg",             15, 80, 40)
    hco3   = st.slider("HCO3 mEq/L",             10, 40, 24)
    base_excess = st.slider("Base Excess",        -20.0, 20.0, 0.0, step=0.5)
    fio2   = st.slider("FiO2",                   0.21, 1.0, 0.21, step=0.01)

    st.markdown("<div class='sidebar-section'>CBC</div>", unsafe_allow_html=True)
    wbc        = st.slider("WBC x10^3/uL",       1.0, 30.0, 7.5, step=0.5)
    hgb        = st.slider("Hgb g/dL",           4.0, 20.0, 14.0, step=0.1)
    hct        = st.slider("Hct %",              15.0, 60.0, 42.0, step=0.5)
    platelets  = st.slider("Platelets x10^3/uL", 20, 600, 220)

    st.markdown("<div class='sidebar-section'>Metabolic / Chemistry</div>", unsafe_allow_html=True)
    glucose    = st.slider("Glucose mg/dL",      40, 400, 105)
    sodium     = st.slider("Sodium mEq/L",       120, 160, 140)
    potassium  = st.slider("Potassium mEq/L",    2.5, 7.0, 4.0, step=0.1)
    chloride   = st.slider("Chloride mEq/L",     85, 120, 102)
    calcium    = st.slider("Calcium mg/dL",      5.0, 14.0, 9.2, step=0.1)
    magnesium  = st.slider("Magnesium mg/dL",    0.5, 5.0, 1.9, step=0.1)
    phosphate  = st.slider("Phosphate mg/dL",    1.0, 10.0, 3.5, step=0.1)

    st.markdown("<div class='sidebar-section'>Renal / Hepatic / Coag</div>", unsafe_allow_html=True)
    creatinine    = st.slider("Creatinine mg/dL",   0.4, 8.0, 0.9, step=0.1)
    bun           = st.slider("BUN mg/dL",           2, 100, 15)
    lactate       = st.slider("Lactate mmol/L",      0.5, 10.0, 1.1, step=0.1)
    bilirubin_total = st.slider("Bilirubin total mg/dL", 0.1, 20.0, 0.6, step=0.1)
    ast           = st.slider("AST U/L",              5, 500, 22)
    ptt           = st.slider("PTT sec",              15, 120, 30)
    fibrinogen    = st.slider("Fibrinogen mg/dL",     50, 800, 280)
    troponin      = st.slider("TroponinI ug/L",       0.0, 5.0, 0.01, step=0.01)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        predict_btn = st.button("Run Prediction", use_container_width=True, type="primary")
    with col_b:
        clear_btn = st.button("Clear History", use_container_width=True)

    if clear_btn:
        st.session_state.patient_history = []
        st.session_state.patient_counter = 1
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='font-size:0.68rem;color:#aaa'>Quick presets:</p>", unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        if st.button("High Risk", use_container_width=True):
            st.session_state['preset'] = 'high'
    with pc2:
        if st.button("Low Risk", use_container_width=True):
            st.session_state['preset'] = 'low'

# Apply presets — override locals
if st.session_state.get('preset') == 'high':
    hr, temp, resp, sbp, dbp, o2sat = 118, 38.7, 24, 88, 55, 93
    lactate, wbc, creatinine = 3.2, 15.0, 1.8
    ph, paco2, hco3, base_excess = 7.28, 50, 18, -6.0
    glucose, bun = 185, 38
    st.session_state.pop('preset')
elif st.session_state.get('preset') == 'low':
    hr, temp, resp, sbp, dbp, o2sat = 72, 36.8, 15, 128, 82, 99
    lactate, wbc, creatinine = 0.9, 7.0, 0.8
    ph, paco2, hco3, base_excess = 7.41, 39, 25, 0.5
    glucose, bun = 95, 12
    st.session_state.pop('preset')

# Pack inputs
inputs = dict(
    age=age, hr=hr, temp=temp, resp=resp, sbp=sbp, dbp=dbp, map_val=map_val,
    o2sat=o2sat, etco2=etco2, lactate=lactate, wbc=wbc, creatinine=creatinine,
    glucose=glucose, bun=bun, hgb=hgb, hct=hct, platelets=platelets,
    potassium=potassium, sodium=sodium, ph=ph, paco2=paco2, hco3=hco3,
    base_excess=base_excess, fio2=fio2, bilirubin_total=bilirubin_total,
    ast=ast, troponin=troponin, ptt=ptt, fibrinogen=fibrinogen,
    magnesium=magnesium, phosphate=phosphate, calcium=calcium, chloride=chloride,
    icu_hours=icu_hours, gender=gender
)


# ─── MAIN ─────────────────────────────────────
st.markdown("# SEPSIS WATCH")
st.markdown("<p style='color:#95a5a6;margin-top:-14px;font-size:0.85rem'>Early Warning System — PhysioNet 2019 · XGBoost + Random Forest Ensemble · For educational use only</p>", unsafe_allow_html=True)
st.divider()

if not models_loaded:
    st.error("Model files not found. Place xgb_sepsis_model.pkl, rf_sepsis_model.pkl, feature_cols.pkl, and ensemble_config.json in the same directory as app.py.")
    st.stop()

prob, xgb_prob, rf_prob = predict(inputs)
label, css_class, tag_cls = risk_info(prob)

if predict_btn:
    entry = {
        'time': datetime.now().strftime("%H:%M:%S"),
        'patient': patient_name,
        'prob': prob,
        'label': label,
        'xgb': xgb_prob,
        'rf': rf_prob,
        'hr': hr, 'temp': temp, 'sbp': sbp,
        'lactate': lactate, 'wbc': wbc,
    }
    st.session_state.patient_history.insert(0, entry)
    st.session_state.patient_counter += 1

# ── TOP ROW
col_risk, col_gauge = st.columns([1.3, 1])

with col_risk:
    st.markdown(f"""
    <div class='risk-banner {css_class}'>
        <div class='risk-label'>Sepsis Probability</div>
        <div class='risk-pct'>{prob:.1%}</div>
        <div class='risk-name'>{label}</div>
        <div style='font-size:0.72rem;color:#95a5a6;margin-top:8px'>
            Threshold: {config['best_threshold']:.0%} &nbsp;&middot;&nbsp; Patient: {patient_name}
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class='model-pill'>
            <div class='model-pill-val' style='color:#2980b9'>{xgb_prob:.1%}</div>
            <div class='model-pill-name'>XGBoost</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='model-pill'>
            <div class='model-pill-val' style='color:#8e44ad'>{rf_prob:.1%}</div>
            <div class='model-pill-name'>Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        qsofa = int(resp >= 22) + int(sbp <= 100)
        st.markdown(f"""<div class='model-pill'>
            <div class='model-pill-val' style='color:{"#c0392b" if qsofa >= 2 else "#d68910" if qsofa == 1 else "#1e8449"}'>{qsofa}/2</div>
            <div class='model-pill-name'>qSOFA</div>
        </div>""", unsafe_allow_html=True)

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 32, 'color': '#1a1a2e', 'family': 'DM Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#bdc3c7', 'tickfont': {'color': '#95a5a6', 'size': 10}},
            'bar':  {'color': '#c0392b' if prob >= 0.6 else '#d68910' if prob >= 0.35 else '#1e8449', 'thickness': 0.2},
            'bgcolor': '#f4f1ec',
            'bordercolor': '#ddd9d0',
            'steps': [
                {'range': [0, 35],  'color': 'rgba(30,132,73,0.06)'},
                {'range': [35, 60], 'color': 'rgba(214,137,16,0.06)'},
                {'range': [60, 100],'color': 'rgba(192,57,43,0.06)'},
            ],
            'threshold': {
                'line': {'color': '#c0392b', 'width': 2},
                'thickness': 0.75,
                'value': config['best_threshold'] * 100
            }
        }
    ))
    fig_gauge.update_layout(
        height=240, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1a1a2e', 'family': 'DM Sans'}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ── VITALS PANEL
st.markdown("<div class='section-header'>Vital Signs</div>", unsafe_allow_html=True)

vital_items = [
    ("HR", hr, "bpm", 'hr'), ("Temp", temp, "C", 'temp'),
    ("Resp", resp, "/min", 'resp'), ("SBP", sbp, "mmHg", 'sbp'),
    ("DBP", dbp, "mmHg", 'dbp'), ("MAP", map_val, "mmHg", 'map_val'),
    ("O2 Sat", o2sat, "%", 'o2sat'), ("EtCO2", etco2, "mmHg", 'etco2'),
]

cols = st.columns(8)
for i, (name, val, unit, key) in enumerate(vital_items):
    lo, hi = RANGES.get(key, (None, None))
    abn = lo is not None and (val < lo or val > hi)
    card_cls = "abn" if abn else "norm"
    val_cls  = "abn-val" if abn else ""
    display  = f"{val:.1f}" if isinstance(val, float) else str(val)
    with cols[i]:
        st.markdown(f"""<div class='vcard {card_cls}'>
            <div class='vcard-val {val_cls}'>{display}</div>
            <div class='vcard-name'>{name}</div>
            <div class='vcard-unit'>{unit}</div>
        </div>""", unsafe_allow_html=True)

# ── LABS PANEL
st.markdown("<div class='section-header'>Laboratory Values</div>", unsafe_allow_html=True)

lab_row1 = [
    ("Lactate", lactate, "mmol/L", 'lactate'), ("WBC", wbc, "k/uL", 'wbc'),
    ("Creatinine", creatinine, "mg/dL", 'creatinine'), ("BUN", bun, "mg/dL", 'bun'),
    ("Glucose", glucose, "mg/dL", 'glucose'), ("Hgb", hgb, "g/dL", 'hgb'),
    ("Hct", hct, "%", 'hct'), ("Platelets", platelets, "k/uL", 'platelets'),
]
lab_row2 = [
    ("pH", ph, "", 'ph'), ("PaCO2", paco2, "mmHg", 'paco2'),
    ("HCO3", hco3, "mEq/L", 'hco3'), ("Base Exc", base_excess, "", 'base_excess'),
    ("FiO2", fio2, "", 'fio2'), ("K+", potassium, "mEq/L", 'potassium'),
    ("Ca", calcium, "mg/dL", 'calcium'), ("Mg", magnesium, "mg/dL", 'magnesium'),
]
lab_row3 = [
    ("Cl-", chloride, "mEq/L", 'chloride'), ("Phos", phosphate, "mg/dL", 'phosphate'),
    ("Bili", bilirubin_total, "mg/dL", 'bilirubin_total'), ("AST", ast, "U/L", 'ast'),
    ("TropI", troponin, "ug/L", 'troponin'), ("PTT", ptt, "sec", 'ptt'),
    ("Fibrin", fibrinogen, "mg/dL", 'fibrinogen'), ("Na+", sodium, "mEq/L", None),
]

def render_lab_row(items):
    cols = st.columns(8)
    for i, (name, val, unit, key) in enumerate(items):
        lo, hi = RANGES.get(key, (None, None)) if key else (None, None)
        abn = lo is not None and (val < lo or val > hi)
        card_cls = "abn" if abn else "norm"
        val_cls  = "abn-val" if abn else ""
        display  = f"{val:.2f}" if isinstance(val, float) and abs(val) < 10 else (f"{val:.1f}" if isinstance(val, float) else str(val))
        with cols[i]:
            st.markdown(f"""<div class='vcard {card_cls}'>
                <div class='vcard-val {val_cls}' style='font-size:1.1rem'>{display}</div>
                <div class='vcard-name'>{name}</div>
                <div class='vcard-unit'>{unit}</div>
            </div>""", unsafe_allow_html=True)

render_lab_row(lab_row1)
render_lab_row(lab_row2)
render_lab_row(lab_row3)

st.divider()

# ── RADAR + HISTORY
col_radar, col_hist = st.columns([1, 1.2])

with col_radar:
    st.markdown("<div class='section-header'>Risk Factor Profile</div>", unsafe_allow_html=True)

    def norm(val, lo, hi):
        return min(max((val - lo) / (hi - lo + 1e-6), 0), 1)

    categories = ['HR', 'Temp', 'Resp', 'BP (low)', 'O2 (low)', 'Lactate', 'WBC', 'Creatinine', 'pH (low)', 'Glucose']
    values = [
        norm(hr, 60, 120),
        norm(temp, 36, 39.5),
        norm(resp, 12, 30),
        norm(140 - sbp, 0, 80),
        norm(100 - o2sat, 0, 10),
        norm(lactate, 0, 5),
        norm(wbc, 4, 18),
        norm(creatinine, 0.6, 4),
        norm(7.45 - ph, 0, 0.3),
        norm(glucose, 70, 300),
    ]

    line_col   = '#c0392b' if prob >= 0.6 else '#d68910' if prob >= 0.35 else '#1e8449'
    fill_col   = 'rgba(192,57,43,0.1)' if prob >= 0.6 else 'rgba(214,137,16,0.1)' if prob >= 0.35 else 'rgba(30,132,73,0.1)'

    values_c = values + [values[0]]
    cats_c   = categories + [categories[0]]

    fig_r = go.Figure(go.Scatterpolar(
        r=values_c, theta=cats_c,
        fill='toself', fillcolor=fill_col,
        line=dict(color=line_col, width=2),
    ))
    fig_r.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], tickfont={'color':'#bdc3c7','size':8}, gridcolor='#ddd9d0'),
            angularaxis=dict(tickfont={'color':'#7f8c8d','size':10}, gridcolor='#ddd9d0')
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(t=20, b=20, l=40, r=40),
        font={'color':'#1a1a2e', 'family':'DM Sans'},
        showlegend=False
    )
    st.plotly_chart(fig_r, use_container_width=True)

with col_hist:
    st.markdown("<div class='section-header'>Prediction History</div>", unsafe_allow_html=True)

    if not st.session_state.patient_history:
        st.markdown("<p style='color:#bdc3c7;text-align:center;padding:40px 0;font-size:0.85rem'>No predictions logged yet.</p>", unsafe_allow_html=True)
    else:
        if len(st.session_state.patient_history) > 1:
            hist_df = pd.DataFrame(st.session_state.patient_history[::-1])
            fig_t = go.Figure(go.Scatter(
                x=hist_df['time'], y=hist_df['prob']*100,
                mode='lines+markers',
                line=dict(color='#2980b9', width=2),
                marker=dict(
                    size=7,
                    color=['#c0392b' if p >= 0.6 else '#d68910' if p >= 0.35 else '#1e8449'
                           for p in hist_df['prob']],
                    line=dict(color='#f4f1ec', width=1)
                ),
                fill='tozeroy',
                fillcolor='rgba(41,128,185,0.05)'
            ))
            fig_t.add_hline(y=config['best_threshold']*100, line_dash='dot',
                            line_color='#c0392b', opacity=0.6,
                            annotation_text='threshold', annotation_font_color='#c0392b',
                            annotation_font_size=10)
            fig_t.update_layout(
                height=160, margin=dict(t=10, b=30, l=40, r=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, tickfont={'color':'#bdc3c7','size':9}),
                yaxis=dict(gridcolor='#ddd9d0', tickfont={'color':'#bdc3c7','size':9},
                           range=[0, 100], title='Risk %', title_font={'color':'#95a5a6','size':9}),
                font={'color':'#1a1a2e'}
            )
            st.plotly_chart(fig_t, use_container_width=True)

        for entry in st.session_state.patient_history[:8]:
            tag_c = 'tag-h' if entry['label'] == 'HIGH RISK' else \
                    'tag-m' if entry['label'] == 'MODERATE' else 'tag-l'
            st.markdown(f"""
            <div class='hist-row'>
                <span style='color:#bdc3c7'>{entry['time']}</span>
                <span style='color:#2c3e50'>{entry['patient']}</span>
                <span class='{tag_c}'>{entry['prob']:.1%}</span>
                <span class='{tag_c}'>{entry['label']}</span>
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align:center;color:#bdc3c7;font-size:0.7rem'>For educational demonstration only &middot; Not for clinical use</p>", unsafe_allow_html=True)