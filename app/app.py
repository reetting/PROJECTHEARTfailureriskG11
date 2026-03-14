import os
import sys
import pickle
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", message=".*use_container_width.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, precision_score, recall_score
)
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from src.data_processing import load_data, handle_outliers, optimize_memory
from src.SHAP import get_shap_explainer, compute_shap_values, plot_waterfall_single, get_top_features
from fpdf import FPDF

# ==========================================
# 1. CONFIGURATION & STATE MANAGEMENT
# ==========================================
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🧡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "app_state" not in st.session_state:
    st.session_state.app_state = "booting"
if "current_page" not in st.session_state:
    st.session_state.current_page = "input"
if "proba" not in st.session_state:
    st.session_state.proba = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# ==========================================
# 2. CHARGEMENT DU MOTEUR ML
# ==========================================
FEATURE_NAMES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "heart_failure_clinical_records_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")

@st.cache_resource(show_spinner=False)
def get_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def get_test_data():
    df = load_data(DATA_PATH)
    df = handle_outliers(df)
    df = optimize_memory(df)
    X, y = df[FEATURE_NAMES], df["DEATH_EVENT"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
MODELS_RESULTS = {
    "Random Forest":       {"Recall": 0.6316, "Precision": 0.8571, "ROC-AUC": 0.9072, "F1-Score": 0.7273, "Accuracy": 0.8500},
    "XGBoost":             {"Recall": 0.5789, "Precision": 0.8462, "ROC-AUC": 0.8238, "F1-Score": 0.6875, "Accuracy": 0.8333},
    "LightGBM":            {"Recall": 0.6842, "Precision": 0.8667, "ROC-AUC": 0.8652, "F1-Score": 0.7647, "Accuracy": 0.8667},
    "LightGBM Equilibre":  {"Recall": 0.9512, "Precision": 0.8864, "ROC-AUC": 0.9780, "F1-Score": 0.9176, "Accuracy": 0.9146},
    "Ensemble Averaging":  {"Recall": 0.8947, "Precision": 0.9444, "ROC-AUC": 0.9769, "F1-Score": 0.9189, "Accuracy": 0.9500},
}
def generate_pdf_report(patient: dict, proba: float) -> str:
    risk_label = (
    "RISQUE CRITIQUE" if proba >= 0.65 else
    "RISQUE MODERE"   if proba >= 0.40 else
    "RISQUE FAIBLE"
    )
    pdf = FPDF()
    pdf.add_page()
    # En-tete
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 15, "CardioCare AI", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 8, "Rapport de Diagnostic", ln=True, align="C")
    pdf.ln(10)
    # Resultat
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, f"Diagnostic : {risk_label}", ln=True)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, f"Probabilite de deces : {proba*100:.1f}%", ln=True)
    pdf.ln(5)
    # Ligne separatrice
    pdf.set_draw_color(41, 128, 185)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    # Profil patient
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 10, "Profil Clinique", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(44, 62, 80)
    labels = {
        "age":                      "Age",
        "sex":                      "Sexe",
        "ejection_fraction":        "Fraction d ejection (%)",
        "serum_creatinine":         "Creatinine serique (mg/dL)",
        "serum_sodium":             "Sodium serique (mEq/L)",
        "creatinine_phosphokinase": "Enzyme CPK (mcg/L)",
        "platelets":                "Plaquettes (k/mL)",
        "anaemia":                  "Anemie",
        "diabetes":                 "Diabete",
        "high_blood_pressure":      "Hypertension",
        "smoking":                  "Fumeur",
        "time":                     "Periode de suivi (jours)",
    }
    for key, label in labels.items():
        value = patient[key]
        if key == "sex":
            value = "Homme" if value == 1 else "Femme"
        elif key in ["anaemia", "diabetes", "high_blood_pressure", "smoking"]:
            value = "Oui" if value == 1 else "Non"
        pdf.cell(100, 8, label, border="B")
        pdf.cell(0, 8, str(value), border="B", ln=True)
    pdf.ln(5)
    # Avertissement
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(127, 140, 141)
    pdf.multi_cell(0, 6,
        "Avertissement : Ce rapport est genere a des fins educatives uniquement. "
        "Il ne se substitue pas a un diagnostic medical professionnel."
    )
    path = "/tmp/cardiocare_rapport.pdf"
    pdf.output(path)
    return path
model = get_model()
X_train, X_test, y_train, y_test = get_test_data()
explainer = get_shap_explainer(model, X_train)

# ==========================================
# 3. CSS — COULEURS NOTRE CODE + STRUCTURE NOUVEAU
# ==========================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif !important;
            color: #4A403B !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stHeader"] { display: none !important; }

        .block-container { max-width: 1300px; padding-top: 2rem; padding-bottom: 2rem; }

        /* FOND — couleurs notre code */
        .stApp {
            background: linear-gradient(-45deg, #f8fafc, #e0eaf5, #eef2f3, #e0eaf5);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            overflow-x: hidden;
        }
        .stApp::before {
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: radial-gradient(#b0c4de 1px, transparent 1px);
            background-size: 30px 30px;
            opacity: 0.4;
            pointer-events: none;
            z-index: 0;
        }
        .block-container { position: relative; z-index: 1; }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* TYPOGRAPHIE */
        h1, h2, h3, h4 { color: #2C3E50 !important; letter-spacing: -0.02em; }
        .accent-text  { color: #2980B9; }
        .danger-text  { color: #E74C3C; }
        .safe-text    { color: #27AE60; }
        .sub-text     { color: #7F8C8D !important; }

        /* ANIMATION PAGE */
        .page-animate {
            animation: slideFadeIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            opacity: 0;
        }
        @keyframes slideFadeIn {
            0%   { opacity: 0; transform: translateY(15px) scale(0.99); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* GLASSMORPHISM CARDS */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.75) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.6) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }

        /* NAVIGATION */
        .nav-container div.stButton > button {
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(41, 128, 185, 0.2);
            color: #7F8C8D !important;
            font-weight: 700;
            border-radius: 10px;
            padding: 12px 24px;
            width: 100%;
            transition: all 0.3s ease;
        }
        .nav-container div.stButton > button:hover {
            background: #FFF;
            border: 1px solid #2980B9;
            color: #2980B9 !important;
            box-shadow: 0 6px 15px rgba(41, 128, 185, 0.15);
        }
        .nav-active div.stButton > button {
            background: linear-gradient(90deg, #2980B9, #3498DB);
            border: 1px solid #2980B9;
            color: #FFFFFF !important;
            box-shadow: 0 8px 20px rgba(41, 128, 185, 0.3);
            animation: shimmerBtn 5s infinite linear;
        }
        @keyframes shimmerBtn {
            0%   { background-position: -200% center; }
            100% { background-position: 200% center; }
        }

        /* BOUTON ACTION */
        .action-btn div.stButton > button {
            background: linear-gradient(90deg, #2980B9 0%, #3498DB 50%, #2980B9 100%);
            background-size: 200% auto;
            color: white !important;
            font-weight: 800; font-size: 1.1rem;
            border-radius: 10px; border: none; padding: 15px; width: 100%;
            box-shadow: 0 8px 25px rgba(41, 128, 185, 0.4);
            transition: all 0.3s;
            animation: shimmerBtn 5s infinite linear;
        }
        .action-btn div.stButton > button:hover {
            box-shadow: 0 12px 30px rgba(41, 128, 185, 0.6);
            transform: scale(1.02);
        }

        /* ECG ANIMATION */
        .ecg-container {
            width: 100%; height: 70px; overflow: hidden; position: relative;
            margin-bottom: -15px;
        }
        .ecg-line {
            fill: none; stroke: #2980B9; stroke-width: 2.5;
            stroke-linecap: round; stroke-linejoin: round;
            stroke-dasharray: 1500; stroke-dashoffset: 1500;
            animation: drawECG 4s linear infinite;
            filter: drop-shadow(0 3px 4px rgba(41, 128, 185, 0.4));
        }
        @keyframes drawECG { to { stroke-dashoffset: 0; } }

        /* SCORE CIRCLE */
        .massive-score-container {
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; padding: 40px; text-align: center;
        }
        .score-circle {
            width: 260px; height: 260px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; flex-direction: column;
            background: #FFFFFF; font-size: 4.5rem; font-weight: 900;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.02);
        }
        @keyframes pulseSoft {
            0%   { transform: scale(0.98); box-shadow: 0 0 0 0 var(--pulse-color); }
            70%  { transform: scale(1.02); box-shadow: 0 0 0 40px rgba(0,0,0,0); }
            100% { transform: scale(0.98); box-shadow: 0 0 0 0 rgba(0,0,0,0); }
        }
        .pulse-slow   { --pulse-color: rgba(39, 174, 96, 0.4);   border: 4px solid #27AE60; color: #27AE60; animation: pulseSoft 2.5s infinite; }
        .pulse-medium { --pulse-color: rgba(243, 156, 18, 0.4);  border: 4px solid #F39C12; color: #F39C12; animation: pulseSoft 1.5s infinite; }
        .pulse-fast   { --pulse-color: rgba(231, 76, 60, 0.5);   border: 4px solid #E74C3C; color: #E74C3C; animation: pulseSoft 0.8s infinite; }

        /* CUSTOM CARDS */
        .custom-card {
            background-color: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px);
            padding: 20px; border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.05);
            border: 1px solid rgba(255,255,255,0.6);
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .custom-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.1);
        }
        .custom-card h4 { color: #2C3E50 !important; }

        /* SCANNER LOADING */
        .scanner-screen {
            height: 85vh; display: flex; flex-direction: column;
            align-items: center; justify-content: center; text-align: center;
        }
        .soft-pulse-loader {
            width: 80px; height: 80px; border-radius: 50%;
            background: #2980B9; margin-bottom: 30px;
            animation: pulseSoftLoader 2s infinite;
        }
        @keyframes pulseSoftLoader {
            0%   { transform: scale(0.8); opacity: 0.8; box-shadow: 0 0 0 0 rgba(41,128,185,0.4); }
            70%  { transform: scale(1);   opacity: 0.2; box-shadow: 0 0 0 40px rgba(41,128,185,0); }
            100% { transform: scale(0.8); opacity: 0.8; box-shadow: 0 0 0 0 rgba(41,128,185,0); }
        }

        [data-testid="stMetricLabel"] { color: #7F8C8D !important; }
        [data-testid="stMetricValue"] { color: #2C3E50 !important; }
        hr { border: 0; border-top: 1px solid rgba(0,0,0,0.05); margin: 1.5rem 0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. ÉCRAN DE CHARGEMENT
# ==========================================
if st.session_state.app_state == "booting":
    placeholder = st.empty()
    placeholder.markdown("""
        <div class="scanner-screen">
            <div class="soft-pulse-loader"></div>
            <h1 style="font-size:3rem; color:#2C3E50;">Préparation de votre espace<span style="color:#2980B9;">.</span></h1>
            <p style="color:#7F8C8D; font-size:1.1rem; margin-top:10px;">Initialisation des algorithmes de santé...</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.5)
    placeholder.empty()
    st.session_state.app_state = "ready"
    st.rerun()

# ==========================================
# 5. EN-TÊTE ECG + NAVIGATION
# ==========================================
st.markdown("""
    <div class="ecg-container">
        <svg viewBox="0 0 1000 60" preserveAspectRatio="none">
            <path class="ecg-line" d="M0,30 L200,30 L210,10 L220,50 L230,20 L240,40 L250,30 L450,30 L460,10 L480,55 L490,5 L500,30 L700,30 L710,20 L720,40 L730,30 L1000,30"></path>
        </svg>
    </div>
    <h1 style='text-align:center; font-weight:900; margin-bottom:30px;'>
        🫀 CardioCare <span class='accent-text'>AI</span>
    </h1>
""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4, nav5 = st.columns(5)
def nav_class(page): return "nav-active nav-container" if st.session_state.current_page == page else "nav-container"

with nav1:
    st.markdown(f"<div class='{nav_class('input')}'>", unsafe_allow_html=True)
    if st.button("1. Bilan Patient", use_container_width=True):
        st.session_state.current_page = "input"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with nav2:
    st.markdown(f"<div class='{nav_class('dashboard')}'>", unsafe_allow_html=True)
    if st.button("2. Vue d'Ensemble", use_container_width=True):
        st.session_state.current_page = "dashboard"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with nav3:
    st.markdown(f"<div class='{nav_class('shap')}'>", unsafe_allow_html=True)
    if st.button("3. Explication Médicale", use_container_width=True):
        st.session_state.current_page = "shap"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with nav4:
    st.markdown(f"<div class='{nav_class('perf')}'>", unsafe_allow_html=True)
    if st.button("4. Fiabilité IA", use_container_width=True):
        st.session_state.current_page = "perf"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with nav5:
    st.markdown(f"<div class='{nav_class('comparaison')}'>", unsafe_allow_html=True)
    if st.button("5. Comparaison", use_container_width=True):
        st.session_state.current_page = "comparaison"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='page-animate'>", unsafe_allow_html=True)
# ==========================================
# 6. VUE 1 : BILAN PATIENT
# ==========================================
if st.session_state.current_page == "input":
    st.markdown("### 📋 Formulaire de Santé")
    st.markdown("<p class='sub-text' style='margin-bottom:30px;'>Renseignez les informations avec soin.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("<h4 class='accent-text'>👤 Profil Personnel</h4>", unsafe_allow_html=True)
            age               = st.slider("Âge (années)", 18, 95, 60)
            sex               = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
            time_fup          = st.slider("Période de suivi (jours)", 4, 285, 100)

        with st.container(border=True):
            st.markdown("<h4 class='accent-text'>🩺 Signes Cardiaques</h4>", unsafe_allow_html=True)
            ejection_fraction   = st.slider("Fraction d'éjection (%)", 14, 80, 38)
            high_blood_pressure = st.toggle("Antécédents d'Hypertension")

    with col2:
        with st.container(border=True):
            st.markdown("<h4 class='accent-text'>🩸 Prises de Sang</h4>", unsafe_allow_html=True)
            serum_creatinine         = st.number_input("Créatinine sérique (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
            serum_sodium             = st.number_input("Sodium sérique (mEq/L)", 110.0, 150.0, 137.0, step=1.0)
            creatinine_phosphokinase = st.number_input("Enzyme CPK (mcg/L)", 20, 8000, 250, step=10)
            platelets                = st.number_input("Plaquettes (k/mL)", 25000, 850000, 265000, step=5000)

        with st.container(border=True):
            st.markdown("<h4 class='accent-text'>⚠️ Antécédents</h4>", unsafe_allow_html=True)
            c_a, c_d, c_s = st.columns(3)
            with c_a: anaemia  = st.toggle("Anémie")
            with c_d: diabetes = st.toggle("Diabète")
            with c_s: smoking  = st.toggle("Fumeur")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
    if st.button("🚀 LANCER L'ÉVALUATION", use_container_width=True):
        patient = {
            "age":                      age,
            "anaemia":                  int(anaemia),
            "creatinine_phosphokinase": creatinine_phosphokinase,
            "diabetes":                 int(diabetes),
            "ejection_fraction":        ejection_fraction,
            "high_blood_pressure":      int(high_blood_pressure),
            "platelets":                platelets,
            "serum_creatinine":         serum_creatinine,
            "serum_sodium":             serum_sodium,
            "sex":                      1 if sex == "Homme" else 0,
            "smoking":                  int(smoking),
            "time":                     time_fup,
        }
        st.session_state.patient_data = patient
        X_input = pd.DataFrame([patient])[FEATURE_NAMES]
        st.session_state.proba = float(model.predict_proba(X_input)[0][1])
        st.session_state.current_page = "dashboard"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 7. VUE 2 : RÉSULTAT
# ==========================================
elif st.session_state.current_page == "dashboard":
    if st.session_state.proba is None:
        st.warning("Veuillez d'abord remplir le Bilan Patient.")
        if st.button("← Retour au formulaire"):
            st.session_state.current_page = "input"
            st.rerun()
    else:
        proba = st.session_state.proba

        if proba >= 0.65:
            pulse_class = "pulse-fast"
            status_text = "RISQUE CRITIQUE"
            desc_text   = "Nos algorithmes détectent des signes d'alerte nécessitant un avis médical rapide."
            color_class = "danger-text"
            color       = "#E74C3C"
        elif proba >= 0.40:
            pulse_class = "pulse-medium"
            status_text = "RISQUE MODÉRÉ"
            desc_text   = "Certains biomarqueurs sont instables. Une surveillance est recommandée."
            color_class = "accent-text"
            color       = "#F39C12"
        else:
            pulse_class = "pulse-slow"
            status_text = "RISQUE FAIBLE"
            desc_text   = "Les indicateurs actuels ne montrent pas de risque critique immédiat."
            color_class = "safe-text"
            color       = "#27AE60"

        c_dash1, c_dash2 = st.columns([1, 1])

        with c_dash1:
            st.markdown("<div class='massive-score-container'>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-weight:bold; letter-spacing:1.5px; margin-bottom:20px;'>INDICE DE RISQUE</p>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='score-circle {pulse_class}'>
                    {proba*100:.1f}<span style='font-size:1.5rem;'>%</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin-top:30px;' class='{color_class}'>{status_text}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='sub-text' style='text-align:center;'>{desc_text}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_dash2:
            st.markdown("<h3 class='accent-text'>Résumé de l'analyse</h3>", unsafe_allow_html=True)
            p = st.session_state.patient_data
            with st.container(border=True):
                s1, s2 = st.columns(2)
                s1.metric("Fraction Éjection", f"{p['ejection_fraction']}%",
                          delta="Normal" if p['ejection_fraction'] > 30 else "À vérifier",
                          delta_color="normal" if p['ejection_fraction'] > 30 else "inverse")
                s2.metric("Créatinine", f"{p['serum_creatinine']} mg/dL",
                          delta="Anormal" if p['serum_creatinine'] > 1.5 else "OK",
                          delta_color="inverse" if p['serum_creatinine'] > 1.5 else "normal")
                s1.metric("Sodium",     f"{p['serum_sodium']} mEq/L")
                s2.metric("Enzyme CPK", f"{p['creatinine_phosphokinase']} mcg/L")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
            if st.button("🔬 COMPRENDRE CES RÉSULTATS", use_container_width=True):
                st.session_state.current_page = "shap"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('espace', unsafe_allow_html=True)
            if st.button('Telecharger le rapport PDF', use_container_width=True):
                pdf_path = generate_pdf_report(
                    st.session_state.patient_data,
                    st.session_state.proba
                )
                with open(pdf_path, 'rb') as f:
                    st.download_button(
                        label='Cliquer pour telecharger',
                        data=f,
                        file_name='cardiocare_rapport.pdf',
                        mime='application/pdf',
                        use_container_width=True)

# ==========================================
# 8. VUE 3 : EXPLICABILITÉ SHAP
# ==========================================
elif st.session_state.current_page == "shap":
    st.markdown("### 🔬 Transparence des Résultats")
    st.markdown("<p class='sub-text'>Découvrez comment l'IA a interprété vos données.</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Analyse de votre profil", "Critères généraux de l'IA"])

    with tab1:
        if st.session_state.patient_data is None:
            st.warning("Veuillez d'abord remplir le Bilan Patient.")
        else:
            X_input = pd.DataFrame([st.session_state.patient_data])[FEATURE_NAMES]

            col_shap1, col_shap2 = st.columns([1, 2])
            with col_shap1:
                with st.container(border=True):
                    st.markdown("<h4 class='accent-text'>Top 5 facteurs</h4>", unsafe_allow_html=True)
                    patient_shap = compute_shap_values(explainer, X_input)
                    if patient_shap.ndim > 1:
                        patient_shap = patient_shap[0:1]
                    top_features = get_top_features(patient_shap, FEATURE_NAMES, top_n=5)
                    for feat, score in top_features:
                        if score > 0:
                            st.markdown(f"<p><span class='danger-text' style='font-weight:bold;'>🔴 ↑ {feat}</span><br><span class='sub-text' style='font-size:0.85rem;'>Augmente le risque</span></p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p><span class='safe-text' style='font-weight:bold;'>🟢 ↓ {feat}</span><br><span class='sub-text' style='font-size:0.85rem;'>Réduit le risque</span></p>", unsafe_allow_html=True)

            with col_shap2:
                with st.container(border=True):
                    waterfall_path = "/tmp/patient_waterfall.png"
                    plot_waterfall_single(explainer, X_input, FEATURE_NAMES, save_path=waterfall_path)
                    st.image(waterfall_path, use_container_width=True)

    with tab2:
        with st.container(border=True):
            shap_values = compute_shap_values(explainer, X_test)
            mean_abs    = np.abs(shap_values).mean(axis=0)
            sorted_idx  = np.argsort(mean_abs)

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            ax.barh(
                [FEATURE_NAMES[i] for i in sorted_idx],
                mean_abs[sorted_idx],
                color="#2980B9", edgecolor="none"
            )
            ax.set_xlabel("Impact Moyen sur la Décision de l'IA", color="#2C3E50", weight="bold")
            ax.tick_params(colors="#2C3E50")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#7F8C8D")
            ax.spines["left"].set_color("#7F8C8D")
            plt.tight_layout()
            st.pyplot(fig, transparent=True, use_container_width=True)

# ==========================================
# 9. VUE 4 : PERFORMANCES
# ==========================================
elif st.session_state.current_page == "perf":
    st.markdown("### 📈 Fiabilité du Diagnostic")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>RECALL ⭐</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='danger-text' style='margin:0;'>{recall_score(y_test, y_pred):.4f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Priorité n°1</p>", unsafe_allow_html=True)
    with col_m2:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>PRECISION</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='accent-text' style='margin:0;'>{precision_score(y_test, y_pred):.4f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Priorité n°2</p>", unsafe_allow_html=True)
    with col_m3:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>ROC-AUC</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='safe-text' style='margin:0;'>{roc_auc_score(y_test, y_proba):.4f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Comparaison modèles</p>", unsafe_allow_html=True)
    with col_m4:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>F1-SCORE</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#8E44AD; margin:0;'>{f1_score(y_test, y_pred):.4f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Équilibre P/R</p>", unsafe_allow_html=True)
    with col_m5:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>ACCURACY</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='accent-text' style='margin:0;'>{accuracy_score(y_test, y_pred)*100:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Moyenne globale</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_conf1, c_conf2 = st.columns([1, 2])

    with c_conf2:
        with st.container(border=True):
            st.markdown("<h4 class='accent-text'>📊 Matrice de Confusion</h4>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(6, 4))
            fig_cm.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            cmap_custom = sns.light_palette("#2980B9", as_cmap=True)
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_custom,
                        cbar=False, annot_kws={"size": 18, "weight": "bold", "color": "#2C3E50"},
                        xticklabels=["Survie Prédite", "Décès Prédit"],
                        yticklabels=["Survie Réelle", "Décès Réel"], ax=ax)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#2C3E50", weight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), color="#2C3E50", weight="bold")
            st.pyplot(fig_cm, transparent=True, use_container_width=True)
elif st.session_state.current_page == "comparaison":
    st.markdown("### 📊 Comparaison des Modèles")
    st.markdown("<p class='sub-text'>Performances comparées de tous les modèles testés.</p>",
                unsafe_allow_html=True)

    metrics_order = ["Recall", "Precision", "ROC-AUC", "F1-Score", "Accuracy"]
    rows = []
    for model_name, scores in MODELS_RESULTS.items():
        row = [model_name] + [f"{scores[m]:.4f}" for m in metrics_order]
        rows.append(row)

    df_compare = pd.DataFrame(rows, columns=["Modèle"] + metrics_order)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"""
        <div class='custom-card' style='border-left:6px solid #27AE60;'>
            <h4 style='color:#27AE60;'>✅ Modèle choisi : LightGBM Equilibre</h4>
            <p style='color:#2C3E50;'>
                Meilleur Recall (0.9512) et meilleur ROC-AUC (0.9780).<br/>
                Dans un contexte médical, le Recall est la métrique prioritaire
                car rater un patient en danger est plus grave qu'une fausse alarme.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 10. PIED DE PAGE
# ==========================================
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:#7F8C8D; font-size:13px; margin-top:20px;'>
        <p><strong>⚠️ Avertissement :</strong> Application développée à des fins éducatives et de recherche.<br>
        Ne se substitue en aucun cas à un diagnostic ou un avis médical professionnel.</p>
    </div>
""", unsafe_allow_html=True)
