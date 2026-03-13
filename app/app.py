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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, precision_score, recall_score
)
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from src.data_processing import load_data, handle_outliers, optimize_memory
from src.SHAP import get_shap_explainer, compute_shap_values, plot_waterfall_single, get_top_features

# ==========================================
# 1. CONFIGURATION & STATE MANAGEMENT
# ==========================================
st.set_page_config(
    page_title="CardioCare",
    page_icon="🧡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

plt.style.use('default')

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

model = get_model()
X_train, X_test, y_train, y_test = get_test_data()
explainer = get_shap_explainer(model, X_train)

# ==========================================
# 3. CSS
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

        .stApp {
            background: linear-gradient(-45deg, #FDFBF7, #FFF5ED, #FEF9F0, #FDFBF7);
            background-size: 400% 400%;
            animation: warmBreeze 20s ease-in-out infinite;
            overflow-x: hidden;
        }
        @keyframes warmBreeze {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        h1, h2, h3, h4 { color: #3A302B !important; letter-spacing: -0.02em; }
        .accent-text { color: #E28743; }
        .danger-text { color: #D65A31; }
        .safe-text { color: #6C9A7A; }
        .sub-text { color: #8B7E74 !important; }

        .page-animate {
            animation: slideFadeIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            opacity: 0;
            transform: translateY(15px);
        }
        @keyframes slideFadeIn {
            0% { opacity: 0; transform: translateY(15px) scale(0.99); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.7) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
            border-radius: 24px !important;
            box-shadow: 0 10px 30px rgba(139, 100, 70, 0.06);
            transition: all 0.3s ease;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 15px 40px rgba(139, 100, 70, 0.12);
            transform: translateY(-2px);
        }

        .nav-container div.stButton > button {
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(226, 135, 67, 0.2);
            color: #8B7E74 !important;
            font-weight: 700;
            border-radius: 16px;
            padding: 12px 24px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(0,0,0,0.02);
        }
        .nav-container div.stButton > button:hover {
            background: #FFF;
            border: 1px solid #E28743;
            color: #E28743 !important;
            box-shadow: 0 6px 15px rgba(226, 135, 67, 0.15);
        }
        .nav-active div.stButton > button {
            background: #E28743;
            border: 1px solid #E28743;
            color: #FFFFFF !important;
            box-shadow: 0 8px 20px rgba(226, 135, 67, 0.25);
        }

        .action-btn div.stButton > button {
            background: linear-gradient(135deg, #E28743, #D65A31);
            color: white !important;
            font-weight: 800; font-size: 1.1rem; letter-spacing: 0.5px;
            border-radius: 16px; border: none; padding: 15px; width: 100%;
            box-shadow: 0 8px 25px rgba(214, 90, 49, 0.3);
            transition: all 0.3s;
        }
        .action-btn div.stButton > button:hover {
            box-shadow: 0 12px 30px rgba(214, 90, 49, 0.4);
            transform: scale(1.02);
        }

        .ecg-container {
            width: 100%; height: 70px; overflow: hidden; position: relative;
            margin-bottom: -15px;
        }
        .ecg-line {
            fill: none; stroke: #E28743; stroke-width: 2.5;
            stroke-linecap: round; stroke-linejoin: round;
            stroke-dasharray: 1500; stroke-dashoffset: 1500;
            animation: drawECG 4s linear infinite;
            filter: drop-shadow(0 3px 4px rgba(226, 135, 67, 0.3));
        }
        @keyframes drawECG { to { stroke-dashoffset: 0; } }

        .massive-score-container {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 40px; text-align: center;
        }
        .score-circle {
            width: 260px; height: 260px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; flex-direction: column;
            background: #FFFFFF;
            font-size: 4.5rem; font-weight: 900;
            position: relative;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.02);
        }
        @keyframes pulseSoft {
            0% { transform: scale(0.98); box-shadow: 0 0 0 0 var(--pulse-color); }
            70% { transform: scale(1.02); box-shadow: 0 0 0 40px rgba(0,0,0,0); }
            100% { transform: scale(0.98); box-shadow: 0 0 0 0 rgba(0,0,0,0); }
        }
        .pulse-slow   { --pulse-color: rgba(108, 154, 122, 0.4); border: 4px solid #6C9A7A; color: #6C9A7A; animation: pulseSoft 2.5s infinite; }
        .pulse-medium { --pulse-color: rgba(226, 135, 67, 0.4);  border: 4px solid #E28743; color: #E28743; animation: pulseSoft 1.5s infinite; }
        .pulse-fast   { --pulse-color: rgba(214, 90, 49, 0.5);   border: 4px solid #D65A31; color: #D65A31; animation: pulseSoft 0.8s infinite; }

        .scanner-screen {
            height: 85vh; display: flex; flex-direction: column; align-items: center; justify-content: center;
            color: #E28743; text-align: center;
        }
        .soft-pulse-loader {
            width: 80px; height: 80px; border-radius: 50%;
            background: #E28743; margin-bottom: 30px;
            animation: pulseSoftLoader 2s infinite;
        }
        @keyframes pulseSoftLoader {
            0%  { transform: scale(0.8); opacity: 0.8; box-shadow: 0 0 0 0 rgba(226, 135, 67, 0.4); }
            70% { transform: scale(1);   opacity: 0.2; box-shadow: 0 0 0 40px rgba(226, 135, 67, 0); }
            100%{ transform: scale(0.8); opacity: 0.8; box-shadow: 0 0 0 0 rgba(226, 135, 67, 0); }
        }

        [data-testid="stMetricLabel"] { color: #8B7E74 !important; }
        [data-testid="stMetricValue"] { color: #3A302B !important; }
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
            <h1 style="font-size: 3rem; color: #3A302B;">Préparation de votre espace<span style="color: #E28743;">.</span></h1>
            <p style="color: #8B7E74; font-size: 1.1rem; margin-top: 10px;">Initialisation des algorithmes de santé...</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.5)
    placeholder.empty()
    st.session_state.app_state = "ready"
    st.rerun()

# ==========================================
# 5. EN-TÊTE & NAVIGATION
# ==========================================
st.markdown("""
    <div class="ecg-container">
        <svg viewBox="0 0 1000 60" preserveAspectRatio="none">
            <path class="ecg-line" d="M0,30 L200,30 L210,10 L220,50 L230,20 L240,40 L250,30 L450,30 L460,10 L480,55 L490,5 L500,30 L700,30 L710,20 L720,40 L730,30 L1000,30"></path>
        </svg>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; font-weight:900; margin-bottom:30px;'>CardioCare <span class='accent-text'>Sérénité</span></h1>", unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.columns(4)
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
    if st.button("LANCER L'ÉVALUATION SÉCURISÉE", use_container_width=True):
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
            pulse_class  = "pulse-fast"
            status_text  = "VIGILANCE REQUISE"
            desc_text    = "Nos algorithmes détectent des signes d'alerte nécessitant un avis médical rapide."
            color_class  = "danger-text"
        elif proba >= 0.40:
            pulse_class  = "pulse-medium"
            status_text  = "ATTENTION MODÉRÉE"
            desc_text    = "Certains biomarqueurs sont instables. Une surveillance est recommandée."
            color_class  = "accent-text"
        else:
            pulse_class  = "pulse-slow"
            status_text  = "ÉTAT RASSURANT"
            desc_text    = "Vos constantes actuelles sont stables et ne montrent pas de risque immédiat."
            color_class  = "safe-text"

        c_dash1, c_dash2 = st.columns([1, 1])

        with c_dash1:
            st.markdown("<div class='massive-score-container'>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-weight:bold; letter-spacing:1.5px; margin-bottom:20px;'>NIVEAU D'ATTENTION</p>", unsafe_allow_html=True)
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
                s1.metric("Sodium", f"{p['serum_sodium']} mEq/L")
                s2.metric("Enzyme CPK", f"{p['creatinine_phosphokinase']} mcg/L")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
            if st.button("COMPRENDRE CES RÉSULTATS", use_container_width=True):
                st.session_state.current_page = "shap"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 8. VUE 3 : EXPLICABILITÉ SHAP
# ==========================================
elif st.session_state.current_page == "shap":
    st.markdown("### 🌿 Transparence des Résultats")
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
                    st.markdown("<h4 class='accent-text'>Les points clés</h4>", unsafe_allow_html=True)
                    patient_shap = compute_shap_values(explainer, X_input)
                    if patient_shap.ndim > 1:
                        patient_shap = patient_shap[0:1]
                    top_features = get_top_features(patient_shap, FEATURE_NAMES, top_n=5)
                    for feat, score in top_features:
                        if score > 0:
                            st.markdown(f"<p><span class='danger-text' style='font-weight:bold;'>↑ {feat}</span><br><span class='sub-text' style='font-size:0.85rem;'>Demande de l'attention</span></p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p><span class='safe-text' style='font-weight:bold;'>↓ {feat}</span><br><span class='sub-text' style='font-size:0.85rem;'>Facteur rassurant</span></p>", unsafe_allow_html=True)

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
                color="#E28743", edgecolor="none"
            )
            ax.set_xlabel("Impact Moyen sur la Décision de l'IA", color="#4A403B", weight="bold")
            ax.tick_params(colors="#4A403B")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#8B7E74")
            ax.spines["left"].set_color("#8B7E74")
            plt.tight_layout()
            st.pyplot(fig, transparent=True)

# ==========================================
# 9. VUE 4 : PERFORMANCES
# ==========================================
elif st.session_state.current_page == "perf":
    st.markdown("### 🛡️ Fiabilité du Diagnostic")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>RECALL ⭐</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='danger-text' style='margin:0;'>{recall_score(y_test, y_pred):.3f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Priorité n°1</p>", unsafe_allow_html=True)
    with col_m2:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>PRÉCISION</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='accent-text' style='margin:0;'>{precision_score(y_test, y_pred):.3f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Priorité n°2</p>", unsafe_allow_html=True)
    with col_m3:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>ROC-AUC</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='safe-text' style='margin:0;'>{roc_auc_score(y_test, y_proba):.3f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Comparaison modèles</p>", unsafe_allow_html=True)
    with col_m4:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>F1-SCORE</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#8E44AD; margin:0;'>{f1_score(y_test, y_pred):.3f}</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Équilibre P/R</p>", unsafe_allow_html=True)
    with col_m5:
        with st.container(border=True):
            st.markdown("<p class='sub-text' style='font-size:12px; font-weight:bold; letter-spacing:1px;'>EXACTITUDE</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#3A302B; margin:0;'>{accuracy_score(y_test, y_pred)*100:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text' style='font-size:11px;'>Moyenne globale</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_conf1, c_conf2 = st.columns([1, 2])
    with c_conf2:
        with st.container(border=True):
            st.markdown("<h4 class='accent-text'>Matrice de Validation</h4>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(6, 4))
            fig_cm.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            cmap_custom = sns.light_palette("#E28743", as_cmap=True)
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_custom,
                        cbar=False, annot_kws={"size": 18, "weight": "bold", "color": "#3A302B"},
                        xticklabels=["Survie Prédite", "Décès Prédit"],
                        yticklabels=["Survie Réelle", "Décès Réel"], ax=ax)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#4A403B", weight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), color="#4A403B", weight="bold")
            st.pyplot(fig_cm, transparent=True)

# ==========================================
# 10. PIED DE PAGE
# ==========================================
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:#8B7E74; font-size:13px; margin-top:20px;'>
        <p><strong>⚠️ Avertissement :</strong> Application développée à des fins éducatives et de recherche.<br>
        Ne se substitue en aucun cas à un diagnostic ou un avis médical professionnel.</p>
    </div>
""", unsafe_allow_html=True)