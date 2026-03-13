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
    page_title="CardioCare | Deep Medical Night",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Forcer le thème sombre pour matplotlib (Explicabilité SHAP lisible)
plt.style.use('dark_background')

# Initialisation de la machine à états (SPA sans rechargements)
if "app_state" not in st.session_state:
    st.session_state.app_state = "booting" # booting, ready
if "current_page" not in st.session_state:
    st.session_state.current_page = "input" # input, dashboard, ai_engine
if "proba" not in st.session_state:
    st.session_state.proba = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# ==========================================
# 2. CHARGEMENT DU MOTEUR ML (Intact)
# ==========================================
FEATURE_NAMES =[
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
# 3. MOTEUR CSS : "DEEP MEDICAL NIGHT" & GLASSMORPHISM
# ==========================================
st.markdown("""
    <style>
        /* Importer typographie Premium */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

        /* Réinitialisation de l'interface Streamlit */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: #E2E8F0 !important;
        }[data-testid="stSidebar"], [data-testid="stHeader"] { display: none !important; }
        .block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 2rem; }

        /* FOND ANIMÉ MESH GRADIENT (Zéro image) */
        .stApp {
            background: #060B14;
            overflow-x: hidden;
            z-index: 0;
        }
        .stApp::before {
            content: ""; position: fixed; top: -50%; left: -50%; width: 200%; height: 200%;
            background: 
                radial-gradient(circle at 20% 30%, rgba(0, 212, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 80% 70%, rgba(255, 75, 75, 0.04) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(0, 255, 157, 0.02) 0%, transparent 50%);
            animation: meshGlow 30s infinite alternate linear;
            z-index: -1; pointer-events: none;
        }
        @keyframes meshGlow {
            0% { transform: rotate(0deg) scale(1); }
            100% { transform: rotate(15deg) scale(1.1); }
        }

        /* TYPOGRAPHIE & COULEURS ACCENTS */
        h1, h2, h3 { color: #FFFFFF !important; letter-spacing: -0.03em; }
        .cyan-text { color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.4); }
        .red-text { color: #ff4b4b; text-shadow: 0 0 10px rgba(255,75,75,0.4); }
        .green-text { color: #00ff9d; text-shadow: 0 0 10px rgba(0,255,157,0.4); }

        /* GLASSMORPHISM CARDS (Remplace les borders par défaut) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(10, 15, 30, 0.4) !important;
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px !important;
            box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
            box-shadow: 0 10px 50px -10px rgba(0, 212, 255, 0.15);
        }

        /* NAVIGATION TOP CUSTOM (Hack CSS sur les boutons) */
        .nav-container div.stButton > button {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255,255,255,0.1);
            color: #94A3B8 !important;
            font-weight: 600;
            border-radius: 12px;
            padding: 10px 24px;
            width: 100%;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        .nav-container div.stButton > button:hover {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid #00d4ff;
            color: #00d4ff !important;
            box-shadow: 0 0 15px rgba(0,212,255,0.3);
            transform: translateY(-2px);
        }
        .nav-active div.stButton > button {
            background: rgba(0, 212, 255, 0.15);
            border: 1px solid #00d4ff;
            color: #FFFFFF !important;
            box-shadow: inset 0 0 10px rgba(0,212,255,0.2), 0 0 20px rgba(0,212,255,0.4);
        }

        /* BOUTON ACTION PRINCIPAL (Pulse) */
        .action-btn div.stButton > button {
            background: linear-gradient(90deg, #00d4ff, #007bb5);
            color: white !important;
            font-weight: 800; font-size: 1.1rem; letter-spacing: 1px;
            border-radius: 12px; border: none; padding: 15px; width: 100%;
            box-shadow: 0 0 20px rgba(0,212,255,0.5);
            transition: all 0.3s;
        }
        .action-btn div.stButton > button:hover {
            box-shadow: 0 0 30px rgba(0,212,255,0.8);
            transform: scale(1.02);
        }

        /* HEARTBEAT VISUALIZER (ECG) CSS Pur */
        .ecg-container {
            width: 100%; height: 60px; overflow: hidden; position: relative; opacity: 0.8;
            margin-bottom: -10px;
        }
        .ecg-line {
            fill: none; stroke: #00d4ff; stroke-width: 2;
            stroke-dasharray: 1500; stroke-dashoffset: 1500;
            animation: drawECG 4s linear infinite;
            filter: drop-shadow(0 0 5px #00d4ff);
        }
        @keyframes drawECG { to { stroke-dashoffset: 0; } }

        /* RESULTAT SPECTACULAIRE : MASSIVE SCORE & PULSE */
        .massive-score-container {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 40px; text-align: center;
        }
        .score-circle {
            width: 250px; height: 250px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; flex-direction: column;
            background: rgba(10, 15, 30, 0.6);
            backdrop-filter: blur(20px);
            font-size: 4rem; font-weight: 900;
            position: relative;
        }
        .score-circle::before {
            content: ''; position: absolute; top: -5px; left: -5px; right: -5px; bottom: -5px;
            border-radius: 50%; z-index: -1;
        }
        /* Vitesses de pulsations cardiaques dynamiques */
        @keyframes pulseHeart {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 var(--pulse-color); }
            70% { transform: scale(1.05); box-shadow: 0 0 0 40px rgba(0,0,0,0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0,0,0,0); }
        }
        .pulse-slow   { --pulse-color: rgba(0, 255, 157, 0.6); border: 2px solid #00ff9d; color: #00ff9d; animation: pulseHeart 1.5s infinite; text-shadow: 0 0 20px #00ff9d; }
        .pulse-medium { --pulse-color: rgba(255, 204, 0, 0.6); border: 2px solid #ffcc00; color: #ffcc00; animation: pulseHeart 0.9s infinite; text-shadow: 0 0 20px #ffcc00; }
        .pulse-fast   { --pulse-color: rgba(255, 75, 75, 0.8); border: 2px solid #ff4b4b; color: #ff4b4b; animation: pulseHeart 0.4s infinite; text-shadow: 0 0 20px #ff4b4b; }

        /* CUSTOM SLIDERS & INPUTS */
        div[data-baseweb="slider"] { filter: drop-shadow(0 0 3px rgba(0,212,255,0.5)); }
        input { background-color: rgba(255,255,255,0.05) !important; color: #00d4ff !important; font-weight: bold !important; border-radius: 8px !important; }

        /* SCANNER LOADING ANIMATION */
        .scanner-screen {
            height: 80vh; display: flex; flex-direction: column; align-items: center; justify-content: center;
            font-family: 'JetBrains Mono', monospace; color: #00d4ff; text-align: center;
        }
        .scanner-line {
            width: 300px; height: 2px; background: #00d4ff; box-shadow: 0 0 20px 5px rgba(0,212,255,0.5);
            animation: scanMove 2s infinite ease-in-out alternate; margin: 40px 0;
        }
        @keyframes scanMove { 0% { transform: translateY(-50px); } 100% { transform: translateY(50px); } }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 4. ÉCRAN DE CHARGEMENT "LANDING PAGE TRANSITION"
# ==========================================
if st.session_state.app_state == "booting":
    st.markdown("""
        <div class="scanner-screen">
            <h1 style="font-size: 3rem; letter-spacing: 5px; text-shadow: 0 0 20px #00d4ff;">CARDIOCARE <span style="font-weight:300;">OS</span></h1>
            <div class="scanner-line"></div>
            <p>INITIALISATION DES SYSTÈMES BIOMÉTRIQUES...</p>
            <p style="color: #64748B; font-size: 0.8rem;">Connexion au moteur LightGBM... [OK]</p>
            <p style="color: #64748B; font-size: 0.8rem;">Calibrage de l'explicabilité SHAP... [OK]</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    st.session_state.app_state = "ready"
    st.rerun()


# ==========================================
# 5. EN-TÊTE & NAVIGATION CUSTOM (Barre Horizontale)
# ==========================================
# Le Heartbeat ECG Graphique
st.markdown("""
    <div class="ecg-container">
        <svg viewBox="0 0 1000 60" preserveAspectRatio="none">
            <path class="ecg-line" d="M0,30 L200,30 L210,10 L220,50 L230,20 L240,40 L250,30 L450,30 L460,10 L480,55 L490,5 L500,30 L700,30 L710,20 L720,40 L730,30 L1000,30"></path>
        </svg>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-weight: 900; margin-bottom: 30px;'>🫀 CardioCare <span class='cyan-text'>AI</span></h1>", unsafe_allow_html=True)

# Navigation via Colonnes (Hack CSS appliqué)
nav1, nav2, nav3, nav4 = st.columns(4)
def nav_class(page): return "nav-active nav-container" if st.session_state.current_page == page else "nav-container"

with nav1:
    st.markdown(f"<div class='{nav_class('input')}'>", unsafe_allow_html=True)
    if st.button("1. Saisie des Constantes", use_container_width=True): 
        st.session_state.current_page = "input"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with nav2:
    st.markdown(f"<div class='{nav_class('dashboard')}'>", unsafe_allow_html=True)
    if st.button("2. Diagnostic Central", use_container_width=True):
        st.session_state.current_page = "dashboard"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with nav3:
    st.markdown(f"<div class='{nav_class('shap')}'>", unsafe_allow_html=True)
    if st.button("3. Explicabilité SHAP", use_container_width=True):
        st.session_state.current_page = "shap"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with nav4:
    st.markdown(f"<div class='{nav_class('perf')}'>", unsafe_allow_html=True)
    if st.button("4. Qualité du Modèle", use_container_width=True):
        st.session_state.current_page = "perf"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)


# ==========================================
# 6. VUE 1 : INPUT EXPERIENCE (Cartes Interactives)
# ==========================================
if st.session_state.current_page == "input":
    st.markdown("### 📋 Interface d'Acquisition Biométrique")
    st.markdown("<p style='color: #94A3B8; margin-bottom: 30px;'>Veuillez calibrer les capteurs virtuels pour générer le profil patient.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True): # Glassmorphism auto-appliqué
            st.markdown("<h4 class='cyan-text'>👤 Profil Démographique</h4>", unsafe_allow_html=True)
            age = st.slider("Âge (années)", 18, 95, 60)
            sex = st.radio("Sexe Biologique", ["Femme", "Homme"], horizontal=True)
            time_fup = st.slider("Période de suivi (jours)", 4, 285, 100)

        with st.container(border=True):
            st.markdown("<h4 class='cyan-text'>🩺 Paramètres Cardiaques</h4>", unsafe_allow_html=True)
            ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38)
            high_blood_pressure = st.toggle("Hypertension Artérielle")

    with col2:
        with st.container(border=True):
            st.markdown("<h4 class='red-text'>🩸 Chimie Sanguine</h4>", unsafe_allow_html=True)
            serum_creatinine = st.number_input("Créatinine sérique (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
            serum_sodium = st.number_input("Sodium sérique (mEq/L)", 110.0, 150.0, 137.0, step=1.0)
            creatinine_phosphokinase = st.number_input("Enzyme CPK (mcg/L)", 20, 8000, 250, step=10)
            platelets = st.number_input("Plaquettes (k/mL)", 25000, 850000, 265000, step=5000)

        with st.container(border=True):
            st.markdown("<h4 class='red-text'>⚠️ Facteurs de Risque</h4>", unsafe_allow_html=True)
            c_a, c_d, c_s = st.columns(3)
            with c_a: anaemia = st.toggle("Anémie")
            with c_d: diabetes = st.toggle("Diabète")
            with c_s: smoking = st.toggle("Fumeur")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
    if st.button("INITIALISER LE SCAN PRÉDICTIF ⚡", use_container_width=True):
        patient = {
            "age": age, "anaemia": int(anaemia), "creatinine_phosphokinase": creatinine_phosphokinase,
            "diabetes": int(diabetes), "ejection_fraction": ejection_fraction,
            "high_blood_pressure": int(high_blood_pressure), "platelets": platelets,
            "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
            "sex": 1 if sex == "Homme" else 0, "smoking": int(smoking), "time": time_fup,
        }
        st.session_state.patient_data = patient
        # Prédiction immédiate
        X_input = pd.DataFrame([patient])[FEATURE_NAMES]
        st.session_state.proba = float(model.predict_proba(X_input)[0][1])
        st.session_state.current_page = "dashboard"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 7. VUE 2 : RÉSULTAT SPECTACULAIRE (Pulse Dashboard)
# ==========================================
elif st.session_state.current_page == "dashboard":
    if st.session_state.proba is None:
        st.warning("⚠️ Aucune donnée patient détectée. Veuillez effectuer la Saisie des Constantes.")
    else:
        proba = st.session_state.proba
        
        # Détermination du statut dynamique
        if proba >= 0.65:
            pulse_class = "pulse-fast"
            status_text = "RISQUE CRITIQUE"
            desc_text = "Défaillance cardiaque imminente détectée. Intervention d'urgence vitale requise."
        elif proba >= 0.40:
            pulse_class = "pulse-medium"
            status_text = "ALERTE MODÉRÉE"
            desc_text = "Instabilité des biomarqueurs. Mise sous surveillance clinique conseillée."
        else:
            pulse_class = "pulse-slow"
            status_text = "RYTHME STABLE"
            desc_text = "Signes vitaux nominaux. Aucun risque létal détecté à court terme."

        c_dash1, c_dash2 = st.columns([1, 1])

        with c_dash1:
            st.markdown("<div class='massive-score-container'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94A3B8; font-weight:bold; letter-spacing:2px; margin-bottom:20px;'>INDICE DE DÉFAILLANCE</p>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='score-circle {pulse_class}'>
                    {proba*100:.1f}<span style='font-size:1.5rem;'>%</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin-top:30px; letter-spacing:1px;' class='{pulse_class.split('-')[1]}-text'>{status_text}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94A3B8; text-align:center;'>{desc_text}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_dash2:
            st.markdown("<h3 class='cyan-text'>Résumé Biomarkers</h3>", unsafe_allow_html=True)
            p = st.session_state.patient_data
            with st.container(border=True):
                s1, s2 = st.columns(2)
                s1.metric("Fraction Éjection", f"{p['ejection_fraction']}%", delta="Normal" if p['ejection_fraction']>30 else "Critique", delta_color="normal" if p['ejection_fraction']>30 else "inverse")
                s2.metric("Créatinine", f"{p['serum_creatinine']} mg/dL", delta="Anormal" if p['serum_creatinine']>1.5 else "OK", delta_color="inverse")
                s1.metric("Sodium", f"{p['serum_sodium']} mEq/L")
                s2.metric("Enzyme CPK", f"{p['creatinine_phosphokinase']} mcg/L")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
            if st.button("DÉCRYPTER L'ANALYSE (SHAP) 🧠", use_container_width=True):
                st.session_state.current_page = "shap"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 8. VUE 3 : EXPLICABILITÉ (Moteur IA SHAP)
# ==========================================
elif st.session_state.current_page == "shap":
    st.markdown("### 🧠 Imagerie Interprétative de l'Intelligence Artificielle")
    st.markdown("<p style='color: #94A3B8;'>Analyse structurelle des poids décisionnels du réseau LightGBM.</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧬 Radiographie du Patient Actuel", "🌍 Matrice d'Importance Globale"])

    with tab1:
        if st.session_state.patient_data is None:
            st.warning("⚠️ Calibrez un patient d'abord.")
        else:
            X_input = pd.DataFrame([st.session_state.patient_data])[FEATURE_NAMES]
            
            col_shap1, col_shap2 = st.columns([1, 2])
            with col_shap1:
                with st.container(border=True):
                    st.markdown("<h4 class='cyan-text'>Vecteurs Dominants</h4>", unsafe_allow_html=True)
                    patient_shap = compute_shap_values(explainer, X_input)
                    top_features = get_top_features(patient_shap, FEATURE_NAMES, top_n=5)
                    for feat, score in top_features:
                        if score > 0:
                            st.markdown(f"<p><span style='color:#ff4b4b; font-weight:bold;'>↑ {feat}</span><br><span style='font-size:0.8rem; color:#94A3B8;'>Aggrave le risque (+{score:.2f})</span></p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p><span style='color:#00ff9d; font-weight:bold;'>↓ {feat}</span><br><span style='font-size:0.8rem; color:#94A3B8;'>Réduit le risque ({score:.2f})</span></p>", unsafe_allow_html=True)

            with col_shap2:
                with st.container(border=True):
                    # SHAP génère du matplotlib, on assure la transparence et lisibilité
                    waterfall_path = "/tmp/patient_waterfall.png"
                    fig = plot_waterfall_single(explainer, X_input, FEATURE_NAMES, save_path=waterfall_path)
                    # Forcer un background transparent sur l'image enregistrée
                    st.image(waterfall_path, use_column_width=True)

    with tab2:
        with st.container(border=True):
            shap_values = compute_shap_values(explainer, X_test)
            mean_abs = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(mean_abs)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_alpha(0) # Fond transparent
            ax.patch.set_alpha(0)
            ax.barh(
                [FEATURE_NAMES[i] for i in sorted_idx],
                mean_abs[sorted_idx],
                color="#00d4ff", edgecolor="rgba(0,212,255,0.5)"
            )
            ax.set_xlabel("Impact Moyen Absolu sur le Modèle (|SHAP|)", color="#E2E8F0", weight="bold")
            ax.tick_params(colors="#E2E8F0")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#475569")
            ax.spines["left"].set_color("#475569")
            plt.tight_layout()
            st.pyplot(fig, transparent=True)


# ==========================================
# 9. VUE 4 : PERFORMANCES DU MODÈLE
# ==========================================
elif st.session_state.current_page == "perf":
    st.markdown("### ⚙️ Diagnostic de Performance du Noyau Algorithmique")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        with st.container(border=True):
            st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:bold; letter-spacing:1px;'>SENSIBILITÉ</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='cyan-text' style='margin:0;'>{recall_score(y_test, y_pred):.3f}</h2>", unsafe_allow_html=True)
    with col_m2:
        with st.container(border=True):
            st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:bold; letter-spacing:1px;'>PRÉCISION</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='cyan-text' style='margin:0;'>{precision_score(y_test, y_pred):.3f}</h2>", unsafe_allow_html=True)
    with col_m3:
        with st.container(border=True):
            st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:bold; letter-spacing:1px;'>ROC-AUC</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='green-text' style='margin:0;'>{roc_auc_score(y_test, y_proba):.3f}</h2>", unsafe_allow_html=True)
    with col_m4:
        with st.container(border=True):
            st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:bold; letter-spacing:1px;'>EXACTITUDE</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#FFFFFF; margin:0;'>{accuracy_score(y_test, y_pred)*100:.1f}%</h2>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_conf1, c_conf2 = st.columns([1, 2])
    
    with c_conf2:
        with st.container(border=True):
            st.markdown("<h4 class='cyan-text'>Matrice de Confusion Neuronale</h4>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(6, 4))
            fig_cm.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            # Custom cmap pour matcher le Deep Medical Night
            cmap_custom = sns.color_palette("dark:cyan_r", as_cmap=True)
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_custom,
                        cbar=False, annot_kws={"size": 18, "weight": "bold", "color": "white"},
                        xticklabels=["Survie Prédite", "Décès Prédit"],
                        yticklabels=["Survie Réelle", "Décès Réel"], ax=ax)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#94A3B8", weight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), color="#94A3B8", weight="bold")
            st.pyplot(fig_cm, transparent=True)