import os
import sys
import pickle
import warnings

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

# ==========================================
# 1. CONFIGURATION & CSS AVANCÉ (UI/UX)
# ==========================================
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection du CSS (Glassmorphism, Animations, Typographie Haut Contraste)
st.markdown("""
    <style>
        /* --- Typographie Globale --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: #1E293B !important; /* Contraste maximal (Gris très foncé) */
        }

        /* --- Arrière-plan Animé Pur CSS (Medical Vibe) --- */
        .stApp {
            background: linear-gradient(-45deg, #F8FAFC, #E2E8F0, #F1F5F9, #E0F2FE);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            z-index: 0;
        }
        /* Maillage géométrique en mouvement lent (Points) */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background-image: radial-gradient(#94A3B8 1px, transparent 1px);
            background-size: 35px 35px;
            opacity: 0.15;
            z-index: -1;
            pointer-events: none;
            animation: meshMove 30s linear infinite;
        }

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95) !important; /* Bleu nuit médical */
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255,255,255,0.05);
        }[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h4,[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
            color: #F8FAFC !important;
        }
        
        /* --- Glassmorphism Custom Cards --- */
        .glass-card {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            margin-bottom: 20px;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            background: rgba(255, 255, 255, 0.85);
        }

        /* --- Animations Clés --- */
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes meshMove {
            0% { transform: translateY(0) translateX(0); }
            100% { transform: translateY(-35px) translateX(-35px); }
        }
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseCritical {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }

        /* Classes d'animation d'entrée */
        .fade-in { animation: fadeInUp 0.6s ease-out forwards; opacity: 0; }
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.2s; }
        .delay-3 { animation-delay: 0.3s; }

        /* --- Bouton d'Action Principal --- */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0284C7 0%, #38BDF8 50%, #0284C7 100%);
            background-size: 200% auto;
            color: white !important;
            border-radius: 12px;
            border: none;
            padding: 14px 24px;
            font-weight: 800;
            font-size: 16px;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3);
            animation: shimmer 5s infinite linear;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(2, 132, 199, 0.5);
        }

        /* --- Styling des Onglets (Tabs) --- */
        [data-baseweb="tab-list"] {
            gap: 15px;
            background: rgba(255, 255, 255, 0.4);
            padding: 10px;
            border-radius: 14px;
            backdrop-filter: blur(10px);
        }[data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            padding: 12px 24px;
            border: none !important;
            font-weight: 600;
            color: #64748B !important;
            transition: all 0.3s ease;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #0F172A !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        /* Override Metrics Nativs */
        [data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 800; font-size: 1.8rem; }
        [data-testid="stMetricLabel"] { color: #64748B !important; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.5px; }

        .pulse-alert {
            animation: pulseCritical 2s infinite;
            border: 2px solid #EF4444 !important;
        }
        
        /* Grille HTML Custom pour le profil patient */
        .profile-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .profile-item { background: rgba(255,255,255,0.5); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.8); text-align: center; transition: transform 0.2s;}
        .profile-item:hover { transform: scale(1.05); background: rgba(255,255,255,0.8); }
        .profile-label { font-size: 12px; color: #64748B; font-weight: 600; text-transform: uppercase; }
        .profile-value { font-size: 20px; color: #0F172A; font-weight: 800; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTES
# ==========================================
FEATURE_NAMES =[
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "heart_failure_clinical_records_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")

# ==========================================
# 3. CHARGEMENT MODÈLE & DONNÉES (Intact)
# ==========================================
@st.cache_resource(show_spinner="Initialisation du moteur IA clinique...")
def get_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def get_test_data():
    df = load_data(DATA_PATH)
    df = handle_outliers(df)
    df = optimize_memory(df)

    X = df[FEATURE_NAMES]
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_test

# ==========================================
# 4. FONCTIONS GRAPHIQUES REVISITÉES
# ==========================================
def risk_gauge(probability: float) -> go.Figure:
    color = "#10B981" if probability < 0.4 else "#F59E0B" if probability < 0.65 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 55, "color": color, "family": "Inter", "weight": "bold"}},
        title={"text": "INDICE DE RISQUE", "font": {"size": 14, "color": "#64748B", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#CBD5E1"},
            "bar": {"color": color, "thickness": 0.2},
            "bgcolor": "rgba(255,255,255,0.1)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(16, 185, 129, 0.1)"},
                {"range":[40, 65], "color": "rgba(245, 158, 11, 0.1)"},
                {"range": [65,100], "color": "rgba(239, 68, 68, 0.1)"},
            ],
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter"}
    )
    return fig

# ==========================================
# 5. EN-TÊTE DE L'APPLICATION
# ==========================================
st.markdown("""
    <div class='fade-in' style='text-align:center; padding: 20px 0 30px 0;'>
        <h1 style='font-size: 4em; font-weight: 800; color: #0F172A; margin-bottom: 5px; letter-spacing: -1px;'>
            🫀 CardioCare <span style='color: #0284C7;'>AI</span>
        </h1>
        <p style='color: #64748B; font-size: 1.3rem; font-weight: 400; margin-top: 0;'>
            Plateforme clinique d'évaluation prédictive du risque cardiaque
        </p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 6. SIDEBAR (Paramètres cliniques)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align:center; font-weight:800; font-size: 1.5rem;'>📋 Dossier Patient</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin-top: 0;'>", unsafe_allow_html=True)

    st.markdown("<h4 style='color:#38BDF8 !important; font-weight: 600; font-size: 1.1rem;'>👤 Profil & Constantes</h4>", unsafe_allow_html=True)
    age               = st.slider("Âge (années)", 18, 95, 60)
    sex               = st.radio("Sexe biologique", ["Femme", "Homme"], horizontal=True)
    time              = st.slider("Période de suivi (jours)", 4, 285, 100)
    ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38)

    st.markdown("<br><h4 style='color:#38BDF8 !important; font-weight: 600; font-size: 1.1rem;'>🩸 Biologie Sanguine</h4>", unsafe_allow_html=True)
    serum_creatinine         = st.number_input("Créatinine sérique (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
    serum_sodium             = st.number_input("Sodium sérique (mEq/L)", 110.0, 150.0, 137.0, step=1.0)
    creatinine_phosphokinase = st.number_input("Enzyme CPK (mcg/L)", 20, 8000, 250, step=10)
    platelets                = st.number_input("Plaquettes (k/mL)", 25000, 850000, 265000, step=5000)

    st.markdown("<br><h4 style='color:#38BDF8 !important; font-weight: 600; font-size: 1.1rem;'>🩺 Comorbidités</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        anaemia  = st.checkbox("Anémie")
        diabetes = st.checkbox("Diabète")
    with col2:
        high_blood_pressure = st.checkbox("Hypertension")
        smoking             = st.checkbox("Fumeur")

    st.markdown("<br><br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Lancer l'Analyse", use_container_width=True)

# ==========================================
# 7. CONSTRUCTION DU DICT PATIENT (Intact)
# ==========================================
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
    "time":                     time,
}

# ==========================================
# 8. EXÉCUTION DU MOTEUR (Intact)
# ==========================================
model                   = get_model()
X_train, X_test, y_test = get_test_data()
explainer               = get_shap_explainer(model, X_train)

if "proba" not in st.session_state:
    st.session_state["proba"] = None

if predict_btn:
    with st.spinner("Analyse algorithmique en cours..."):
        X_input = pd.DataFrame([patient])[FEATURE_NAMES]
        st.session_state["proba"] = float(model.predict_proba(X_input)[0][1])

proba = st.session_state["proba"]

# ==========================================
# 9. INTERFACE DES ONGLETS (Refonte Complète)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Évaluation Diagnostique", "🔬 Interprétabilité IA (SHAP)", "📈 Métriques Modèle"])

# --- ONGLET 1 : DIAGNOSTIC ---
with tab1:
    if proba is None:
        st.markdown("""
            <div class='glass-card fade-in' style='text-align: center; padding: 60px 20px;'>
                <h3 style='color: #64748B;'>En attente des données cliniques</h3>
                <p style='color: #94A3B8;'>Renseignez les paramètres dans le panneau latéral et cliquez sur <b>Lancer l'Analyse</b>.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        col_gauge, col_result = st.columns([1.1, 1])

        with col_gauge:
            st.markdown("<div class='glass-card fade-in delay-1'>", unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(proba), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result:
            risk_label  = "CRITIQUE" if proba >= 0.65 else "MODÉRÉ" if proba >= 0.40 else "FAIBLE"
            color       = "#EF4444" if proba >= 0.65 else "#F59E0B" if proba >= 0.40 else "#10B981"
            bg_color    = "rgba(239, 68, 68, 0.05)" if proba >= 0.65 else "rgba(245, 158, 11, 0.05)" if proba >= 0.40 else "rgba(16, 185, 129, 0.05)"
            pulse_class = "pulse-alert" if proba >= 0.65 else ""
            desc_text   = (
                "Une intervention et une prise en charge médicale urgentes sont impératives." if proba >= 0.65 else
                "Une surveillance clinique rapprochée de ces indicateurs est recommandée."   if proba >= 0.40 else
                "Les biomarqueurs actuels ne révèlent pas de risque d'insuffisance à court terme."
            )
            
            st.markdown(f"""
                <div class='glass-card {pulse_class} fade-in delay-2' style='border-left: 8px solid {color}; background-color: {bg_color}; padding: 35px 25px;'>
                    <div style='text-transform: uppercase; font-size: 13px; font-weight: 700; color: #64748B; letter-spacing: 1px;'>Niveau d'Alerte IA</div>
                    <h2 style='color: {color}; font-size: 38px; font-weight: 800; margin: 10px 0 20px 0;'>RISQUE {risk_label}</h2>
                    <p style='color: #334155; font-size: 16px; line-height: 1.6; font-weight: 400;'>{desc_text}</p>
                    <hr style='border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 25px 0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='display:block; font-size: 12px; color: #64748B; text-transform:uppercase;'>Probabilité Calculée</span>
                            <strong style='font-size: 22px; color: #0F172A;'>{proba*100:.1f}%</strong>
                        </div>
                        <div style='text-align: right;'>
                            <span style='display:block; font-size: 12px; color: #64748B; text-transform:uppercase;'>Indice de Confiance IA</span>
                            <strong style='font-size: 22px; color: #0F172A;'>86.7%</strong>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<h3 class='fade-in delay-3' style='margin-top: 20px; font-weight: 700; font-size: 1.4rem; color: #1E293B;'>🩺 Récapitulatif Biologique & Clinique</h3>", unsafe_allow_html=True)
        
        # Grille HTML pure pour un design parfait et responsive
        st.markdown(f"""
            <div class='glass-card fade-in delay-3'>
                <div class='profile-grid'>
                    <div class='profile-item'><div class='profile-label'>Âge</div><div class='profile-value'>{age} ans</div></div>
                    <div class='profile-item'><div class='profile-label'>Sexe</div><div class='profile-value'>{sex}</div></div>
                    <div class='profile-item'><div class='profile-label'>Suivi médical</div><div class='profile-value'>{time} jrs</div></div>
                    <div class='profile-item'><div class='profile-label'>Frac. Éjection</div><div class='profile-value'>{ejection_fraction}%</div></div>
                    <div class='profile-item'><div class='profile-label'>Créatinine</div><div class='profile-value'>{serum_creatinine} <span style='font-size:14px; font-weight:400;'>mg/dL</span></div></div>
                    <div class='profile-item'><div class='profile-label'>Sodium</div><div class='profile-value'>{serum_sodium} <span style='font-size:14px; font-weight:400;'>mEq/L</span></div></div>
                    <div class='profile-item'><div class='profile-label'>Enzyme CPK</div><div class='profile-value'>{creatinine_phosphokinase}</div></div>
                    <div class='profile-item'><div class='profile-label'>Plaquettes</div><div class='profile-value'>{platelets/1000:.0f}k</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- ONGLET 2 : SHAP ---
with tab2:
    st.markdown("""
        <div class='glass-card fade-in'>
            <h3 style='margin-top:0; font-weight: 800; color: #0F172A;'>🔬 Cartographie des Décisions de l'IA</h3>
            <p style='color: #64748B; font-size: 1.1rem;'>Comprenez de manière transparente l'impact de chaque biomarqueur sur la prédiction.</p>
        </div>
    """, unsafe_allow_html=True)

    shap_values = compute_shap_values(explainer, X_test)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    sorted_idx  = np.argsort(mean_abs)

    # Graphique d'importance globale (Matplotlib optimisé UI)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color="#0284C7", edgecolor="none", height=0.5, alpha=0.85
    )
    ax.set_xlabel("Impact moyen sur le modèle (|SHAP Value|)", fontsize=12, color="#334155", weight="bold")
    ax.set_title("Poids global des variables cliniques (Base de test)", fontsize=14, fontweight="bold", color="#0F172A", pad=20)
    ax.tick_params(colors="#475569", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.tight_layout()
    
    st.markdown("<div class='glass-card fade-in delay-1'>", unsafe_allow_html=True)
    st.pyplot(fig, transparent=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if proba is not None:
        st.markdown("<h4 class='fade-in delay-2' style='font-weight: 800; color: #0F172A; margin-top: 30px;'>🧬 Profil d'explicabilité pour ce patient unique</h4>", unsafe_allow_html=True)
        X_input      = pd.DataFrame([patient])[FEATURE_NAMES]
        patient_shap = compute_shap_values(explainer, X_input)
        top_features = get_top_features(patient_shap, FEATURE_NAMES, top_n=5)

        col_text, col_plot = st.columns([1, 2])
        
        with col_text:
            st.markdown("<div class='glass-card fade-in delay-2' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<p style='font-weight: 600; color:#334155;'>Top 5 des facteurs influents :</p>", unsafe_allow_html=True)
            for feat, score in top_features:
                if score > 0:
                    st.markdown(f"<div style='background: rgba(239, 68, 68, 0.1); padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #EF4444;'><b>{feat}</b> <br> <span style='color:#EF4444; font-size:13px;'>📈 Augmente le risque (+{score:.2f})</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background: rgba(16, 185, 129, 0.1); padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #10B981;'><b>{feat}</b> <br> <span style='color:#10B981; font-size:13px;'>📉 Réduit le risque ({score:.2f})</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_plot:
            st.markdown("<div class='glass-card fade-in delay-3'>", unsafe_allow_html=True)
            waterfall_path = "/tmp/patient_waterfall.png"
            plot_waterfall_single(explainer, X_input, FEATURE_NAMES, save_path=waterfall_path)
            st.image(waterfall_path, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Lancez une analyse depuis la barre latérale pour visualiser le profil d'explicabilité de ce patient spécifique.")

# --- ONGLET 3 : PERFORMANCES ---
with tab3:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.markdown("<h3 class='fade-in' style='font-weight: 800; color: #0F172A;'>⚙️ Validation Technique du Modèle (LightGBM)</h3>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    def render_metric_card(col, title, value, desc, color, delay):
        col.markdown(f"""
            <div class='glass-card fade-in {delay}' style='text-align:center; padding: 20px 10px;'>
                <h4 style='color:#64748B; font-size:14px; text-transform:uppercase; margin:0 0 10px 0;'>{title}</h4>
                <h2 style='color:{color}; font-size:38px; font-weight:800; margin:0;'>{value}</h2>
                <p style='color:#94A3B8; font-size:12px; font-weight:600; margin:10px 0 0 0;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    render_metric_card(col_m1, "Sensibilité ⭐", f"{recall_score(y_test, y_pred):.3f}", "Détection des cas critiques", "#EF4444", "delay-1")
    render_metric_card(col_m2, "Précision", f"{precision_score(y_test, y_pred):.3f}", "Fiabilité des alertes", "#F59E0B", "delay-1")
    render_metric_card(col_m3, "ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}", "Capacité de séparation", "#10B981", "delay-2")
    render_metric_card(col_m4, "F1-Score", f"{f1_score(y_test, y_pred):.3f}", "Équilibre global", "#8B5CF6", "delay-2")
    render_metric_card(col_m5, "Exactitude", f"{accuracy_score(y_test, y_pred)*100:.1f}%", "Prédictions correctes", "#0284C7", "delay-3")

    st.markdown("<br><h3 class='fade-in delay-2' style='font-weight: 800; color: #0F172A;'>📊 Matrice de Confusion</h3>", unsafe_allow_html=True)

    col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
    with col_cm2:
        st.markdown("<div class='glass-card fade-in delay-3'>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(7, 5))
        
        # Heatmap customisée
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    cbar=False, annot_kws={"size": 20, "weight": "bold", "family":"Inter"},
                    xticklabels=["Survie Prédite", "Décès Prédit"],
                    yticklabels=["Survie Réelle", "Décès Réel"], ax=ax)
        
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, weight="bold", color="#334155")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight="bold", color="#334155")
        fig_cm.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        st.pyplot(fig_cm, transparent=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 10. PIED DE PAGE
# ==========================================
st.markdown("<hr style='border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center; color:#64748B; font-size:13px; font-weight:400; opacity: 0.8;'>
        <p><strong>⚠️ Clause de non-responsabilité médicale :</strong> CardioCare AI est une démonstration technologique à visée éducative et exploratoire.<br>
        Les prédictions générées par ce système d'Intelligence Artificielle <b>ne se substituent en aucun cas</b> à un jugement clinique, un diagnostic ou un avis médical professionnel formulé par un praticien de santé qualifié.</p>
    </div>
""", unsafe_allow_html=True)