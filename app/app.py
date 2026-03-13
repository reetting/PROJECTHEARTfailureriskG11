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
# 1. CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(-45deg, #f8fafc, #e0eaf5, #eef2f3, #e0eaf5);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
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
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #1A2530, #2C3E50, #1A2530);
            background-size: 200% 200%;
            animation: gradientBG 12s ease infinite;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .block-container {
            position: relative;
            z-index: 1;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(30px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseCritical {
            0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); }
            70% { box-shadow: 0 0 0 15px rgba(231, 76, 60, 0); }
            100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
        }
        @keyframes shimmerBtn {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        .fade-in {
            animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            opacity: 0;
        }
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.3s; }
        .delay-3 { animation-delay: 0.5s; }
        .pulse-alert {
            animation: pulseCritical 2s infinite;
            border: 2px solid #E74C3C !important;
        }
        .animated-title {
            background: linear-gradient(270deg, #2C3E50, #2980B9, #8E44AD, #2C3E50);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientBG 8s ease infinite;
            font-weight: 900;
        }
        h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #2980B9 0%, #3498DB 50%, #2980B9 100%);
            background-size: 200% auto;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 12px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(41, 128, 185, 0.4);
            animation: shimmerBtn 5s infinite linear;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 20px rgba(41, 128, 185, 0.6);
        }
        .custom-card {
            background-color: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.6);
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        } 
        .custom-card h4 { color: #2C3E50 !important; }[data-testid="stMetricLabel"] { color: #5D6D7E !important; }
[data-testid="stMetricValue"] { color: #2C3E50 !important; }
        .custom-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
            background-color: rgba(255, 255, 255, 0.85);
        }
        hr { border: 0; border-top: 1px solid rgba(0,0,0,0.05); margin: 1.5rem 0;}
        .text-muted { color: #7F8C8D; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTES
# ==========================================
FEATURE_NAMES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "heart_failure_clinical_records_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")

# ==========================================
# 3. CHARGEMENT MODÈLE & DONNÉES
# ==========================================
@st.cache_resource(show_spinner="Chargement du modèle LightGBM...")
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
# 4. FONCTIONS GRAPHIQUES
# ==========================================
def risk_gauge(probability: float) -> go.Figure:
    color = "#2ECC71" if probability < 0.4 else "#F1C40F" if probability < 0.65 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 45, "color": color, "family": "Arial"}},
        title={"text": "Indice de Risque", "font": {"size": 18, "color": "#7F8C8D"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.5)",
            "borderwidth": 2,
            "bordercolor": "#80FFFF",
            "steps": [
                {"range": [0,  40], "color": "rgba(46, 204, 113, 0.15)"},
                {"range": [40, 65], "color": "rgba(241, 196, 15, 0.15)"},
                {"range": [65,100], "color": "rgba(231, 76, 60, 0.15)"},
            ],
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ==========================================
# 5. EN-TÊTE
# ==========================================
st.markdown("""
    <div class='fade-in' style='text-align:center; padding: 10px 0;'>
        <h1 class='animated-title' style='font-size: 4em; margin-bottom: 5px;'>🫀 CardioCare AI</h1>
        <p class='text-muted' style='margin-top: 0; font-size: 1.2rem;'>
            Système d'évaluation avancée du risque d'insuffisance cardiaque
        </p>
    </div>
    <hr/>
""", unsafe_allow_html=True)

# ==========================================
# 6. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#FFFFFF; font-weight:300;'>📋 Données Patient</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<h4 style='color:#AAB7B8;'>👤 Profil & Constantes</h4>", unsafe_allow_html=True)
    age               = st.slider("Âge (années)", 18, 95, 60)
    sex               = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
    time              = st.slider("Période de suivi (jours)", 4, 285, 100)
    ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#AAB7B8;'>🩸 Analyses Sanguines</h4>", unsafe_allow_html=True)
    serum_creatinine         = st.number_input("Créatinine sérique (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
    serum_sodium             = st.number_input("Sodium sérique (mEq/L)", 110.0, 150.0, 137.0, step=1.0)
    creatinine_phosphokinase = st.number_input("Enzyme CPK (mcg/L)", 20, 8000, 250, step=10)
    platelets                = st.number_input("Plaquettes (k/mL)", 25000, 850000, 265000, step=5000)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#AAB7B8;'>🩺 Comorbidités & Profil</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])
    with col1:
        anaemia  = st.checkbox("Anémie")
        diabetes = st.checkbox("Diabète")
    with col2:
        high_blood_pressure = st.checkbox("Hypertension")
        smoking             = st.checkbox("Fumeur")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Analyser le risque", use_container_width=True)

# ==========================================
# 7. PATIENT DICT
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
# 8. EXÉCUTION
# ==========================================
model                   = get_model()
X_train, X_test, y_test = get_test_data()
explainer               = get_shap_explainer(model, X_train)

if "proba" not in st.session_state:
    st.session_state["proba"] = None

if predict_btn:
    X_input = pd.DataFrame([patient])[FEATURE_NAMES]
    st.session_state["proba"] = float(model.predict_proba(X_input)[0][1])

proba = st.session_state["proba"]

# ==========================================
# 9. ONGLETS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Estimé", "🔬 Explicabilité de l'IA", "📈 Performances techniques"])

# --- ONGLET 1 ---
with tab1:
    if proba is None:
        st.info("👈 Remplissez les paramètres du patient et cliquez sur **Analyser le risque**")
    else:
        col_gauge, col_result = st.columns([1.2, 1])

        with col_gauge:
            st.plotly_chart(risk_gauge(proba), use_container_width=True)

        with col_result:
            risk_label  = "RISQUE CRITIQUE" if proba >= 0.65 else "RISQUE MODÉRÉ" if proba >= 0.40 else "RISQUE FAIBLE"
            color       = "#E74C3C" if proba >= 0.65 else "#F39C12" if proba >= 0.40 else "#27AE60"
            pulse_class = ""
            desc_text   = (
                "Une prise en charge médicale urgente est fortement recommandée." if proba >= 0.65 else
                "Une surveillance régulière de ces indicateurs est conseillée."   if proba >= 0.40 else
                "Les indicateurs actuels ne montrent pas de risque critique immédiat."
            )
            st.markdown(f"""
                <div class='custom-card {pulse_class} fade-in delay-1' style='border-left:6px solid {color}; padding:30px; margin-top:30px;'>
                    <h4 style='color:#7F8C8D; margin:0; font-size:14px; letter-spacing:1px;'>DIAGNOSTIC ESTIMÉ</h4>
                    <h2 style='color:{color}; margin:15px 0; font-size:32px;'>{risk_label}</h2>
                    <p style='color:#34495E; margin-bottom:15px;'>{desc_text}</p>
                    <hr style='margin:15px 0;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='font-size:14px; color:#7F8C8D;'>Probabilité : <strong style='color:#2C3E50;'>{proba*100:.1f}%</strong></span>
                        <span style='font-size:14px; color:#7F8C8D;'>Fiabilité : <strong style='color:#2C3E50;'>86.7%</strong></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top:20px;'>📝 Profil clinique</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Âge",                 f"{age} ans")
            c2.metric("Sexe",                sex)
            c3.metric("Suivi",               f"{time} jours")
            c4.metric("Fraction d'éjection", f"{ejection_fraction}%")
            st.divider()
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Créatinine", f"{serum_creatinine} mg/dL")
            c6.metric("Sodium",     f"{serum_sodium} mEq/L")
            c7.metric("CPK",        f"{creatinine_phosphokinase}")
            c8.metric("Plaquettes", f"{platelets/1000:.0f}k")

# --- ONGLET 2 ---
with tab2:
    st.markdown("""
        <div class='custom-card fade-in'>
            <h3 style='margin-top:0;'>🔬 Analyse de l'impact des variables</h3>
            <p class='text-muted'>Ce graphique montre comment l'IA a pris sa décision.</p>
        </div>
    """, unsafe_allow_html=True)

    shap_values = compute_shap_values(explainer, X_test)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    sorted_idx  = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color="#E74C3C", edgecolor="white", height=0.6
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Importance globale des features — SHAP", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig, transparent=True, use_container_width=True)

    if proba is not None:
        st.markdown("---")
        st.markdown("#### 🧬 Explication individuelle — Ce patient")
        X_input      = pd.DataFrame([patient])[FEATURE_NAMES]
        patient_shap = compute_shap_values(explainer, X_input)
        top_features = get_top_features(patient_shap, FEATURE_NAMES, top_n=5)

        st.markdown("**Top 5 facteurs de risque pour ce patient :**")
        for feat, score in top_features:
            direction = "🔴 augmente" if score > 0 else "🟢 réduit"
            st.markdown(f"- **{feat}** → {direction} le risque (`{score}`)")

        waterfall_path = "/tmp/patient_waterfall.png"
        plot_waterfall_single(explainer, X_input, FEATURE_NAMES, save_path=waterfall_path)
        st.image(waterfall_path, use_column_width=True)
    else:
        st.info("👈 Faites une prédiction d'abord pour voir l'explication individuelle")

# --- ONGLET 3 ---
with tab3:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.markdown("<h3 class='fade-in'>⚙️ Métriques de Validation</h3>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    col_m1.markdown(f"""
        <div class='custom-card fade-in delay-1' style='text-align:center;'>
            <h4>Recall ⭐</h4>
            <h2 style='color:#E74C3C; font-size:36px; margin:0;'>{recall_score(y_test, y_pred):.4f}</h2>
            <p style='color:#7F8C8D; font-size:12px;'>Priorité n°1</p>
        </div>
    """, unsafe_allow_html=True)

    col_m2.markdown(f"""
        <div class='custom-card fade-in delay-2' style='text-align:center;'>
            <h4>Precision</h4>
            <h2 style='color:#E67E22; font-size:36px; margin:0;'>{precision_score(y_test, y_pred):.4f}</h2>
            <p style='color:#7F8C8D; font-size:12px;'>Priorité n°2</p>
        </div>
    """, unsafe_allow_html=True)

    col_m3.markdown(f"""
        <div class='custom-card fade-in delay-3' style='text-align:center;'>
            <h4>ROC-AUC</h4>
            <h2 style='color:#27AE60; font-size:36px; margin:0;'>{roc_auc_score(y_test, y_proba):.4f}</h2>
            <p style='color:#7F8C8D; font-size:12px;'>Comparaison modèles</p>
        </div>
    """, unsafe_allow_html=True)

    col_m4.markdown(f"""
        <div class='custom-card fade-in delay-3' style='text-align:center;'>
            <h4>F1-Score</h4>
            <h2 style='color:#8E44AD; font-size:36px; margin:0;'>{f1_score(y_test, y_pred):.4f}</h2>
            <p style='color:#7F8C8D; font-size:12px;'>Équilibre P/R</p>
        </div>
    """, unsafe_allow_html=True)

    col_m5.markdown(f"""
        <div class='custom-card fade-in delay-3' style='text-align:center;'>
            <h4>Accuracy</h4>
            <h2 style='color:#2980B9; font-size:36px; margin:0;'>{accuracy_score(y_test, y_pred)*100:.1f}%</h2>
            <p style='color:#7F8C8D; font-size:12px;'>Moyenne globale</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Matrice de Confusion</h3>", unsafe_allow_html=True)

    col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
    with col_cm2:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    cbar=False, annot_kws={"size": 18, "weight": "bold"},
                    xticklabels=["Survécu", "Décédé"],
                    yticklabels=["Survécu", "Décédé"], ax=ax)
        ax.set_ylabel("Résultat Réel", fontsize=12, fontweight="bold", color="#34495E")
        ax.set_xlabel("Prédiction de l'IA", fontsize=12, fontweight="bold", color="#34495E")
        fig_cm.patch.set_alpha(0)
        st.pyplot(fig_cm, transparent=True, use_container_width=True)

# ==========================================
# 10. PIED DE PAGE
# ==========================================
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:#7F8C8D; font-size:13px; margin-top:20px;'>
        <p><strong>⚠️ Avertissement :</strong> Application développée à des fins éducatives et de recherche.<br>
        Ne se substitue en aucun cas à un diagnostic ou un avis médical professionnel.</p>
    </div>
""", unsafe_allow_html=True)