import os
import sys
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Assure-toi que ces imports fonctionnent toujours dans ton arborescence
from src.data_processing import load_data, handle_outliers, optimize_memory

# ── Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio Risk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Personnalisé (Animations & Design Chaleureux) ──────────
def inject_custom_css():
    st.markdown("""
        <style>
        /* Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(184, 64, 64, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(184, 64, 64, 0); }
            100% { box-shadow: 0 0 0 0 rgba(184, 64, 64, 0); }
        }
        
        /* Classes d'animation */
        .fade-in {
            animation: fadeInUp 0.8s ease-out forwards;
        }
        .delay-1 { animation-delay: 0.2s; opacity: 0; }
        .delay-2 { animation-delay: 0.4s; opacity: 0; }
        
        .card-high-risk {
            animation: pulse 2s infinite;
        }

        /* Style des textes et conteneurs */
        .title-text {
            color: #4A3C31;
            font-family: 'Georgia', serif;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitle-text {
            color: #8D7B68;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #FDFBF7;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #F0EAE1;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ── Constantes & Chemins ───────────────────────────────────────
FEATURE_NAMES =[
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "heart_failure_clinical_records_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")

# Palette de couleurs chaleureuses
COLOR_SAFE = "#6B8E7B"  # Sauge
COLOR_WARN = "#D99A5B"  # Ambre
COLOR_DANGER = "#B84040" # Terre cuite / Sanguine


# ── Fonctions de chargement ────────────────────────────────────
@st.cache_resource
def get_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def get_test_data():
    df = load_data(DATA_PATH)
    df = handle_outliers(df)
    df = optimize_memory(df)
    X, y = df[FEATURE_NAMES], df["DEATH_EVENT"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


# ── Visualisations ─────────────────────────────────────────────
def risk_gauge(probability: float) -> go.Figure:
    if probability < 0.4:
        color = COLOR_SAFE
    elif probability < 0.65:
        color = COLOR_WARN
    else:
        color = COLOR_DANGER

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 45, "color": color, "family": "Arial"}},
        title={"text": "Probabilité de Mortalité", "font": {"size": 16, "color": "#8D7B68"}},
        gauge={
            "axis": {"range":[0, 100], "tickwidth": 1, "tickcolor": "#8D7B68"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range":[0,  40], "color": "#EAF0EC"}, # Sauge très clair
                {"range":[40, 65], "color": "#FBF3EB"}, # Ambre très clair
                {"range":[65,100], "color": "#F5EBEB"}, # Sanguine très clair
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#4A3C31"})
    return fig

def shap_plot(model, X_test: pd.DataFrame) -> plt.Figure:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": mean_shap}).sort_values("Importance", ascending=True)

    # Noms plus lisibles
    importance_df["Feature"] = importance_df["Feature"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('transparent')
    ax.set_facecolor('transparent')
    
    ax.barh(importance_df["Feature"], importance_df["Importance"], color=COLOR_DANGER, alpha=0.85, edgecolor="white", height=0.5)
    ax.set_xlabel("Impact moyen sur la prédiction (|SHAP|)", fontsize=10, color="#4A3C31")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color('#8D7B68')
    ax.spines["left"].set_color('#8D7B68')
    ax.tick_params(colors='#4A3C31')
    
    plt.tight_layout()
    return fig


# ── Header Animé ───────────────────────────────────────────────
st.markdown("""
    <div class='fade-in'>
        <h1 class='title-text'>🫀 Évaluation du Risque Cardiaque</h1>
        <p class='subtitle-text'>
            Assistance au diagnostic médical · Modèle d'Intelligence Artificielle Explicable
        </p>
    </div>
""", unsafe_allow_html=True)


# ── Sidebar (Interface d'entrée) ───────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color: #4A3C31;'>Dossier Patient</h3>", unsafe_allow_html=True)
    
    with st.expander("📊 Paramètres Biologiques", expanded=True):
        age                      = st.slider("Âge (années)", 40, 95, 60)
        ejection_fraction        = st.slider("Fraction d'éjection (%)", 14, 80, 38)
        serum_creatinine         = st.number_input("Créatinine sérique (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
        serum_sodium             = st.number_input("Sodium sérique (mEq/L)", 110.0, 150.0, 137.0, step=1.0)
        creatinine_phosphokinase = st.number_input("Enzyme CPK (mcg/L)", 20, 8000, 250, step=10)
        platelets                = st.number_input("Plaquettes (k/mL)", 25000, 850000, 265000, step=5000)
    
    with st.expander("🩺 Antécédents & Mode de vie", expanded=True):
        sex                 = st.radio("Sexe",["Femme", "Homme"], horizontal=True)
        col1, col2 = st.columns(2)
        with col1:
            anaemia             = st.checkbox("Anémie")
            diabetes            = st.checkbox("Diabète")
        with col2:
            high_blood_pressure = st.checkbox("Hypertension")
            smoking             = st.checkbox("Fumeur")
            
    time = st.slider("Période de suivi estimée (jours)", 4, 285, 100)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Lancer l'Analyse 🔍", use_container_width=True, type="primary")

# ── Préparation des données Patient ────────────────────────────
patient = {
    "age": age,
    "anaemia": int(anaemia),
    "creatinine_phosphokinase": creatinine_phosphokinase,
    "diabetes": int(diabetes),
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": int(high_blood_pressure),
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": 1 if sex == "Homme" else 0,
    "smoking": int(smoking),
    "time": time,
}

model = get_model()
X_test, y_test = get_test_data()

if predict_btn or "proba" not in st.session_state:
    X_input = pd.DataFrame([patient])[FEATURE_NAMES]
    st.session_state["proba"] = float(model.predict_proba(X_input)[0][1])

proba = st.session_state["proba"]


# ── Tabs (Contenu Principal) ───────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Bilan du Risque", "🔬 Explicabilité (SHAP)", "📈 Métriques du Modèle"])

with tab1:
    col_gauge, col_text = st.columns([1.2, 1])
    
    with col_gauge:
        st.markdown("<div class='fade-in delay-1'>", unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(proba), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_text:
        # Logique de couleur et de statut
        if proba >= 0.65:
            risk_label, color, bg_color, pulse_class = "RISQUE ÉLEVÉ", COLOR_DANGER, "#F5EBEB", "card-high-risk"
            icon = "🔴"
        elif proba >= 0.40:
            risk_label, color, bg_color, pulse_class = "RISQUE MODÉRÉ", COLOR_WARN, "#FBF3EB", ""
            icon = "🟠"
        else:
            risk_label, color, bg_color, pulse_class = "RISQUE FAIBLE", COLOR_SAFE, "#EAF0EC", ""
            icon = "🟢"

        # Carte de résultat animée
        st.markdown(f"""
            <div class='fade-in delay-2 {pulse_class}' style='
                background-color: {bg_color}; 
                border-left: 6px solid {color};
                padding: 25px; 
                border-radius: 10px; 
                margin-top: 40px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);'>
                <h2 style='color: {color}; margin: 0; font-family: "Georgia", serif;'>
                    {icon} {risk_label}
                </h2>
                <p style='font-size: 1.1rem; color: #4A3C31; margin-top: 10px;'>
                    La probabilité estimée d'événement indésirable est de <b>{proba*100:.1f}%</b>.
                </p>
                <p style='font-size: 0.9rem; color: #8D7B68; margin-bottom:0;'>
                    *Basé sur l'analyse combinée des biomarqueurs et des antécédents médicaux.
                </p>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div class='fade-in'>
            <h3 style='color:#4A3C31;'>Quels facteurs influencent cette prédiction ?</h3>
            <p style='color:#8D7B68;'>Le graphique ci-dessous illustre l'importance de chaque variable clinique dans la prise de décision de l'Intelligence Artificielle.</p>
        </div>
    """, unsafe_allow_html=True)
    fig_shap = shap_plot(model, X_test)
    st.pyplot(fig_shap, use_container_width=True)

with tab3:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#4A3C31;'>Performances Techniques</h3>", unsafe_allow_html=True)
    
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Style des métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Précision (Accuracy)", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
    c2.metric("Score ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    c3.metric("Score F1", f"{f1_score(y_test, y_pred):.3f}")

    st.markdown("<hr style='border-color: #F0EAE1;'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8D7B68; font-weight:bold;'>Matrice de Confusion</p>", unsafe_allow_html=True)
    
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(6, 4))
    fig_cm.patch.set_facecolor('transparent')
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", # Palette chaude "Orange-Red"
                xticklabels=["Survie", "Décès"],
                yticklabels=["Survie", "Décès"], ax=ax, 
                cbar_kws={'label': 'Nombre de patients'})
    ax.set_ylabel("Réalité", color="#4A3C31", fontweight='bold')
    ax.set_xlabel("Prédiction de l'IA", color="#4A3C31", fontweight='bold')
    ax.tick_params(colors='#4A3C31')
    st.pyplot(fig_cm)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
    <div style='margin-top: 50px; text-align: center; color: #B0A69B; font-size: 0.85rem; border-top: 1px solid #F0EAE1; padding-top: 20px;'>
        ⚠️ <b>Avis Important :</b> Cet outil est fourni à des fins de recherche et de démonstration. 
        Il ne se substitue en aucun cas à un diagnostic ou à un avis médical professionnel.
    </div>
""", unsafe_allow_html=True)