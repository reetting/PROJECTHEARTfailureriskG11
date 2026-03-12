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

# ── Import des modules locaux ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_data, handle_outliers, optimize_memory

# ── Config de la page ──────────────────────────────────────────
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constantes & Couleurs (Lumineuses et Douces) ───────────────
FEATURE_NAMES =[
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "heart_failure_clinical_records_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")

# Couleurs chaleureuses mais claires
COLOR_SAFE   = "#2A9D8F"  # Bleu/Vert canard (médical et rassurant)
COLOR_WARN   = "#F4A261"  # Orange sable (chaleureux)
COLOR_DANGER = "#E76F51"  # Terre cuite (alerte sans être agressif)


# ── CSS & Animation d'Introduction (Splash Screen) ─────────────
def inject_ui_styles():
    st.markdown("""
        <style>
        /* 1. ANIMATION D'OUVERTURE (SPLASH SCREEN) */
        #splash-screen {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background-color: #FAFAFA;
            z-index: 999999;
            display: flex; justify-content: center; align-items: center;
            flex-direction: column;
            animation: hideSplashScreen 3s ease-in-out forwards;
        }
        .splash-heart {
            font-size: 90px;
            animation: heartbeat 1s infinite;
            margin-bottom: 20px;
        }
        .splash-text {
            color: #2b2b2b;
            font-family: 'Inter', sans-serif;
            font-weight: 300;
            letter-spacing: 2px;
        }
        
        @keyframes heartbeat {
            0% { transform: scale(1); }
            15% { transform: scale(1.15); }
            30% { transform: scale(1); }
            45% { transform: scale(1.15); }
            100% { transform: scale(1); }
        }
        @keyframes hideSplashScreen {
            0% { opacity: 1; }
            80% { opacity: 1; }
            100% { opacity: 0; visibility: hidden; pointer-events: none; }
        }

        /* 2. STYLES GLOBAUX & ANIMATIONS DE LA PAGE */
        .stApp {
            background-color: #F8F9FA; /* Fond global très clair */
        }
        
        @keyframes slideUpFade {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade {
            animation: slideUpFade 0.8s ease-out forwards;
        }
        .delay-1 { animation-delay: 2.5s; opacity: 0; } /* Attend la fin du splash screen */
        .delay-2 { animation-delay: 2.8s; opacity: 0; }
        .delay-3 { animation-delay: 3.1s; opacity: 0; }

        /* 3. CARTES (CARDS) LUMINEUSES */
        .glass-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.04);
            border: 1px solid #EAEAEA;
            color: #333333;
        }
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }
        .stat-label {
            font-size: 1rem;
            color: #7A7A7A;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        </style>

        <!-- HTML DU SPLASH SCREEN -->
        <div id="splash-screen">
            <div class="splash-heart">🫀</div>
            <h2 class="splash-text">CardioCare AI</h2>
            <p style="color: #888;">Initialisation du modèle clinique...</p>
        </div>
    """, unsafe_allow_html=True)


# ── Fonctions Backend (Cachées) ────────────────────────────────
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
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


# ── Visualisations (Refaites en mode clair/lumineux) ───────────
def create_gauge_chart(probability):
    if probability < 0.4:
        color = COLOR_SAFE
    elif probability < 0.65:
        color = COLOR_WARN
    else:
        color = COLOR_DANGER

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range":[0, 100], "tickwidth": 1, "tickcolor": "#E2D3D3"},
            "bar": {"color": color, "thickness": 0.2},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#D1FFF8"},  # Vert d'eau très clair
                {"range":[40, 65], "color": "#FEF5EC"}, # Sable très clair
                {"range":[65, 100], "color": "#FFB19F"},# Terre cuite très clair
            ],
        }
    ))
    fig.update_layout(
        height=250, margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", font={'color': "#BBA9A9"}
    )
    return fig

def create_shap_plot(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": mean_shap}).sort_values("Importance", ascending=True)
    df_imp["Feature"] = df_imp["Feature"].str.replace("_", " ").str.capitalize()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('none')  # Fond transparent corrigé
    ax.set_facecolor('none')         # Fond transparent corrigé
    
    # Barres douces et chaleureuses
    ax.barh(df_imp["Feature"], df_imp["Importance"], color=COLOR_DANGER, alpha=0.75, height=0.5, edgecolor="white")
    ax.set_xlabel("Impact sur la prédiction (Valeur SHAP absolue moyenne)", color="#555", fontsize=10)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#FFD0D0")
    ax.spines["left"].set_color("#666262")
    ax.tick_params(colors='#555')
    
    plt.tight_layout()
    return fig


# ── MAIN APP ───────────────────────────────────────────────────
def main():
    inject_ui_styles()
    
    model = get_model()
    X_test, y_test = get_test_data()

    # --- EN-TÊTE ---
    st.markdown("""
        <div class="animate-fade delay-1" style="text-align: center; margin-bottom: 40px; margin-top: 20px;">
            <h1 style="color: #2b2b2b; font-weight: 800; letter-spacing: -1px;">CardioCare AI</h1>
            <p style="color: #888; font-size: 1.2rem;">Système d'évaluation avancée du risque d'insuffisance cardiaque</p>
        </div>
    """, unsafe_allow_html=True)

    # --- BARRE LATÉRALE (SAISIE DES DONNÉES) ---
    with st.sidebar:
        st.markdown("### 📋 Dossier du Patient")
        st.markdown("Veuillez ajuster les constantes cliniques :")
        
        age = st.slider("Âge (années)", 40, 95, 60)
        ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38)
        
        st.markdown("---")
        serum_creatinine = st.number_input("Créatinine sérique (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
        serum_sodium = st.number_input("Sodium sérique (mEq/L)", 110.0, 150.0, 137.0, step=1.0)
        creatinine_phosphokinase = st.number_input("Enzyme CPK (mcg/L)", 20, 8000, 250, step=10)
        platelets = st.number_input("Plaquettes (k/mL)", 25000, 850000, 265000, step=5000)
        
        st.markdown("---")
        st.markdown("**Comorbidités & Profil**")
        col1, col2 = st.columns(2)
        with col1:
            anaemia = st.checkbox("Anémie")
            diabetes = st.checkbox("Diabète")
        with col2:
            high_blood_pressure = st.checkbox("Hypertension")
            smoking = st.checkbox("Fumeur")
            
        sex = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
        time = st.slider("Période de suivi (jours)", 4, 285, 100)

    # Constitution des données
    patient_data = {
        "age": age, "anaemia": int(anaemia), "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": int(diabetes), "ejection_fraction": ejection_fraction,
        "high_blood_pressure": int(high_blood_pressure), "platelets": platelets,
        "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
        "sex": 1 if sex == "Homme" else 0, "smoking": int(smoking), "time": time,
    }

    # Calcul de la prédiction
    X_input = pd.DataFrame([patient_data])[FEATURE_NAMES]
    proba = float(model.predict_proba(X_input)[0][1])

    # --- CONTENU PRINCIPAL ---
    
    # 1. Zone de Résultat Supérieure (Mise en avant visuelle)
    col_gauge, col_result = st.columns([1, 1.2])
    
    with col_gauge:
        st.markdown("<div class='glass-card animate-fade delay-2'>", unsafe_allow_html=True)
        st.markdown("<div class='stat-label'>Indice de Risque</div>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge_chart(proba), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_result:
        # Logique de diagnostic
        if proba >= 0.65:
            lbl, col, txt = "RISQUE ÉLEVÉ", COLOR_DANGER, "Une surveillance médicale rapprochée est fortement recommandée au vu des biomarqueurs actuels."
        elif proba >= 0.40:
            lbl, col, txt = "RISQUE MODÉRÉ", COLOR_WARN, "Des facteurs de risque sont présents. Une consultation préventive est conseillée."
        else:
            lbl, col, txt = "RISQUE FAIBLE", COLOR_SAFE, "Les indicateurs actuels ne montrent pas de risque critique immédiat."
            
        st.markdown(f"""
            <div class='glass-card animate-fade delay-2' style='border-left: 8px solid {col}; height: 93%;'>
                <div class='stat-label'>Diagnostic Estimé</div>
                <h2 style='color: {col}; font-weight: 800; font-size: 2.2rem; margin: 10px 0;'>{lbl}</h2>
                <p style='color: #555; font-size: 1.1rem; line-height: 1.6; margin-top: 20px;'>{txt}</p>
                <hr style='border: none; border-top: 1px solid #EAEAEA; margin: 25px 0;'>
                <div style='display: flex; justify-content: space-between; color: #888; font-size: 0.9rem;'>
                    <span>Probabilité exacte : <b>{proba*100:.1f}%</b></span>
                    <span>Fiabilité du modèle : <b>{accuracy_score(y_test, model.predict(X_test))*100:.1f}%</b></span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 2. Zone d'Explications et Métriques (Onglets)
    st.markdown("<div class='animate-fade delay-3' style='margin-top: 30px;'>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔬 Explicabilité de l'IA (SHAP)", "📈 Évaluation du Modèle"])
    
    with tab1:
        st.markdown("""
            <div style="padding: 15px 0;">
                <h4 style="color: #333;">Comment l'IA a-t-elle pris sa décision ?</h4>
                <p style="color: #777;">Ce graphique montre quelles variables influencent le plus le risque prédit.</p>
            </div>
        """, unsafe_allow_html=True)
        st.pyplot(create_shap_plot(model, X_test), transparent=True)
        
    with tab2:
        y_pred = model.predict(X_test)
        y_proba_all = model.predict_proba(X_test)[:, 1]
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='glass-card'><div class='stat-label'>Précision</div><div class='stat-value' style='color:{COLOR_SAFE}'>{accuracy_score(y_test, y_pred)*100:.1f}%</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='glass-card'><div class='stat-label'>ROC-AUC</div><div class='stat-value' style='color:{COLOR_WARN}'>{roc_auc_score(y_test, y_proba_all):.3f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card'><div class='stat-label'>Score F1</div><div class='stat-value' style='color:{COLOR_DANGER}'>{f1_score(y_test, y_pred):.3f}</div></div>", unsafe_allow_html=True)
        
        st.markdown("<h5 style='color:#555; margin-top:20px;'>Matrice de Confusion</h5>", unsafe_allow_html=True)
        fig_cm, ax = plt.subplots(figsize=(6, 3))
        fig_cm.patch.set_facecolor('none') # Fond transparent corrigé
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlOrRd", 
                    cbar=False, annot_kws={"size": 14}, 
                    xticklabels=["Survie", "Décès"], yticklabels=["Survie", "Décès"], ax=ax)
        ax.set_ylabel("Réalité", color="#555", fontweight="bold")
        ax.set_xlabel("Prédiction", color="#555", fontweight="bold")
        ax.tick_params(colors='#555')
        st.pyplot(fig_cm, transparent=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
