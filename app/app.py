import os
import sys
import pickle
import warnings
import uuid

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
# 1. CONFIGURATION INITIALE
# ==========================================
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CONSTANTES & CHARGEMENT
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
        model = pickle.load(f)
    return model

@st.cache_data(show_spinner=False)
def get_test_data():
    df = load_data(DATA_PATH)
    df = handle_outliers(df)
    df = optimize_memory(df)
    X = df[FEATURE_NAMES]
    y = df["DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_test

model = get_model()
X_train, X_test, y_test = get_test_data()
explainer = get_shap_explainer(model, X_train)

# ==========================================
# 3. ÉTAT DE SESSION & NAVIGATION
# ==========================================
if "proba" not in st.session_state:
    st.session_state["proba"] = None

# Menu de navigation principal
st.sidebar.markdown("<h2 style='color: white; font-weight: 800; text-align: center; margin-bottom: 20px;'>🫀 CardioCare<span style='color: #38BDF8;'>AI</span></h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Menu Principal</p>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation", 
    ["🏥 Tableau de Bord Patient", "🧠 Explicabilité IA (SHAP)", "⚙️ Santé du Modèle"],
    label_visibility="collapsed"
)
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# ==========================================
# 4. MOTEUR D'ANIMATION & CSS AVANCÉ
# ==========================================
# Astuce : On génère un ID unique basé sur le nom de la page pour forcer Streamlit 
# à rejouer l'animation CSS à CHAQUE changement de rubrique dans le menu.
anim_id = hash(page)

st.markdown(f"""
    <style>
        /* --- Importation Typographie --- */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: #0F172A;
        }}

        /* --- Arrière-plan Animé : Orbes Biomédicales en lévitation --- */
        .stApp {{
            background-color: #F8FAFC;
            overflow-x: hidden;
        }}
        .bg-orb {{
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            z-index: -1;
            animation: float 20s infinite ease-in-out alternate;
            opacity: 0.6;
        }}
        .orb-1 {{ top: -10%; left: -10%; width: 50vw; height: 50vw; background: radial-gradient(circle, #E0F2FE 0%, transparent 70%); animation-delay: 0s; }}
        .orb-2 {{ bottom: -20%; right: -10%; width: 60vw; height: 60vw; background: radial-gradient(circle, #DBEAFE 0%, transparent 70%); animation-delay: -5s; }}
        .orb-3 {{ top: 40%; left: 40%; width: 30vw; height: 30vw; background: radial-gradient(circle, #F1F5F9 0%, transparent 70%); animation-delay: -10s; }}
        
        @keyframes float {{
            0% {{ transform: translate(0, 0) scale(1); }}
            100% {{ transform: translate(50px, -50px) scale(1.1); }}
        }}

        /* --- Animation de Changement de Page (Se rejoue à chaque clic) --- */
        [data-testid="block-container"] {{
            animation: pageEnter_{anim_id} 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}
        @keyframes pageEnter_{anim_id} {{
            0% {{ opacity: 0; transform: translateY(40px) scale(0.98); filter: blur(4px); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }}
        }}

        /* --- Sidebar Redesign --- */[data-testid="stSidebar"] {{
            background: rgba(15, 23, 42, 0.98) !important;
            border-right: 1px solid rgba(255,255,255,0.05);
        }}
        div[role="radiogroup"] > label {{
            background: rgba(255,255,255,0.05);
            padding: 12px 15px;
            border-radius: 10px;
            margin-bottom: 8px;
            transition: all 0.2s ease;
            cursor: pointer;
        }}
        div[role="radiogroup"] > label:hover {{ background: rgba(255,255,255,0.1); transform: translateX(5px); }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {{ color: #F1F5F9 !important; }}

        /* --- Bouton Lancer Analyse --- */
        .stButton > button {{
            background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 15px;
            font-weight: 700;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
            width: 100%;
        }}
        .stButton > button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5);
            background: linear-gradient(135deg, #1D4ED8 0%, #0891B2 100%);
        }}

        /* --- Glassmorphism Cards --- */
        .glass-panel {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 1);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
        }}
        
        /* Alertes Pulsantes */
        @keyframes pulseAlert {{
            0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }}
            70% {{ box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        .alert-critical {{ animation: pulseAlert 2s infinite; border: 2px solid #EF4444; }}

        /* Headers customisés */
        h1, h2, h3 {{ color: #0F172A !important; font-weight: 800 !important; letter-spacing: -0.5px; }}
        .subtitle {{ color: #64748B; font-size: 1.1rem; font-weight: 500; margin-top: -10px; margin-bottom: 30px; }}
    </style>

    <!-- Injection des Orbes de fond -->
    <div class="bg-orb orb-1"></div>
    <div class="bg-orb orb-2"></div>
    <div class="bg-orb orb-3"></div>
""", unsafe_allow_html=True)


# ==========================================
# 5. SIDEBAR : ENTRÉE DES DONNÉES
# ==========================================
st.sidebar.markdown("<p style='color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Paramètres Cliniques</p>", unsafe_allow_html=True)

with st.sidebar.expander("👤 Profil & Constantes", expanded=True):
    age = st.slider("Âge", 18, 95, 60)
    sex = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
    time = st.slider("Suivi (jours)", 4, 285, 100)
    ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38)

with st.sidebar.expander("🩸 Biologie Sanguine", expanded=False):
    serum_creatinine = st.number_input("Créatinine (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
    serum_sodium = st.number_input("Sodium (mEq/L)", 110.0, 150.0, 137.0, step=1.0)
    creatinine_phosphokinase = st.number_input("CPK (mcg/L)", 20, 8000, 250, step=10)
    platelets = st.number_input("Plaquettes (k/mL)", 25000, 850000, 265000, step=5000)

with st.sidebar.expander("🩺 Comorbidités", expanded=False):
    anaemia = st.checkbox("Anémie")
    diabetes = st.checkbox("Diabète")
    high_blood_pressure = st.checkbox("Hypertension")
    smoking = st.checkbox("Fumeur")

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("🚀 Lancer l'Analyse"):
    patient = {
        "age": age, "anaemia": int(anaemia), "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": int(diabetes), "ejection_fraction": ejection_fraction,
        "high_blood_pressure": int(high_blood_pressure), "platelets": platelets,
        "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
        "sex": 1 if sex == "Homme" else 0, "smoking": int(smoking), "time": time,
    }
    with st.spinner("Exécution des algorithmes en cours..."):
        X_input = pd.DataFrame([patient])[FEATURE_NAMES]
        st.session_state["proba"] = float(model.predict_proba(X_input)[0][1])

proba = st.session_state["proba"]


# ==========================================
# 6. VUES PRINCIPALES (SPA)
# ==========================================

# --- PAGE 1 : TABLEAU DE BORD ---
if page == "🏥 Tableau de Bord Patient":
    st.markdown("<h1>Tableau de Bord Clinique</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Aperçu global de l'état du patient et évaluation immédiate du risque.</div>", unsafe_allow_html=True)

    # Ligne 1 : Résumé Patient (Cartes KPI horizontales)
    st.markdown(f"""
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 25px;'>
            <div class='glass-panel' style='text-align: center; padding: 15px;'>
                <div style='color: #64748B; font-size: 13px; font-weight: 700; text-transform: uppercase;'>Âge Patient</div>
                <div style='font-size: 28px; font-weight: 800; color: #0F172A;'>{age} <span style='font-size:16px; font-weight:500;'>ans</span></div>
            </div>
            <div class='glass-panel' style='text-align: center; padding: 15px;'>
                <div style='color: #64748B; font-size: 13px; font-weight: 700; text-transform: uppercase;'>Sexe</div>
                <div style='font-size: 28px; font-weight: 800; color: #0F172A;'>{sex}</div>
            </div>
            <div class='glass-panel' style='text-align: center; padding: 15px;'>
                <div style='color: #64748B; font-size: 13px; font-weight: 700; text-transform: uppercase;'>Frac. Éjection</div>
                <div style='font-size: 28px; font-weight: 800; color: #0284C7;'>{ejection_fraction}%</div>
            </div>
            <div class='glass-panel' style='text-align: center; padding: 15px;'>
                <div style='color: #64748B; font-size: 13px; font-weight: 700; text-transform: uppercase;'>Durée de suivi</div>
                <div style='font-size: 28px; font-weight: 800; color: #0F172A;'>{time} <span style='font-size:16px; font-weight:500;'>jrs</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if proba is None:
        st.info("👈 Renseignez le profil complet dans le menu latéral et cliquez sur **Lancer l'Analyse**.")
    else:
        col1, col2 = st.columns([1, 1.2])
        
        # Jauge Plotly
        with col1:
            st.markdown("<div class='glass-panel' style='height: 100%;'>", unsafe_allow_html=True)
            color = "#10B981" if proba < 0.4 else "#F59E0B" if proba < 0.65 else "#EF4444"
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(proba * 100, 1),
                number={"suffix": "%", "font": {"size": 50, "color": color, "family": "Plus Jakarta Sans"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "transparent"},
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0.05)", "borderwidth": 0,
                    "steps":[
                        {"range": [0, 40], "color": "rgba(16, 185, 129, 0.1)"},
                        {"range":[40, 65], "color": "rgba(245, 158, 11, 0.1)"},
                        {"range": [65, 100], "color": "rgba(239, 68, 68, 0.1)"},
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(t=20, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Plus Jakarta Sans"})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Boîte d'alerte dynamique
        with col2:
            risk_label = "CRITIQUE" if proba >= 0.65 else "MODÉRÉ" if proba >= 0.40 else "FAIBLE"
            alert_class = "alert-critical" if proba >= 0.65 else ""
            desc = (
                "Intervention médicale urgente requise. Fort risque de défaillance à court terme." if proba >= 0.65 else
                "Surveillance clinique rapprochée conseillée. Paramètres instables." if proba >= 0.40 else
                "Paramètres vitaux dans les normes. Pas de risque immédiat détecté."
            )
            bg_grad = "linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.02))" if proba >= 0.65 else "linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.02))" if proba >= 0.40 else "linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.02))"
            
            st.markdown(f"""
                <div class='glass-panel {alert_class}' style='background: {bg_grad}; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='text-transform: uppercase; font-size: 14px; font-weight: 800; color: {color}; letter-spacing: 2px;'>Alerte Système IA</div>
                    <h2 style='font-size: 46px; font-weight: 800; color: #0F172A; margin: 5px 0 15px 0;'>RISQUE {risk_label}</h2>
                    <p style='font-size: 18px; color: #334155; line-height: 1.5; font-weight: 500;'>{desc}</p>
                    <div style='margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(0,0,0,0.1); display: flex; justify-content: space-between;'>
                        <span style='color: #64748B; font-weight: 600;'>Fiabilité du modèle : <span style='color:#0F172A; font-weight:800;'>86.7%</span></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- PAGE 2 : EXPLICABILITÉ (SHAP) ---
elif page == "🧠 Explicabilité IA (SHAP)":
    st.markdown("<h1>Explicabilité et Transparence</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Comprenez le raisonnement de l'Intelligence Artificielle de A à Z.</div>", unsafe_allow_html=True)

    if proba is None:
        st.warning("⚠️ Veuillez lancer une analyse depuis le menu latéral pour débloquer l'explicabilité de ce profil.")
    else:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h3>🧬 Impact Spécifique au Patient (Waterfall)</h3>", unsafe_allow_html=True)
        col_w1, col_w2 = st.columns([2, 1])
        
        with col_w1:
            # Récupération des données du Sidebar pour SHAP
            patient_data = {
                "age": age, "anaemia": int(anaemia), "creatinine_phosphokinase": creatinine_phosphokinase,
                "diabetes": int(diabetes), "ejection_fraction": ejection_fraction,
                "high_blood_pressure": int(high_blood_pressure), "platelets": platelets,
                "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
                "sex": 1 if sex == "Homme" else 0, "smoking": int(smoking), "time": time,
            }
            X_input = pd.DataFrame([patient_data])[FEATURE_NAMES]
            waterfall_path = "/tmp/patient_waterfall.png"
            plot_waterfall_single(explainer, X_input, FEATURE_NAMES, save_path=waterfall_path)
            st.image(waterfall_path, use_column_width=True)
            
        with col_w2:
            patient_shap = compute_shap_values(explainer, X_input)
            top_features = get_top_features(patient_shap, FEATURE_NAMES, top_n=5)
            st.markdown("<div style='background: rgba(255,255,255,0.5); padding: 20px; border-radius: 15px;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0;'>Top 5 Facteurs Déterminants</h4>", unsafe_allow_html=True)
            for feat, score in top_features:
                if score > 0:
                    st.markdown(f"<div style='border-left: 4px solid #EF4444; padding-left: 10px; margin-bottom: 12px;'><b>{feat}</b><br><span style='color:#EF4444; font-size:14px; font-weight:600;'>↗ Aggrave le risque (+{score:.2f})</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='border-left: 4px solid #10B981; padding-left: 10px; margin-bottom: 12px;'><b>{feat}</b><br><span style='color:#10B981; font-size:14px; font-weight:600;'>↘ Réduit le risque ({score:.2f})</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h3>🌍 Importance Globale des Biomarqueurs (Cohorte)</h3>", unsafe_allow_html=True)
        shap_values = compute_shap_values(explainer, X_test)
        mean_abs = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.barh([FEATURE_NAMES[i] for i in sorted_idx], mean_abs[sorted_idx], color="#0284C7", alpha=0.9, height=0.6)
        ax.set_xlabel("Poids moyen dans la décision de l'IA (|SHAP|)", fontsize=11, fontweight="bold", color="#334155")
        ax.tick_params(colors="#0F172A", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("transparent")
        ax.spines["bottom"].set_color("#CBD5E1")
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- PAGE 3 : PERFORMANCES ---
elif page == "⚙️ Santé du Modèle":
    st.markdown("<h1>Évaluation Technique du Moteur LightGBM</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analyse de la robustesse et des métriques statistiques de l'algorithme.</div>", unsafe_allow_html=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Cartes de métriques
    st.markdown(f"""
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px;'>
            <div class='glass-panel' style='text-align:center; padding: 20px;'>
                <div style='color:#EF4444; font-size:36px; font-weight:800;'>{recall_score(y_test, y_pred):.3f}</div>
                <div style='color:#64748B; font-weight:700; font-size:14px;'>SENSIBILITÉ (RECALL)</div>
            </div>
            <div class='glass-panel' style='text-align:center; padding: 20px;'>
                <div style='color:#F59E0B; font-size:36px; font-weight:800;'>{precision_score(y_test, y_pred):.3f}</div>
                <div style='color:#64748B; font-weight:700; font-size:14px;'>PRÉCISION</div>
            </div>
            <div class='glass-panel' style='text-align:center; padding: 20px;'>
                <div style='color:#10B981; font-size:36px; font-weight:800;'>{roc_auc_score(y_test, y_proba):.3f}</div>
                <div style='color:#64748B; font-weight:700; font-size:14px;'>ROC-AUC</div>
            </div>
            <div class='glass-panel' style='text-align:center; padding: 20px;'>
                <div style='color:#8B5CF6; font-size:36px; font-weight:800;'>{f1_score(y_test, y_pred):.3f}</div>
                <div style='color:#64748B; font-weight:700; font-size:14px;'>SCORE F1</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Matrice de Confusion</h3>", unsafe_allow_html=True)
    col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
    with col_cm2:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 22, "weight": "bold"},
                    xticklabels=["Survie Prédite", "Décès Prédit"], yticklabels=["Survie Réelle", "Décès Réel"], ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11, weight="bold", color="#334155")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, weight="bold", color="#334155")
        fig_cm.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        st.pyplot(fig_cm, transparent=True)
    st.markdown("</div>", unsafe_allow_html=True)