"""
app.py — Interface Streamlit pour Heart Failure Risk Predictor
Corrections :
  - get_model() appelé une seule fois, résultat partagé entre tabs
  - SHAP values : gestion du cas liste (XGBoost binaire)
  - predict_btn correctement utilisé via st.session_state
  - Suppression des imports dupliqués dans les tabs
  - Typage et formatage cohérents
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
import shap  # import global, pas dans la fonction
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.data_processing import load_data, handle_outliers, optimize_memory
from src.train_model import train_all_models
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix
)
import seaborn as sns

# ── Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

FEATURE_NAMES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",   "heart_failure_clinical_records_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")


# ── Load / Train Model (cached) ────────────────────────────────
@st.cache_resource
def get_model():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = load_data(DATA_PATH)
    df = handle_outliers(df)
    df = optimize_memory(df)

    X = df[FEATURE_NAMES]
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_NAMES)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_NAMES)

    trained_models = train_all_models(X_train_scaled, y_train)
    best_model = trained_models["XGBoost"]

    return best_model, scaler, X_train_scaled, X_test_scaled, y_test


# ── Gauge ──────────────────────────────────────────────────────
def risk_gauge(probability: float) -> go.Figure:
    color = "#27AE60" if probability < 0.4 else "#F39C12" if probability < 0.65 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        title={"text": "Mortality Risk", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#D5F5E3"},
                {"range": [40, 65], "color": "#FDEBD0"},
                {"range": [65,100], "color": "#FADBD8"},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


# ── SHAP Global Importance ─────────────────────────────────────
def shap_plot(model, X_train: pd.DataFrame) -> plt.Figure:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # XGBoost binaire peut renvoyer une liste [classe_0, classe_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature":    FEATURE_NAMES,
        "Importance": mean_shap
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color="#E74C3C", edgecolor="white", height=0.6)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── UI Header ──────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#2C3E50;'>🫀 Heart Failure Risk Predictor</h1>
    <p style='text-align:center; color:#7F8C8D; font-size:16px;'>
        AI-powered clinical decision support · XGBoost + SHAP Explainability
    </p><hr/>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Patient Parameters")
    st.markdown("---")

    age                      = st.slider("Age (years)", 40, 95, 60)
    ejection_fraction        = st.slider("Ejection Fraction (%)", 14, 80, 38)
    serum_creatinine         = st.number_input("Serum Creatinine (mg/dL)",  0.5,   10.0,  1.2,    step=0.1)
    serum_sodium             = st.number_input("Serum Sodium (mEq/L)",     110.0, 150.0, 137.0,   step=1.0)  # FIX: float pour cohérence
    creatinine_phosphokinase = st.number_input("CPK Enzyme (mcg/L)",        20,   8000,  250,     step=10)
    platelets                = st.number_input("Platelets (kiloplatelets/mL)", 25000, 850000, 265000, step=5000)
    time                     = st.slider("Follow-up period (days)", 4, 285, 100)

    st.markdown("---")
    st.markdown("**Comorbidities**")
    anaemia             = st.checkbox("Anaemia")
    diabetes            = st.checkbox("Diabetes")
    high_blood_pressure = st.checkbox("High Blood Pressure")
    smoking             = st.checkbox("Smoking")
    sex                 = st.radio("Sex", ["Female", "Male"], horizontal=True)

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ── Patient dict ───────────────────────────────────────────────
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
    "sex":                      1 if sex == "Male" else 0,
    "smoking":                  int(smoking),
    "time":                     time,
}

# ── Charger le modèle une seule fois ───────────────────────────
model, scaler, X_train, X_test, y_test = get_model()

# Stocker la prédiction en session state pour éviter de recalculer sur chaque interaction
if predict_btn or "proba" not in st.session_state:
    X_input        = pd.DataFrame([patient])[FEATURE_NAMES]
    X_input_scaled = pd.DataFrame(scaler.transform(X_input), columns=FEATURE_NAMES)
    st.session_state["proba"] = float(model.predict_proba(X_input_scaled)[0][1])

proba = st.session_state["proba"]

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔬 SHAP Explanation", "📈 Model Performance"])

# ── Tab 1 : Prediction ─────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.plotly_chart(risk_gauge(proba), use_container_width=True)

    with col2:
        risk_label = (
            "🔴 HIGH RISK"     if proba >= 0.65 else
            "🟡 MODERATE RISK" if proba >= 0.40 else
            "🟢 LOW RISK"
        )
        color = (
            "#E74C3C" if proba >= 0.65 else
            "#F39C12" if proba >= 0.40 else
            "#27AE60"
        )
        st.markdown(f"""
            <div style='background:{color}22; border-left:5px solid {color};
                padding:20px; border-radius:8px; margin-top:30px;'>
                <h2 style='color:{color}; margin:0;'>{risk_label}</h2>
                <p style='font-size:18px; margin:8px 0 0 0;'>
                    Predicted mortality probability: <b>{proba*100:.1f}%</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Patient Summary**")

        # FIX: formatage des valeurs numériques pour affichage propre
        summary_df = pd.DataFrame({
            "Parameter": list(patient.keys()),
            "Value":     [str(v) for v in patient.values()]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Tab 2 : SHAP ───────────────────────────────────────────────
with tab2:
    st.markdown("#### Global Feature Importance (SHAP)")
    fig_shap = shap_plot(model, X_train)
    st.pyplot(fig_shap, use_container_width=True)

# ── Tab 3 : Performance ────────────────────────────────────────
with tab3:
    st.markdown("#### Model Evaluation Metrics")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
    c2.metric("ROC-AUC",  f"{roc_auc_score(y_test, y_proba):.4f}")
    c3.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Survived", "Died"],
                yticklabels=["Survived", "Died"], ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig_cm)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#BDC3C7; font-size:12px;'>"
    "⚠️ For research and educational purposes only. Not a substitute for medical advice."
    "</p>",
    unsafe_allow_html=True
)
