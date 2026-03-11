"""
SHAP Explainability Module
Generates SHAP values and visualizations for heart failure prediction model.
"""

import os
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Explainer ─────────────────────────────────────────────────────────────────

def get_shap_explainer(model, X_train):
    """
    Crée le bon explainer SHAP selon le type de modèle.
    - TreeExplainer  → RandomForest, XGBoost, LightGBM
    - LinearExplainer → LogisticRegression (si ajouté plus tard)
    """
    name = type(model).__name__.lower()
    if any(k in name for k in ["forest", "xgb", "lgbm", "gradient", "tree"]):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, X_train)


# ── Calcul des SHAP values ────────────────────────────────────────────────────

def compute_shap_values(explainer, X):
    """
    Calcule les SHAP values.
    Retourne un array 2D (n_samples, n_features) — toujours pour la classe 1.
    Gère les 3 formats possibles selon la version de SHAP/sklearn :
      - list [classe0, classe1]     → ancien RandomForest/LightGBM
      - array 3D (n, features, 2)   → nouveau sklearn RandomForest
      - array 2D (n, features)      → XGBoost / déjà correct
    """
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        # Ancien format : liste [classe0, classe1]
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Nouveau format sklearn : (n_samples, n_features, n_classes)
        shap_values = shap_values[:, :, 1]

    return shap_values


# ── Plots globaux (sur tout le dataset d'entraînement) ────────────────────────

def plot_summary(shap_values, X, feature_names: list, save_path: str = None):
    """Beeswarm plot — impact de chaque feature sur l'ensemble des prédictions."""
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, plot_size=(10, 6))
    plt.title("SHAP Summary Plot — Impact des features sur le risque", pad=12)
    plt.tight_layout()
    _save(save_path, "Summary plot")
    plt.close()


def plot_bar_importance(shap_values, feature_names: list, save_path: str = None):
    """Bar plot — importance globale (mean |SHAP value|)."""
    mean_abs   = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color="#c0392b"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Importance globale des features — SHAP")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()
    _save(save_path, "Bar importance")
    plt.close()


# ── Plot individuel (un seul patient) ────────────────────────────────────────

def plot_waterfall_single(explainer, X_patient, feature_names: list, save_path: str = None):
    """
    Waterfall plot pour UN patient.
    X_patient : DataFrame d'une ligne (comme renvoyé par l'interface Streamlit).
    """
    exp = explainer(X_patient)

    # Normalisation pour classification binaire
    if exp.values.ndim == 3:
        vals = exp.values[0, :, 1]
        base = float(exp.base_values[0, 1])
    elif exp.values.ndim == 2 and exp.values.shape[-1] == 2:
        vals = exp.values[0, 1]
        base = float(exp.base_values[0])
    else:
        vals = exp.values[0] if exp.values.ndim > 1 else exp.values
        base = float(exp.base_values[0]) if np.ndim(exp.base_values) > 0 else float(exp.base_values)

    data = X_patient.values.flatten() if hasattr(X_patient, "values") else np.array(X_patient).flatten()

    explanation = shap.Explanation(
        values=vals,
        base_values=base,
        data=data,
        feature_names=feature_names
    )

    plt.figure()
    shap.waterfall_plot(explanation, show=False, max_display=12)
    plt.title("SHAP Waterfall — Explication individuelle patient", pad=10)
    plt.tight_layout()
    _save(save_path, "Waterfall plot")
    plt.close()


# ── Utilitaires ───────────────────────────────────────────────────────────────

def get_top_features(shap_values, feature_names: list, top_n: int = 5) -> list:
    """
    Retourne les top_n features les plus importantes (mean |SHAP|).
    Retourne : liste de tuples (nom_feature, score).
    """
    mean_abs   = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    return [(feature_names[i], round(float(mean_abs[i]), 4)) for i in sorted_idx[:top_n]]


def explain_patient(model, explainer, X_patient, feature_names: list,
                    save_dir: str = "outputs/shap") -> dict:
    """
    Pipeline complet d'explication pour un patient :
    1. Probabilité de décès
    2. Waterfall plot sauvegardé
    3. Top 5 features contributives

    Retourne : dict {probability, top_features, waterfall_path}
    """
    os.makedirs(save_dir, exist_ok=True)
    prob = model.predict_proba(X_patient)[0][1]

    waterfall_path = os.path.join(save_dir, "patient_waterfall.png")
    plot_waterfall_single(explainer, X_patient, feature_names, save_path=waterfall_path)

    patient_shap = compute_shap_values(explainer, X_patient)
    if patient_shap.ndim > 1:
        patient_shap = patient_shap[0]

    top = sorted(zip(feature_names, patient_shap), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "probability" : round(float(prob), 4),
        "top_features": [(n, round(float(v), 4)) for n, v in top],
        "waterfall_path": waterfall_path,
    }


def _save(path, label):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] {label} sauvegardé → {path}")