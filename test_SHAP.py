"""
tests/test_shap_explainability.py
Tests automatisés — coordonnés avec train_model.py et shap_explainability.py.
Exécution : pytest tests/ -v
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.SHAP import (
    get_shap_explainer,
    compute_shap_values,
    get_top_features,
    plot_summary,
    plot_bar_importance,
    explain_patient,
)

# ── Noms de features identiques au dataset UCI ────────────────────────────────
FEATURE_NAMES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time",
]

# ── Fixture : modèle entraîné sur données synthétiques ───────────────────────
@pytest.fixture(scope="module")
def artifacts():
    X_np, y = make_classification(
        n_samples=120, n_features=12, n_informative=6, random_state=42
    )
    X = pd.DataFrame(X_np, columns=FEATURE_NAMES)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


# ── Tests : get_shap_explainer ────────────────────────────────────────────────

def test_explainer_is_tree_for_random_forest(artifacts):
    """RandomForest → doit produire un TreeExplainer."""
    import shap
    model, X, _ = artifacts
    explainer = get_shap_explainer(model, X)
    assert isinstance(explainer, shap.TreeExplainer)


# ── Tests : compute_shap_values ───────────────────────────────────────────────

def test_shap_values_shape(artifacts):
    """SHAP values doit avoir la forme (n_samples, n_features)."""
    model, X, _ = artifacts
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)
    assert shap_values.shape == (len(X), X.shape[1])


def test_shap_values_not_all_zero(artifacts):
    """Les SHAP values ne doivent pas être toutes nulles."""
    model, X, _ = artifacts
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)
    assert np.abs(shap_values).sum() > 0


# ── Tests : get_top_features ──────────────────────────────────────────────────

def test_top_features_count_and_order(artifacts):
    """get_top_features doit retourner top_n features triées par importance desc."""
    model, X, _ = artifacts
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)

    top5 = get_top_features(shap_values, FEATURE_NAMES, top_n=5)
    assert len(top5) == 5

    scores = [s for _, s in top5]
    assert scores == sorted(scores, reverse=True), "Pas triées par ordre décroissant"

    for name, _ in top5:
        assert name in FEATURE_NAMES


# ── Tests : plots sauvegardés ────────────────────────────────────────────────

def test_plot_summary_creates_file(tmp_path, artifacts):
    """plot_summary doit créer un fichier PNG non vide."""
    model, X, _ = artifacts
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)

    out = str(tmp_path / "summary.png")
    plot_summary(shap_values, X, FEATURE_NAMES, save_path=out)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_bar_importance_creates_file(tmp_path, artifacts):
    """plot_bar_importance doit créer un fichier PNG non vide."""
    model, X, _ = artifacts
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)

    out = str(tmp_path / "bar.png")
    plot_bar_importance(shap_values, FEATURE_NAMES, save_path=out)
    assert os.path.exists(out) and os.path.getsize(out) > 0


# ── Test : explain_patient (pipeline complet) ─────────────────────────────────

def test_explain_patient_output(tmp_path, artifacts):
    """
    explain_patient doit retourner probability, top_features et waterfall_path,
    et le fichier waterfall doit exister.
    """
    model, X, _ = artifacts
    explainer   = get_shap_explainer(model, X)
    patient     = X.iloc[[0]]   # un seul patient (DataFrame d'une ligne)

    result = explain_patient(
        model, explainer, patient, FEATURE_NAMES,
        save_dir=str(tmp_path)
    )

    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert "top_features" in result
    assert len(result["top_features"]) == 5
    assert "waterfall_path" in result
    assert os.path.exists(result["waterfall_path"])


# ── Test : compatibilité avec save_model / load_model de train_model.py ───────

def test_model_pickle_roundtrip(tmp_path, artifacts):
    """
    Le modèle sauvegardé avec pickle (comme dans save_model) doit être
    rechargeable et produire les mêmes prédictions.
    """
    model, X, _ = artifacts
    path = str(tmp_path / "best_model.pkl")

    with open(path, "wb") as f:
        pickle.dump(model, f)

    with open(path, "rb") as f:
        loaded = pickle.load(f)

    preds_original = model.predict(X)
    preds_loaded   = loaded.predict(X)
    assert np.array_equal(preds_original, preds_loaded), \
        "Le modèle rechargé ne produit pas les mêmes prédictions"