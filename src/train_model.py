"""
train_model.py — Entraînement des modèles
Entraîne 3 modèles, compare les performances, sauvegarde le meilleur
"""

import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from src.data_processing import load_data, handle_outliers, optimize_memory, prepare_data


MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, eval_metric="logloss", random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, verbose=-1
    ),
}


def train_all_models(X_train, y_train) -> dict:
    trained = {}
    for name, model in MODELS.items():
        print(f"Entraînement : {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def select_best_model(trained_models: dict, X_test, y_test) -> tuple:
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1":      f1_score(y_test, y_pred),
        }
        print(f"  {name} → ROC-AUC: {results[name]['roc_auc']:.3f} | F1: {results[name]['f1']:.3f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n✓ Meilleur modèle : {best_name}")
    return best_name, trained_models[best_name]


def save_model(model, path: str = "models/best_model.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé → {path}")


def load_model(path: str = "models/best_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    df = load_data("data/heart_failure_clinical_records_dataset.csv")
    df = handle_outliers(df)
    df = optimize_memory(df)
    X_train, X_test, y_train, y_test = prepare_data(df)

    trained_models = train_all_models(X_train, y_train)
    best_name, best_model = select_best_model(trained_models, X_test, y_test)
    save_model(best_model)
    print("\nEntraînement terminé ✓")