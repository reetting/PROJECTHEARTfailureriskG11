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
from sklearn.model_selection import train_test_split

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


def train_all_models(X_train, y_train):
    trained = {}
    for name, model in MODELS.items():
        print(f"Entraînement : {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def evaluate_all_models(trained_models: dict, X_test, y_test) -> None:
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(
            model, X_test, y_test, cv=kf, scoring='roc_auc'
        )

        print(f"\n{'='*35}")
        print(f"  Modèle : {name}")
        print(f"{'='*35}")
        print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.3f}")
        print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
        print(f"  Precision : {precision_score(y_test, y_pred):.3f}")
        print(f"  Recall    : {recall_score(y_test, y_pred):.3f}")
        print(f"  F1-Score  : {f1_score(y_test, y_pred):.3f}")
        print(f"  CV-Score  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f} ➕")


def save_model(model, path: str = "models/best_model.pkl") -> None:
    """Sauvegarde le modèle choisi."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModèle sauvegardé → {path}")


def load_model(path: str = "models/best_model.pkl"):
    """Charge le modèle sauvegardé."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Chargement et preprocessing
    df = load_data("data/heart_failure_clinical_records_dataset.csv")
    df = handle_outliers(df)
    df = optimize_memory(df)

    # Séparation features / cible
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']

    # Split 80% entraînement / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entraînement des 3 modèles
    trained_models = train_all_models(X_train, y_train)

    # Affichage des métriques — à toi de choisir le meilleur !
    evaluate_all_models(trained_models, X_test, y_test)

    print("\nEntraînement terminé ✓")