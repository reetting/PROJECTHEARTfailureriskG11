import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from src.data_processing import load_data, handle_outliers, optimize_memory
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
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

def compare_base_vs_balanced(X_train, X_test, y_train, y_test):
    """
    Compare LightGBM entraîné sur données de base
    vs données équilibrées (SMOTE) — avec split correct
    """
    # ── Modèle de base ─────────────────────────────────────
    trained_base = train_all_models(X_train, y_train)
    lgbm_base    = trained_base["LightGBM"]
    pred_base    = lgbm_base.predict(X_test)
    proba_base   = lgbm_base.predict_proba(X_test)[:, 1]

    # ── Modèle sur données équilibrées — AVEC SPLIT ────────
    df_balanced  = load_data("data/heart_failure_balanced_dataset.csv")
    df_balanced  = handle_outliers(df_balanced)
    df_balanced  = optimize_memory(df_balanced)

    X_bal = df_balanced.drop(columns=["DEATH_EVENT"])
    y_bal = df_balanced["DEATH_EVENT"]

    # Split correct avant entraînement
    X_bal_train, X_bal_test, y_bal_train, y_bal_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    trained_balanced = train_all_models(X_bal_train, y_bal_train)
    lgbm_balanced    = trained_balanced["LightGBM"]
    pred_balanced    = lgbm_balanced.predict(X_bal_test)
    proba_balanced   = lgbm_balanced.predict_proba(X_bal_test)[:, 1]

    # ── Tableau comparatif ─────────────────────────────────
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   COMPARAISON LightGBM — Base vs Equilibre       ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║ {'Metrique':<12} {'Base':>10} {'Equilibre':>12} {'Diff':>8} ║")
    print("╠══════════════════════════════════════════════════╣")

    metrics = {
        "Recall":    (recall_score,    pred_base,  pred_balanced,  y_test, y_bal_test),
        "Precision": (precision_score, pred_base,  pred_balanced,  y_test, y_bal_test),
        "ROC-AUC":   (roc_auc_score,   proba_base, proba_balanced, y_test, y_bal_test),
        "F1-Score":  (f1_score,        pred_base,  pred_balanced,  y_test, y_bal_test),
        "Accuracy":  (accuracy_score,  pred_base,  pred_balanced,  y_test, y_bal_test),
    }

    for name, (fn, base_val, bal_val, y_base, y_bal) in metrics.items():
        score_base = fn(y_base, base_val)
        score_bal  = fn(y_bal,  bal_val)
        diff       = score_bal - score_base
        signe      = "+" if diff >= 0 else ""
        print(f"║ {name:<12} {score_base:>10.4f} {score_bal:>12.4f} {signe}{diff:>7.4f} ║")

    print("╚══════════════════════════════════════════════════╝")
