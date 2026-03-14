import sys
from sklearn.model_selection import GridSearchCV #teste toutes les combinaisons possibles pour garder la meilleure performance 
import os
import pickle
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
def ensemble_averaging(trained_models, X_test, y_test):
    """
    Combine les 3 modeles par moyenne des probabilites.
    Compare le resultat avec LightGBM seul.
    """
    X_test = X_test.copy()
    X_test.columns = X_test.columns.str.upper()
    proba_rf   = trained_models["RandomForest"].predict_proba(X_test)[:, 1]
    proba_xgb  = trained_models["XGBoost"].predict_proba(X_test)[:, 1]
    proba_lgbm = trained_models["LightGBM"].predict_proba(X_test)[:, 1]

    proba_ensemble  = (proba_rf + proba_xgb + proba_lgbm) / 3
    y_pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    lgbm_pred       = trained_models["LightGBM"].predict(X_test)

    print("\n=== ENSEMBLE AVERAGING vs LightGBM seul ===")
    print(f"{'Metrique':<12} {'LightGBM':>10} {'Ensemble':>10} {'Diff':>8}")
    print("-" * 45)

    metrics = {
        "Recall":    (recall_score,    lgbm_pred,  y_pred_ensemble),
        "Precision": (precision_score, lgbm_pred,  y_pred_ensemble),
        "ROC-AUC":   (roc_auc_score,   proba_lgbm, proba_ensemble),
        "F1-Score":  (f1_score,        lgbm_pred,  y_pred_ensemble),
        "Accuracy":  (accuracy_score,  lgbm_pred,  y_pred_ensemble),
    }

    for name, (fn, lgbm_val, ens_val) in metrics.items():
        s_lgbm = fn(y_test, lgbm_val)
        s_ens  = fn(y_test, ens_val)
        diff   = s_ens - s_lgbm
        signe  = "+" if diff >= 0 else ""
        print(f"{name:<12} {s_lgbm:>10.4f} {s_ens:>10.4f} {signe}{diff:>7.4f}")

    return proba_ensemble, y_pred_ensemble

def optimize_lightgbm(X_train, y_train):

    """
    Cette fonction applique GridSearchCV pour trouver
    les meilleurs hyperparamètres du modèle LightGBM.

    Objectif :
    Maximiser le Recall, car dans un contexte médical
    il est important de détecter le maximum de patients à risque.
    """


    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'min_child_samples': [10, 20, 30]
    }

    # Création du modèle LightGBM de base
    
    model = LGBMClassifier(

        class_weight='balanced',

        random_state=42,

        # supprimer les logs inutiles
        verbose=-1
    )

    # Configuration de GridSearchCV
 
    grid_search = GridSearchCV(

        estimator=model,

        # grille de paramètres définie plus haut
        param_grid=param_grid,

        # métrique utilisée pour choisir le meilleur modèle
        scoring='recall',

        # validation croisée 5-fold
        cv=5,

        # utiliser tous les CPU disponibles
        n_jobs=-1,

        # afficher la progression dans le terminal
        verbose=1
    )

    # Entrainement du GridSearch


    # GridSearch va entraîner plusieurs modèles
    # avec différentes combinaisons d'hyperparamètres
    grid_search.fit(X_train, y_train)

    # -----------------------------------------------------
    # Affichage des meilleurs paramètres trouvés
    # -----------------------------------------------------

    print("\nMeilleurs hyperparamètres trouvés :")

    for param, value in grid_search.best_params_.items():
        print(f"{param} : {value}")

    print(f"\nMeilleur Recall (Cross Validation) : {grid_search.best_score_:.4f}")

    # -----------------------------------------------------
    # Retourner le meilleur modèle trouvé
    # -----------------------------------------------------

    return grid_search.best_estimator_
