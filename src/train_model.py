
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from src.data_processing import load_data, handle_outliers, optimize_memory
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
