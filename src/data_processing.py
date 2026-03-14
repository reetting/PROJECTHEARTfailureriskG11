import pandas as pd
import numpy as np

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise l'usage mémoire du DataFrame en ajustant les types de données.
    Exemple: float64 vers float32, int64 vers int32.
    """
    # Copie pour ne pas modifier l'original par accident
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Optimisation des entiers
        if str(col_type).startswith('int'):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        # Optimisation des nombres à virgule
        elif str(col_type).startswith('float'):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            
    return df_optimized

def load_data(file_path: str) -> pd.DataFrame:
    """Charge le dataset médical et applique l'optimisation."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    return optimize_memory(df)
def handle_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return df
from sklearn.model_selection import train_test_split

def prepare_data(df):

    # séparation features / target
    X = df.drop("death_event", axis=1)
    y = df["death_event"]

    # division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
