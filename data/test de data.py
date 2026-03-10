
from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
heart_failure_clinical_records = fetch_ucirepo(id=519)

# data
X = heart_failure_clinical_records.data.features
y = heart_failure_clinical_records.data.targets

# metadata
print(heart_failure_clinical_records.metadata)

# variable information
print(heart_failure_clinical_records.variables)

# combiner X et y
data = pd.concat([X, y], axis=1)

# vérifier les valeurs manquantes
missing_values = data.isnull().sum()

print("Missing values per column:")
print(missing_values)

print("\nTotal missing values:", missing_values.sum())