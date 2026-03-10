
from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset 
heart_failure_clinical_records = fetch_ucirepo(id=519) 
  
# data (as pandas dataframes) 
X = heart_failure_clinical_records.data.features 
y = heart_failure_clinical_records.data.targets 
  
# metadata 
print(heart_failure_clinical_records.metadata) 
  
# variable information 
print(heart_failure_clinical_records.variables) 


# combiner X et y
data = pd.concat([X, y], axis=1)
# Calcul des quartiles
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)

# Calcul de l'IQR
IQR = Q3 - Q1

# bornes
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# détection des outliers
outliers = ((data < lower_bound) | (data > upper_bound))

# nombre d'outliers par variable
print("Number of outliers per feature:")
print(outliers.sum())