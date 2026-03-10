from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import matplotlib.pyplot as plt
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
data.columns = [col.lower() for col in data.columns]


sns.countplot(x="death_event", data=data)
plt.title("Class Distribution of DEATH_EVENT")
plt.xlabel("DEATH_EVENT")
plt.ylabel("Number of Patients")
plt.show()