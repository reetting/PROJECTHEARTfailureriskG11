## Critical Questions

### 1. Was the dataset balanced?
No. The dataset is imbalanced:
- ~68% survived (class 0) → 203 patients
- ~32% deceased (class 1) → 96 patients

**Solution:** We applied SMOTE (Synthetic Minority Oversampling Technique)
to balance the classes before training.

**Impact:** SMOTE improved recall for the minority class (deceased patients),
which is critical in a medical context where missing a high-risk patient
is more dangerous than a false alarm.

---

### 2. Which ML model performed best?
*(À compléter après le Jour 3)*
| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|---------|
| Random Forest | - | - | - |
| XGBoost | - | - | - |
| LightGBM | - | - | - |
| Logistic Regression | - | - | - |

**Best model:** XGBoost *(à confirmer)*

---

### 3. Which medical features most influenced predictions?
*(À compléter après SHAP — Jour 3)*
Based on SHAP analysis, the top features were:
1. `time` — follow-up period
2. `serum_creatinine` — kidney function indicator
3. `ejection_fraction` — heart pumping efficiency

---

### 4. Prompt Engineering
**Task selected:** Data preprocessing pipeline

**Prompt used:**
> "Write a Python function called optimize_memory(df) that converts
> float64 columns to float32 and int64 columns to int32 in a pandas
> DataFrame. Show memory usage before and after."

**Result:** The function correctly reduced memory usage by ~50%.

**What worked well:** Being specific about input/output types.

**What could be improved:** Adding handling for categorical columns.

###Choix du modèle final : LightGBM (francais)
Comprendre les métriques
Avant de choisir notre modèle, on a d’abord pris le temps de comprendre ce que chaque métrique mesure vraiment. Voici ce qu’on a retenu :
ROC-AUC
C’est la capacité de notre modèle à distinguer deux cas : un patient qui va survivre et un patient qui va décéder. Plus ce score est proche de 1, plus le modèle est bon pour séparer les deux. Si le score est à 0.5, c’est comme si le modèle répondait au hasard.
Accuracy
C’est le pourcentage de fois où le modèle donne la bonne réponse sur l’ensemble des patients. Par exemple si le modèle a juste pour 85 patients sur 100, l’accuracy est de 85%. C’est une bonne métrique en général, mais elle peut être trompeuse quand les données sont déséquilibrées — dans notre cas 68% de survivants et 32% de décès. Le modèle pourrait dire “survie” tout le temps et avoir quand même 68% d’accuracy sans être utile du tout.
Precision
La precision répond à cette question : parmi tous les patients que le modèle a dit “à risque de décès”, combien étaient vraiment à risque ? C’est utile pour éviter les fausses alertes — on ne veut pas stresser un patient qui va bien.
Recall
Le recall répond à une autre question : parmi tous les patients qui allaient vraiment décéder, combien le modèle a réussi à les identifier ? C’est la métrique la plus importante dans un contexte médical. Si le recall est faible, ça veut dire que le modèle rate des patients en danger — et ça, dans un hôpital.
F1-Score
Le F1-score c’est un équilibre entre la precision et le recall. Il est particulièrement utile quand les données sont déséquilibrées car il évite qu’un modèle soit noté “bien” juste parce qu’il répond souvent la même chose.
Cross-Validation Score (CV-Score)
C’est une métrique qu’on a ajoutée en plus de ce qui était demandé. Elle nous permet de vérifier que le modèle ne fait pas que “apprendre par cœur” les données d’entraînement. Concrètement on divise les données en 5 parties, on teste le modèle 5 fois sur des portions différentes, et on fait la moyenne des scores. Si le modèle est bon à chaque fois, on peut lui faire confiance.

Résultats obtenus
Voici les résultats après entraînement des 3 modèles :



|Métrique |Random Forest|XGBoost|LightGBM |
|---------|-------------|-------|---------|
|ROC-AUC  |0.907        |0.824  |0.865    |
|Accuracy |0.850        |0.833  |0.867    |
|Precision|0.857        |0.846  |0.867    |
|Recall   |0.632        |0.579  |**0.684**|
|F1-Score |0.727        |0.688  |0.765    |
|CV-Score |0.784        |0.817  |0.760    |

Pourquoi on a choisi LightGBM
Dans un contexte médical, la question la plus importante n’est pas “est-ce que le modèle a souvent raison ?” mais plutôt “est-ce que le modèle rate des patients en danger ?”.
Rater un patient qui allait décéder est bien plus grave que dire à un patient en bonne santé qu’il est à risque. C’est pour ça qu’on a accordé le plus d’importance au Recall.
LightGBM obtient le meilleur Recall avec 0.684 — c’est le modèle qui détecte le plus de vrais cas de décès parmi tous les patients. Il a aussi la meilleure Accuracy et Precision avec 0.867 chacune, ce qui montre qu’il est performant de manière globale.
Même si Random Forest a un meilleur ROC-AUC (0.907), son Recall est nettement plus faible (0.632) — ce qui veut dire qu’il rate plus de patients en danger.
LightGBM est donc le meilleur équilibre entre détecter les vrais dangers et limiter les fausses alertes.