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