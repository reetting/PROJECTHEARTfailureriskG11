# 🫀 CardioCare AI — Prédiction du Risque d'Insuffisance Cardiaque

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Application de support à la décision médicale basée sur le Machine Learning.  
> Développée dans le cadre de la Coding Week — École Centrale Casablanca, Promotion 2028.

---

## 📋 Description

CardioCare AI est une application web qui prédit le risque de décès par insuffisance cardiaque à partir de données cliniques d'un patient. Elle utilise un modèle LightGBM entraîné sur le dataset UCI Heart Failure Clinical Records, et fournit des explications via SHAP.

---

## 👥 Équipe — Groupe 11

| Membre | Rôle |
|--------|------|
| Laouine Youssef | EDA & Preprocessing |
| maleu ivan emmanuel rabé(ps : reetting) | Entraînement des modèles |
| mostafa amine zakanie | Évaluation & SHAP |
| Aboufouzia mohamed yahya  | Interface Streamlit |
| Latifi Chady | DevOps, Tests & Documentation |

---

## 📊 Dataset

- **Source** : [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Taille** : 299 patients, 13 variables cliniques
- **Distribution** : 68% survie / 32% décès
- **Dataset équilibré** : 406 patients après SMOTE (50% / 50%)

---

## 🧠 Modèles entraînés

| Modèle | Recall | Precision | ROC-AUC | F1-Score | Accuracy |
|--------|--------|-----------|---------|----------|----------|
| Random Forest | 0.632 | 0.857 | 0.907 | 0.727 | 0.850 |
| XGBoost | 0.579 | 0.846 | 0.824 | 0.688 | 0.833 |
| **LightGBM ✅** | **0.684** | **0.867** | 0.865 | **0.765** | **0.867** |

---

## 🏆 Pourquoi LightGBM ?

Dans un contexte médical, la métrique la plus importante est le **Recall** — il mesure la capacité du modèle à détecter les vrais cas de décès parmi tous les patients à risque.

Rater un patient en danger est bien plus grave qu'une fausse alarme. C'est pourquoi on a priorisé :

1. **Recall** → LightGBM : 0.684 ✅ (le plus élevé)
2. **Precision** → LightGBM : 0.867 ✅ (le plus élevé)
3. **ROC-AUC** → Random Forest : 0.907 (mais Recall trop faible)

LightGBM offre le meilleur équilibre entre détecter les vrais dangers et limiter les fausses alertes.

---

## ⚖️ Comparaison Base vs Données Équilibrées (SMOTE)

L'application du **rééchantillonnage SMOTE** améliore significativement les performances du modèle sur toutes les métriques clés :

| Métrique  | Base   | Équilibré | Différence |
|-----------|--------|-----------|------------|
| Recall    | 0.6842 | 0.9512    | **+0.2670** |
| Precision | 0.8667 | 0.8864    | +0.0197    |
| ROC-AUC   | 0.8652 | 0.9780    | **+0.1128** |
| F1-Score  | 0.7647 | 0.9176    | **+0.1529** |
| Accuracy  | 0.8667 | 0.9146    | +0.0480    |

> 💡 Le modèle entraîné sur les données équilibrées détecte **95% des patients en danger** contre 68% pour le modèle de base — soit un gain de **+27 points de Recall**.

![Comparaison Base vs SMOTE]│file:///C:/Users/LENOVO/Downloads/smote_comparison_chart.html

---

##  Structure du projet

```
HEARTTfailureriskG5/
├── .github/workflows/      # CI/CD GitHub Actions
│
├── data/
│   ├── heart_failure_clinical_records_dataset.csv   # Dataset original
│   └── heart_failure_balanced_dataset.csv           # Dataset équilibré (SMOTE)
├── models/
│   └── best_model.pkl      # Modèle LightGBM sauvegardé
├── notebooks/
│   └── eda.ipynb           # Analyse exploratoire
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Chargement, outliers, mémoire
│   ├── train_model.py      # Entraînement & évaluation
│   ├── evaluate_model.py   # Métriques détaillées
│   └── SHAP.py             # Explicabilité SHAP
├── app/
│   └── app.py              # Interface Streamlit
├── tests/                  # Tests unitaires
├── requirements.txt
├── 
├── 
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le repo
git clone https://github.com/reetting/HEARTTfailureriskG5.git
cd HEARTTfailureriskG5

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
pip install -e .
```

---

##  Utilisation

### Entraîner le modèle
```bash
python -m src.train_model
```

### Lancer l'application
```bash
streamlit run app/app.py
```

### Lancer les tests
```bash
pytest tests/
```

---

##  Déploiement

L'application est déployée sur Streamlit Cloud :  
👉 **[hearttfailureriskg5.streamlit.app](https://hearttfailureriskg5.streamlit.app)**
 si vous rencontrez une erreure essayer ce lien : https://hearttfailureriskg5-jzjtrfnukh7yce7lsvxsyk.streamlit.app/
 si il y a toujours un probleme écrivez a @reetting pour avoir l'accés direct .
---

##  Explicabilité — SHAP

Le module SHAP permet de comprendre les décisions du modèle :

- **Global** : quelles variables influencent le plus les prédictions
- **Individuel** : pourquoi le modèle a prédit ce résultat pour CE patient

Les variables les plus importantes identifiées :
1. `time` — Période de suivi
2. `serum_creatinine` — Créatinine sérique
3. `ejection_fraction` — Fraction d'éjection

---

##  Avertissement

> Cette application est développée à des fins **éducatives et de recherche uniquement**.  
> Elle ne se substitue en aucun cas à un diagnostic ou un avis médical professionnel.

---

##  Licence

— École Centrale Casablanca, Promotion 2028

