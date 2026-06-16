# AI Hallucination Risk Platform

Plateforme professionnelle (Streamlit) + système ML multi-prédiction, basée sur
le dataset `Charlotin-hallucination_cases.csv` (cas judiciaires impliquant des
hallucinations d'IA générative).

##  Structure du projet attendue

```
projet/
├── app.py
├── train_models.py
├── requirements.txt
├── data/
│   └── Charlotin-hallucination_cases.csv
├── images/
│   └── logo_university.png   (optionnel)
├── risk_model.pkl       (généré par train_models.py)
├── sanction_model.pkl   (généré par train_models.py)
├── penalty_model.pkl    (généré par train_models.py)
└── outcome_model.pkl    (généré par train_models.py)
```

##  Installation

```bash
pip install -r requirements.txt
```

##  Étape 1 : Entraîner les modèles ML

Place ton CSV dans `data/Charlotin-hallucination_cases.csv`, puis lance :

```bash
python train_models.py
```

Ce script va :
1. Nettoyer les données (suppression des lignes vides).
2. Construire 4 cibles à partir des colonnes existantes :
   - **risk_target** : Low / Medium / High (calculé à partir de la sanction,
     de la pénalité et de la décision)
   - **sanction_target** : Yes / No (colonne "Professional Sanction")
   - **penalty_target** : Yes / No (dérivé de "Monetary Penalty")
   - **outcome_target** : catégorie simplifiée de la colonne "Outcome"
     (Case dismissed, Sanction, Filing struck, Warning, Monetary penalty, Other)
3. Entraîner 4 pipelines TF-IDF + classifieur (LogisticRegression / RandomForest).
4. Sauvegarder 4 fichiers `.pkl` à la racine du projet.
5. Afficher dans la console le `classification_report` de chaque modèle
   (précision, rappel, f1-score) pour évaluer la qualité.

##  Étape 2 : Lancer l'application

```bash
streamlit run app.py
```

##  Pages de l'application

- **🏠 Dashboard** : KPIs (nombre de cas, outils IA, tribunaux, sanctions),
  graphiques interactifs (Plotly) sur les outils IA et tribunaux les plus
  fréquents, aperçu du dataset.
- **📈 Analyse** : répartition des sanctions, top décisions, évolution
  temporelle des cas, répartition géographique.
- **🤖 Prédiction** : zone de texte où l'utilisateur décrit un cas
  d'hallucination IA. L'application affiche **4 prédictions simultanées** :
  niveau de risque, sanction professionnelle, pénalité financière probable,
  et catégorie de décision — chacune avec ses probabilités détaillées.
- **ℹ️ À propos** : présentation du projet.

## 🔧 Personnalisation

- Pour ajouter une 5e prédiction (ex : prédire l'outil IA impliqué), duplique
  le bloc dans `train_models.py` avec une nouvelle cible (`tool_target`), puis
  ajoute le modèle correspondant dans `app.py`.
- Tu peux remplacer `RandomForestClassifier` par `LinearSVC`,
  `GradientBoostingClassifier`, etc. selon les performances obtenues.
- Si ton dataset évolue (nouvelles colonnes, nouvelles valeurs), adapte les
  fonctions `simplify_outcome()` et `risk_level()` dans `train_models.py`.

