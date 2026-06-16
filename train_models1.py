"""
train_models.py
================
Entraîne PLUSIEURS modèles de Machine Learning à partir du dataset
"Charlotin-hallucination_cases.csv" afin de prédire, à partir du
texte décrivant un cas (colonnes "Hallucination" + "Details") :

    1. risk_model.pkl     -> Niveau de risque global (Low / Medium / High)
    2. sanction_model.pkl -> Sanction professionnelle (Yes / No)
    3. penalty_model.pkl  -> Pénalité financière probable (Yes / No)
    4. outcome_model.pkl  -> Catégorie de décision de justice (Outcome)

Chaque modèle est un Pipeline scikit-learn (TF-IDF + classifieur)
sauvegardé avec joblib, prêt à être chargé par l'app Streamlit (app.py).

Utilisation :
    python train_models.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


# -------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------
DATA_PATH = "data/Charlotin-hallucination_cases.csv"

df = pd.read_csv(DATA_PATH)

# Supprimer les lignes totalement vides
df.dropna(how="all", inplace=True)


# -------------------------------------------------
# 2. Nettoyage léger
# -------------------------------------------------
if "AI Tool" in df.columns:
    df["AI Tool"] = df["AI Tool"].astype(str).str.strip()
    # Harmoniser "implied"/"Implied" etc.
    df["AI Tool"] = df["AI Tool"].replace({"implied": "Implied"})


# -------------------------------------------------
# 3. Construction du texte d'entrée (feature unique)
# -------------------------------------------------

df["text"] = (
    df["Hallucination"].fillna("")
    + " "
    + df["Details"].fillna("")
).astype(str)



# -------------------------------------------------
# 4. Construction des 4 cibles (targets)
# -------------------------------------------------

# --- Cible 1 : Sanction professionnelle (Yes / No) ---
df["Professional Sanction"] = df["Professional Sanction"].fillna("No")
df["sanction_target"] = df["Professional Sanction"].apply(
    lambda x: "Yes" if str(x).strip().lower() == "yes" else "No"
)

# --- Cible 2 : Pénalité financière (Yes / No) ---

df["Monetary Penalty"] = df["Monetary Penalty"].fillna("")

df["penalty_target"] = df["Monetary Penalty"].apply(
    lambda x: "Yes" if str(x).strip() != "" else "No"
)

# --- Cible 3 : Catégorie de décision (Outcome simplifié) ---

def simplify_outcome(value):

    if pd.isna(value):
        return "Other"

    text = str(value).lower()

    if "dismiss" in text:
        return "Case dismissed"

    if "warning" in text:
        return "Warning"

    if (
        "sanction" in text
        or "struck" in text
        or "stricken" in text
        or "fine" in text
        or "penalty" in text
        or "monetary" in text
    ):
        return "Sanction"

    return "Other"
df["outcome_target"] = df["Outcome"].apply(simplify_outcome)
    
# --- Cible 4 : Niveau de risque global (Low / Medium / High) ---

def risk_level(row):

    if (
        row["sanction_target"] == "Yes"
        and row["penalty_target"] == "Yes"
    ):
        return "High"

    if (
        row["sanction_target"] == "Yes"
        or row["penalty_target"] == "Yes"
        or row["outcome_target"] == "Sanction"
    ):
        return "Medium"

    return "Low"

df["risk_target"] = df.apply(risk_level, axis=1)

print("Répartition des cibles construites :\n")
for col in ["sanction_target", "penalty_target", "outcome_target", "risk_target"]:
    print(f"--- {col} ---")
    print(df[col].value_counts())
    print()


# -------------------------------------------------
# 5. Entraînement des 4 modèles
# -------------------------------------------------
X = df["text"]

# Pour chaque cible : (nom du fichier de sortie, classifieur utilisé)
targets = {
    "risk_target": (
        "risk_model.pkl",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
}

for target_col, (filename, classifier) in targets.items():
    y = df[target_col]

    # stratify uniquement si chaque classe a au moins 2 exemples
    can_stratify = y.value_counts().min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000 , ngram_range=(1,2) )),
        ('clf', classifier)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=========== {target_col} ===========")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    #confusion matrix 
    
    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(y_test, y_pred))
    for col in [
    "risk_target",
    "sanction_target",
    "penalty_target",
    "outcome_target"
    ]:
     print("\n", col)
     print(df[col].value_counts())
     
    
    import os

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{filename}")
    print(f"-> Modèle sauvegardé : {filename}")

print("\n✅ Tous les modèles ont été entraînés et sauvegardés avec succès.")
