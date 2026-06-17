import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Hallucination Risk Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS (look "plateforme pro")
# =========================
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }

    .metric-card {
        background: linear-gradient(135deg, #5c1a1a 0%, #8b2e2e 100%);
        padding: 18px 20px;
        border-radius: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .metric-card h2 { margin: 0; font-size: 28px; }
    .metric-card p { margin: 0; font-size: 13px; opacity: 0.85; }

    .pred-card {
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }
    .pred-high   { background: linear-gradient(135deg, #7a0c0c 0%, #b71c1c 100%); }
    .pred-medium { background: linear-gradient(135deg, #a14e00 0%, #d9822b 100%); }
    .pred-low    { background: linear-gradient(135deg, #2e5e2e 0%, #4f8f4f 100%); }
    .pred-neutral{ background: linear-gradient(135deg, #4a4a4a 0%, #757575 100%); }

    section[data-testid="stSidebar"] {
        background-color: #5c1a1a;
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    .page-header {
        background: linear-gradient(135deg, #5c1a1a 0%, #8b2e2e 100%);
        padding: 22px 28px;
        border-radius: 14px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .page-header h1 { margin: 0; font-size: 26px; }
    .page-header p { margin: 4px 0 0 0; font-size: 13px; opacity: 0.9; }
</style>
 """, unsafe_allow_html=True)


# =========================
# CHARGEMENT DES DONNÉES
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/Charlotin-hallucination_cases.csv")


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


df = load_data()
df.dropna(how="all", inplace=True)

# Modèles multi-prédiction

risk_model = load_model("models/risk_model.pkl")

# =========================
# SIDEBAR
# =========================
#un seul logo pas 2

with st.sidebar:
    if os.path.exists("images/logo.png"):
        st.image("images/logo.png", use_container_width=True)
    st.markdown(
     """ <div style="text-align:center; color:white; font-size:13px;"> FS UCD University-20/Juin/2026 </div>
        </div>
    
      """, unsafe_allow_html=True,
     )
    st.markdown("## ⚖️ AI Hallucination Risk Platform")
    st.markdown("---")
    st.markdown("**Étudiant**")
    st.write("Ikram Khaddoum")
    st.markdown("**Encadrant**")
    st.write("Pr. Aaroud")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📈 Analyse", "🤖 Prédiction", "ℹ️ À propos"],
    )


# =========================
# HEADER PRINCIPAL
# =========================
st.markdown(
"""<div class="page-header">
    <div>
        <h1>⚖️ AI Hallucination Risk Platform</h1>
        <p>Université Chouaib Doukkali — Faculté des Sciences El Jadida | Projet Data Science 2026</p>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# PAGE : DASHBOARD
# =========================
if page == "🏠 Dashboard":
    st.title("📊 Tableau de bord - Hallucinations IA devant les tribunaux")
    st.caption("Vue d'ensemble des cas répertoriés où une IA générative a produit des erreurs/fabrications dans une procédure judiciaire.")

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f'<div class="metric-card"><h2>{len(df)}</h2><p>Cas totaux</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{df["AI Tool"].nunique()}</h2><p>Outils IA identifiés</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{df["Court"].nunique()}</h2><p>Tribunaux</p></div>', unsafe_allow_html=True)
    with c4:
        nb_sanctions = (df["Professional Sanction"].astype(str).str.lower() == "yes").sum()
        st.markdown(f'<div class="metric-card"><h2>{nb_sanctions}</h2><p>Sanctions professionnelles</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Outils IA les plus impliqués")
        tool_counts = df["AI Tool"].value_counts().head(10).reset_index()
        tool_counts.columns = ["AI Tool", "Nombre de cas"]
        fig = px.bar(tool_counts, x="Nombre de cas", y="AI Tool", orientation="h",
                      color="Nombre de cas", color_continuous_scale="Blues")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Répartition par tribunal (top 10)")
        court_counts = df["Court"].value_counts().head(10).reset_index()
        court_counts.columns = ["Court", "Nombre de cas"]
        fig = px.bar(court_counts, x="Court", y="Nombre de cas",
                      color="Nombre de cas", color_continuous_scale="Oranges")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head(10), use_container_width=True)


# =========================
# PAGE : ANALYSE
# =========================
elif page == "📈 Analyse":
    st.title("📈 Analyse approfondie")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sanctions professionnelles")
        sanction_counts = df["Professional Sanction"].fillna("No").value_counts().reset_index()
        sanction_counts.columns = ["Sanction", "Nombre"]
        fig = px.pie(sanction_counts, names="Sanction", values="Nombre",
                      color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 des décisions (Outcome)")
        outcome_counts = df["Outcome"].dropna().value_counts().head(10).reset_index()
        outcome_counts.columns = ["Outcome", "Nombre"]
        fig = px.bar(outcome_counts, x="Nombre", y="Outcome", orientation="h",
                      color="Nombre", color_continuous_scale="Tealgrn")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Évolution des cas dans le temps")
    if "Date" in df.columns:
        df_time = df.copy()
        df_time["Date"] = pd.to_datetime(df_time["Date"], errors="coerce")
        df_time = df_time.dropna(subset=["Date"])
        df_time["Mois"] = df_time["Date"].dt.to_period("M").astype(str)
        timeline = df_time.groupby("Mois").size().reset_index(name="Nombre de cas")
        fig = px.line(timeline, x="Mois", y="Nombre de cas", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Répartition géographique (États)")
    if "State(s)" in df.columns:
        state_counts = df["State(s)"].dropna().value_counts().head(15).reset_index()
        state_counts.columns = ["State(s)", "Nombre de cas"]
        fig = px.bar(state_counts, x="State(s)", y="Nombre de cas",
                      color="Nombre de cas", color_continuous_scale="Purples")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# =========================
# PAGE : PRÉDICTION (multi-prédiction)
# =========================
elif page == "🤖 Prédiction":
    st.title("🤖 Prédiction multi-critères")
    st.write(
        "Décrivez le cas (type d'hallucination constatée par le tribunal, "
        "contexte, citations fabriquées, etc.). Le système prédira "
        "**plusieurs informations en même temps** :"
    )
    st.markdown(""" 
    - 🟥 **Niveau de risque global** (Low / Medium / High)
    - ⚖️ **Probabilité de sanction professionnelle** (Yes / No)
    - 💰 **Probabilité de pénalité financière** (Yes / No)
    - 📄 **Catégorie probable de la décision** (Outcome)"""
    )

    text = st.text_area(
        "Description du cas",
        height=160,
        placeholder="Ex: The attorney submitted a brief citing fabricated case law generated by ChatGPT, "
                     "which the court found to be entirely non-existent..."
    )
    models_missing = risk_model is None
    if models_missing:
        st.warning(
            "⚠️ Le modèle risk_model.pkl est introuvable. Lance d'abord train_models.py."
        )

    if st.button("🔍 Lancer la prédiction", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Merci de saisir une description du cas.")
        elif models_missing:
            st.error("Modèles manquants : impossible de lancer la prédiction.")
        else:
           text_lower = text.lower()

           # Prédiction ML unique
           risk_pred = risk_model.predict([text])[0]
           
           # Résultats dérivés du risque

           if risk_pred == "Low":

                sanction_pred = "No"
                penalty_pred = "No"
                outcome_pred = "Warning"

           elif risk_pred == "Medium":

                sanction_pred = "No"
                penalty_pred = "Possible"
                outcome_pred = "Other"

           else:  # High

                sanction_pred = "Yes"
                penalty_pred = "Yes"
                outcome_pred = "Sanction"

          # Mots-clés sanction
           if any(word in text_lower for word in [
             "disciplinary",
             "misconduct",
             "professional sanction",
             "bar association"
             ]):
              sanction_pred = "Yes"
              risk_pred = "High"
              outcome_pred = "Sanction"

          # Mots-clés pénalité
           if any(word in text_lower for word in [
              "fine",
              "financial penalty",
              "monetary penalty",
              "ordered to pay"
             ]):
               penalty_pred = "Yes"

           

          # Cohérence finale
           if sanction_pred == "Yes":
              risk_pred = "High"
              outcome_pred = "Sanction"
           # =========================
           # RULES MÉTIER (OPTION 2)
           # =========================

            

           
        
           st.markdown("---")
           st.subheader("Résultats de la prédiction")

           r1c1, r1c2 = st.columns(2)
           r2c1, r2c2 = st.columns(2)

           # --- Niveau de risque ---
           risk_class = {"High": "pred-high", "Medium": "pred-medium", "Low": "pred-low"}.get(risk_pred, "pred-neutral")
           risk_icon = {"High": "🚨", "Medium": "⚠️", "Low": "✅"}.get(risk_pred, "ℹ️")
           with r1c1:
               st.markdown(
                   f'<div class="pred-card {risk_class}">{risk_icon} Risque global : {risk_pred}</div>',
                   unsafe_allow_html=True,
               )

           # --- Sanction professionnelle ---
           sanction_class = "pred-high" if sanction_pred == "Yes" else "pred-low"
           with r1c2:
               st.markdown(
                   f'<div class="pred-card {sanction_class}">⚖️ Sanction professionnelle : {sanction_pred}</div>',
                   unsafe_allow_html=True,
               )

           # --- Pénalité financière ---
           penalty_class = "pred-medium" if penalty_pred == "Yes" else "pred-low"
           with r2c1:
               st.markdown(
                   f'<div class="pred-card {penalty_class}">💰 Pénalité financière probable : {penalty_pred}</div>',
                   unsafe_allow_html=True,
               )

           # --- Catégorie de décision ---
           with r2c2:
               st.markdown(
                   f'<div class="pred-card pred-neutral">📄 Décision probable : {outcome_pred}</div>',
                   unsafe_allow_html=True,
               )

           # --- Probabilités (si disponibles) ---
           
           st.markdown("---")
           st.subheader("Interprétation")

           st.info(f"Niveau de risque prédit : {risk_pred}")

           if risk_pred == "Low":
             st.success(
             "Le cas présente peu de signaux de risque. "
             "Aucune sanction ou pénalité n'est probable."
             )

           elif risk_pred == "Medium":
             st.warning(
             "Le cas contient certains éléments pouvant poser problème. "
             "Une vérification supplémentaire est recommandée."
            )

           else:
             st.error(
             "Le cas présente des signaux forts d'hallucination ou de mauvaise conduite. "
             "Une sanction professionnelle ou une décision défavorable est possible."
             )

# =========================
# PAGE : À PROPOS
# =========================
elif page == "ℹ️ À propos":
    st.title("ℹ️ À propos de ce projet")
    st.markdown("""
     ### ⚖️ AI Hallucination Risk Platform

Cette plateforme analyse un jeu de données recensant des **cas judiciaires impliquant des hallucinations générées par des systèmes d’intelligence artificielle** (ChatGPT, Claude, Gemini, Microsoft Copilot, etc.).

Une hallucination correspond à la production d’informations erronées ou fictives présentées comme réelles, telles que des citations fabriquées, des jurisprudences inexistantes, des références juridiques incorrectes ou des faits inventés. Dans le domaine juridique, ces erreurs peuvent avoir des conséquences importantes sur les procédures et les décisions de justice.

### Fonctionnalités

- 📊 **Tableau de bord interactif** présentant les principaux indicateurs et statistiques du dataset.
- 📈 **Analyse exploratoire des données** : répartition des outils d’IA, sanctions professionnelles, décisions judiciaires, évolution temporelle et analyse géographique.
- 🤖 **Module de prédiction du niveau de risque** basé sur un modèle de Machine Learning utilisant le traitement automatique du langage naturel (NLP), la vectorisation TF-IDF et la régression logistique.
- ⚖️ **Évaluation automatique des conséquences potentielles** d’un cas analysé :
  - Niveau de risque global (Low / Medium / High)
  - Sanction professionnelle probable
  - Pénalité financière probable
  - Catégorie probable de décision judiciaire
- 📋 Interface simple et intuitive permettant l’analyse de nouveaux cas à partir d’une description textuelle.

### Technologies Utilisées

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Plotly
- TF-IDF
- Régression Logistique

---

**Projet de Data Science 2025-2026**  
**Réalisé par : Ikram Khaddoum**  
**Encadré par : Pr. Aaroud**  
**Faculté des Sciences – Université Chouaïb Doukkali**
    """ )   


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
   """ <div style="text-align:center; color:#888; font-size:13px;">
    AI Hallucination Risk Platform • Projet académique 2026 • Ikram Khaddoum
    </div>
    
    """, unsafe_allow_html=True,
)

