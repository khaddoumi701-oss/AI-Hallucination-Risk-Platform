import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Hallucination Dashboard",
    layout="wide"
)

# =========================
# LOAD MODEL + DATA
# =========================
model = joblib.load("Hallucination_model.pkl")

df = pd.read_csv("../data/Charlotin-hallucination_cases.csv")

# =========================
# SIDEBAR DESIGN
# =========================
with st.sidebar:

    st.image("../images/logo_university.png", width=150)

    st.title("AI Hallucination System")

    st.markdown("### Student")
    st.write("Ikram Khaddoum")

    st.markdown("### Supervisor")
    st.write("Pr. Arroud")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Analysis", "Prediction"]
    )

# =========================
# DASHBOARD PAGE
# =========================
if page == "Dashboard":

    st.title("📊 AI Hallucination Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Cases", len(df))
    col2.metric("AI Tools", df["AI Tool"].nunique())
    col3.metric("Courts", df["Court"].nunique())
    col4.metric("States", df["State(s)"].nunique())

    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    st.subheader("AI Tools Distribution")

    tool_counts = df["AI Tool"].value_counts().head(10)

    fig, ax = plt.subplots()
    ax.barh(tool_counts.index, tool_counts.values)
    ax.invert_yaxis()
    st.pyplot(fig)

    # -------------------------
    st.subheader("Court Distribution")

    court_counts = df["Court"].value_counts().head(10)

    fig, ax = plt.subplots()
    ax.bar(court_counts.index, court_counts.values)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =========================
# ANALYSIS PAGE
# =========================
elif page == "Analysis":

    st.title("📈 Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        if "Severity" in df.columns:

            risk_counts = df["Severity"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(risk_counts.values,
                   labels=risk_counts.index,
                   autopct="%1.1f%%")

            st.pyplot(fig)

    with col2:

        st.subheader("Top Outcomes")

        outcome_counts = df["Outcome"].value_counts().head(10)

        fig, ax = plt.subplots()
        ax.barh(outcome_counts.index, outcome_counts.values)
        st.pyplot(fig)

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":

    st.title("🤖 AI Hallucination Prediction")

    st.markdown("Enter a description of AI hallucination case:")

    text = st.text_area("Input Text")

    if st.button("Predict Risk"):

        prediction = model.predict([text])[0]

        # OPTIONAL probability (if supported)
        try:
            proba = model.predict_proba([text])[0]
            confidence = max(proba)
        except:
            confidence = None

        st.markdown("---")

        if prediction == "High":
            st.error("🚨 HIGH RISK")
        elif prediction == "Medium":
            st.warning("⚠️ MEDIUM RISK")
        else:
            st.success("✅ LOW RISK")

        if confidence:
            st.info(f"Confidence: {confidence:.2f}")

        st.markdown("---")
        st.write("Prediction result based on trained ML model.")

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
"""
**AI Hallucination Risk Prediction System**  
Data Science Project 2026  
Developed by Ikram Khaddoum  
Supervised by Pr. Arroud
"""
)