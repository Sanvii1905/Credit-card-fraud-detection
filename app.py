import os
import joblib
import pandas as pd
import streamlit as st

from preprocessing import FEATURES

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
MODEL_PATH = "fraud_model.joblib"
SCALER_PATH = "scaler.joblib"
DATA_PATH = "creditcard_sample.csv"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.error("Model files are missing. Please run: py train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@st.cache_data
def load_sample_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


sample_df = load_sample_data()

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
st.sidebar.caption("Machine Learning Project")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Fraud Detection", "📁 Upload Data", "📊 Model Performance", "ℹ️ About"],
)

st.sidebar.divider()
st.sidebar.caption("Academic prototype • Streamlit + Machine Learning")


# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
if page == "🏠 Dashboard":
    st.title("💳 Credit Card Fraud Detection")
    st.write("Machine-learning based academic demonstration for classifying credit-card transactions.")

    st.info(
        "🛡️ **Secure Transactions. Safer Tomorrow.**  \n"
        "This application demonstrates how machine learning can identify potentially fraudulent transactions."
    )

    if not sample_df.empty and "Class" in sample_df.columns:
        total = len(sample_df)
        fraud = int((sample_df["Class"] == 1).sum())
        legitimate = int((sample_df["Class"] == 0).sum())
        fraud_rate = fraud / total if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraudulent Transactions", f"{fraud:,}")
        c3.metric("Legitimate Transactions", f"{legitimate:,}")
        c4.metric("Fraud Rate", f"{fraud_rate:.2%}")

        st.subheader("Transaction Class Distribution")
        chart_df = pd.DataFrame(
            {"Transactions": [legitimate, fraud]},
            index=["Legitimate (0)", "Fraud (1)"],
        )
        st.bar_chart(chart_df)

        left, right = st.columns(2)

        with left:
            st.subheader("Dataset Summary")
            st.write(
                "The included sample dataset contains labelled transactions used "
                "for this academic demonstration."
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Class": ["Legitimate (0)", "Fraud (1)"],
                        "Count": [legitimate, fraud],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader("Detection Flow")
            st.markdown(
                """
                **Transaction data**  
                ↓  
                **Preprocessing / Scaling**  
                ↓  
                **Logistic Regression Model**  
                ↓  
                **Fraud Probability**  
                ↓  
                **Fraud / Legitimate classification**
                """
            )
    else:
        st.warning("Sample dataset not found. Check data/creditcard_sample.csv.")


# ------------------------------------------------------------
# Fraud Detection
# ------------------------------------------------------------
elif page == "🔍 Fraud Detection":
    st.title("🔍 Fraud Detection")
    st.write("Enter transaction features or load an example fraud transaction.")

    if "time" not in st.session_state:
        st.session_state["time"] = 36000.0
    if "amount" not in st.session_state:
        st.session_state["amount"] = 100.0
    for i in range(1, 29):
        st.session_state.setdefault(f"V{i}", 0.0)

    if st.button("Load Sample Fraud Transaction"):
        if not sample_df.empty and "Class" in sample_df.columns:
            fraud_rows = sample_df[sample_df["Class"] == 1]
            if not fraud_rows.empty:
                row = fraud_rows.iloc[0]
                st.session_state["time"] = float(row["Time"])
                st.session_state["amount"] = float(row["Amount"])
                for i in range(1, 29):
                    st.session_state[f"V{i}"] = float(row[f"V{i}"])
                st.session_state["sample_loaded"] = True
                st.rerun()
            else:
                st.error("No fraud transaction (Class = 1) was found.")
        else:
            st.error("Sample dataset not found.")

    if st.session_state.get("sample_loaded", False):
        st.success("Sample fraud transaction loaded automatically.")

    with st.form("transaction_form"):
        st.subheader("Transaction Details")

        a, b = st.columns(2)
        with a:
            time = st.number_input(
                "Time (seconds)",
                min_value=0.0,
                key="time",
            )
        with b:
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                key="amount",
            )

        st.subheader("Transaction Features")

        values = []
        cols = st.columns(4)

        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                value = st.number_input(
                    f"V{i}",
                    key=f"V{i}",
                    format="%.5f",
                )
                values.append(value)

        predict = st.form_submit_button("Predict Transaction", type="primary")

    if predict:
        X = pd.DataFrame([[time] + values + [amount]], columns=FEATURES)
        p = float(model.predict_proba(scaler.transform(X))[0][1])
        pred = int(p >= 0.5)

        st.divider()

        if pred:
            st.error("🚨 POTENTIAL FRAUD DETECTED")
        else:
            st.success("✅ TRANSACTION CLASSIFIED AS LEGITIMATE")

        c1, c2 = st.columns(2)
        c1.metric("Fraud Probability", f"{p:.2%}")
        c2.metric("Legitimate Probability", f"{1 - p:.2%}")

        if pred:
            st.warning(
                "The model assigned a fraud probability at or above the 0.50 "
                "classification threshold used by this prototype."
            )
        else:
            st.info(
                "The model assigned a fraud probability below the 0.50 "
                "classification threshold used by this prototype."
            )


# ------------------------------------------------------------
# Upload Data
# ------------------------------------------------------------
elif page == "📁 Upload Data":
    st.title("📁 Upload Transaction Data")
    st.write("Upload a CSV containing the model features to classify multiple transactions.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Uploaded rows: **{len(df):,}**")

        missing = [feature for feature in FEATURES if feature not in df.columns]

        if missing:
            st.error("The uploaded file is missing required columns:")
            st.write(", ".join(missing))
        else:
            X = df[FEATURES].copy()
            probabilities = model.predict_proba(scaler.transform(X))[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            result = df.copy()
            result["Fraud Probability"] = probabilities
            result["Prediction"] = predictions
            result["Prediction Label"] = result["Prediction"].map(
                {0: "Legitimate", 1: "Fraud"}
            )

            fraud_count = int(predictions.sum())
            legitimate_count = len(predictions) - fraud_count

            c1, c2, c3 = st.columns(3)
            c1.metric("Total", f"{len(result):,}")
            c2.metric("Predicted Fraud", f"{fraud_count:,}")
            c3.metric("Predicted Legitimate", f"{legitimate_count:,}")

            st.subheader("Prediction Results")
            st.dataframe(result, use_container_width=True)

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )


# ------------------------------------------------------------
# Model Performance
# ------------------------------------------------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    if sample_df.empty or "Class" not in sample_df.columns:
        st.warning("Sample dataset not available.")
    else:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        X = sample_df[FEATURES]
        y = sample_df["Class"].astype(int)

        predictions = model.predict(scaler.transform(X))

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.2%}")
        c2.metric("Precision", f"{precision:.2%}")
        c3.metric("Recall", f"{recall:.2%}")
        c4.metric("F1 Score", f"{f1:.2%}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, predictions)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.caption(
            "These metrics are calculated on the included sample dataset. "
            "For a production system, performance should be evaluated on a "
            "properly separated validation/test set and monitored over time."
        )


# ------------------------------------------------------------
# About
# ------------------------------------------------------------
else:
    st.title("ℹ️ About the Project")

    st.subheader("Objective")
    st.write(
        "Detect potentially fraudulent credit-card transactions using supervised machine learning."
    )

    st.subheader("Machine Learning Model")
    st.write("Logistic Regression with balanced class weights.")

    st.subheader("Features")
    st.write("Time, V1-V28 and Amount.")

    st.subheader("Technologies")
    st.write("Python • Pandas • Scikit-learn • Joblib • Streamlit")

    st.subheader("How it works")
    st.markdown(
        """
        1. Transaction information is entered or uploaded.
        2. The transaction features are scaled using the project's scaler.
        3. The trained Logistic Regression model calculates a fraud probability.
        4. A 0.50 threshold is used by this prototype for classification.
        5. The application displays the classification and probability.
        """
    )

    st.warning(
        "Academic note: the included dataset is synthetic and intended for demonstration. "
        "A production banking system would require validated real-world data, secure integration, "
        "privacy and security controls, threshold tuning, monitoring and human review."
    )
