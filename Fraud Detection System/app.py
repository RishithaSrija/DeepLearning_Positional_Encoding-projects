import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)
def set_background():

    with open("background.png", "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background:
            linear-gradient(
                rgba(0,0,0,0.65),
                rgba(0,0,0,0.65)
            ),
            url("data:image/png;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
set_background()

st.title("💳 Deep Learning Fraud Detection System")

# Load model and scaler
@st.cache_resource
def load_resources():
    model = load_model(
        "attention_fraud_model.keras",
        compile=False
    )

    scaler = joblib.load("scaler.pkl")

    return model, scaler

model, scaler = load_resources()

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    expected_features = 30

    if df.shape[1] < expected_features:
        st.error(
            f"Expected at least {expected_features} columns."
        )
        st.stop()

    try:

        data = df.copy()

        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        if "Amount" in data.columns:
            data["Amount"] = scaler.transform(
                data[["Amount"]]
            )

        sequence_length = 5

        X_seq = []

        for i in range(len(data) - sequence_length):

            X_seq.append(
                data.iloc[
                    i:i+sequence_length
                ].values
            )

        X_seq = np.array(X_seq)

        st.write("Generated Sequences:", X_seq.shape)

        predictions = model.predict(X_seq)

        fraud_probability = predictions.flatten()

        result_df = pd.DataFrame({
            "Transaction_Index":
            np.arange(sequence_length, len(data)),

            "Fraud_Probability":
            fraud_probability
        })

        result_df["Risk_Level"] = np.where(
            result_df["Fraud_Probability"] > 0.5,
            "High Risk",
            "Low Risk"
        )

        st.subheader("Fraud Prediction Results")

        st.dataframe(result_df)

        st.subheader("High Risk Transactions")

        high_risk = result_df[
            result_df["Risk_Level"] == "High Risk"
        ]

        st.dataframe(high_risk)

        st.metric(
            "High Risk Count",
            len(high_risk)
        )

        st.subheader("Fraud Probability Distribution")

        st.bar_chart(
            result_df["Fraud_Probability"]
        )

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(str(e))