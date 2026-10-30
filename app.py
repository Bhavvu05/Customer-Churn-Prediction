
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the saved model
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "churn_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Upload customer data to predict whether customers "
    "are likely to leave the company."
)

# Upload CSV file
uploaded_file = st.file_uploader(
    "Upload your customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # Remove columns not used as model inputs
    data = data.drop(
        columns=["customer_id", "churn"],
        errors="ignore"
    )

    # Get the original training feature names
    preprocessor = model.named_steps["preprocessor"]
    expected_columns = list(preprocessor.feature_names_in_)

    # Check for missing columns
    missing_columns = [
        col for col in expected_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")

    else:
        # Keep the same features and order as training
        data = data[expected_columns]

        st.subheader("Uploaded Customer Data")
        st.dataframe(data.head())

        if st.button("Predict Churn"):

            predictions = model.predict(data)

            results = data.copy()

            results["Prediction"] = [
                "Churn" if value == 1 else "Not Churn"
                for value in predictions
            ]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data)
                classes = list(model.classes_)
                churn_index = classes.index(1)

                results["Churn Probability (%)"] = (
                    probabilities[:, churn_index] * 100
                ).round(2)

            st.subheader("Prediction Results")
            st.dataframe(results)

            st.download_button(
                label="Download Predictions",
                data=results.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
