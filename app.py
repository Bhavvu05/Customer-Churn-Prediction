import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "model" / "churn_pipeline.joblib"

DATASET_PATH = (
    BASE_DIR / "data" / "customer_churn_cleaned.csv"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Load original synthetic training dataset for feature guidance
@st.cache_data
def load_training_data():
    return pd.read_csv(DATASET_PATH)

training_data = load_training_data()

# Get required model input columns
preprocessor = model.named_steps["preprocessor"]

expected_columns = list(
    preprocessor.feature_names_in_
)

# Feature descriptions
feature_meanings = {
    "age": "Age of the customer",
    "gender": "Gender of the customer",
    "geography": "Customer's geographical location",
    "account_type": "Type of customer account",
    "education": "Customer's education level",
    "employment": "Customer's employment category",
    "marital_status": "Customer's marital status",
    "income": "Customer's income",
    "credit_score": "Customer's credit score",
    "tenure_years": "Number of years the customer has been with the company",
    "balance": "Customer's account balance",
    "num_products": "Number of products held by the customer",
    "has_credit_card": "Whether the customer has a credit card",
    "is_active": "Whether the customer is an active customer",
    "estimated_salary": "Estimated customer salary",
    "monthly_spend": "Customer's monthly spending",
    "transactions_last_3m": "Number of transactions in the last 3 months",
    "support_tickets": "Number of customer support tickets",
    "late_payments": "Number of late payments",
    "web_logins": "Number of web logins",
    "satisfaction_score": "Customer satisfaction score",
    "days_since_last_login": "Days since the customer's last login"
}

# Feature units
feature_units = {
    "age": "Years",
    "gender": "Category",
    "geography": "Category",
    "account_type": "Category",
    "education": "Category",
    "employment": "Category",
    "marital_status": "Category",
    "income": "Monetary units (currency unspecified)",
    "credit_score": "Points",
    "tenure_years": "Years",
    "balance": "Monetary units (currency unspecified)",
    "num_products": "Count",
    "has_credit_card": "0 = No, 1 = Yes",
    "is_active": "0 = No, 1 = Yes",
    "estimated_salary": "Monetary units (currency unspecified)",
    "monthly_spend": "Monetary units (currency unspecified)",
    "transactions_last_3m": "Count",
    "support_tickets": "Count",
    "late_payments": "Count",
    "web_logins": "Count",
    "satisfaction_score": "Score",
    "days_since_last_login": "Days"
}

# Build feature guide using the training dataset
feature_guide = []

for column in expected_columns:

    series = training_data[column]

    if pd.api.types.is_numeric_dtype(series):

        value_range = (
            f"{series.min():g} to {series.max():g}"
        )

    else:

        categories = sorted(
            series.dropna().astype(str).unique().tolist()
        )

        value_range = ", ".join(categories)

    feature_guide.append({
        "Column Name": column,
        "Meaning": feature_meanings.get(
            column, "Customer feature"
        ),
        "Unit": feature_units.get(
            column, "As defined in the dataset"
        ),
        "Observed Range / Allowed Values": value_range
    })

# Application title
st.title("📊 Customer Churn Prediction")

st.write(
    "Upload customer information to predict whether "
    "customers are likely to churn."
)

# CSV template download
st.subheader("1. Download CSV Template")

st.write(
    "Download the template, enter your customer data "
    "under the correct columns, and upload it below."
)

template = pd.DataFrame(columns=expected_columns)

st.download_button(
    label="📥 Download Customer CSV Template",
    data=template.to_csv(index=False),
    file_name="customer_churn_template.csv",
    mime="text/csv"
)

# Feature guide
st.subheader("2. Required Feature Guide")

st.write(
    "Use the meanings, units, and observed training-data "
    "ranges below when entering your values."
)

st.dataframe(
    pd.DataFrame(feature_guide),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Numeric ranges are observed in the synthetic training "
    "dataset, not guaranteed real-world limits. "
    "Categorical values should match the listed categories. "
    "Monetary currency is not specified by this dataset."
)

# CSV upload
st.subheader("3. Upload Customer Data")

uploaded_file = st.file_uploader(
    "Upload your completed customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        # Remove columns not used as model inputs
        data = data.drop(
            columns=["customer_id", "churn"],
            errors="ignore"
        )

        # Check for missing columns
        missing_columns = [
            col for col in expected_columns
            if col not in data.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

        elif data.empty:

            st.error("Your uploaded CSV contains no customer records.")

        else:

            # Keep required features in training order
            data = data[expected_columns]

            validation_rules = {
                "age": (18, 80),
                "credit_score": (300, 850),
                "income": (0, None),
                "satisfaction_score": (1, 10)
            }

            errors = []

            for column, (minimum, maximum) in validation_rules.items():

                values = pd.to_numeric(data[column], errors="coerce")

                 # Check non-numeric values
                invalid_format = data[column].notna() & values.isna()

                if invalid_format.any():
                    errors.append(f"{column}: Please enter numeric values.")

                # Check allowed range
                invalid_range = values.notna() & (values < minimum)

                if maximum is not None:
                    invalid_range = invalid_range | (
                        values.notna() & (values > maximum)
                    )

                if invalid_range.any():
                    errors.append(
                        f"{column}: Allowed values are "
                        f"{minimum} to {maximum if maximum is not None else 'above'}."
                    )

                # Convert valid numeric strings into numbers
                data[column] = values

            if errors:
                for error in errors:
                    st.error(error)

                st.stop()


            # -------------------------------
            # 2. CATEGORICAL VALIDATION
            # -------------------------------

            categorical_columns = [
                "gender",
                "geography",
                "account_type",
                "education",
                "employment",
                "marital_status"
            ]

            categorical_errors = False

            for column in categorical_columns:

                # Get allowed categories from cleaned training data
                valid_categories = (
                    training_data[column].dropna().astype(str).unique()
                )

                # Create case-insensitive lookup
                category_lookup = {
                    category.strip().casefold(): category
                    for category in valid_categories
                }

                # Normalize capitalization and extra spaces
                normalized_values = data[column].apply(
                    lambda value: (
                        value if pd.isna(value)
                        else category_lookup.get(
                            str(value).strip().casefold(), value
                        )
                    )
                )

                # Check unsupported categories
                invalid_values = (
                    data[column].notna()
                    & ~data[column].astype(str).str.strip().str.casefold().isin(
                        category_lookup
                    )
                )

                if invalid_values.any():
                    categorical_errors = True

                    invalid_categories = data.loc[
                        invalid_values, column
                    ].unique()

                    st.error(
                        f"{column}: Invalid categories "
                        f"{list(invalid_categories)}. "
                        f"Allowed values: {list(valid_categories)}"
                    )

                data[column] = normalized_values

            if categorical_errors:
                st.stop()

            st.success("CSV uploaded successfully!")

            st.subheader("Uploaded Customer Data")

            st.dataframe(data.head(), use_container_width=True)

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

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.download_button(
                    label="📥 Download Predictions",
                    data=results.to_csv(index=False),
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(f"Unable to process the uploaded CSV: {e}")
