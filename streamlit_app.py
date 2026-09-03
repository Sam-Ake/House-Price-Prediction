import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.pkl")
    preprocessor = joblib.load("house_price_preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🏠 House Price Prediction")
st.write(
    "Enter the details of a property below to estimate its expected "
    "sale price using our machine learning model."
)

st.divider()


# ============================================================
# PROPERTY INFORMATION
# ============================================================

st.header("🏡 Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1.0,
        max_value=20.0,
        value=3.0,
        step=1.0
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.25
    )

    sqft_living = st.number_input(
        "Living Area (sqft)",
        min_value=200,
        max_value=20000,
        value=1800,
        step=50
    )

    sqft_lot = st.number_input(
        "Lot Size (sqft)",
        min_value=500,
        max_value=1000000,
        value=5000,
        step=500
    )


with col2:
    floors = st.number_input(
        "Floors",
        min_value=1.0,
        max_value=4.0,
        value=1.0,
        step=0.5
    )

    waterfront = st.selectbox(
        "Waterfront",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    view = st.number_input(
        "View Rating",
        min_value=0,
        max_value=4,
        value=0,
        step=1
    )

    condition = st.number_input(
        "Condition Rating",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )


with col3:
    sqft_above = st.number_input(
        "Above Ground Area (sqft)",
        min_value=200,
        max_value=20000,
        value=1500,
        step=50
    )

    sqft_basement = st.number_input(
        "Basement Area (sqft)",
        min_value=0,
        max_value=10000,
        value=300,
        step=50
    )

    yr_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=1995,
        step=1
    )

    yr_renovated = st.number_input(
        "Year Renovated",
        min_value=0,
        max_value=2026,
        value=0,
        step=1
    )


# ============================================================
# LOCATION INFORMATION
# ============================================================

st.divider()

st.header("📍 Location Information")

col1, col2, col3 = st.columns(3)

with col1:
    street = st.text_input(
        "Street",
        value="123 Example Street"
    )

with col2:
    city = st.text_input(
        "City",
        value="Seattle"
    )

with col3:
    statezip = st.text_input(
        "State/ZIP",
        value="WA 98101"
    )

country = st.text_input(
    "Country",
    value="USA"
)


# ============================================================
# SALE INFORMATION
# ============================================================

st.divider()

st.header("📅 Sale Information")

col1, col2 = st.columns(2)

with col1:
    sale_year = st.number_input(
        "Sale Year",
        min_value=1900,
        max_value=2026,
        value=2014,
        step=1
    )

with col2:
    sale_month = st.selectbox(
        "Sale Month",
        options=list(range(1, 13)),
        format_func=lambda x: pd.Timestamp(
            year=2000,
            month=x,
            day=1
        ).strftime("%B")
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

house_age = sale_year - yr_built

was_renovated = 1 if yr_renovated > 0 else 0


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.header("💰 Price Prediction")

if st.button("Predict House Price", type="primary"):

    input_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "waterfront": [waterfront],
        "view": [view],
        "condition": [condition],
        "sqft_above": [sqft_above],
        "sqft_basement": [sqft_basement],
        "yr_built": [yr_built],
        "yr_renovated": [yr_renovated],
        "was_renovated": [was_renovated],
        "sale_year": [sale_year],
        "sale_month": [sale_month],
        "house_age": [house_age],
        "street": [street],
        "city": [city],
        "statezip": [statezip],
        "country": [country]
    })

    try:

        # Transform input using the saved preprocessor
        transformed_data = preprocessor.transform(input_data)

        # Generate prediction
        prediction = model.predict(transformed_data)[0]

        st.success("Prediction generated successfully!")

        st.metric(
            label="Estimated House Sale Price",
            value=f"${prediction:,.2f}"
        )

        st.info(
            f"Estimated house age at the time of sale: "
            f"{house_age} years"
        )

    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header("📊 Feature Importance")

st.write(
    "This chart shows which features contributed most strongly "
    "to the Gradient Boosting model's predictions."
)

try:

    if hasattr(model, "feature_importances_"):

        feature_names = preprocessor.get_feature_names_out()

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1]
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top 15 Feature Importances")

        st.pyplot(fig)

    else:

        st.info(
            "Feature importance is not available for this model."
        )

except Exception as e:

    st.warning(
        "Feature importance could not be displayed."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "House Price Prediction | Machine Learning Project"
)