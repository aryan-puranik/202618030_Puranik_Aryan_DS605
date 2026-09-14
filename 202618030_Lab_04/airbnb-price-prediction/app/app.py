"""
Streamlit application for the Airbnb (NYC) nightly price prediction project.

Run with:
    streamlit run app/app.py
(run from the project root so the relative paths below resolve correctly)
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Airbnb NYC Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


model = load_model()
metrics = load_metrics()

st.title("🏠 NYC Airbnb Nightly Price Predictor")
st.write(
    "Enter the details of a listing below to get an estimated nightly price, "
    "based on a model trained on the Kaggle *New York City Airbnb Open Data* dataset."
)

if model is None:
    st.error(
        "No trained model found at `models/best_model.joblib`. "
        "Run `python src/train.py` or the training notebook first."
    )
    st.stop()

if metrics:
    with st.expander("ℹ️ About the model in use"):
        st.write(f"**Best model:** {metrics['best_model']}")
        best_res = metrics["results"][metrics["best_model"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Test RMSE ($)", f"{best_res['test_rmse']:.2f}")
        c2.metric("Test MAE ($)", f"{best_res['test_mae']:.2f}")
        c3.metric("Test R²", f"{best_res['test_r2']:.3f}")

NEIGHBOURHOOD_GROUPS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]

st.subheader("Listing details")

col1, col2 = st.columns(2)
with col1:
    neighbourhood_group = st.selectbox("Borough (neighbourhood group)", NEIGHBOURHOOD_GROUPS)
    room_type = st.selectbox("Room type", ROOM_TYPES)
    latitude = st.number_input("Latitude", value=40.7128, format="%.5f")
    longitude = st.number_input("Longitude", value=-74.0060, format="%.5f")
    minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=3)

with col2:
    number_of_reviews = st.number_input("Number of reviews", min_value=0, value=10)
    reviews_per_month = st.number_input("Reviews per month", min_value=0.0, value=1.0, step=0.1)
    calculated_host_listings_count = st.number_input(
        "Host's total listings count", min_value=1, value=1
    )
    availability_365 = st.slider("Availability (days/year)", 0, 365, 180)
    has_reviews = 1 if number_of_reviews > 0 else 0
    days_since_last_review = st.number_input(
        "Days since last review (use a large number, e.g. 500+, if never reviewed)",
        min_value=0,
        value=30,
    )

if st.button("Predict nightly price", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "latitude": latitude,
                "longitude": longitude,
                "minimum_nights": minimum_nights,
                "number_of_reviews": number_of_reviews,
                "reviews_per_month": reviews_per_month,
                "calculated_host_listings_count": calculated_host_listings_count,
                "availability_365": availability_365,
                "days_since_last_review": days_since_last_review,
                "neighbourhood_group": neighbourhood_group,
                "room_type": room_type,
            }
        ]
    )
    # keep column order consistent with training
    input_df = input_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated nightly price: **${prediction:,.2f}**")
    st.caption(
        "This is an estimate from a machine learning model trained on historical "
        "2019 NYC Airbnb listings and should be used only as a rough guide."
    )

st.divider()
st.caption(
    "Model trained with a Linear Regression / XGBoost comparison pipeline — "
    "see `notebooks/01_airbnb_price_prediction.ipynb` for the full analysis."
)
