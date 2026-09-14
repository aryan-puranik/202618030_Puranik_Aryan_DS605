"""
Shared data-cleaning and feature-engineering logic for the Airbnb price
prediction project.

Importing this module from BOTH the training notebook and the Streamlit app
guarantees that new listings passed through the app are transformed in
EXACTLY the same way as the training data, which avoids train/serve skew.
"""

import numpy as np
import pandas as pd

# Columns actually used as model inputs (after cleaning/engineering)
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "days_since_last_review",
]

CATEGORICAL_FEATURES = [
    "neighbourhood_group",
    "room_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"

# Reference date used to compute "days since last review" for rows that are
# missing a review date. Fixed so behaviour is reproducible.
REFERENCE_DATE = pd.Timestamp("2019-07-08")  # day after the last date in the dataset


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw AB_NYC_2019 dataframe.

    Steps:
      1. Drop irrelevant / high-cardinality identifier columns.
      2. Handle missing values (reviews_per_month, last_review).
      3. Engineer `days_since_last_review` and `has_reviews`.
      4. Remove invalid / extreme outlier prices.
    """
    data = df.copy()

    # --- Missing value handling -------------------------------------------------
    # A missing reviews_per_month / last_review simply means the listing has
    # never been reviewed -> 0 reviews per month is the correct fill value.
    data["reviews_per_month"] = data["reviews_per_month"].fillna(0)

    data["last_review"] = pd.to_datetime(data["last_review"], errors="coerce")
    data["days_since_last_review"] = (
        REFERENCE_DATE - data["last_review"]
    ).dt.days
    # No review at all -> use a large sentinel value (never reviewed)
    data["days_since_last_review"] = data["days_since_last_review"].fillna(
        data["days_since_last_review"].max() + 1
    )
    data["has_reviews"] = (data["number_of_reviews"] > 0).astype(int)

    # name / host_name have a handful of missing values and are not used as
    # model features, so we simply drop them along with other identifiers.
    drop_cols = [
        "id",
        "name",
        "host_id",
        "host_name",
        "last_review",
        "neighbourhood",  # high-cardinality (~220 values); neighbourhood_group is used instead
    ]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # --- Outlier handling ---------------------------------------------------
    # price == 0 is not a valid nightly rate (listing error / blocked calendar)
    data = data[data["price"] > 0]
    # Cap extreme high-end prices at the 99th percentile to reduce the
    # influence of luxury outliers on the model without discarding rows.
    upper_cap = data["price"].quantile(0.99)
    data = data[data["price"] <= upper_cap]

    # minimum_nights has some unrealistic values (e.g. 1000+ nights); cap at
    # a sensible ceiling (1 year).
    data = data[data["minimum_nights"] <= 365]

    data = data.reset_index(drop=True)
    return data


def get_feature_target(df: pd.DataFrame):
    """Return (X, y) using only the columns the model was trained on."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
