# NYC Airbnb Nightly Price Prediction

End-to-end machine learning project (DS605 — Lab 4) that cleans and analyzes the Kaggle
*New York City Airbnb Open Data* dataset, trains and compares **Linear Regression** and
**XGBoost** regression models, and serves the best model through a **Streamlit** app.

## Folder structure

```
airbnb-price-prediction/
├── data/
│   └── AB_NYC_2019.csv                # raw dataset
├── notebooks/
│   └── 01_airbnb_price_prediction.ipynb   # full EDA -> cleaning -> training -> evaluation walkthrough
├── src/
│   ├── preprocessing.py               # shared cleaning / feature engineering (used by notebook + app)
│   └── train.py                       # standalone script: train, compare, save the best pipeline
├── app/
│   └── app.py                         # Streamlit price-prediction UI
├── models/
│   ├── best_model.joblib              # saved preprocessing + model pipeline (auto-generated)
│   └── metrics.json                   # saved train/test metrics for the compared models
├── assets/                            # EDA plots referenced in the notebook / README
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Clone the repo and move into it
git clone <your-repo-url>
cd airbnb-price-prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## How to run

### 1. Run the notebook (Task 1, 2 & 4)

```bash
jupyter notebook notebooks/01_airbnb_price_prediction.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`). The notebook will:
- Load and explore `data/AB_NYC_2019.csv` (missing values, price distribution, outliers, correlations)
- Clean the data and engineer features (see `src/preprocessing.py`)
- Train **Linear Regression** and **XGBoost** (with `GridSearchCV` hyperparameter tuning)
- Evaluate both models with RMSE / MAE / R² on train and test sets and check for over/underfitting
- Save the best-performing pipeline to `models/best_model.joblib` and its metrics to `models/metrics.json`
- Save all EDA/comparison plots to `assets/`

### 2. Or just retrain from the command line

If you don't need the notebook's explanations/plots and just want to (re)generate the model:

```bash
python src/train.py
```

This produces the same `models/best_model.joblib` and `models/metrics.json`.

### 3. Run the Streamlit app (Task 3)

```bash
streamlit run app/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), fill in the listing details
(borough, room type, location, reviews, availability, etc.), and click **Predict nightly price**.

> The app loads `models/best_model.joblib`, so run the notebook or `src/train.py` **at least once**
> before launching the app.

## Results

| Model | Test RMSE ($) | Test MAE ($) | Test R² |
|---|---|---|---|
| Linear Regression | 84.50 | 53.40 | 0.350 |
| XGBoost (tuned) | *populated after you run the notebook / `train.py` with xgboost installed* | | |

The `LinearRegression` numbers above were generated end-to-end in this environment and are what
ships with the repo's `models/best_model.joblib` so the app works out of the box. **XGBoost could
not be installed in the environment used to prepare this handover (no internet access)**, so its
row is left for you to fill in — simply run `python src/train.py` (or the notebook) after
`pip install -r requirements.txt` on a machine with internet access, and the script will:
1. Train XGBoost with hyperparameter tuning,
2. Compare it against Linear Regression on test RMSE,
3. Automatically overwrite `models/best_model.joblib` with whichever model performs better.

In practice, XGBoost is expected to outperform Linear Regression here because nightly price depends
on non-linear interactions between location and room type that a linear model can't capture on its own.

## Key preprocessing decisions

- Missing `reviews_per_month` → filled with `0` (means "never reviewed", not unknown)
- Missing `last_review` → engineered into `days_since_last_review` (+ a `has_reviews` flag)
- Dropped `id`, `host_id`, `name`, `host_name` (identifiers, no predictive value)
- Dropped fine-grained `neighbourhood` (223 unique values) in favor of `neighbourhood_group` (5
  values) + `latitude`/`longitude`, to avoid excessive one-hot dimensionality
- Removed `price == 0` listings and prices above the 99th percentile (outliers)
- Removed `minimum_nights > 365` (data-entry errors)

Full details and the reasoning for each decision are in the notebook (Section 1.7) and in
`src/preprocessing.py`.

## Limitations

- Data is from 2019 and does not reflect current pricing, regulation, or post-pandemic demand.
- No text (listing title/description) or image features are used.
- Fine-grained `neighbourhood` was dropped in favor of coarser location features.
- The model does not account for seasonality, local events, or dynamic host pricing strategies.

## Notes on reproducibility

`src/preprocessing.py` is imported by **both** the notebook/training script and the Streamlit app,
so a listing entered in the app is guaranteed to be transformed identically to how the training data
was transformed — this avoids train/serve skew.
