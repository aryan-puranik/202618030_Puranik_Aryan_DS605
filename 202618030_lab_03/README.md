#                                                    Hotel Booking Cancellation Prediction

## Name : Puranik Aryan Hitesh  
## Student ID: 202618030
## Overview

This project focuses on building and evaluating machine learning classification models to predict whether a hotel booking will be canceled.

The assignment demonstrates a complete machine learning workflow, including data preprocessing, missing-value handling, feature transformation, consistent train-test splitting, model training, performance evaluation, confusion-matrix analysis, and overfitting assessment.

Two numerical preprocessing strategies are compared:

* **Pipeline A:** KNN Imputation + StandardScaler
* **Pipeline B:** KNN Imputation + MinMaxScaler



---

## Objectives

The main objectives of this assignment are to:

1. Prepare the hotel booking dataset for machine learning.
2. Identify and handle missing values appropriately.
3. Remove features that directly reveal the final booking outcome.
4. Detect and remove only clear/extreme numerical outliers.
5. Create a reproducible train-test split.
6. Apply appropriate preprocessing to numerical and categorical features.
7. Compare StandardScaler and MinMaxScaler.
8. Train Logistic Regression and Decision Tree classifiers.
9. Evaluate all four model-pipeline combinations using multiple classification metrics.
10. Analyze confusion matrices for the best Logistic Regression and Decision Tree models.
11. Compare train-test performance to identify possible overfitting.
12. Determine which preprocessing-model combination provides the best overall performance.

---

## Dataset

The project uses the **Hotel Booking Demand Dataset**, containing information about hotel reservations and their associated characteristics.

The target variable is:

```text
is_canceled
```

where:

* `0` → Booking was not canceled
* `1` → Booking was canceled

The dataset contains both numerical and categorical features, making it suitable for demonstrating mixed-type preprocessing using `ColumnTransformer`.

---

## Data Preprocessing

### 1. Removal of Target-Leaking Features

The following columns were removed before model training:

```text
reservation_status
reservation_status_date
```

### 2. Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split(
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

This produces:

* **80% training data**
* **20% testing data**

`stratify=y` preserves approximately the same class distribution in both datasets.

The same split is used for **all four experiments** to ensure a fair comparison.


### 3. Numerical Features

Missing values in numerical columns are handled using:

```python
KNNImputer(n_neighbors=5)
```

KNN imputation estimates missing values based on neighboring observations rather than simply replacing missing values with the mean or median.

Two scaling methods are evaluated.

#### Pipeline A — StandardScaler

```text
Numerical Features
        ↓
KNNImputer(n_neighbors=5)
        ↓
StandardScaler
```

`StandardScaler` standardizes numerical features using their mean and standard deviation.

#### Pipeline B — MinMaxScaler

```text
Numerical Features
        ↓
KNNImputer(n_neighbors=5)
        ↓
MinMaxScaler
```

`MinMaxScaler` transforms numerical features to a common range, typically `[0, 1]`.

---

### 4. Categorical Features

Categorical missing values are handled using:

python
SimpleImputer(strategy="most_frequent")

The categorical features are then transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The `handle_unknown="ignore"` option ensures that previously unseen categories in the test data do not cause errors during transformation.

---

## Preprocessing Architecture

`ColumnTransformer` is used to apply different preprocessing operations to numerical and categorical columns.

Conceptually, the preprocessing workflow is:

```text
                         Input Features
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          Numerical Features          Categorical Features
                 │                           │
           KNN Imputation             Most-Frequent Imputation
                 │                           │
        StandardScaler /              OneHotEncoder
         MinMaxScaler                       │
                 │                           │
                 └─────────────┬─────────────┘
                               │
                       Transformed Features
                               │
                             Model
```

All preprocessing operations are placed inside `Pipeline` and `ColumnTransformer`.

This ensures that preprocessing is **fitted only on the training data**, preventing information from the test set from leaking into the training process.

---

## Machine Learning Models

Two classification algorithms are evaluated.

### Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

Logistic Regression is a linear classification algorithm that estimates the probability of an observation belonging to a particular class.

The same model configuration is used with both preprocessing pipelines.

### Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

Decision Tree is a non-linear, tree-based classification algorithm.

The same model configuration is used with both preprocessing pipelines.

---

## Four Experimental Combinations

The assignment evaluates the following four combinations:

| Experiment | Model               | Numerical Preprocessing     |
| ---------- | ------------------- | --------------------------- |
| 1          | Logistic Regression | KNNImputer + StandardScaler |
| 2          | Logistic Regression | KNNImputer + MinMaxScaler   |
| 3          | Decision Tree       | KNNImputer + StandardScaler |
| 4          | Decision Tree       | KNNImputer + MinMaxScaler   |

The train-test split and model settings remain unchanged across all experiments.

---

## Evaluation Metrics

Each model is evaluated using:

### Training Accuracy

Measures the proportion of correctly classified training observations.

### Testing Accuracy

Measures the proportion of correctly classified observations in the unseen test set.

### Precision

Measures how many observations predicted as canceled were actually canceled.

### Recall

Measures how many of the actual canceled bookings were correctly identified.

### F1-Score

The F1-score combines precision and recall and provides a balanced measure of classification performance.

For this assignment, the **test F1-score** is particularly useful when comparing the overall classification performance of the models.

---

## Confusion Matrix

Confusion matrices are generated for:

1. The best Logistic Regression result
2. The best Decision Tree result

The confusion matrix contains:

```text
                    Predicted
                 Not Canceled   Canceled
Actual
Not Canceled          TN           FP
Canceled              FN           TP
```

Where:

* **TN — True Negative:** Correctly predicted not canceled
* **FP — False Positive:** Predicted canceled but actually not canceled
* **FN — False Negative:** Predicted not canceled but actually canceled
* **TP — True Positive:** Correctly predicted canceled

The confusion matrices provide additional insight into the types of errors made by the models.

---

## Overfitting Analysis

Possible overfitting is assessed using the difference between training and testing accuracy:

```text
Train-Test Difference =
Training Accuracy - Testing Accuracy
```

A small difference indicates that the model's performance is relatively consistent between training and unseen data.

A large difference may indicate overfitting, where the model performs very well on training observations but generalizes poorly to unseen observations.

The Decision Tree is expected to be more susceptible to overfitting because it can learn complex decision boundaries and may fit the training data very closely.


## Key Analysis Questions

The experiments are designed to answer the following questions:

### 1. Which preprocessing-model combination performs best?

The best combination is the model-pipeline pair achieving the strongest overall test performance, particularly based on F1-score.

### 2. Does StandardScaler or MinMaxScaler affect Logistic Regression more?

Logistic Regression is sensitive to feature scaling because the model uses numerical feature coefficients during optimization. Therefore, differences between StandardScaler and MinMaxScaler can affect its performance.

The actual effect is determined from the experimental results.

### 3. Does scaling significantly affect Decision Tree performance?

Decision Trees generally do not depend strongly on feature scale because their decisions are based on feature thresholds and ordering.

Therefore, StandardScaler and MinMaxScaler are generally expected to produce similar Decision Tree performance.

### 4. Does the model show overfitting?

The train-test accuracy difference is examined to determine whether a model performs substantially better on training data than on unseen test data.

---

## Expected Observations

The final analysis should contain approximately 4–5 observations supported by the performance table and confusion matrices.

Examples of observations include:

* The model with the highest test F1-score provides the strongest overall classification performance.
* The difference between StandardScaler and MinMaxScaler can be compared directly for Logistic Regression.
* Decision Tree performance is generally less sensitive to numerical scaling.
* A large train-test accuracy gap may indicate overfitting.
* The confusion matrices reveal whether the models produce more false positives or false negatives when predicting booking cancellations.

These observations should be updated based on the actual experimental results.

---

## Technologies and Libraries

The project was implemented using Python in JupyterLab.

### Libraries

```text
Python
Pandas
NumPy
Scikit-learn
Matplotlib
JupyterLab
```

### Scikit-learn Components

```python
train_test_split
ColumnTransformer
Pipeline
KNNImputer
SimpleImputer
StandardScaler
MinMaxScaler
OneHotEncoder
LogisticRegression
DecisionTreeClassifier
accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
ConfusionMatrixDisplay
```

---

## Reproducibility

The experiments use:

```python
random_state=42
```

for the train-test split and Decision Tree.

The train-test split is performed once and reused across all experiments:

```python
X_train, X_test, y_train, y_test
```

This ensures that differences in model performance are attributable to the preprocessing and model choices rather than differences in the training/testing observations.

---

## Project Workflow

```text
Load Dataset
     ↓
Initial Data Inspection
     ↓
Missing-Value Analysis
     ↓
Remove Target-Leaking Columns
     ↓
Outlier Detection and Treatment
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Identify Numerical & Categorical Features
     ↓
Create Preprocessing Pipelines
     ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Pipeline A             Pipeline B
KNN Imputer            KNN Imputer
StandardScaler         MinMaxScaler
 │                       │
 └──────────┬────────────┘
            ↓
     Model Training
            ↓
 ┌──────────┴───────────┐
 │                      │
 ▼                      ▼
Logistic Regression   Decision Tree
 │                      │
 └──────────┬───────────┘
            ↓
   Model Evaluation
            ↓
 Accuracy / Precision
 Recall / F1-score
            ↓
 Confusion Matrices
            ↓
 Overfitting Analysis
            ↓
 Final Comparison
```

---

## Conclusion

This assignment demonstrates how preprocessing choices can influence the performance of machine learning classification models. By keeping the train-test split and model configurations consistent, the four experiments provide a fair comparison between StandardScaler and MinMaxScaler and between Logistic Regression and Decision Tree classifiers.

The use of `Pipeline` and `ColumnTransformer` ensures that preprocessing is performed correctly without data leakage. Evaluation using multiple metrics and confusion matrices provides a more complete understanding of model performance than accuracy alone.

The final model selection should be based on the observed test-set performance, with particular attention to F1-score, precision, recall, and the train-test performance gap.

---

## Author

**Puranik Aryan Hitesh**

**Course:** Fundamentals of Machine Learning

**Assignment:** Classification Model Comparison — Hotel Booking Cancellation Prediction
