# Vectorized Programming with NumPy and Data Wrangling with Pandas
Name: Puranik Aryan Hitesh
ID: 202618030


# DS605 — Fundamentals of Machine Learning

## Lab Assignment 2: Vectorized Programming with NumPy and Data Wrangling with Pandas


  <b>NumPy • Pandas • Data Wrangling • Statistical Analysis • Data Visualization</b>

## Assignment Overview

This repository contains the implementation of **Lab Assignment 2** for the course **DS605: Fundamentals of Machine Learning**.

The assignment focuses on two fundamental components of data science:

1. **Vectorized numerical computing using NumPy**
2. **Data wrangling and exploratory analysis using Pandas**

The second part of the assignment uses the **Kaggle Titanic `train.csv` dataset** to perform data inspection, filtering, aggregation, missing-value analysis, feature engineering, pivot-table analysis, and visualization.

The primary objective is to develop practical familiarity with efficient NumPy operations and essential Pandas data-analysis workflows.

---

##  Student Information

| Field          | Details                                  |
| -------------- | ---------------------------------------- |
| **Name**       | Aryan Puranik                            |
| **Student ID** | 202618030                                |
| **Course**     | DS605 — Fundamentals of Machine Learning |
| **Lab**        | Lab Assignment 2                         |
| **Topic**      | NumPy and Pandas                         |
| **Submission** | Public GitHub Repository                 |

---

## Objectives

The assignment is designed to develop proficiency in:

* Creating and manipulating NumPy arrays
* Performing vectorized numerical operations
* Computing descriptive statistics
* Understanding array dimensions, indexing, slicing, and reshaping
* Performing matrix and linear-algebra operations
* Generating and analyzing normally distributed data
* Loading and inspecting datasets with Pandas
* Selecting data using `loc` and `iloc`
* Filtering data using Boolean indexing and `query()`
* Performing grouped aggregations with `groupby()`
* Handling missing values
* Detecting outliers using the IQR method
* Creating new features from existing columns
* Constructing pivot tables
* Performing correlation analysis
* Creating meaningful data visualizations
* Drawing observations from numerical and graphical analysis

These objectives directly follow the requirements specified in the assignment document.

---

# Dataset

### Titanic Dataset

**Dataset:** Kaggle Titanic Dataset
**File:** `train.csv`

The Titanic dataset is used for the Pandas portion of the assignment. It contains passenger-level information such as:

* Passenger ID
* Survival status
* Passenger class
* Name
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Ticket
* Fare
* Cabin
* Port of embarkation

The dataset is used to demonstrate practical data-wrangling and exploratory-analysis techniques.

---

# Assignment Tasks

## Part A — Vectorized Programming with NumPy

### Task 1 — Arrays, Statistics, and Indexing



### Task 2 — Vectorized Arithmetic and Linear Algebra


---

#  Part B — Data Wrangling with Pandas

## Task 4 — Load and Inspect Data

The Titanic dataset is loaded using Pandas and examined using:

```python
head()
tail()
shape
columns
info()
describe()
```

The notebook also demonstrates row and column selection using:

```python
loc
iloc
```

and explains the difference between the two approaches.

---

## Task 5 — Filtering and Querying

Boolean indexing and/or `query()`:


T
## Task 6 — Grouping and Aggregation

`groupby()` is used to analyze:

### Survival Rate

* Survival rate by Sex
* Survival rate by Pclass

### Average Values

* Average Age by Pclass
* Average Fare by Pclass

### Combined Groups

* Passenger count by Sex-Pclass
* Survival rate by Sex-Pclass

### Embarkation Analysis

* Passenger count by Embarked
* Average Fare by Embarked
* Survival rate by Embarked

---

## Task 7 — Missing Values and Fare Outliers

The dataset is analyzed for missing values by calculating:

* Missing-value count for every column
* Missing-value percentage for every column

A bar chart is generated to visualize missing values.

Different imputation strategies are explored, including:

* Mean
* Median
* Mode
* Random value

For the `Age` column, missing values are filled using the mean as required.

### Fare Outlier Detection

The IQR method is used to identify potential Fare outliers.


### IsAlone

A passenger is considered alone when:

`

# Task 9 — Visualizations and Observations

The analysis includes the following visualizations:

### Correlation Heatmap

A correlation heatmap is generated for relevant numerical variables to identify strong positive and negative relationships.

### Survival Rate by Sex

A visualization compares male and female survival rates.

### Age vs Fare

A scatter plot compares:

* Age
* Fare
* Survival status

Survivors and non-survivors are distinguished in the visualization.

### Numerical and Visual Observations

The final analysis contains **5–7 short observations** supported by the calculated results and visualizations, as required by the assignment.

---

# 🛠️ Technologies and Libraries

The project is implemented in Python using the following tools:

| Technology           | Purpose                                       |
| -------------------- | --------------------------------------------- |
| **Python**           | Programming language                          |
| **NumPy**            | Numerical computing and vectorized operations |
| **Pandas**           | Data manipulation and analysis                |
| **Matplotlib**       | Data visualization                            |
| **Seaborn**          | Statistical visualization and heatmaps        |
| **Jupyter Notebook** | Interactive development and presentation      |


# 🔍 Key Learning Outcomes

Through this assignment, the following practical concepts are demonstrated:

* Efficient vectorized computation using NumPy
* Difference between element-wise and matrix multiplication
* Matrix transformations and linear algebra
* Statistical properties of random distributions
* Pandas DataFrame inspection
* Boolean filtering and conditional querying
* Grouped statistical analysis
* Missing-data handling
* IQR-based outlier detection
* Feature engineering
* Pivot-table analysis
* Correlation analysis
* Data visualization
* Data-driven interpretation of results

---

# 📌 Key Observations

The final observations are derived from the numerical analysis and visualizations performed on the Titanic dataset.

Typical areas examined include:

1. Differences in survival rates between male and female passengers.
2. Relationship between passenger class and survival probability.
3. Survival patterns across Sex-Pclass groups.
4. Differences in Fare distributions between passenger classes.
5. Relationships among numerical variables identified through correlation analysis.
6. The influence of family size and travelling alone on survival.
7. Patterns between passenger Age, Fare, and survival status.

> **Important:** Replace these general observation areas with the exact numerical findings produced by your notebook before submitting the repository.

---

#  Author

**Puranik Aryan Hitesh**

**Student ID:** 202618030

**Course:** DS605 — Fundamentals of Machine Learning

**Assignment:** Lab 2 — NumPy and Pandas
