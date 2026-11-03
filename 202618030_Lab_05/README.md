# Garment Employee Productivity Prediction

### DS605 — Fundamentals of Machine Learning | Lab Assignment 05

**Student:** Puranik Aryan  
**Student ID:** 202618030  
**Notebook:** `202618030_Puranik_Aryan_lab05.ipynb`

---

## Project Overview

Employee productivity is a critical operational measure in labour-intensive manufacturing. In garment production, actual output can be influenced by production targets, staffing levels, work-in-progress, overtime, incentives, and interruptions to daily operations. Understanding how these factors relate to performance can help frame useful forecasting and target-attainment questions.

This assignment studies garment employee productivity from two complementary machine learning perspectives:

- **Regression:** Estimate an employee team's continuous *actual productivity* score.
- **Binary classification:** Predict whether actual productivity **meets or exceeds its assigned target**.

Rather than treating a library-trained model as the end of the exercise, the study compares established Scikit-learn estimators with independently developed NumPy/Pandas counterparts. It then examines how an alternative optimization strategy changes logistic regression's accuracy, convergence behaviour, and measured runtime.

The emphasis is on **predictive performance, consistency of experimental conditions, interpretability of evaluation metrics, and the trade-offs between optimization approaches**. The notebook contains the computational work; this README focuses on its objectives, experimental design, observations, and implications.

## Objectives

1. Formulate continuous productivity prediction and target-attainment classification as distinct supervised learning problems.
2. Evaluate linear and logistic regression on the same underlying garment production records.
3. Compare standard machine learning estimators with independently constructed numerical alternatives under consistent experimental conditions.
4. Assess model performance using task-appropriate metrics, rather than relying on a single score.
5. Investigate whether a second-order optimization approach changes logistic regression convergence, runtime, or prediction quality.
6. Identify limitations of the results and the practical questions that remain unanswered.

## Dataset

**Source:** [UCI Machine Learning Repository — Productivity Prediction of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)  
**Dataset file:** `garments_worker_productivity.csv`

The notebook uses **1,197 observations and 15 original columns**, describing garment production operations and recorded productivity. The data contain both numerical production measures and categorical descriptors of the operating environment.

| Dataset characteristic | Description |
|---|---|
| Observations | 1,197 |
| Original attributes | 15 |
| Operational context | Garment manufacturing teams and production performance |
| Regression outcome | `actual_productivity` |
| Classification outcome | `MeetsTarget`, derived from actual and targeted productivity |
| Main missing-data issue | 506 missing entries in `wip` |
| Cleaned department groups | Sewing (691 records) and finishing (506 records) |

### Feature groups

| Group | Representative variables | Relevance |
|---|---|---|
| Production planning | `targeted_productivity`, `smv`, `wip` | Expected output, standard work content, and work in progress |
| Workforce and scheduling | `team`, `no_of_workers`, `over_time` | Team assignment, staffing, and additional labour time |
| Operational interruptions | `idle_time`, `idle_men`, `no_of_style_change` | Production disruptions and changes in style |
| Incentives | `incentive` | Financial incentive information associated with production |
| Production context | `quarter`, `department`, `day` | Timing and departmental context |

The original date attribute is excluded from the modelling features in the notebook. The retained predictors are a mixture of numerical and categorical attributes.

### Understanding the two targets

**Actual productivity** is a continuous recorded measure. The regression problem asks how closely the model can estimate this value from available operational information.

**Target attainment** is defined as:

\[
\text{MeetsTarget}=\begin{cases}
1 & \text{if actual productivity} \geq \text{targeted productivity}\\
0 & \text{otherwise}
\end{cases}
\]

Approximately **73.10%** of records meet or exceed the target; **26.90%** do not. This imbalance matters when interpreting classification accuracy: predicting the majority outcome is already a meaningful baseline to consider.

Importantly, although the classification label is derived from actual productivity, **actual productivity is not included as a classification input**. This distinction is essential to evaluating whether target attainment can be predicted from the other available attributes.

## Study Design

The assignment is organized into three connected parts.

### Part A — Scikit-learn benchmarks

The first stage establishes reference performance for **linear regression** and **logistic regression** using established machine learning tools. It produces held-out predictions and records regression error, classification quality, and model execution times. These results serve as the comparison point for the rest of the study.

### Part B — Independent numerical models

The second stage investigates whether the two learning problems can be solved independently using **NumPy and Pandas**, without relying on Scikit-learn's model-training and evaluation utilities. The aim is to validate understanding of the underlying mathematical models and determine how closely independently obtained predictions and metrics align with the reference estimators.

### Part C — Model comparison and optimization

The final stage compares results across approaches and focuses on optimization of the logistic regression model. In addition to the initial gradient-descent solution, it evaluates a **Newton/Iteratively Reweighted Least Squares (IRLS)** alternative. The comparison considers convergence, predictive metrics, and runtime rather than assuming that a faster optimizer must also produce a more accurate classifier.

### Experimental consistency

All reported comparisons use the same **80:20 training–test partition**: **957 training records** and **240 test records**, with a fixed random state of 42. Missing values, categorical variables, and numerical scale differences are addressed consistently across the experiments. Statistics used for data preparation are learned from training data, not the held-out test records.

This setup supports a controlled comparison of modelling approaches on a shared evaluation sample. Because the notebook reports a **single random split**, the results should be interpreted as a reproducible experiment rather than evidence of performance stability across every possible partition or future production period.

## Results and Evaluation

The following results are taken from the notebook's recorded outputs. Runtime values reflect that particular execution environment and are not universal model-speed benchmarks.

### Regression — Predicting actual productivity

| Metric | Scikit-learn | Independent NumPy/Pandas model |
|---|---:|---:|
| Mean Absolute Error (MAE) | 0.1088 | 0.1088 |
| Root Mean Squared Error (RMSE) | 0.1481 | 0.1481 |
| R² score | 0.1736 | 0.1736 |
| Training time | 0.052641 s | 0.001446 s |
| Prediction time | 0.007643 s | 0.000098 s |

**Interpretation.** Both regression approaches deliver virtually identical test-set errors and explained variance. This provides strong evidence that their reported least-squares solutions are consistent under the shared experimental setup.

However, an **R² of approximately 0.174** means that the linear model explains only about **17.4% of the observed variance in actual productivity on the test set**. Matching the reference implementation is therefore a validation of computational agreement, **not** proof that the predictive model is sufficiently expressive for operational forecasting. Unmodelled relationships, nonlinear effects, data quality, and additional contextual factors may be important.

The difference between MAE and RMSE also deserves attention: RMSE penalizes larger errors more heavily, so considering both is more informative than reporting either alone.

### Classification — Predicting target attainment

| Metric | Scikit-learn | Independent gradient descent | Optimized Newton/IRLS |
|---|---:|---:|---:|
| Accuracy | 0.7500 | **0.7583** | 0.7500 |
| Precision | 0.7696 | **0.7767** | 0.7696 |
| Recall | 0.9435 | 0.9435 | 0.9435 |
| F1-score | 0.8477 | **0.8520** | 0.8477 |
| Training time | 0.032983 s | 0.082934 s | **0.002810 s** |
| Prediction time | 0.009188 s | **0.000130 s** | 0.000149 s |

*The bold values indicate the strongest observed value within each metric, not a statistically established winner.*

**Interpretation.** The independently trained gradient-descent classifier produces marginally higher accuracy, precision, and F1-score on this particular test split, while all three approaches achieve the same reported recall of approximately **94.35%**. The results do not establish that the manual classifier generalizes better: the test set is small, class proportions are uneven, and no repeated-split or statistical uncertainty analysis is reported.

The optimized Newton/IRLS classifier matches Scikit-learn's reported classification metrics at the displayed precision. This is a useful outcome because it demonstrates that different optimization strategies can reach predictions of comparable quality, even when their computational paths differ.

### Convergence and execution time

The notebook records **3,000 gradient-descent iterations** for the initial manual logistic regression experiment, compared with **8 Newton/IRLS iterations** for the optimized alternative. In the recorded run, optimized training takes **0.002810 s**, versus **0.082934 s** for gradient descent.

This illustrates a key optimization trade-off: second-order methods can require substantially fewer iterations, but each iteration may be more computationally demanding. Accordingly, fewer iterations do not automatically guarantee lower runtime for larger datasets or higher-dimensional problems.

The extremely short training and prediction times should also be interpreted cautiously. Library pipelines and manually evaluated numerical operations do not necessarily measure identical overhead, and single-run timings are sensitive to hardware, software versions, execution conditions, and measurement noise. The values are useful for this assignment's **within-run comparison**, but not as a definitive performance ranking across computing environments.

## Key Findings

**1. Numerical agreement does not imply predictive adequacy.**  
The regression models agree almost exactly on the held-out data, yet their relatively low R² points to limited explanatory performance under a linear specification.

**2. Classification results require class-aware interpretation.**  
With roughly 73% positive examples in the full dataset, accuracy alone can be misleading. Precision, recall, and F1-score reveal more about the classifier's treatment of teams that meet productivity targets. In particular, high positive-class recall does not automatically imply equally strong identification of teams that miss their targets.

**3. The choice of optimization method affects computational behaviour.**  
The optimized logistic regression experiment reaches classification results comparable to the reference method using far fewer recorded iterations than the initial gradient-descent approach.

**4. Small metric differences need proportionate conclusions.**  
The manual gradient-descent classifier's improvement in accuracy is about **0.83 percentage points** over the Scikit-learn model. Without evaluation across multiple splits, this should be described as a result of the present experiment, not evidence of systematic superiority.

**5. The two tasks answer different operational questions.**  
Regression asks *how productive a team will be*; classification asks *whether it will satisfy its assigned target*. The classification problem may produce stronger-looking metrics while providing less detailed information about the magnitude of underperformance or overperformance.

## Limitations and Considerations

- **Limited regression fit:** The tested linear specification leaves most test-set productivity variability unexplained.
- **Class imbalance:** Positive target-attainment examples dominate the dataset, making a majority-class baseline and negative-class diagnostics important for any practical interpretation.
- **One train–test split:** Reported results do not quantify variation across alternative partitions or random seeds.
- **No temporal validation:** The notebook uses a random split; it does not evaluate how performance transfers to later production dates. This matters if the intended application is forecasting future operations.
- **Limited diagnostic coverage:** The reported evaluation includes aggregate regression and classification metrics, but does not establish calibration, subgroup fairness, or class-specific error rates for teams missing their targets.
- **Timing comparability:** Reported runtimes include different levels of framework overhead and reflect only one execution context.
- **Observational data:** Relationships between operational variables and productivity should not be interpreted as causal effects without an appropriate causal research design.

## Potential Extensions

Several improvements could extend the analysis beyond the scope of this assignment:

1. **Stronger regression benchmarks:** Compare linear regression with regularized and nonlinear models to examine whether the current R² reflects model limitations, feature limitations, or both.
2. **More robust validation:** Use repeated or cross-validated evaluation and report variability, not only point estimates from one partition.
3. **Time-aware evaluation:** Assess whether a model trained on earlier observations can generalize to later production periods, if the date coverage supports this design.
4. **Class-specific diagnostics:** Examine confusion matrices, specificity, balanced accuracy, and precision–recall behaviour, especially for failure to meet targets.
5. **Operational interpretability:** Study which recorded characteristics are associated with productivity predictions while avoiding unsupported causal claims.
6. **Error analysis:** Identify the operating conditions in which productivity estimates or target-attainment predictions are least reliable.

These are proposed directions, **not experiments claimed to have been completed in the notebook**.

## Repository Contents

```text
.
├── 202618030_Puranik_Aryan_lab05.ipynb   # Assignment notebook and recorded results
├── garments_worker_productivity.csv     # Dataset (obtain separately if not included)
└── README.md                            # Project overview and findings
```

### Reproducing the study

The work is presented in a Jupyter Notebook and uses Python, NumPy, Pandas, and Scikit-learn. Open the notebook in a compatible Jupyter environment and ensure the dataset CSV is accessible at the path referenced by the notebook before running its cells in order. Software versions and execution hardware may affect measured timings; the notebook's recorded predictive metrics are the reference values summarized above.

## Conclusion

This assignment compares conventional machine learning estimators with independent numerical approaches for garment employee productivity prediction. **Linear regression exhibits excellent agreement between the two approaches but modest predictive fit**, while **logistic regression achieves substantially higher target-attainment recall and similar classification quality across the tested optimization strategies**.

The most consequential result is not that one implementation is categorically better than another. It is that **computational correctness, predictive usefulness, and optimization efficiency are separate questions**. A model can reproduce a reference solution almost perfectly while still leaving significant variation unexplained; conversely, a more efficient optimization method may match established classification results without changing the underlying task's predictive limitations.

Together, the three assignment parts provide a practical comparison of supervised learning objectives, evaluation choices, and optimizer behaviour on a real operational dataset.

---

**Course:** DS605 — Fundamentals of Machine Learning  
**Assignment:** Lab 05  
**Author:** Puranik Aryan (202618030)
