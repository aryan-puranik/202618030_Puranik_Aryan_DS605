# Banknote Authentication: Exploring Decision Trees and Support Vector Machines

### A comparative machine learning study | Decision Trees · Linear SVM · Kernel SVM

## Name : Puranik Aryan Hitesh
## student id : 202618030

## Project Overview

How can we distinguish a genuine banknote from a forged one using measurable characteristics of its image? This project explores that question through two widely used classification approaches: **Decision Trees** and **Support Vector Machines (SVMs)**.

Rather than focusing only on which model produces the highest accuracy, the study looks at **how different models make decisions**, how those decisions change with model settings, and what trade-offs appear between accuracy, interpretability, flexibility, and computational cost.

The work is organized into four connected parts. It begins with an interpretable Decision Tree, moves to a linear SVM and its margin-based perspective, explores the limits of linear boundaries using nonlinear kernels, and ends with a consistent comparison on the original dataset.

## Dataset

**Source:** [UCI Machine Learning Repository — Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

The dataset contains **1,372 observations**, **four numerical predictors**, and **two classes**. The predictors are statistics extracted from wavelet-transformed images of banknotes:

| Feature | What it represents |
| --- | --- |
| Variance | Variation in the transformed image |
| Skewness | Asymmetry in the feature distribution |
| Curtosis | Shape or tail characteristics of the distribution |
| Entropy | Information content of the image features |

The target is a binary class label used to distinguish the two banknote categories. The dataset contains no missing attribute values. For the main experiments, the data is divided into training and test sets, with the same split reused for the final comparison.

## Part A — Understanding Decision Trees

The first part investigates how a Decision Tree can classify banknotes through a sequence of understandable decisions. A tree divides the feature space into regions and assigns a class based on the path a sample follows.

**Approach.** A Decision Tree is developed independently and assessed alongside Scikit-learn's Decision Tree classifier. Both Gini impurity and entropy are considered as ways to evaluate the quality of decisions. Different tree depths are examined to understand how greater flexibility affects training and test performance.

To make the results easier to interpret, **variance and skewness** are used in separate two-feature experiments. Their decision boundaries show how a tree divides a two-dimensional feature space into distinct regions. The learned tree and its split rules are also visualized.

**Observations and takeaways.**

- Trees are particularly appealing when a prediction needs a clear, traceable explanation.
- Their decision regions are made of axis-aligned splits, which produces a characteristic stepped or rectangular boundary.
- Deeper trees can capture more detail, but additional complexity may fit noise rather than general patterns. Training performance therefore needs to be considered alongside test performance.
- Comparing the independent tree with Scikit-learn helps assess whether the two models arrive at similar predictions, even when their internal split choices are not identical.

## Part B — Exploring Linear SVMs

The second part approaches classification from a different angle: instead of building a hierarchy of rules, a **linear Support Vector Machine** looks for a separating boundary with a margin between the classes.

**Approach.** A soft-margin linear SVM is studied alongside Scikit-learn's linear SVM. The features are standardized using information from the training set, because the relative scale of measurements matters for margin-based models. The study visualizes the separating line, the positive and negative margins, and samples that fall within or violate those margins.

The experiments also explore the influence of **regularization strength (`C`)**, **learning rate**, and **training iterations**. These settings provide different perspectives: regularization affects how strongly the model penalizes classification or margin violations, while the learning rate and iteration count influence optimization behavior.

**Observations and takeaways.**

- Linear SVMs provide a compact decision rule, but their straight-line boundary may be restrictive when the relationship between classes is nonlinear.
- A wider-margin solution may tolerate some violations in exchange for better generalization; insisting too strongly on fitting the training data is not always beneficial.
- Proper standardization makes the features more comparable and prevents differences in measurement scale from dominating the model.
- Changes in optimization settings can affect convergence and model quality. A fair comparison therefore considers both predictive performance and whether the training process has sufficiently converged.

## Part C — When Linear Boundaries Are Not Enough

Not every classification problem can be separated effectively with a straight line. Part C illustrates this limitation using a **synthetic two-dimensional `make_moons` dataset**, whose two classes follow curved, interleaving shapes.

**Approach.** Three Scikit-learn SVMs are trained and compared on the same nonlinear dataset:

| Kernel | Perspective |
| --- | --- |
| **Linear** | Tests what a straight boundary can achieve |
| **Polynomial** | Allows curved relationships shaped by polynomial interactions |
| **RBF (Radial Basis Function)** | Allows flexible boundaries based on similarity between observations |

Their decision boundaries are plotted side by side, and predictions are evaluated on held-out observations. This makes the differences between kernels visible rather than relying only on aggregate metrics.

**Observations and takeaways.**

- A linear boundary can struggle with interleaving shapes even when the classes follow a recognizable pattern.
- Polynomial and RBF kernels can represent curved separation patterns without manually creating an extensive set of transformed predictors.
- Greater flexibility is useful only when it captures meaningful structure; it can also increase the risk of overfitting.
- The most appropriate kernel depends on the geometry, noise, and complexity of the data. Nonlinear models are not automatically the best choice for every dataset.

> **Important distinction:** The nonlinear kernel comparison uses synthetic data to demonstrate decision-boundary behavior. The final model comparison in Part D returns to the original Banknote dataset.

## Part D — Final Model Comparison

The last part brings the earlier ideas together in a single evaluation of **five classifiers** using **all four Banknote features** and a consistent train-test split.

| Model | Role in the comparison |
| --- | --- |
| Decision Tree — independent implementation | Interpretable rule-based baseline |
| Decision Tree — Scikit-learn | Reference tree implementation |
| Linear SVM — independent implementation | Margin-based linear classifier |
| Linear SVM — Scikit-learn | Reference linear SVM implementation |
| RBF SVM — Scikit-learn | Flexible nonlinear classifier |

**Evaluation criteria.** The models are compared on **accuracy, precision, recall, F1-score, training time, and prediction time**. Training-versus-test performance is also considered to identify possible overfitting. SVM features are standardized consistently using the training set; the Decision Trees use the original feature values.

**Key themes emerging from the comparison:**

- **Interpretability:** A shallow Decision Tree is the easiest to explain as a sequence of rules. A linear SVM offers a simple mathematical boundary, while an RBF SVM is generally less transparent.
- **Scaling:** Trees usually do not need feature standardization, whereas SVMs are sensitive to feature magnitudes.
- **Overfitting:** Greater complexity can improve training fit without improving generalization. Tree depth and SVM regularization are important controls.
- **Computational cost:** Linear SVM prediction is typically lightweight. Kernel models can depend on the number of support vectors, and highly complex trees require more decision paths or stored nodes.
- **Fairness:** A shared split and consistent evaluation metrics matter more than isolated headline accuracy figures. Timing comparisons should also be interpreted carefully because independently developed educational models and optimized library implementations have different overheads.

## Overall Reflection

The central lesson of this project is that **there is no universally best classifier**. Each approach brings a different strength to the same problem.

A Decision Tree is useful when readable decisions and feature-based explanations are the priority. A linear SVM is attractive when a relatively simple boundary can separate the classes effectively. A nonlinear SVM offers extra flexibility when the underlying structure cannot be captured with a straight line, though that flexibility introduces additional complexity and tuning considerations.



