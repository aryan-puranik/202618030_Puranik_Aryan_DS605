# Feature Extraction and Machine Learning with Image and Text Data

**DS605 — Fundamentals of Machine Learning | Lab Assignment 06**  
**Student:** Puranik Aryan Hitesh  
**Student ID:** 202618030

## Project Overview

Machine learning models depend not only on the choice of classifier but also on how information is represented. Images contain pixels, textures, edges, and spatial structures; documents contain words with different frequencies and levels of relevance. Neither representation automatically tells a conventional machine learning model which signals matter for prediction.

This assignment studies feature representation across **two different data modalities—images and text**—and then revisits image classification to investigate the practical trade-off between predictive performance and computational efficiency. Three connected investigations make up the project:

- **Part A — Image Feature Extraction and Classification:** identify cracks in surface images using engineered visual features and compare Logistic Regression with Random Forest.
- **Part B — Text Vectorization and Spam Classification:** compare word-count and TF–IDF representations using Multinomial Naive Bayes, Logistic Regression, and a linear Support Vector Classifier.
- **Part C — Improving the Image Representation:** examine changes in image resolution, edge detection, and feature dimensionality, and determine how they affect crack-detection accuracy and processing time.

The emphasis throughout is on **why a representation is useful, how fairly it is evaluated, what the observed results mean, and where the findings may not generalize**. The accompanying notebooks contain the complete experiments and implementation.

## Objectives

The work aims to:

1. Understand how raw visual and textual information can be transformed into informative numerical features.
2. Assess how different learning algorithms respond to the same feature representation.
3. Evaluate predictions beyond accuracy, particularly through precision, recall, F1-score, confusion matrices, and ROC–AUC where appropriate.
4. Compare word-frequency information with term importance in document classification.
5. Examine whether reducing visual resolution and feature dimensionality can lower computational cost without severely compromising detection quality.
6. Apply consistent train/test separation and cross-validation to support meaningful comparisons.

---

## Part A — Image Feature Extraction and Crack Classification

### Problem and Motivation

Detecting cracks in surfaces is a relevant visual inspection problem. A practical detection system must distinguish thin, dark, irregular crack structures from ordinary surface patterns, shadows, noise, and other visual variations. Although image-based deep learning approaches are common, this experiment investigates whether **interpretable, manually designed image descriptors** can provide strong classification results using conventional machine learning algorithms.

The task is **binary classification**:

| Label | Class | Meaning |
|---|---|---|
| `1` | `crack` | A visible crack is present in the surface image. |
| `0` | `no_crack` | The image represents a surface without a labeled crack. |

### Dataset and Experimental Design

The notebook evaluates **400 images**, balanced equally between the two classes. The original images have dimensions of **448 × 448 pixels** and three color channels. A stratified division preserves this class balance in the training and evaluation sets.

| Dataset characteristic | Value |
|---|---:|
| Total images | 400 |
| Crack images | 200 |
| Non-crack images | 200 |
| Training images | 320 (160 per class) |
| Test images | 80 (40 per class) |
| Engineered features per image | 50 |

The same training and test examples are used for both classifiers. Model selection is based on **five-fold stratified cross-validation using the training set**, with the held-out test set reserved for final evaluation. This design reduces the likelihood that differences in results are caused by inconsistent evaluation data.

### Understanding the Visual Representation

A crack is not defined simply by a dark pixel. Surface illumination, texture, and noise can create dark regions unrelated to a structural defect. The representation therefore combines several complementary types of visual evidence.

**Image enhancement and crack emphasis.** Images are resized and converted to grayscale so brightness variations can be studied consistently. Gaussian smoothing reduces local noise. Black-hat morphological processing emphasizes relatively dark features against their surrounding surface, making crack-like structures easier to identify. Normalization improves the visibility of the resulting response, while thresholding produces a candidate crack mask.

**Intensity descriptors.** Brightness, contrast, pixel distributions, and related statistics provide a broad description of the surface appearance. They can help distinguish relatively uniform areas from surfaces containing strong local variations.

**Edge descriptors.** Edge-based measurements capture the amount and density of image boundaries. They provide evidence about lines and irregular structures, although not every detected edge necessarily corresponds to a crack.

**Morphology and spatial descriptors.** Black-hat and mask-derived features describe the extent, connected regions, shapes, and locations of candidate dark structures. Spatial measurements are useful because the distribution of surface irregularities can be as informative as their total area.

Together, these descriptors produce **50 numerical features per image**, offering a compact and comparatively interpretable alternative to feeding raw pixel arrays into a conventional classifier.

### Classifiers and Comparison Rationale

**Logistic Regression** provides a linear decision model and serves as a useful reference for determining whether the engineered representation already separates the classes well. Feature scaling supports its sensitivity to differing numerical ranges.

**Random Forest** combines decision trees and can capture nonlinear patterns and interactions between intensity, edge, and morphology features. It also offers feature-importance estimates for exploring which descriptors contribute most to classification decisions.

Both approaches are assessed under a consistent data split and model-selection procedure.

### Observed Results

| Evaluation metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Training cross-validation F1 | 0.9623 | **0.9655** |
| Held-out test accuracy | **98.75%** | 95.00% |
| Test crack precision | **100.00%** | 95.00% |
| Test crack recall | **97.50%** | 95.00% |
| Test crack F1-score | **0.9873** | 0.9500 |
| Missed cracks | **1** | 2 |
| False crack alarms | **0** | 2 |

The cross-validation procedure ranked **Random Forest slightly higher** on the training data. On the independent 80-image test set, however, **Logistic Regression performed better**, making only one error. This difference is important: the model preferred by cross-validation is not guaranteed to achieve the highest score on a particular held-out sample.

The test confusion matrices reinforce this interpretation. Logistic Regression correctly identified all 40 non-crack images and 39 of the 40 crack images; Random Forest correctly classified 38 of each class. On a small test set, a difference of one image corresponds to **2.5 percentage points of crack recall**, so these results should not be interpreted as a definitive ranking across all surface conditions.

### Key Observations

- **Feature quality matters:** strong test performance indicates that the combination of intensity, edge, and morphological descriptors captures substantial discriminatory information in this dataset.
- **A simpler classifier can be competitive:** Logistic Regression outperforming Random Forest on the held-out test set suggests that the engineered features may already provide a relatively separable representation.
- **Detection errors have unequal consequences:** missed cracks may be particularly concerning in safety-related inspection settings; recall should therefore be examined alongside precision and false alarms.
- **Feature interpretability is valuable:** engineered descriptors provide a clearer link to observable image properties than raw high-dimensional pixels, though feature importance alone does not establish causality.

**Part A takeaway:** meaningful image features enable both a linear and a nonlinear classifier to detect cracks effectively. The experiment also establishes the baseline against which changes to feature representation can be evaluated in Part C.

---

## Part B — Text Vectorization and Spam Classification

### Problem and Motivation

Unlike images, emails are composed of words and document-level patterns. Machine learning classifiers require a numerical representation that preserves useful differences between spam and legitimate communication. This part compares two foundational approaches to document representation: **raw word counts** and **term frequency–inverse document frequency (TF–IDF)**.

The objective is to identify spam (`1`) and non-spam (`0`) emails while examining how the choice of text representation affects predictive quality and model efficiency.

### Dataset and Important Representation Constraint

The source contains **5,172 emails** represented by **3,000 word-count columns**, alongside an email identifier and the target label. It is not a collection of intact raw email messages: the original word ordering and sentence structure are unavailable.

| Dataset characteristic | Value |
|---|---:|
| Initial emails | 5,172 |
| Initial non-spam emails | 3,672 (71.0%) |
| Initial spam emails | 1,500 (29.0%) |
| Vocabulary size | 3,000 terms |
| Exact duplicate rows removed | 541 |
| Emails retained for modeling | 4,631 |
| Training emails | 3,704 |
| Test emails | 927 |
| Spam proportion after deduplication | Approximately 31.5% |

Because the dataset already provides a **bag-of-words count representation**, the experiment reconstructs *pseudo-text* that retains each word's frequency but not its original position. This permits a direct and comparable evaluation of CountVectorizer and TF–IDF without claiming to recover the original email bodies. The notebook verifies that count vectorization reproduces the supplied word counts exactly.

This distinction is central to interpreting the study: the experiment compares **frequency-based representations of the available vocabulary**, not full natural-language understanding, sentence context, or semantic meaning.

### Comparing the Text Representations

**Bag of Words / CountVectorizer** describes an email by the number of times each vocabulary term appears. It preserves direct frequency information and is straightforward to interpret, but frequently occurring terms can dominate the numerical representation.

**TF–IDF** changes the relative importance of terms by considering both their frequency within an email and their prevalence across the document collection. Terms common to many emails generally receive less distinctive weight, while relatively less common terms may carry greater discriminative value. The notebook's TF–IDF representation also applies sublinear term-frequency scaling and normalization.

Both representations cover the **same 3,000-word vocabulary** and exhibit the **same sparsity (94.09% in the reported training matrices)**. Their difference is therefore principally in feature **weighting**, not the number of features or the locations of non-zero entries.

### Models Evaluated

The experiment compares three established text classifiers:

| Model | Motivation |
|---|---|
| Multinomial Naive Bayes | A fast probabilistic baseline naturally suited to non-negative word-frequency features. |
| Logistic Regression | A discriminative linear model that can learn weighted relationships between terms and spam labels. |
| Linear Support Vector Classifier | A margin-based classifier commonly effective on high-dimensional sparse text representations. |

Each classifier is evaluated with both vectorization schemes, allowing the effect of representation to be examined separately from the choice of learning algorithm. The evaluation gives particular attention to **spam precision, spam recall, and spam F1-score**, since a majority-class prediction can produce deceptively reasonable accuracy in an imbalanced dataset.

### Held-Out Test Results

| Text representation | Classifier | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---|---:|---:|---:|---:|
| CountVectorizer | Multinomial Naive Bayes | 95.15% | 0.8946 | 0.9589 | 0.9256 |
| CountVectorizer | Logistic Regression | 97.20% | 0.9375 | 0.9760 | 0.9564 |
| CountVectorizer | LinearSVC | 97.63% | 0.9530 | 0.9726 | 0.9627 |
| TF–IDF | Multinomial Naive Bayes | 93.74% | 0.8874 | 0.9178 | 0.9024 |
| TF–IDF | Logistic Regression | 97.20% | 0.9209 | **0.9966** | 0.9572 |
| **TF–IDF** | **LinearSVC** | **98.38%** | **0.9695** | **0.9795** | **0.9744** |

**TF–IDF with LinearSVC** achieved the strongest spam F1-score and overall test accuracy. It offered a particularly favorable balance between catching spam and avoiding the incorrect classification of legitimate emails as spam.

TF–IDF with Logistic Regression had the **highest spam recall (99.66%)**, but its lower precision means that it generated relatively more false spam alarms. In an email product, that trade-off matters: missing a spam message and incorrectly hiding a legitimate message create different user costs.

Multinomial Naive Bayes performed better with **raw counts** than with TF–IDF in this particular experiment. These findings reinforce that TF–IDF is not automatically superior for every model; its effectiveness depends on the classifier and the dataset.

### Cross-Validation and Robustness

To assess whether the single train/test split gave a misleading impression, the notebook also reports five-fold stratified cross-validation, with text representation fitted separately within each training fold.

| Representation + model | Mean spam F1 | F1 standard deviation |
|---|---:|---:|
| **TF–IDF + LinearSVC** | **0.9750** | 0.0052 |
| TF–IDF + Logistic Regression | 0.9530 | 0.0053 |
| CountVectorizer + Logistic Regression | 0.9479 | 0.0040 |
| CountVectorizer + LinearSVC | 0.9441 | 0.0036 |
| CountVectorizer + Multinomial Naive Bayes | 0.9160 | 0.0073 |
| TF–IDF + Multinomial Naive Bayes | 0.8963 | 0.0088 |

The highest-ranking combination remains **TF–IDF + LinearSVC**, with a mean F1-score close to its held-out test result. This agreement supports the conclusion that the combination performs well across the evaluated dataset splits, rather than merely on one favorable test partition.

### Interpretation and Practical Caveats

The results establish several useful lessons:

- **Weighting changes what the classifier learns:** equal-sized representations can produce meaningfully different decision boundaries and predictive outcomes.
- **There is no universal winner for all objectives:** TF–IDF with LinearSVC gives the best overall F1 here, while TF–IDF with Logistic Regression prioritizes recall.
- **The vocabulary limits generalization:** reconstructed pseudo-text contains only the dataset's original words and frequencies. Real emails may include unfamiliar vocabulary, phrasing, and contextual signals that this representation cannot express.
- **Performance on sample sentences is less convincing:** the notebook's final free-text demonstration labels all four example messages as non-spam, including obviously promotional or urgent account-related examples. This does not invalidate the measured benchmark results, but it **does warn against interpreting them as proof of reliable performance on naturally written, unseen emails**.
- **Time comparisons are experiment-specific:** model training and prediction timings are useful for comparing the reported configurations, but they should not be treated as universal speed benchmarks across machines or corpora.

**Part B takeaway:** text representation is a modeling decision in its own right. On this dataset, TF–IDF paired with a linear Support Vector Classifier achieves the strongest balance of spam detection metrics, while remaining subject to the limitations of the preconstructed bag-of-words vocabulary.

---

## Part C — Improving Feature Representation for Crack Detection

### Research Question

Part A established that engineered image features can support highly accurate crack classification. Part C asks a different question:

> Can the same task be performed with a less expensive visual representation while keeping most of the useful predictive information?

This is a practical concern for large-scale inspection, batch processing, and resource-constrained systems, where image processing can cost considerably more than the classifier's final prediction.

### Changes Investigated

The notebook compares alternative feature representations across three dimensions:

**1. Image resolution.** The original **256 × 256** processed image is compared with a reduced **128 × 128** version. Smaller images contain fewer pixels and can lower the cost of smoothing, morphological operations, edge detection, and statistical feature extraction. The risk is that fine or faint crack details could be lost.

**2. Edge detection thresholds.** Fixed Canny thresholds are compared with thresholds adjusted according to the image's brightness statistics. Adaptive thresholds may improve consistency across lighting conditions, although their value depends on the actual image distribution.

**3. Feature dimensionality.** The study evaluates whether all 50 descriptors are needed. It examines the removal of spatial-grid descriptors and progressively smaller groups of features chosen for their statistical relationship with the class label. Fewer features can simplify the representation, but removing useful complementary cues may harm classification.

These changes are evaluated using the **same class labels and consistent image partitions** as the original experiment. The choice of improved representation is made using training-set cross-validation rather than by repeatedly selecting changes based on held-out test scores.

### Representation Selection Results

The original image representation requires approximately **13.4 milliseconds per image** for preprocessing and feature extraction. Reducing image resolution yields a substantially lower extraction cost:

| Visual representation | Engineered features before selection | Approximate extraction time per image |
|---|---:|---:|
| 256 px, fixed Canny (original) | 50 | 13.4 ms |
| 256 px, adaptive Canny | 50 | 13.6 ms |
| 128 px, fixed Canny | 50 | **5.8 ms** |
| 128 px, adaptive Canny | 50 | **5.8 ms** |

Training-set evaluation selected **128 px with fixed Canny thresholds and all original feature families**, based on a rule that favors lower extraction cost among configurations whose average cross-validation F1 is sufficiently close to the best result.

A subsequent feature-count comparison selected **30 of the 50 features**, a **40% reduction in feature dimensionality**. The selected 30-feature configuration achieved a mean training cross-validation F1 of **0.9660**, compared with **0.9677** when all 50 features were retained at the selected resolution.

This outcome suggests that many of the original descriptors contain overlapping or partially redundant information. Nevertheless, feature reduction is not free: removing some variables can still affect precision, ranking quality, or downstream model behavior.

### Original vs Improved: Held-Out Test Comparison

| Classifier | Representation | Features | Test accuracy | Crack recall | Crack F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Original | 50 | **98.75%** | 97.50% | **0.9873** |
| Logistic Regression | Improved | 30 | 96.25% | 97.50% | 0.9630 |
| Random Forest | Original | 50 | **95.00%** | 95.00% | **0.9500** |
| Random Forest | Improved | 30 | 93.75% | 95.00% | 0.9383 |

The improved representation is **not a pure accuracy improvement**. Both models lose some F1-score on the held-out set, predominantly through reduced precision. However, their crack recall remains unchanged:

- Logistic Regression still detects **39 of 40** labeled cracks.
- Random Forest still detects **38 of 40** labeled cracks.

In the evaluated test set, reducing the representation therefore **does not increase the number of missed cracks**, although it produces more false crack alarms. This distinction matters when inspection systems place greater weight on minimizing missed defects than on minimizing follow-up checks.

### Efficiency Gains and Trade-Offs

| Efficiency measure | Observed change |
|---|---|
| Number of features | **50 → 30** (40% fewer) |
| Preprocessing + feature extraction | **13.4 → 5.8 ms/image** |
| Extraction speed-up | **About 2.33×** |
| Logistic Regression end-to-end speed-up | **About 2.33×** |
| Random Forest end-to-end speed-up | **About 2.03×** |

The results show that **image processing dominates the per-image cost** in these experiments. Reducing image resolution produces most of the practical speed gain, whereas reducing the number of model inputs alone does not guarantee faster model fitting or inference. Indeed, the improved Random Forest configuration has slower measured prediction time than its original counterpart, but the lower feature-extraction time still produces a faster end-to-end pipeline.

The notebook also records increased final model-fitting times for the improved representation. Therefore, the appropriate conclusion is **faster per-image classification throughput due to cheaper extraction**, not universally faster training and prediction at every stage.

### Interpretation

Part C demonstrates the difference between **optimizing a classifier score** and **optimizing an entire prediction pipeline**.

If the only objective is to maximize held-out test F1, the original representation is stronger in this experiment. If the objective includes inference efficiency and maintaining crack recall, the improved representation becomes attractive: it processes images much faster while detecting the same number of cracks in the current 80-image test set.

However, preserving recall on 40 positive examples is limited evidence. Evaluation on more images, especially those with subtle cracks, different surfaces, varying lighting, and different image capture conditions, would be needed before claiming that the reduced-resolution representation is equally safe for practical inspection.

**Part C takeaway:** efficient feature engineering is a deliberate trade-off. Reducing input resolution and removing redundant variables can substantially decrease processing cost, but the quality of the compromise must be judged across accuracy, false alarms, missed detections, and end-to-end latency—not any single metric.

---

## Synthesis: Lessons Across All Three Parts

Although the tasks differ, the experiments point to a common principle: **the numerical representation of a problem often matters as much as the algorithm used to solve it**.

| Theme | Evidence from the experiments | General lesson |
|---|---|---|
| Informative features | Engineered visual descriptors support strong crack classification. | Conventional models can work well when the representation exposes relevant structure. |
| Representation weighting | Count features and TF–IDF produce different spam-classification results despite the same vocabulary. | The meaning of feature values is as important as their number. |
| Model dependence | Naive Bayes favors counts; LinearSVC performs best with TF–IDF; Logistic Regression competes strongly in crack detection. | No classifier or representation is universally optimal. |
| Robust evaluation | Training cross-validation and held-out testing can favor different image models. | Model-selection scores and final generalization estimates play different roles. |
| Efficiency | Lower-resolution image features reduce extraction time by more than half. | End-to-end system cost should inform representation choices. |
| Trade-offs | Reduced image features maintain test crack recall but decrease precision and F1. | A more efficient representation may be preferable even when it is not the most accurate. |

A particularly important insight is that **more features are not necessarily better**, just as more detailed images are not always essential for a given classification task. The appropriate representation depends on which information must be preserved and what errors and computational costs are acceptable.

## Limitations

This work is an educational, controlled comparison rather than a deployment-validated inspection or email-security system.

**Image data limitations.** The crack dataset is relatively small and balanced. Real-world crack prevalence may be much lower, and surface conditions may differ substantially from those represented in the available images. The 80-image test set makes small score differences sensitive to individual examples. The experiments do not establish robustness to severe lighting changes, camera motion, unseen materials, or subtle defects.

**Text data limitations.** The spam dataset is supplied as a fixed vocabulary of word counts, without the original sentence order or complete email contents. Reconstructed pseudo-text cannot restore linguistic context. Predictions on arbitrary new emails may therefore be unreliable, as suggested by the notebook's free-text examples. Exact duplicate removal improves dataset quality but does not rule out near-duplicate messages across partitions.

**Evaluation limitations.** Results are tied to the particular train/test partitions, cross-validation arrangements, class definitions, and available hardware. Metric differences should not be overgeneralized without repeated evaluation, external datasets, and uncertainty estimates.

**Operational limitations.** The experiments do not assess production considerations such as model monitoring, calibration, changing data distributions, consequences of false alarms, or human review workflows.

## Potential Extensions

Future work could investigate **external validation** with image collections from different surfaces and lighting conditions; precision–recall and threshold analysis based on the actual cost of missing defects; more robust text benchmarks using genuine raw emails; and sensitivity analyses across alternative dataset partitions.

For the image task, the most relevant next step would be to test whether the reduced-resolution representation continues to maintain crack recall on a substantially larger and more diverse sample. For text classification, evaluating on untouched real-world messages would help establish whether the strong benchmark scores translate into useful spam detection outside the provided bag-of-words dataset.

## Conclusion

This assignment demonstrates how thoughtful feature representation connects raw data to predictive modeling.

**Part A** shows that a compact set of intensity, edge, and morphological descriptors can support accurate crack detection with traditional classifiers, with Logistic Regression achieving **98.75% test accuracy** in the reported evaluation.

**Part B** shows that word weighting significantly affects classification. Among the evaluated configurations, **TF–IDF with LinearSVC** achieves the strongest spam F1-score (**0.9744**) and test accuracy (**98.38%**), while also revealing the limits of a fixed-vocabulary pseudo-text representation.

**Part C** demonstrates that a smaller and faster image representation can maintain the observed crack recall while reducing feature count by **40%** and accelerating feature extraction by approximately **2.33×**, at the cost of lower test precision and F1-score.

Taken together, the experiments reinforce a practical machine learning principle: **good results come from selecting representations, models, evaluation methods, and efficiency trade-offs that match the actual problem—not simply from using more complex algorithms or larger feature sets.**

---

## Notebook Guide

| Notebook | Assignment part | Focus |
|---|---|---|
| [`crack_classifier_stepwise.ipynb`](crack_classifier_stepwise.ipynb) | Part A | Image feature extraction, crack classification, baseline model evaluation. |
| [`spam_vectorization_partB_emails_csv.ipynb`](spam_vectorization_partB_emails_csv.ipynb) | Part B | Bag-of-words and TF–IDF comparison for spam detection. |
| [`crack_classifier_stepwise_partC.ipynb`](crack_classifier_stepwise_partC.ipynb) | Part C | Image-representation improvements and original-versus-improved evaluation; the notebook also contains the Part A baseline. |

**Note:** Reported values are taken from the saved notebook outputs. Timing results describe the recorded experimental runs and may vary across environments. The notebooks contain the detailed experimental procedures, visualizations, and reproducible analysis; this README focuses on their motivation, findings, interpretation, and limitations.
