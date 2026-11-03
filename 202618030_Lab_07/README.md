# Image Representation, Principal Component Analysis, Eigenfaces and Face Recognition

**DS605 — Fundamentals of Machine Learning | Lab Assignment 07**  
**Student:** Puranik Aryan Hitesh  
**Student ID:** 202618030

---

## Overview

Recognizing faces from images is a useful case study in the challenges of high-dimensional machine learning. A face image contains thousands of pixel values, but much of the variation across images can often be represented using a smaller number of underlying patterns. This assignment explores that relationship through **Principal Component Analysis (PCA)** and the **eigenfaces** approach to facial representation.

The notebook progresses from understanding images as numerical data to discovering principal directions of facial variation, reconstructing images from compressed representations, and assessing the effectiveness of those representations for recognizing individuals. It then compares nearest-neighbor recognition with more advanced classifiers, examines the effects of hyperparameter selection, and evaluates the distinction between reducing feature dimensionality and reducing total storage requirements.

The central question is: **How much of the information in a high-dimensional face image is necessary for effective recognition, and what is gained or lost by retaining only its most important principal components?**

## Learning Objectives

The assignment investigates five connected ideas:

1. **Image representation:** Understand grayscale face images as numerical arrays and describe their pixel-intensity characteristics.
2. **Dimensionality reduction:** Identify the main directions of variation in face images and compare equivalent PCA formulations.
3. **Information preservation:** Analyze explained variance and assess how accurately compressed representations can reconstruct the original faces.
4. **Face recognition:** Evaluate identity classification in the original pixel space and in a lower-dimensional PCA feature space.
5. **Model selection and efficiency:** Compare K-nearest neighbors and support vector machines, examine cross-validated configurations, and quantify representation and storage trade-offs.

## Dataset and Experimental Setting

The notebook uses a collection of **400 grayscale face images from 40 individuals**, with **10 images per individual**, organized into subject folders. Each image has a resolution of **112 × 92 pixels**. The images are represented as vectors of **10,304 pixel features**, giving a high-dimensional input space relative to the number of available examples.

| Property | Value |
|---|---:|
| Total face images | 400 |
| Distinct identities | 40 |
| Images per identity | 10 |
| Image resolution | 112 × 92 |
| Features per original image | 10,304 |
| Training samples | 320 |
| Test samples | 80 |
| Train–test division | 80% / 20% |
| Splitting strategy | Stratified by identity |

The stratified split preserves the representation of identities in both subsets. The same held-out test set is used for the reported classifier evaluations, making the comparisons easier to interpret.

### Image Characteristics

The preliminary investigation examines image dimensions, grayscale intensity distributions, and individual image statistics. All images share the same dimensions. Across the dataset, pixel intensities span **0 to 251**, with a mean of approximately **112.63** and a standard deviation of **49.92**.

Such variation reflects differences in appearance, lighting, shading, expression, and other visible properties. Treating every pixel independently provides a detailed representation, but it does not directly distinguish useful identity information from redundant or distracting variation.

---

## Part A — Image Representation and Dataset Exploration

Part A establishes the foundation for the subsequent experiments by viewing face images as both visual objects and points in a numerical feature space.

A grayscale image is a two-dimensional arrangement of intensity values. Flattening it creates a vector in which each coordinate corresponds to a pixel location. Consequently, the collection of images becomes a matrix with one observation per image and one feature per pixel.

This representation is mathematically convenient for similarity analysis and dimensionality reduction, but it introduces a fundamental challenge: **10,304 features are available for each image, while only 320 training images are used.** The resulting feature space is far larger than the sample space.

The exploration highlights three considerations:

- **Consistent image structure** makes pixel-level comparison meaningful because corresponding coordinates refer to the same locations in each image.
- **Variation in illumination and facial appearance** can strongly influence the geometry of the original pixel space.
- **High dimensionality** motivates learning a smaller representation that captures dominant variation without requiring every original pixel coordinate.

**Part A takeaway:** Raw pixel features provide a complete starting representation, but their size and sensitivity to visual variation motivate PCA-based feature extraction.

---

## Part B — PCA, Eigenfaces, and Face Reconstruction

Part B examines the mathematical and visual structure of face images through PCA. Rather than treating all pixel coordinates as equally informative, PCA identifies orthogonal directions along which the training images vary most strongly.

### Mean Face and Centered Variation

The **mean face** summarizes the average intensity at each pixel location across the training images. Visually, it combines shared facial structure while smoothing differences between individuals.

Each face can then be understood in relation to this average. Centered images represent deviations from the mean face, allowing PCA to focus on patterns of variation instead of the overall average appearance.

The notebook reports a centered training mean extremely close to zero (**approximately 7.06 × 10⁻¹⁷**), consistent with successful mean centering.

### Eigenfaces: Principal Directions of Facial Variation

An **eigenface** is a principal direction of variation reshaped into the original image dimensions. These patterns are not individual identities or complete portraits. Instead, they describe changes in image appearance that explain variation among the training faces.

Leading eigenfaces can reflect broad structures and contrasts, while later directions account for progressively smaller portions of the overall variation. Together, they define a coordinate system in which a face can be represented through a comparatively small collection of component scores.

### Covariance Eigen-Decomposition and Singular Value Decomposition

The assignment studies PCA through two mathematically related perspectives: eigen-decomposition and **Singular Value Decomposition (SVD)**. Comparing their principal directions and eigenvalues provides a valuable numerical consistency check.

For the first 20 principal components, the notebook reports:

| Consistency measure | Observed result |
|---|---:|
| Maximum relative eigenvalue error | 3.05 × 10⁻¹⁵ |
| Minimum absolute direction similarity | Approximately 1.0000 |

These results demonstrate near-perfect agreement for the compared leading components, within numerical precision. The absolute similarity measure appropriately treats sign-reversed principal directions as equivalent.

### Explained Variance and Component Selection

Explained variance measures how much of the training data's total variation is captured by a principal component. **Cumulative explained variance** indicates how much information is retained when several leading directions are considered together.

The notebook selects the smallest number of components that retains at least **90% of the training variance**.

<blockquote>

**Observed PCA result:** **98 principal components** retain **90.0581%** of the variance, reducing each image from **10,304 features to 98 component scores** — a **99.05% reduction in feature count**.

</blockquote>

This is a major reduction in representation size. It does not mean 99.05% of the image's meaningful content has been discarded, nor does retaining 90% variance guarantee that 90% of the identity-related information has been preserved. Explained variance and classification usefulness are related but distinct quantities.

### Image Reconstruction

To assess what is retained by the compressed representation, the notebook reconstructs test images from their PCA coordinates and compares them with the original faces.

| Reconstruction measure | Result |
|---|---:|
| PCA dimensions used | 98 |
| Training variance retained | 90.0581% |
| Test-image reconstruction MSE | 334.99 |

The reconstructed faces provide a visual assessment of how well the leading principal components reproduce facial structure. The remaining error reflects details and variation not captured by the selected PCA subspace.

Reconstruction quality is an important complement to explained variance: the latter summarizes variation in the training set, while the reported reconstruction error measures approximation on held-out test images.

**Part B takeaway:** PCA extracts a compact and interpretable basis for face images. The eigen-decomposition and SVD results agree numerically, and 98 components retain approximately 90% of training variance while supporting recognizable image reconstruction.

---

## Part C — Face Recognition in PCA Space

Part C evaluates whether the compressed face representations preserve sufficient information to distinguish individuals.

The central recognition experiment uses **one-nearest-neighbor (1-NN)** classification with Euclidean distance: each test face is assigned the identity of its closest training face. The notebook compares this strategy using original pixel features and the PCA-derived representation.

### Recognition Results

| Representation | Features per image | Test accuracy | Correct / 80 |
|---|---:|---:|---:|
| Original pixel representation (1-NN) | 10,304 | 2.50% | 2 |
| PCA representation (1-NN) | 98 | **96.25%** | **77** |

The PCA-based approach correctly identifies **77 of the 80 test faces**, with three misclassifications. This demonstrates that the selected components provide a highly effective representation for the recognition experiment.

### Interpreting the Raw-Pixel Baseline

The recorded raw-pixel accuracy of **2.50%** is unexpectedly low. This should **not** be interpreted as general evidence that nearest-neighbor classification using raw pixels necessarily fails on face-recognition data.

An important technical caveat is that the images originate as unsigned integer arrays. Direct subtraction and distance computation on unsigned pixel values can produce numerical wraparound, which may invalidate Euclidean distance calculations. The notebook's raw-pixel result should therefore be verified with a numerically safe floating-point distance calculation before drawing a strong conclusion about the superiority of PCA over the raw-pixel baseline.

Even without relying on that unusually low baseline, the **96.25% PCA 1-NN result** is a meaningful finding: a representation containing fewer than 1% of the original feature coordinates supports accurate identity recognition on the held-out dataset.

### Interpretation

PCA can benefit distance-based recognition by focusing comparisons on dominant patterns of image variation and reducing the influence of directions that contribute less to total variance. However, PCA is an **unsupervised** technique: its directions maximize variance, not necessarily separation between identities. As a result, the best dimensionality for reconstruction is not guaranteed to be the best dimensionality for recognition.

**Part C takeaway:** The compact eigenface representation performs strongly for nearest-neighbor face recognition. The raw-pixel baseline needs numerical verification before the size of the apparent performance gain can be treated as reliable.

---

## Part D — Classification, Cross-Validation, and Efficiency

Part D extends face recognition beyond the basic nearest-neighbor experiment. The notebook compares **K-nearest neighbors (KNN)** and an **RBF-kernel Support Vector Machine (SVM)**, then explores how their performance changes with the dimensionality of the PCA representation and the choice of model parameters.

### Initial Classifier Comparison

Both classifiers are initially evaluated using the same **98-component PCA representation**.

| Classifier | PCA components | Test accuracy |
|---|---:|---:|
| KNN (3 neighbors) | 98 | 95.00% |
| SVM (RBF kernel) | 98 | **98.75%** |

The RBF SVM achieves the higher initial accuracy. This is consistent with the idea that a nonlinear decision boundary can sometimes distinguish identities more effectively than a local majority vote among nearby examples. The finding applies to this specific split and configuration; it is not a general ranking of these classifier families.

### Cross-Validation and Model Selection

The next experiment investigates whether different PCA dimensions and classifier settings yield better results. The notebook considers **20, 50, 100, and 150 principal components**, evaluates several KNN neighbor counts and SVM regularization strengths, and compares configurations using **three-fold stratified cross-validation** on the training data.

The PCA representation is derived separately within each cross-validation training fold, reducing the risk of information leakage from the validation fold into feature extraction.

| Model | Selected components | Selected parameter | Mean CV accuracy |
|---|---:|---|---:|
| KNN | 50 | Neighbors = 1 | 94.37% |
| RBF SVM | 50 | C = 10 | **96.25%** |

Both selected configurations use **50 components**, fewer than the 98 needed to preserve 90% of variance in the earlier experiment. This illustrates an important distinction: maximizing retained variance is not the same as maximizing classification accuracy.

Cross-validation accuracy is a **model-selection estimate** from the training folds; it should not be confused with final test-set performance.

### Final Held-Out Evaluation

After model selection, the notebook reports the following results on the original 80-image test set.

| Method | Features | Test accuracy | Macro F1 |
|---|---:|---:|---:|
| Raw-pixel 1-NN | 10,304 | 2.50% | 0.002 |
| PCA 1-NN | 98 | 96.25% | 0.954 |
| Tuned KNN | 50 | **98.75%** | **0.987** |
| Tuned RBF SVM | 50 | **98.75%** | **0.987** |

The tuned KNN and tuned SVM each correctly classify **79 of 80 test images**. Their macro F1 scores are also high, indicating that their performance is not captured by overall accuracy alone.

Macro F1 gives equal weight to each identity when averaging the class-specific F1 scores. This is particularly relevant in a multiclass recognition problem containing 40 distinct identities.

Although the tuned methods achieve the same final test accuracy, that does not imply equivalent behavior across all data distributions. The test set is relatively small, and a difference of just one prediction changes accuracy by **1.25 percentage points**.

### Feature Reduction Versus Storage Savings

The final experiment considers how PCA affects the amount of data needed to represent the faces. The distinction between compact **per-image coefficients** and the **shared PCA basis** is essential.

| Storage measurement | Recorded size |
|---|---:|
| Original training images (8-bit representation) | 3.14 MiB |
| PCA coefficients for training images (32-bit floats) | 0.12 MiB |
| Shared PCA basis and mean face | 3.89 MiB |
| **Total stored PCA representation** | **4.01 MiB** |

The coefficient vectors are extremely compact relative to the original images. However, the basis vectors and mean face must also be stored if the full PCA representation or reconstruction is required. Under the recorded storage assumptions, **the total PCA representation is larger than the original training images**.

This outcome does not contradict the 99.05% reduction in feature count. Dimensionality reduction can lower the number of values used for classification while still introducing substantial shared-model storage overhead.

The balance could change for substantially larger collections, where the basis cost is spread across more images, or under different numerical precision and representation assumptions. The notebook does not benchmark runtime, so computational speedups should not be claimed as a measured result.

**Part D takeaway:** With cross-validation-based selection, both KNN and SVM achieve 98.75% test accuracy using just 50 PCA features per image. However, a compact feature vector does not automatically imply lower total storage when the PCA basis is included.

---

## Consolidated Findings

| Question | What the notebook demonstrates |
|---|---|
| Can face images be described with far fewer features? | Yes. A 98-component PCA representation retains 90.06% of training variance using 99.05% fewer feature coordinates. |
| Do eigen-decomposition and SVD agree? | Yes, for the compared leading components, to near machine precision. |
| Can compressed representations reconstruct recognizable faces? | The notebook visually compares reconstructions and reports a test MSE of 334.99. |
| Are eigenfaces useful for identity recognition? | PCA 1-NN achieves 96.25% test accuracy. |
| Does classifier selection affect recognition performance? | Yes. The initial RBF SVM outperforms the initial three-neighbor KNN. |
| Is the 90%-variance PCA dimension necessarily optimal for classification? | No. Cross-validation selects 50 components for both tuned classifiers. |
| Which evaluated models have the strongest final test performance? | Tuned KNN and tuned RBF SVM tie at 98.75% accuracy and 0.987 macro F1. |
| Does PCA necessarily reduce total storage? | No. In the recorded comparison, the shared basis makes total PCA storage larger than the original 8-bit training images. |

## Discussion

### Why the Assignment Matters

This notebook connects three often-separated areas of machine learning: **representation learning, geometric dimensionality reduction, and supervised classification**. Face recognition is especially well suited to this comparison because the original features have an intuitive visual meaning, while PCA features expose patterns that are not directly visible in the raw feature table.

The experiments show that a carefully selected representation can preserve substantial information and support strong recognition accuracy even after a drastic reduction in dimensionality. They also show that the right representation depends on the objective: reconstructing an image, classifying an identity, and minimizing storage are related but different problems.

### Limitations and Considerations

- **Dataset scale:** Results are based on 400 images from 40 identities, so generalization to larger and more diverse populations is not established.
- **Raw-pixel numerical issue:** The unusually low raw-pixel accuracy warrants re-evaluation using appropriate numeric types before making a definitive raw-versus-PCA comparison.
- **Limited held-out evidence:** The final test set has 80 images, making accuracy sensitive to individual misclassifications.
- **Unsupervised PCA objective:** PCA captures directions of greatest variation, which are not guaranteed to be the directions most discriminative for identity.
- **Reconstruction versus classification:** High explained variance does not automatically imply high recognition accuracy, and good recognition accuracy does not ensure perfect visual reconstruction.
- **Storage accounting:** The basis and mean must be considered alongside the component coefficients; the outcome depends on dataset size and representation precision.
- **Scope of validation:** Cross-validation supports model selection within the available training data, but the notebook does not report performance across independent datasets or real-world acquisition conditions.

### Potential Extensions

Future investigations could compare multiple variance thresholds against recognition performance, repeat evaluation over several stratified splits, report per-identity confusion patterns, examine sensitivity to lighting and pose, and verify distance-based recognition on correctly typed raw-pixel features. It would also be informative to measure actual training and prediction times and to test how the storage balance changes as the number of face images increases.

These are proposed extensions, not experiments claimed to have been completed in the supplied notebook.

---

## Conclusion

Lab Assignment 07 demonstrates how PCA and eigenfaces can transform high-dimensional facial images into compact, informative representations. Starting with **10,304 pixel values per image**, the notebook identifies **98 components retaining approximately 90% of training variance**, validates the agreement between two PCA formulations, and examines how reduced representations affect reconstruction and identity recognition.

The **PCA-based nearest-neighbor approach reaches 96.25% test accuracy**, while **cross-validated KNN and RBF SVM configurations each achieve 98.75% accuracy and 0.987 macro F1 with 50 components**. These findings support the practical value of low-dimensional features in the evaluated recognition task.

Equally important are the qualifications: the raw-pixel baseline requires numerical verification, retained variance should not be confused with identity information, and reduced dimensionality does not necessarily reduce total storage once the PCA basis is counted.

**The broader lesson is that effective machine learning requires evaluating representation quality, predictive performance, and resource costs separately rather than treating any single metric as a complete measure of success.**

---

*This README summarizes the objectives, experimental evidence, interpretations, and limitations of the accompanying notebook. Code-level implementation details are intentionally omitted.*
