#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer
# ## description:
# The Breast Cancer dataset [1] is a widely used dataset for learning and practicing machine learning techniques. It contains diagnostic data for breast cancer cases, including features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei, such as radius, texture, and smoothness, and are compiled into a convenient dataset.
# ## Goal:
# Develop a machine learning model to accurately classify breast cancer cases as malignant or benign.
# ## Citations:
# [1] Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE International Symposium on Electronic Imaging: Science and Technology. Retrieved from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# ## Business Understanding

# ### Business Background
# 
# Breast cancer diagnosis can be supported by data extracted from digitized images of Fine Needle Aspirate (FNA) samples. The provided dataset contains numerical features that describe characteristics of cell nuclei, such as radius, texture, and smoothness, together with a label indicating whether the case is malignant or benign.
# 
# stakeholders may include clinicians/pathologists who need consistent decision support, patients who benefit from timely and clear outcomes, and the healthcare organization that wants a reproducible process.
# 
# The problem in this assignment is a supervised machine learning task: given a vector of extracted nucleus features for each case, build a model that classifies the case as malignant or benign. The objective is to produce a measurable and reproducible classification approach using the provided features, documented in a Jupyter notebook.
# 

# ### Business Objectives
# The objective of this project is to develop a machine learning model that classifies breast cancer cases as malignant or benign using numerical features extracted from digitized FNA images. Because FNA-based diagnosis requires considering many correlated features and can involve subjective judgement, a data-driven model can support more consistent and measurable decision-making. The scope is limited to classical machine learning methods on the provided dataset.

# ### Business Success Criteria
# - A model can classify each case as malignant or benign using the provided FNA-derived features.
# - The results are reported transparently, including a confusion matrix and key metrics, so false negatives and false positives are explicitly visible.
# - A clear, measurable performance target is defined and met, with emphasis on detecting malignant cases (e.g., high recall for the malignant class), while still maintaining reasonable overall performance.
# - The full workflow is reproducible in a single Jupyter notebook, with documented assumptions, decisions, and results.

# ### Assessing the situation:
# #### Resource Inventory
# - Data: Breast Cancer (Diagnostic) dataset containing FNA-derived numerical features and labels.
# - Tools: Jupyter Notebook and standard ML libraries
# - People: single student developer/analyst (Faraz Yazdanibiuki)
# - Time: limited by course schedule and assignment deadline. 
# #### Requirements
# - Provide a working classifier and a clear evaluation report in the notebook.
# - Provide clear documentation in Markdown for each CRISP-DM phase.
# ####  Assumptions
# - Dataset labels are correct and can be treated as ground truth for supervised learning.
# - The extracted features are sufficient to support the classification task.
# - The model is used as decision support, not as a standalone medical diagnosis.
# ####  Constraints
# - Deep learning is not allowed
# - Deployment is out of scope
# - The solution must be deliverable as a reproducible Jupyter notebook.
# #### Risks and Contingencies
# - Risk: performance is unstable across splits (overfitting / variance).
#     Mitigation: use cross-validation and report variance, prefer simpler models if needed.
# - Risk: class imbalance affects learning and metric interpretation.
#     Mitigation: use stratified splitting and include metrics beyond accuracy.
# - Risk: results are hard to interpret or justify.
#     Mitigation: include feature relevance analysis and explain evaluation choices.
# 
# #### Terminology
# This glossary aligns terminology across stakeholders (domain context) and the data mining workflow.
# 
# | Term | Meaning in this project | Why it matters |
# |---|---|---|
# | FNA (Fine Needle Aspiration) | A sampling procedure where a small tissue/cell sample is taken from a breast mass and digitized for analysis | Defines the data source and the scope of what the model can learn |
# | Case / Sample / Record | One patient case represented as one row in the dataset | Clarifies unit of prediction and evaluation |
# | Feature | A numeric variable describing nucleus characteristics (e.g., radius, texture, smoothness) | Input to the model; confusion here breaks interpretation |
# | Label / Class | Ground-truth category: malignant or benign | Target variable the model predicts |
# | Classification | Predicting malignant vs benign from features | The core task (not regression, not clustering) |
# | Malignant (Positive class) | The class treated as “positive” for metrics (unless stated otherwise) | Metric definitions (recall/precision) depend on this |
# | Confusion Matrix | Table of TP, FP, TN, FN counts | Makes the error types explicit, not hidden behind accuracy |
# | True Positive (TP) | Malignant case predicted as malignant | Correct detection of malignant cases |
# | False Negative (FN) | Malignant case predicted as benign | Critical error type in this domain context |
# | False Positive (FP) | Benign case predicted as malignant | Leads to unnecessary follow-up actions |
# | True Negative (TN) | Benign case predicted as benign | Needed to compute specificity and accuracy |
# | Accuracy | (TP + TN) / All cases | Can look good even when FN is high; not sufficient alone |
# | Precision (Malignant) | TP / (TP + FP) | How many predicted malignant are truly malignant |
# | Recall / Sensitivity (Malignant) | TP / (TP + FN) | How many malignant cases are correctly detected |
# | Specificity (Benign) | TN / (TN + FP) | How many benign cases are correctly recognized |
# | F1-score | Harmonic mean of precision and recall | Useful when balancing FP and FN matters |
# | Threshold / Decision boundary | The cutoff that converts a probability/score into a class label | Moving it changes sensitivity vs specificity trade-off |
# | Generalization | Performance on unseen cases (not the training data) | The real goal; avoids overfitting |
# | Overfitting | Model performs well on training but poorly on unseen data | Causes misleading “high scores” |
# | Cross-validation (k-fold) | Evaluation method that trains/tests on multiple splits | Provides a more stable estimate of generalization |
# | Feature scaling | Standardizing feature ranges (e.g., z-score) | Some models (SVM, kNN) can fail without it |
# | Feature selection | Choosing a subset of features | Can reduce complexity and improve stability |
# | Baseline model | A simple reference model to compare against | Prevents claiming improvement without a benchmark |
# | mean  | Average value of a feature across all measured cell nuclei in a sample | For each feature (e.g., radius), calculate the mean over all nuclei measurements <br> Captures the overall/typical characteristics of the tumor cells. Good baseline signal for many models                               |
# | worst | “Most extreme” value of a feature in a sample (not just a simple max)  | Typically the mean of the largest values (often top ~3 nuclei) for that feature <br> Highlights abnormal extremes. Malignancy often shows up in the most irregular / largest nuclei even if the average looks normal |
# | error | Variation (spread) of a feature within a sample, not measurement error | Usually the standard deviation of that feature across nuclei in the same sample <br> Measures heterogeneity inside the tumor. Malignant tumors are often less uniform, so higher variation can be informative        |
# 
# 
# #### Cost/Benefit Analysis
# - Cost: Because, this is a study case, if i can not get the enough grade based on rubric, i gonna fail and then retake. Time and losing oppurtunity for graduating on time. 
# - Benefit: Learning new concepts, applyig on real problem, passing Q3

# ### Determining Data Mining Goals
# #### Data Mining Goals
# - Build a binary classification model that predicts the label (malignant vs benign) from the provided numerical nucleus features.
# #### Data Mining Problem Type
# This is a supervised binary classification problem:
# - Input: numerical features extracted from digitized FNA images of a breast mass (nucleus measurements).
# - Output: a predicted class label: malignant or benign (optionally with a probability/score).
# #### Data Mining Success Criteria
# To translate the business goal of "reliable diagnosis" into technical terms, the model must meet specific performance benchmarks derived from the domain and the original study by Street et al.:
# 
# 1.  Coverage:
#     * Produce a valid prediction score for 100% of the cases in the test set.
# 
# 2.  Accuracy (Benchmark):
#     * Achieve a classification accuracy comparable to the state-of-the-art benchmark of 97% established by Street et al. (1993) on this dataset.
# 
# 3.  Sensitivity (Recall):
#     * Primary Metric: Maximize Recall for the Malignant class.
#     * Target: Achieve a Recall of > 0.95, ensuring that fewer than 5% of actual cancer cases are missed (minimizing False Negatives).
# 
# 4.  Parsimony & Interpretability:
#     * Identify the minimal subset of features required to maintain high performance. Medical practitioners prefer simpler rules over complex "black box" models.
# 
# 

# ### Produce Project Plan
# 
# ### Project Timeline
# This project follows the FDS module timeline (5 weeks), structured around the CRISP-DM phases:
# 
# * Phase 1: Understanding (Weeks 3.1 - 3.3)
#     * Define business and data mining goals (Current Phase).
#     * Perform Exploratory Data Analysis (EDA) to understand feature distributions.
#     * Identify data quality issues (missing values, outliers).
# 
# * Phase 2: Preparation (Week 3.3)
#     * Clean data and handle any irregularities.
#     * Perform feature engineering or selection (aiming for the "parsimony" goal).
#     * Split data into Train/Test sets using Stratified Sampling.
# 
# * Phase 3: Modeling & Evaluation (Weeks 3.3 - 3.5)
#     * Train candidate models (Logistic Regression, SVM, Random Forest).
#     * Evaluate using Cross-Validation (k-fold).
#     * Compare performance against the success criteria (Recall > 0.95, Accuracy ~97%).
# 
# * Final Deadline: March 9, 2025 (Hand-in summative deliverable).
# 
# #### Assessment of Tools & Techniques
# * Environment: Python 3.x , Anaconda(Jupyter Notebook).
# * Key Libraries:
#     * `pandas` & `numpy` for data manipulation.
#     * `matplotlib` & `seaborn` for visualization.
#     * `scikit-learn` for modeling and evaluation metrics.
# * Technique Constraints: Deep Learning (Neural Networks) is explicitly excluded from the scope of this assignment.
