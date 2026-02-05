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
# 
# 
# #### Cost/Benefit Analysis
# - Cost: Because, this is a study case, if i can not get the enough grade based on rubric, i gonna fail and then retake. Time and losing oppurtunity for graduating on time. 
# - Benefit: Learning new concepts, applyig on real problem, passing Q3

# ### Determining Data Mining Goals
# #### Data Mining Goals
# - Build a binary classification model that predicts the label (malignant vs benign) from the provided numerical nucleus features.
# ### Data Mining Problem Type
# This is a supervised binary classification problem:
# - Input: numerical features extracted from digitized FNA images of a breast mass (nucleus measurements).
# - Output: a predicted class label: malignant or benign (optionally with a probability/score).
# ### Technical Data Mining Goals
# The technical solution should:
# 
# 1. Build a classifier that predicts malignant vs benign for 100% of cases in the dataset (each record receives a label).
# 2. Produce a prediction score (probability or decision score) for each case so that the decision boundary can be adjusted if needed.
# 3. Support stable generalization by using a validation strategy that estimates performance on unseen data (e.g., stratified k-fold cross-validation).
# 4. Enable interpretability at feature level by reporting which features contribute most to the classification (model coefficients or feature importance).
# 
# 
