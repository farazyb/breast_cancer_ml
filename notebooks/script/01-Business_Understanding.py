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

# ### Context and Problem Statement
# 
# Breast cancer diagnosis can be supported by data extracted from digitized images of Fine Needle Aspirate (FNA) samples. The provided dataset contains numerical features that describe characteristics of cell nuclei, such as radius, texture, and smoothness, together with a label indicating whether the case is malignant or benign.
# 
# The problem in this assignment is a supervised machine learning task: given a vector of extracted nucleus features for each case, build a model that classifies the case as malignant or benign. The objective is to produce a measurable and reproducible classification approach using the provided features, documented in a Jupyter notebook.
# 

# ### Business Objectives
# The objective of this project is to develop a machine learning model that classifies breast cancer cases as malignant or benign using numerical features extracted from digitized FNA images. Because FNA-based diagnosis requires considering many correlated features and can involve subjective judgement, a data-driven model can support more consistent and measurable decision-making. The scope is limited to classical machine learning methods on the provided dataset.

# ### Business Success Criteria
# - A model can classify each case as **malignant** or **benign** using the provided FNA-derived features.
# - The results are reported transparently, including a **confusion matrix** and key metrics, so **false negatives and false positives** are explicitly visible.
# - A clear, measurable performance target is defined and met, with emphasis on **detecting malignant cases** (e.g., high recall for the malignant class), while still maintaining reasonable overall performance.
# - The full workflow is **reproducible** in a single Jupyter notebook, with documented assumptions, decisions, and results.

# 
