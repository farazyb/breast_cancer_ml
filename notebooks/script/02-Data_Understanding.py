#!/usr/bin/env python
# coding: utf-8

# # Data Understanding
# In this phase, we focus on acquiring the data, understanding its structure, and verifying its quality. This foundation is crucial before we move on to preprocessing or modeling.

# ## Collecting Initial Data
# - The dataset used in this project originates from the University of Wisconsin Hospitals, Madison. It was created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian.
# - Dataset Specifics:
#     - Format: CSV (Comma Separated Values)
#     - File Name: dataset.csv
#     - Target Variable: Diagnosis (Benign vs. Malignant)

# ## Downloading Data

from modules.dataset import load, download_dataset2
URL="https://drive.google.com/file/d/1MaNL7FS7rpX4GLxgGnXX0iD72kb1PR61/view?usp=sharing"
download_dataset2(URL)


# ## Loading and have an overview

# load the data from breast_cancer/data/raw/
df = load()
# show for sample from the Head 
df.head()


#load some sample from the end of data set
df.tail()


# ## Describing Data

# This section examines the structure, data types, and basic statistics of the dataset.

# - Impoerting necessary library for understanding data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# * config out notebook for good visualisation

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Dataset Shape and Info

print(f"Dataset Shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")


df.info()


# Findings:
# - Every column shows “569 non-null”:
#     - That means 0 missing values in all 32 columns.
#     - So I don’t need dropna() or imputation.
# - Data types:
#     - I have three dtype groups:
#         - float64(30) → numeric features
#             - 30 columns are continuous numeric measurements (radius, texture, area, etc.).
#         - int64(1) → target
#             - 0/1 (benign vs malignant).
# - Memory:
#     - The whole dataset uses ~138 KB

# ### Statistical Summary

df.describe()


# ### Result
# - For each column:
#   - count: number of non-missing rows (all are 569 → no missing)
#   - mean: average value
#   - std: spread (standard deviation)
#   - min / max: smallest / largest
#   - 25% / 50% / 75%: quartiles (median is 50%)
# 
# - Describing 31 feature is kind of hard. but by looking very fastly, we can understand that the scale of the data are not same, which means if we use data whitout standardizing them can make some problem in our predections.
#     - Examples of scale mismatch
#         - mean area:
#             mean ≈ 654.9, max ≈ 2501
#         - mean smoothness:
#             mean ≈ 0.096, max ≈ 0.163
#         - perimeter error:
#             mean ≈ 2.87, max ≈ 21.98
#         - area error:
#             mean ≈ 40.3, max ≈ 542.2
# ---

# ## Exploring Data 
# In this section, we explore the data to understand relationships and distributions. We will start by looking at how features relate to each other to identify redundancy, and then dive into specific feature-target relationships.

# ### Target Variable Distribution

# Create readable labels
df['diagnosis_label'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

# Plot class distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='diagnosis_label', hue='diagnosis_label', palette='viridis', legend=False, ax=ax)
ax.set_title('Class Distribution')
ax.set_xlabel('Diagnosis')
ax.set_ylabel('Count')

# Add counts and percentages
counts = df['diagnosis_label'].value_counts()
pct = df['diagnosis_label'].value_counts(normalize=True) * 100
text = f"Malignant: {counts['Malignant']} ({pct['Malignant']:.1f}%)\nBenign: {counts['Benign']} ({pct['Benign']:.1f}%)"
ax.text(0.4, 0.95, text, transform=ax.transAxes, va='top', ha='right', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='gray'))

plt.tight_layout()
plt.show()


# The dataset has class imbalance: 37.3% Malignant (212 samples) vs 62.7% Benign (357 samples). Stratified sampling should be used during train/test split.

# ### Feature Correlation Analysis

numeric_df = df.drop(['diagnosis_label', 'target'], axis=1)
corr_matrix = numeric_df.corr()

# Clustered (sorted) correlation heatmap
sns.clustermap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    figsize=(18, 18)
)
plt.show()


# Observation:
# The correlation heatmap shows two dominant redundant clusters and a set of independent features that add genuinely new information.
# 1. Shape irregularity cluster
#     - Top-left block:
#         * mean concavity, mean concave points, worst concave points, mean compactness, worst compactness, worst concavity
#         * These are strongly positively correlated with each other (deep red). They measure related “boundary irregularity”.
# 2. Size cluster
#     - Large central red block:
#         * mean radius, mean perimeter, mean area, plus their worst radius, worst perimeter, worst area
#         * These are almost the same information (very high correlations). This is the classic multicollinearity “size” group.
# 3. Size error cluster
#     - Small red block near the size group:
#         * area error, radius error, perimeter error
#         * These errors correlate strongly with each other, and moderately with size (bigger tumors tend to have bigger measurement errors).
# 4. Texture subcluster
#     - A tight 2×2 red block:
#         * mean texture ↔ worst texture
#         * Very strong mutual correlation, but comparatively weaker ties to the size/shape blocks.
# 5. Smoothness / symmetry / fractal dimension group
#     - Mid-lower area shows moderate correlations among:
#         * mean smoothness, worst smoothness, mean symmetry, worst symmetry, worst fractal dimension 
#         * They form a looser cluster (not as redundant as size).    
# 6. Concavity/compactness error cluster
#     - Bottom-right strong block:
#         * concave points error, compactness error, concavity error
#         * These error-features move together.

# ### Feature Correlation with Target

feature_cols = df.select_dtypes(include="number").columns.drop("target")
# Correlation with target
corr_target = df[feature_cols].corrwith(df['target']).sort_values(key=lambda x: abs(x), ascending=False)
# Plot top 10
plt.figure(figsize=(10, 5))
corr_target.head(10).plot(kind='bar')
plt.title('Top 10 Features by Correlation with Target')
plt.xlabel('Feature')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Observation:
# - The top 10 features most correlated with the target are all negatively correlated, meaning higher values of these features are associated with malignant tumors (target=0).
# - The strongest correlations come from two groups:
#   - Concavity/shape features: worst concave points,mean concave points, worst concavity, mean concavity — irregular, indented cell boundaries are strong indicators of malignancy.
#   - Size features: worst perimeter, worst radius, worst area, mean perimeter, mean radius, mean area — larger tumors tend to be malignant.
# - The "worst" (largest/most extreme) measurements generally correlate more strongly than the "mean" measurements, suggesting that the most extreme cells in a sample are more diagnostically informative.
# - This aligns with the earlier feature-feature correlation heatmap, where these same features formed dominant clusters (size cluster and shape irregularity cluster).
# - These top correlated features will be strong candidates for feature selection in the model

# ### Feature Distributions by Class

import matplotlib.gridspec as gridspec
# Select top 10 features by correlation
top_features = corr_target.head(10).index.tolist()
top_features.append("texture error")
top_features.append("symmetry error")
n = len(top_features)
print(n)
cols = 2
rows = (n + cols - 1) // cols

# Plot distributions
fig = plt.figure(figsize=(16, 5 * rows))
# axes = axes.flatten()


for i, feature in enumerate(top_features):
    # Create a 2-row sub-grid per feature: boxplot (1/4 height) + histogram (3/4 height)
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=fig.add_gridspec(rows, cols)[i // cols, i % cols],
        height_ratios=[1, 3], hspace=0.05
    )

    ax_box = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax_box)

    # Boxplot on top
    sns.boxplot(data=df, x=feature, hue='diagnosis_label', ax=ax_box, palette='viridis', legend=False)
    ax_box.set(xlabel='', ylabel='')
    ax_box.tick_params(labelbottom=False)
    ax_box.set_title(f'Distribution: {feature}')

    # Histogram + KDE below
    sns.histplot(data=df, x=feature, hue='diagnosis_label', kde=True, element='step', ax=ax_hist, palette='viridis')
    ax_hist.set_ylabel('Count')

plt.tight_layout()
plt.show()


# ### Observation: Combined Distribution & Boxplot Analysis (Top 10 Features)
# 
# The combined histogram + boxplot visualization for the top 10 features most correlated with the target reveals:
# 
# 1. Clear Class Separation (Strong Predictors)
#    - worst concave points, mean concave points, worst perimeter, worst radius, worst area: The Malignant and Benign distributions are visibly shifted apart with minimal overlap. The boxplots confirm non-overlapping IQRs, making these the most discriminative features.
# 
# 2. Moderate Separation
#    - mean perimeter, mean radius, mean area, mean concavity, worst concavity: Distributions overlap partially, but the medians (visible in boxplots) are clearly displaced. These features still carry useful predictive signal.
# 
# 3. Skewness & Outliers
#    - The boxplots expose right-skewed distributions in area-based features (mean area, worst area), with several high-value outliers in the Malignant class. This suggests that extreme tumor sizes are strongly indicative of malignancy.
# 
# 4. Distribution Shape
#    - Malignant cases tend to have wider, flatter distributions (higher variance), while Benign cases are more tightly concentrated around lower values. This pattern is consistent across nearly all top features.
# 5. I intentionally added "texture error" and "symmetry error"  to show the reader that these two mostly are not able to depart the Malignant and Benign.
# 
# 
# 6. Note
#    - The "worst" (largest cell) measurements consistently show better class separation than the "mean" measurements, confirming 

# ### Outlier Analysis

# Count outliers using IQR method
outlier_counts = {}
for col in feature_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()

# Display features with most outliers
outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)
print("Features with most outliers:")
print(outlier_series.head(10))


# Outliers are present in error-related features. In medical data, outliers often represent clinically significant cases and should not be automatically removed. Use robust scaling techniques instead.

# ## Verify Data Quality

# 1. Compute the Metrics
features = df.columns.drop(["target","diagnosis_label"]) # Assuming 'target' is the label column

# a. Missing Values
total_missing = df.isna().sum().sum()

# b. Duplicates
duplicate_rows = df.duplicated().sum()

# c. Unnamed Columns (often artifacts from saving CSVs without index=False)
unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]

# d. Negative Values (Physical measurements like Radius shouldn't be negative)
# We check if ANY value in the feature columns is less than 0
has_negative_values = (df[features] < 0).any().any()

# e. Zero Variance (Columns that have the same value for every single row)
variances = df[features].var()
zero_variance_cols = variances[variances == 0].index.tolist()

# 2. Prepare Data for the Table
# We create a list of results with a "Pass/Fail" logic
quality_report = [
    ["Missing Values", total_missing, "Pass" if total_missing == 0 else "Fail"],
    ["Duplicate Rows", duplicate_rows, "Pass" if duplicate_rows == 0 else "Fail"],
    ["Unnamed Columns", len(unnamed_cols), "Pass" if len(unnamed_cols) == 0 else "Fail"],
    ["Negative Values", "Yes" if has_negative_values else "No", "Pass" if not has_negative_values else "Fail"],
    ["Zero-Variance Features", len(zero_variance_cols), "Pass" if len(zero_variance_cols) == 0 else "Fail"]
]

# Create a DataFrame for the plot
report_df = pd.DataFrame(quality_report, columns=["Quality Check", "Result", "Verdict"])

# 3. Plot the Table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=report_df.values, 
                 colLabels=report_df.columns, 
                 loc='center', 
                 cellLoc='center',
                 colColours=["#f2f2f2"] * 3) # Grey header background

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8) # Adjust row height

# Color code the 'Verdict' column
# (Rows are numbered 1 to 5, Column 2 is 'Verdict')
for i in range(len(quality_report)):
    cell = table[(i + 1, 2)] 
    if quality_report[i][2] == "Pass":
        cell.set_facecolor("#d9f7be") # Green for Pass
    else:
        cell.set_facecolor("#ffccc7") # Red for Fail

plt.title("Data Quality Sanity Check", fontsize=14, y=0.98)
plt.show()


# Conclusion: The dataset is clean with no missing values, duplicates, or invalid entries.
