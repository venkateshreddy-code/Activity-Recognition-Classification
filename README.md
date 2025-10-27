# 🧠 Activity Recognition Classification

This project explores, preprocesses, and classifies **sensor-based activity data** using multiple machine learning models.  
It demonstrates the full data science workflow — from data cleaning and feature engineering to dimensionality reduction, feature selection, and model evaluation.

---

## 📋 Objective

To analyze a real-world **Activity Recognition Dataset** and evaluate the performance of different classifiers under:
- Dimensionality Reduction (PCA, Truncated SVD)
- Feature Selection (Variance Threshold, Correlation Filtering, Chi-Square, Mutual Information, Sequential Feature Selection)
- Custom Bayesian Classifier Implementation

---

## 📂 Dataset

**File:** `Activity_data.csv`  
**Samples:** 75,128  
**Features:** 9 columns including:
- Time  
- Tri-axial accelerations (`frontal_axis_Acceleration`, `vertical_axis+_Acceleration`, `lateral_axis_Acceleration`)  
- Signal features (`RSSI`, `Phase`, `Frequency`, `Antenna_ID_Sensor`)  
- **Target label:** `Activity`

Each record represents sensor readings captured during different physical activities.

---

## ⚙️ Data Preprocessing

- Loaded dataset using `pandas`
- Checked and handled missing values
- Standardized numeric features using `StandardScaler`
- Encoded categorical columns (if any)
- Split dataset into **Training (80%)** and **Testing (20%)**

---

## 🧮 Models Implemented

### 1️⃣ **Bayesian Classifier (from scratch)**
- Estimated priors and Gaussian likelihoods per class
- Used Bayes’ theorem to compute posterior probabilities
- Predicted activities based on maximum posterior probability

### 2️⃣ **Scikit-learn Classifiers**
| Model | Description |
|--------|--------------|
| **SVM (Linear, RBF, Polynomial)** | Support Vector Machines with different kernels |
| **KNN** | Tested multiple k values, selected best k |
| **Naive Bayes (GaussianNB)** | Probabilistic classifier assuming feature independence |
| **Decision Tree** | Recursive partitioning using Gini criterion |
| **Random Forest** | Ensemble of 200 trees, class-weight balanced |
| **Voting Ensemble** | Combines diverse classifiers using soft voting |
| **Stacking Ensemble** | Meta-classifier (Logistic Regression) on top of base models |

---

## 🔍 Dimensionality Reduction

### **PCA (Principal Component Analysis)**
- Retained components explaining **95% variance**
- Visualized first two principal components
- Compared Random Forest results before and after PCA

### **Truncated SVD**
- Applied on standardized data
- Compared results and explained variance with PCA

---

## 🎯 Feature Selection (FS)

Applied and compared five FS methods:
| Method | Description |
|---------|-------------|
| **Variance Thresholding** | Removes low-variance features |
| **Correlation Filtering** | Drops highly correlated (|ρ| > 0.9) features |
| **Chi-Square (χ²)** | Selects top features most related to class labels |
| **Mutual Information** | Ranks features by information gain |
| **Sequential FS (SFS/SBS)** | Selects optimal subset iteratively with Random Forest |

---

## 📊 Evaluation Metrics

All models evaluated using:
- **Accuracy**
- **Precision (Macro)**
- **Recall (Macro)**
- **F1-Score (Macro)**
- **Confusion Matrices (Seaborn Heatmaps)**

Example:
```text
Accuracy : 0.9892
Precision: 0.9760
Recall   : 0.9340
F1-Score : 0.9515
