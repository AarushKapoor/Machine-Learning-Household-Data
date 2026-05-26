# Household Survey ML Analysis

**Math 4050 — Course Project**  
Aarush Kapoor · March 2026

---

## Overview

This project applies five machine learning techniques to a household survey dataset of 8,000 observations. The dataset captures annual household spending across 12 categories alongside demographic and housing characteristics (household size, building materials, asset ownership, urban/rural location). Each technique targets a different response variable and includes a tuning step to manage the bias-variance tradeoff. All models are benchmarked against a null model baseline on held-out test data.

---

## Dataset

| Property | Value |
|---|---|
| Total observations | 8,000 |
| Numerical variables | 15 |
| Categorical variables | 16 |
| Training set | 6,000 (75%) |
| Test set | 2,000 (25%) |
| Random seed | 123 |

Spending variables are labeled `exp_01` through `exp_12`. Categorical variables include `urbrur`, `floor`, `walls`, `roof`, `electricity`, `cook_fuel`, `phone`, `cell`, `car`, `bicycle`, `motorcycle`, `refrigerator`, `tv`, `radio`, `bank`, and `statocc`.

---

## Methods & Results

### 1. Linear Regression — Healthcare Spending (`exp_06`)

Forward stepwise selection with BIC identified 20 predictors.

| Metric | Value |
|---|---|
| R² | 0.899 |
| Test MSE | 5,379 |
| Null Model MSE | 58,193 |
| Improvement | 90.8% reduction |

Key predictors: urban location (+$54), household size (+$7.70/person), electricity access (+$75), communication spending (`exp_08`).

---

### 2. Decision Tree — Bank Account Ownership (`bank`)

Cost-complexity pruning via cross-validation; full 10-node tree was already optimal.

| Metric | Value |
|---|---|
| Test Error Rate | 10.25% |
| Null Model Error Rate | 42.7% |
| Correct predictions | 1,795 / 2,000 |

Root split: communication spending (`exp_08`). Additional splits use transport spending, radio ownership, and miscellaneous spending.

---

### 3. Random Forest — Transport Spending (`exp_07`)

OOB error used to tune `mtry` over values 1–10; best `mtry = 10`, `ntree = 200`.

| Metric | Value |
|---|---|
| Test MSE | 80,130 |
| Null Model MSE | 2,130,154 |
| Improvement | 96.2% reduction |

Top predictors by `%IncMSE`: car ownership, alcohol/tobacco spending (`exp_02`), miscellaneous spending (`exp_12`).

---

### 4. K-Nearest Neighbors — Urban vs. Rural (`urbrur`)

Run on 15 numerical spending variables only (KNN requires numeric input). `k` tuned over {1, 3, 5, 7, 10, 15, 20, 25, 30}; best `k = 5`.

| Metric | Value |
|---|---|
| Test Error Rate | 16.7% |
| Null Model Error Rate | 44.8% |
| Correct predictions | 1,667 / 2,000 |

---

### 5. Logistic Regression — Car Ownership (`car`)

Model fit using selected predictors; 0.5 probability threshold for classification.

| Metric | Value |
|---|---|
| Test Error Rate | 1.55% |
| Null Model Error Rate | 25.65% |
| Correct predictions | 1,969 / 2,000 |

Strongest predictors: transport spending (`exp_07`, p < 2e-16), bank account ownership (coef = +2.169), urban location (coef = −1.506).

---

## Summary

| Technique | Target | Test Performance | Null Baseline | Improvement |
|---|---|---|---|---|
| Linear Regression | `exp_06` (Healthcare) | MSE = 5,379 | MSE = 58,193 | 90.8% ↓ |
| Decision Tree | `bank` | Error = 10.25% | Error = 42.7% | 32.5 pp ↓ |
| Random Forest | `exp_07` (Transport) | MSE = 80,130 | MSE = 2,130,154 | 96.2% ↓ |
| KNN (k=5) | `urbrur` | Error = 16.7% | Error = 44.8% | 28.1 pp ↓ |
| Logistic Regression | `car` | Error = 1.55% | Error = 25.65% | 24.1 pp ↓ |

Every model substantially outperformed its null baseline, with logistic regression and random forest delivering the strongest results.

---

## Files

```
├── Math4050-CourseProject-Kapoor.Rmd   # Full analysis with code and write-up
├── Math4050-CourseProject-Kapoor.pdf   # Rendered report
└── projectData.csv                     # Source dataset (not included in repo)
```

---

## Requirements

R packages used:

- `leaps` — forward stepwise selection
- `tree` — decision tree and cross-validation pruning
- `randomForest` — random forest and variable importance
- `class` — KNN classification

Install all at once:

```r
install.packages(c("leaps", "tree", "randomForest", "class"))
```

---

## Usage

Open `Math4050-CourseProject-Kapoor.Rmd` in RStudio and place `projectData.csv` in the same working directory. Knit to HTML or PDF to reproduce all results. The random seed is set to `123` throughout for reproducibility.
