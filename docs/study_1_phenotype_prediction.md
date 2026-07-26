# Study 1: Phenotypic Machine Learning Prediction of ADHD Diagnosis

## Overview

This study investigates whether machine learning models can predict ADHD diagnosis from phenotypic characteristics available in the ADHD-200 Consortium dataset.

The goal was not to replace clinical diagnosis, but to investigate what information contained within the dataset contributes most strongly to ADHD classification.

---

# Dataset

## ADHD-200 Consortium Phenotypic Dataset

Source:
ADHD-200 Consortium

Dataset size:

- Original participants: 973
- Clean binary classification dataset: 797

Class distribution:

| Class | Participants |
|---|---:|
| No ADHD | 585 |
| ADHD | 212 |

Target variable:

ADHD


Binary encoding:


0 = No ADHD
1 = ADHD


---

# Feature Groups

Two experiments were performed.

---

# Experiment 1

## Full Phenotypic Model

Features:

### Demographic

- Age
- Gender
- Handedness


### Cognitive

- Verbal IQ
- Performance IQ
- Full Scale IQ


### ADHD Symptom Variables

- ADHD Measure
- ADHD Index
- Inattentive symptoms
- Hyperactive/Impulsive symptoms


Question:

> Can machine learning predict ADHD diagnosis using all available phenotypic information?

---

# Experiment 2

## Discovery Model

Symptom variables were removed.

Remaining features:

- Age
- Gender
- Handedness
- Verbal IQ
- Performance IQ
- Full Scale IQ


Question:

> Can ADHD diagnosis be predicted from general demographic and cognitive variables alone?

---

# Machine Learning Models

Four classifiers were evaluated:

1. Logistic Regression

2. Support Vector Machine

3. Random Forest

4. XGBoost


Evaluation method:

- Stratified 5-fold cross validation
- ROC-AUC metric
- Independent test evaluation

---

# Results

## Experiment 1: Full Phenotypic Model

## Model Performance

| Model | ROC-AUC |
|-|-:|
| XGBoost | 0.970 |
| Random Forest | 0.950 |
| SVM | 0.902 |
| Logistic Regression | 0.737 |


Best model:


XGBoost


Cross-validation performance:


Mean ROC-AUC:
0.9698

Standard deviation:
0.012


---

# Experiment 2: Discovery Model

XGBoost without symptom variables:

Features:


Age
Gender
Handedness
IQ measurements


Performance:


Mean ROC-AUC:
0.7248

Test ROC-AUC:
0.7295


---

# Feature Importance

The strongest predictors in the full model were:

1. Hyper/Impulsive symptoms

2. ADHD Index

3. Inattentive symptoms

4. Age

5. Full Scale IQ


This indicates that behavioral symptom measurements contain the majority of predictive information.

---

# Interpretation

The large performance difference between the two experiments demonstrates that:

- ADHD symptom variables provide strong predictive signal.
- Demographic and cognitive variables alone have limited ability to distinguish ADHD status.
- Machine learning successfully captures patterns already present in behavioral assessments.

The model appears to learn clinical phenotype representation rather than a purely demographic or cognitive signature.

---

# Limitations

## Dataset limitations

- Multi-site dataset
- Potential site-specific effects
- Missing values in several variables


## Clinical limitations

Machine learning predictions should not be interpreted as diagnosis.

The model learns statistical associations within this dataset and does not replace clinical assessment.

---

# Conclusion

Machine learning models can accurately predict ADHD diagnosis from ADHD-200 phenotypic data when symptom-derived variables are included.

XGBoost achieved a ROC-AUC of approximately 0.97, demonstrating strong predictive performance.

However, removing symptom information reduced performance to approximately 0.72 ROC-AUC, suggesting that the majority of predictive information originates from behavioral symptom measures rather than demographic or cognitive characteristics.

These findings establish a baseline for future investigations using neuroimaging data from the ADHD-200 dataset.