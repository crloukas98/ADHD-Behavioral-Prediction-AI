# ADHD Behavioral Prediction AI

Machine Learning Research Project for ADHD Prediction Using Phenotypic and Neuroimaging Data

---

## Overview

This project investigates whether artificial intelligence and machine learning can identify patterns associated with Attention-Deficit/Hyperactivity Disorder (ADHD) using publicly available research datasets.

The project is structured as a series of reproducible studies, beginning with phenotypic prediction and progressing toward computational neuroscience approaches using neuroimaging data.

The long-term goal is to explore how AI can assist understanding of ADHD through:

- behavioral data analysis
- machine learning
- explainable AI
- neuroimaging analysis
- computational neuroscience

---

# Current Study

# Study 1: Phenotypic ADHD Prediction

## Research Question

> Can machine learning models predict ADHD diagnosis from clinical and cognitive characteristics contained within the ADHD-200 phenotypic dataset?

---

# Dataset

## ADHD-200 Consortium Phenotypic Dataset

Dataset characteristics:


Original participants:
973

Processed binary classification dataset:
797


Class distribution:

| Class | Participants |
|---|---:|
| No ADHD | 585 |
| ADHD | 212 |

Target:


ADHD diagnosis


Encoding:


0 = No ADHD
1 = ADHD


---

# Methodology

The dataset was processed using a reproducible machine learning pipeline.

Steps:


Raw dataset
|
↓
Data cleaning
|
↓
Missing value handling
|
↓
Feature preprocessing
|
↓
Model training
|
↓
Cross-validation
|
↓
Performance evaluation


---

# Models Evaluated

The following machine learning algorithms were compared:

| Model | Purpose |
|-|-|
| Logistic Regression | Linear baseline |
| Support Vector Machine | Nonlinear classifier |
| Random Forest | Ensemble tree model |
| XGBoost | Gradient boosted tree model |

---

# Results

## Full Phenotypic Model

Features included:

### Demographics

- Age
- Gender
- Handedness

### Cognitive Measures

- Verbal IQ
- Performance IQ
- Full Scale IQ

### ADHD Symptom Measures

- ADHD Measure
- ADHD Index
- Inattentive symptoms
- Hyperactive/Impulsive symptoms

---

## Model Performance

| Model | ROC-AUC |
|---|---:|
| XGBoost | **0.9698** |
| Random Forest | 0.9503 |
| SVM | 0.9018 |
| Logistic Regression | 0.7372 |

---

# Best Model

## XGBoost

Cross-validation:


Mean ROC-AUC:
0.9698


Standard deviation:


0.0120


Held-out test:


ROC-AUC:
0.9356

Accuracy:
0.925


---

# Feature Importance

Most influential features:

| Feature | Importance |
|-|-:|
| Hyper/Impulsive symptoms | 0.1937 |
| ADHD Index | 0.1910 |
| Inattentive symptoms | 0.1862 |
| Age | 0.1385 |
| Full Scale IQ | 0.0853 |

---

# Discovery Experiment

## Question

> Can ADHD diagnosis be predicted without symptom information?

To investigate whether the model was learning general demographic/cognitive patterns, symptom variables were removed.

Removed:

- ADHD Measure
- ADHD Index
- Inattentive symptoms
- Hyper/Impulsive symptoms

Remaining:

- Age
- Gender
- Handedness
- Verbal IQ
- Performance IQ
- Full Scale IQ

---

# Discovery Model Results

Algorithm:


XGBoost


Performance:


Mean ROC-AUC:
0.7248

Test ROC-AUC:
0.7295


---

# Interpretation

The performance difference demonstrates that most predictive information in the ADHD-200 phenotypic dataset comes from behavioral symptom measurements.

The model is highly effective at recognizing ADHD-related behavioral patterns.

However, demographic and cognitive variables alone provide limited predictive power.

This suggests:

- symptom measurements contain the strongest predictive signal
- demographic variables alone are insufficient
- machine learning is capturing phenotype patterns rather than discovering an independent ADHD biomarker

---

# Project Structure


ADHD-Behavioral-Prediction-AI/

├── data/
│ ├── raw/
│ └── processed/
│
├── models/
│
├── reports/
│
├── src/
│ ├── data_processing/
│ ├── model_training/
│ └── evaluation/
│
├── docs/
│
└── README.md


---

# Technologies Used

## Programming

- Python

## Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy

## Explainable AI

- Feature importance analysis
- Future SHAP analysis

## Development Environment

- GitHub Codespaces
- Git

---

# Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt

Run preprocessing:

python src/process_dataset.py

Train models:

python src/train_baseline_model.py
python src/model_benchmark.py
python src/xgboost_discovery_model.py
Limitations
Dataset
Multi-site dataset
Missing values
Potential site-specific effects
Machine Learning
Models learn statistical associations
Results may not generalize to unseen populations
High performance may partially reflect similarity between symptoms and diagnostic criteria
Clinical

This project is a research investigation.

The models are not diagnostic tools and cannot replace clinical evaluation.

Future Directions
Study 2: Explainable ADHD Prediction

Planned:

SHAP explanations
Feature interaction analysis
Individual prediction explanations
Study 3: Neuroimaging ADHD Prediction

Future work will investigate ADHD-200 neuroimaging data.

Planned pipeline:

fMRI data

↓

Brain connectivity features

↓

Machine learning models

↓

ADHD prediction

↓

Explainable AI
Research Goal

The ultimate goal of this project is to explore how artificial intelligence can contribute to understanding ADHD through the combination of:

machine learning
neuroscience
behavioral science
explainable AI