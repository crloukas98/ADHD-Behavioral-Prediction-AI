# ADHD Behavioral Prediction AI

## Machine Learning Research Project for ADHD Prediction Using Phenotypic and Neuroimaging Data

---

# Abstract

This project develops a complete, reproducible, and explainable artificial intelligence pipeline for predicting ADHD-related patterns from clinical phenotypic data.

Using the publicly available **ADHD-200 Consortium dataset**, participant demographic, cognitive, and symptom-related features were processed and evaluated through multiple machine learning approaches.

Four classification algorithms were compared:

- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost

The final models were evaluated using:

- Cross-validation
- ROC-AUC
- Accuracy
- Precision
- Recall
- Confusion matrices

XGBoost achieved the strongest performance, demonstrating excellent discrimination between ADHD and non-ADHD classes.

Beyond prediction accuracy, this project investigates model transparency through **Explainable AI (XAI)** techniques.

SHAP analysis was applied to identify which features contributed most strongly to model decisions, revealing that ADHD symptom measurements and related clinical variables were the primary drivers of predictions.

The result is a complete machine learning workflow integrating:

- Data preprocessing
- Model development
- Performance evaluation
- Explainability analysis

into a reproducible research framework.

This project demonstrates how artificial intelligence can be applied responsibly to healthcare-related datasets by combining predictive capability with interpretability.

**Key achievement:**

Built an end-to-end explainable AI system that predicts ADHD-related patterns from clinical data while identifying the features influencing its decisions.

---

# Overview

This project investigates whether artificial intelligence and machine learning can identify patterns associated with Attention-Deficit/Hyperactivity Disorder (ADHD) using publicly available research datasets.

The project is structured as a series of reproducible studies, beginning with phenotypic prediction and progressing toward computational neuroscience approaches using neuroimaging data.

The long-term goal is to explore how AI can assist understanding of ADHD through:

- Behavioral data analysis
- Machine learning
- Explainable AI
- Neuroimaging analysis
- Computational neuroscience

---

# Current Study

# Study 1: Phenotypic ADHD Prediction

## Research Question

> Can machine learning models predict ADHD diagnosis from clinical and cognitive characteristics contained within the ADHD-200 phenotypic dataset?

---

# Dataset

## ADHD-200 Consortium Phenotypic Dataset

### Dataset Characteristics

| Description | Value |
|---|---:|
| Original participants | 973 |
| Processed binary classification dataset | 797 |

### Class Distribution

| Class | Participants |
|---|---:|
| No ADHD | 585 |
| ADHD | 212 |

### Target Variable

ADHD diagnosis

Encoding:

```text
0 = No ADHD
1 = ADHD
```

---

# Methodology

The dataset was processed using a reproducible machine learning pipeline.

```text
Raw Dataset
     |
     ↓
Data Cleaning
     |
     ↓
Missing Value Handling
     |
     ↓
Feature Preprocessing
     |
     ↓
Model Training
     |
     ↓
Cross-validation
     |
     ↓
Performance Evaluation
```

---

# Models Evaluated

The following machine learning algorithms were compared:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline |
| Support Vector Machine | Nonlinear classifier |
| Random Forest | Ensemble tree model |
| XGBoost | Gradient boosted tree model |

---

# Results

## Full Phenotypic Model

Features included:

## Demographics

- Age
- Gender
- Handedness

## Cognitive Measures

- Verbal IQ
- Performance IQ
- Full Scale IQ

## ADHD Symptom Measures

- ADHD Measure
- ADHD Index
- Inattentive symptoms
- Hyperactive/Impulsive symptoms

---

# Model Performance

| Model | ROC-AUC |
|---|---:|
| XGBoost | **0.9698** |
| Random Forest | 0.9503 |
| SVM | 0.9018 |
| Logistic Regression | 0.7372 |

---

# Best Model

## XGBoost

## Cross-validation Performance

Mean ROC-AUC:

```text
0.9698
```

Standard deviation:

```text
0.0120
```

## Held-out Test Performance

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9356 |
| Accuracy | 0.925 |

---

# Feature Importance

Most influential features:

| Feature | Importance |
|---|---:|
| Hyper/Impulsive symptoms | 0.1937 |
| ADHD Index | 0.1910 |
| Inattentive symptoms | 0.1862 |
| Age | 0.1385 |
| Full Scale IQ | 0.0853 |

---

# Discovery Experiment

## Research Question

> Can ADHD diagnosis be predicted without symptom information?

To investigate whether the model was learning general demographic and cognitive patterns, symptom variables were removed.

---

## Removed Features

- ADHD Measure
- ADHD Index
- Inattentive symptoms
- Hyper/Impulsive symptoms

---

## Remaining Features

- Age
- Gender
- Handedness
- Verbal IQ
- Performance IQ
- Full Scale IQ

---

# Discovery Model Results

## Algorithm

XGBoost

## Performance

| Metric | Score |
|---|---:|
| Mean ROC-AUC | 0.7248 |
| Test ROC-AUC | 0.7295 |

---

# Interpretation

The performance difference demonstrates that most predictive information in the ADHD-200 phenotypic dataset comes from behavioral symptom measurements.

The model is highly effective at recognizing ADHD-related behavioral patterns.

However, demographic and cognitive variables alone provide limited predictive power.

This suggests:

- Symptom measurements contain the strongest predictive signal
- Demographic variables alone are insufficient
- Machine learning is capturing phenotype patterns rather than discovering an independent ADHD biomarker

---

# Project Structure

```text
ADHD-Behavioral-Prediction-AI/

├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── src/
│   ├── data_processing/
│   ├── model_training/
│   └── evaluation/
│
├── docs/
│
└── README.md
```

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
- SHAP analysis

## Development Environment

- GitHub Codespaces
- Git

---

# Reproducibility

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Preprocessing

```bash
python src/process_dataset.py
```

## Train Models

```bash
python src/train_baseline_model.py

python src/model_benchmark.py

python src/xgboost_discovery_model.py
```

---

# Limitations

## Dataset

- Multi-site dataset
- Missing values
- Potential site-specific effects

## Machine Learning

- Models learn statistical associations
- Results may not generalize to unseen populations
- High performance may partially reflect similarity between symptoms and diagnostic criteria

## Clinical

This project is a research investigation.

The models are **not diagnostic tools** and cannot replace professional clinical evaluation.

---

# Study 2: Explainable ADHD Prediction

## Research Question

> Which features contribute most to machine learning ADHD predictions, and can model decisions be interpreted in a clinically meaningful way?

---

# Motivation

Although Study 1 demonstrated strong predictive performance, machine learning models can behave as black boxes.

Study 2 introduces **Explainable Artificial Intelligence (XAI)** methods to investigate:

- Which features drive predictions
- How strongly each feature contributes
- Whether model behavior aligns with known ADHD clinical characteristics

---

# Methodology

The final XGBoost ADHD prediction model from Study 1 was analyzed using:

## SHAP

### SHapley Additive exPlanations

SHAP estimates how much each feature contributes to an individual prediction.

Pipeline:

```text
Final XGBoost Model

        |
        ↓

SHAP Value Calculation

        |
        ↓

Global Feature Importance

        |
        ↓

Prediction Interpretation
```

---

# SHAP Feature Importance

| Rank | Feature | Mean Absolute SHAP |
|---|---|---:|
| 1 | ADHD Index | 1.6816 |
| 2 | Inattentive symptoms | 0.9543 |
| 3 | Full Scale IQ | 0.7540 |
| 4 | Hyper/Impulsive symptoms | 0.6431 |
| 5 | Age | 0.4999 |
| 6 | Gender | 0.4249 |
| 7 | Verbal IQ | 0.3525 |
| 8 | Performance IQ | 0.2854 |
| 9 | ADHD Measure | 0.2715 |

---

# Interpretation

The explainability analysis confirms that the model primarily relies on behavioral symptom measurements.

The strongest contributors were:

- ADHD Index
- Inattentive symptoms
- Hyperactive/Impulsive symptoms

These features correspond directly to established ADHD clinical domains.

Demographic and cognitive variables contributed less strongly.

---

# Scientific Interpretation

The model does not appear to discover an independent biological marker of ADHD from phenotypic data.

Instead, it learns statistical patterns within clinically measured characteristics.

This demonstrates that:

- Machine learning can reproduce ADHD-related behavioral patterns
- Model decisions can be interpreted using explainable AI methods
- Explainability is essential when applying AI to healthcare research

---

# Study 2 Outputs

Generated:

```text
reports/

├── shap_feature_importance.csv

└── figures/

    └── shap_summary.png
```

---

# Study 2 Status

Completed:

✅ Final XGBoost model explanation  
✅ SHAP integration  
✅ Global feature importance analysis  
✅ Explainability visualization  
✅ Interpretation of model behavior
