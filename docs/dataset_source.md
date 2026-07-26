# Dataset Source Documentation

## Project

ADHD Behavioral Prediction AI

## Dataset

ADHD-200 Consortium Dataset

## Purpose

This project investigates whether machine learning models can identify behavioral and cognitive patterns associated with ADHD status.

## Data Used

Initial version:

- Phenotypic data
- Demographic variables
- Behavioral measurements
- Clinical labels

Excluded initially:

- MRI imaging data
- fMRI preprocessing
- High-dimensional neuroimaging features

## Reason for Exclusion

The goal of Version 1 is to build a reproducible explainable AI pipeline using tabular clinical/behavioral data.

Neuroimaging extensions may be added in future versions.

## Data Handling

Raw files are stored locally in:

data/raw/

Raw datasets are not modified.

All preprocessing steps are performed through documented scripts.

## Research Limitations

This model is not intended for diagnosis.

It investigates statistical patterns in a research dataset and does not replace clinical assessment.