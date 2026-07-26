"""
ADHD-200 Symptom Enhanced Model

Adds ADHD symptom measurements.

Model:
Logistic Regression
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)


ROOT = Path(__file__).resolve().parent.parent


DATA = (
    ROOT
    /
    "data"
    /
    "processed"
    /
    "adhd_binary_dataset.csv"
)


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA
    )


    FEATURES = [
        "Age",
        "Gender",
        "Handedness",
        "Verbal IQ",
        "Performance IQ",
        "Full4 IQ",
        "ADHD Measure",
        "ADHD Index",
        "Inattentive",
        "Hyper/Impulsive"
    ]


    X = df[FEATURES]

    y = df["ADHD"]


    categorical_features = [
        "Handedness"
    ]


    numeric_features = [
        "Age",
        "Gender",
        "Verbal IQ",
        "Performance IQ",
        "Full4 IQ",
        "ADHD Measure",
        "ADHD Index",
        "Inattentive",
        "Hyper/Impulsive"
    ]


    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    model = Pipeline(
        [
            (
                "preprocessing",
                preprocessing
            ),

            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000
                )
            )
        ]
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print("\nTraining symptom model...")

    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    probabilities = model.predict_proba(
        X_test
    )[:,1]


    print("\nAccuracy:")
    print(
        accuracy_score(
            y_test,
            predictions
        )
    )


    print("\nROC-AUC:")
    print(
        roc_auc_score(
            y_test,
            probabilities
        )
    )


    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":

    main()