"""
ADHD-200 Model Evaluation Pipeline

Purpose:
Robust evaluation of the Random Forest ADHD classifier.
"""

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
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


REPORTS = (
    ROOT
    /
    "reports"
)


MODEL_FILE = (
    ROOT
    /
    "models"
    /
    "adhd_random_forest.pkl"
)


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA)


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


    categorical = [
        "Handedness"
    ]


    numeric = [
        c for c in FEATURES
        if c not in categorical
    ]


    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        )
                    ]
                ),
                numeric
            ),

            (
                "categorical",
                Pipeline(
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
                ),
                categorical
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
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )


    print("\nCross validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )


    print("\nROC-AUC scores:")
    print(scores)


    print("\nMean ROC-AUC:")
    print(scores.mean())


    print("\nStd:")
    print(scores.std())


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print("\nTraining final model...")


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


    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )


    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )


    plt.title(
        "ADHD Prediction ROC Curve"
    )


    plt.savefig(
        REPORTS
        /
        "roc_curve.png"
    )


    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        model,
        MODEL_FILE
    )


    print("\nModel saved:")
    print(
        MODEL_FILE
    )


if __name__ == "__main__":
    main()