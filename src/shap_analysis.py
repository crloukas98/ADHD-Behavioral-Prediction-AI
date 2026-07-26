"""
ADHD-200 SHAP Explainability Analysis

Purpose:
Explain why the Random Forest predicts ADHD.
"""

from pathlib import Path

import pandas as pd

import shap

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier


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


    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
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
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric
            ),

            (
                "categorical",
                categorical_pipeline,
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


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print("Training Random Forest...")

    model.fit(
        X_train,
        y_train
    )


    # Transform test data
    X_test_processed = (
        model
        .named_steps["preprocessing"]
        .transform(X_test)
    )


    feature_names = (
        model
        .named_steps["preprocessing"]
        .get_feature_names_out()
    )


    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    )


    classifier = (
        model
        .named_steps["classifier"]
    )


    print("Calculating SHAP values...")


    explainer = shap.TreeExplainer(
        classifier
    )


    shap_values = explainer.shap_values(
        X_test_processed
    )


    # For binary classification
    if isinstance(shap_values, list):

        shap_values = shap_values[1]


    print("Creating SHAP summary plot...")


    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.figure(
        figsize=(10,6)
    )


    shap.summary_plot(
        shap_values,
        X_test_processed,
        show=False
    )


    plt.tight_layout()


    plt.savefig(
        REPORTS
        /
        "shap_summary.png",
        bbox_inches="tight"
    )


    print("Saved:")
    print(
        REPORTS
        /
        "shap_summary.png"
    )


if __name__ == "__main__":

    main()