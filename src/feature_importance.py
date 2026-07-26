"""
ADHD-200 Random Forest Feature Importance

Purpose:
Identify which variables contribute most to ADHD prediction.
"""

from pathlib import Path

import pandas as pd

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


OUTPUT = (
    ROOT
    /
    "reports"
    /
    "feature_importance.csv"
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


    print("Training model...")

    model.fit(
        X_train,
        y_train
    )


    classifier = (
        model
        .named_steps["classifier"]
    )


    preprocessor = (
        model
        .named_steps["preprocessing"]
    )


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_
        }
    )


    importance = importance.sort_values(
        by="importance",
        ascending=False
    )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    importance.to_csv(
        OUTPUT,
        index=False
    )


    print("\nFeature importance:")
    print(
        importance.head(15)
    )


    print("\nSaved:")
    print(
        OUTPUT
    )


    plt.figure(
        figsize=(10,6)
    )


    top = importance.head(10)


    plt.barh(
        top["feature"],
        top["importance"]
    )


    plt.gca().invert_yaxis()

    plt.title(
        "ADHD Prediction Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        ROOT
        /
        "reports"
        /
        "feature_importance.png"
    )


if __name__ == "__main__":

    main()