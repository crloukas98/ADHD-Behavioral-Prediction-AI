"""
ADHD-200 Random Forest Model

Nonlinear classifier baseline.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

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
                "num",
                SimpleImputer(
                    strategy="median"
                ),
                numeric
            ),

            (
                "cat",
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
                "prep",
                preprocessing
            ),

            (
                "clf",
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


    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]


    print("\nAccuracy:")
    print(
        accuracy_score(
            y_test,
            pred
        )
    )


    print("\nROC-AUC:")
    print(
        roc_auc_score(
            y_test,
            prob
        )
    )


    print("\nReport:")
    print(
        classification_report(
            y_test,
            pred
        )
    )


if __name__ == "__main__":
    main()