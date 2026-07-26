"""
ADHD-200 XGBoost Discovery Model

Purpose:
Predict ADHD diagnosis using only
non-symptom features.

Removed:
- ADHD Measure
- ADHD Index
- Inattentive
- Hyper/Impulsive
"""

from pathlib import Path

import pandas as pd

import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


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


MODEL_PATH = (
    ROOT
    /
    "models"
    /
    "xgboost_discovery_model.pkl"
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
        "Full4 IQ"
    ]


    X = df[FEATURES]

    y = df["ADHD"]


    print("\nFeatures:")
    print(FEATURES)


    print("\nClass distribution:")
    print(
        y.value_counts()
    )


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
                        ),

                        (
                            "scaler",
                            StandardScaler()
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


    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )


    model = Pipeline(
        [
            (
                "preprocessing",
                preprocessing
            ),

            (
                "classifier",
                classifier
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


    print("\nTraining XGBoost discovery model...")


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


    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        model,
        MODEL_PATH
    )


    print("\nSaved:")
    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()