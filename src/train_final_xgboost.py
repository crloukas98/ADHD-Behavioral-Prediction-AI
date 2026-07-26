"""
Study 1 Final Model

Train final XGBoost ADHD prediction model
and save for explainability (Study 2)
"""

from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent.parent


DATA_PATH = (
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
    "xgboost_final_model.pkl"
)


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH
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


    TARGET = "ADHD"


    X = df[FEATURES]

    y = df[TARGET]


    print("\nClass distribution:")
    print(
        y.value_counts()
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


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


    categorical_features = [
        "Handedness"
    ]


    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
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
        transformers=[
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


    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )


    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing
            ),
            (
                "classifier",
                model
            )
        ]
    )


    print("\nTraining final XGBoost model...")


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]


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


    joblib.dump(
        pipeline,
        MODEL_PATH
    )


    print("\nSaved model:")

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()
    