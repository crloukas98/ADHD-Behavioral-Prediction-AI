"""
ADHD-200 Model Benchmark

Compare:
- Logistic Regression
- Random Forest
- SVM
- XGBoost

using 5-fold ROC-AUC.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

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


    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ),


        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42
            ),


        "SVM":
            SVC(
                probability=True,
                class_weight="balanced"
            ),


        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
    }


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    results = []


    for name, classifier in models.items():

        print("\nTraining:")
        print(name)


        pipeline = Pipeline(
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


        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="roc_auc"
        )


        results.append(
            {
                "Model": name,
                "Mean ROC-AUC": scores.mean(),
                "Std": scores.std()
            }
        )


        print(
            scores
        )

        print(
            "Mean:",
            scores.mean()
        )


    results_df = pd.DataFrame(
        results
    )


    results_df = results_df.sort_values(
        by="Mean ROC-AUC",
        ascending=False
    )


    print("\n================")
    print("FINAL BENCHMARK")
    print("================")

    print(results_df)


    output = (
        ROOT
        /
        "reports"
        /
        "model_benchmark.csv"
    )


    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    results_df.to_csv(
        output,
        index=False
    )


    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()