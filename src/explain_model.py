"""
Study 2: Explainable ADHD Prediction

Uses SHAP to explain ADHD prediction models.
"""

from pathlib import Path

import pandas as pd
import numpy as np

import joblib
import shap

import matplotlib.pyplot as plt


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


OUTPUT_DIR = (
    ROOT
    /
    "reports"
    /
    "figures"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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


    X = df[FEATURES]


    print("Loading model...")


    model = joblib.load(
        MODEL_PATH
    )


    preprocessing = model.named_steps[
        "preprocessing"
    ]


    classifier = model.named_steps[
        "classifier"
    ]


    print("Transforming data...")


    X_processed = preprocessing.transform(
        X
    )


    #
    # Convert sparse/object matrices
    # into pure numeric numpy array
    #

    if hasattr(
        X_processed,
        "toarray"
    ):

        X_processed = X_processed.toarray()


    X_processed = np.asarray(
        X_processed,
        dtype=float
    )


    print(
        "Processed shape:",
        X_processed.shape
    )


    print(
        "Calculating SHAP values..."
    )


    explainer = shap.TreeExplainer(
        classifier
    )


    shap_values = explainer.shap_values(
        X_processed
    )


    #
    # New SHAP versions return Explanation objects
    #

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[1]


    elif len(
        shap_values.shape
    ) == 3:

        shap_values = shap_values[:,:,1]


    print(
        "SHAP shape:",
        shap_values.shape
    )


    feature_names = (
        preprocessing
        .get_feature_names_out()
    )


    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap":
                np.abs(shap_values)
                .mean(axis=0)
        }
    )


    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    )


    print(
        "\nTop SHAP features:"
    )

    print(
        importance.head(15)
    )


    output_csv = (
        ROOT
        /
        "reports"
        /
        "shap_feature_importance.csv"
    )


    importance.to_csv(
        output_csv,
        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        output_csv
    )


    print(
        "\nCreating SHAP plot..."
    )


    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False
    )


    plot_path = (
        OUTPUT_DIR
        /
        "shap_summary.png"
    )


    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:"
    )

    print(
        plot_path
    )


if __name__ == "__main__":
    main()