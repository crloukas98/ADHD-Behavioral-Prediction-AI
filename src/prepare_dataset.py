"""
ADHD-200 Dataset Preparation

Creates binary ADHD classification dataset.

0 = Control
1 = ADHD
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "adhd200_preprocessed_phenotypics.tsv"
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "adhd_binary_dataset.csv"
)


def prepare_dataset():

    print("Loading raw dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t"
    )


    print(
        "Original shape:",
        df.shape
    )


    print("\nDX raw values:")
    print(
        df["DX"].value_counts()
    )


    print("\nDX data type:")
    print(
        df["DX"].dtype
    )


    # Convert DX to string for filtering
    df["DX"] = (
        df["DX"]
        .astype(str)
        .str.strip()
    )


    # Keep only confirmed control and ADHD
    df = df[
        df["DX"].isin(
            ["0", "1"]
        )
    ].copy()


    # Convert target back to integer
    df["DX"] = df["DX"].astype(int)


    df.rename(
        columns={
            "DX": "ADHD"
        },
        inplace=True
    )


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\nClean dataset:")
    print(
        df.shape
    )


    print("\nClass distribution:")
    print(
        df["ADHD"].value_counts()
    )


    print("\nSaved:")
    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    prepare_dataset()