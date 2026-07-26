"""
ADHD-200 Dataset Inspector

Purpose:
Load and validate the raw phenotypic dataset.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "adhd200_preprocessed_phenotypics.tsv"
)


def inspect_dataset():

    print("Loading dataset:")
    print(DATA_FILE)

    df = pd.read_csv(
        DATA_FILE,
        sep="\t"
    )


    print("\n====================")
    print("Dataset shape")
    print("====================")

    print(df.shape)


    print("\n====================")
    print("Columns")
    print("====================")

    for column in df.columns:
        print(column)


    print("\n====================")
    print("First rows")
    print("====================")

    print(
        df.head()
    )


    print("\n====================")
    print("Missing values")
    print("====================")

    print(
        df.isna().sum()
    )


if __name__ == "__main__":

    inspect_dataset()