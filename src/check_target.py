"""
Check ADHD-200 target variable.
"""

from pathlib import Path
import pandas as pd


DATA = (
    Path(__file__).resolve().parent.parent
    /
    "data/raw/adhd200_preprocessed_phenotypics.tsv"
)


df = pd.read_csv(
    DATA,
    sep="\t"
)


print("DX distribution:")
print(df["DX"].value_counts())


print("\nDX percentages:")
print(
    df["DX"]
    .value_counts(normalize=True)
    * 100
)