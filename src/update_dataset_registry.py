"""
Dataset Registry Updater

Purpose:
Update dataset provenance information
after source verification.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = PROJECT_ROOT / "config" / "datasets.json"


def update_registry(
    source,
    version
):

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        registry = json.load(file)


    dataset = registry["datasets"]["adhd200_phenotypic"]

    dataset["source"] = source
    dataset["version"] = version
    dataset["status"] = "source_verified"


    with open(
        DATASET_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )


    print("Dataset registry updated.")
    print()
    print(dataset)


if __name__ == "__main__":

    print(
        """
This script updates dataset provenance.

Usage:
Call update_registry()
with verified source information.
        """
    )