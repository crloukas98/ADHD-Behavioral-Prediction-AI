"""
ADHD Dataset Source Locator

Purpose:
Track and verify the official source
of the ADHD research dataset.
"""

import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"

DATASET_FILE = CONFIG_DIR / "datasets.json"


def show_paths():

    print("Project root:")
    print(PROJECT_ROOT)

    print("\nLooking for:")
    print(DATASET_FILE)

    print("\nExists:")
    print(DATASET_FILE.exists())


def load_registry():

    if not DATASET_FILE.exists():
        print("\nDataset registry not found.")
        return None

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def show_dataset_status():

    registry = load_registry()

    if registry is None:
        return

    dataset = registry["datasets"]["adhd200_phenotypic"]

    print("\nDataset:")
    print(dataset["name"])

    print("\nCurrent status:")
    print(dataset["status"])

    print("\nSource:")
    print(dataset["source"])

    print("\nVersion:")
    print(dataset["version"])


def main():

    print("ADHD Dataset Source Verification\n")

    show_paths()

    show_dataset_status()

    print(
        "\nLast checked:",
        datetime.now().isoformat()
    )


if __name__ == "__main__":
    main()