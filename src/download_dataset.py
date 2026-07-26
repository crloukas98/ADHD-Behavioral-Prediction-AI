"""
ADHD Dataset Download Utility

Purpose:
Download and organize raw research data.

Raw data is stored in:
data/raw/

Raw files should never be modified directly.
"""

from pathlib import Path
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

METADATA_FILE = PROJECT_ROOT / "data" / "dataset_metadata.json"


def create_directories():
    """
    Create required dataset directories.
    """

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Raw data directory ready: {RAW_DATA_DIR}"
    )


def create_metadata():

    metadata = {
        "dataset": "ADHD-200 Consortium Phenotypic Data",
        "version": "pending",
        "source": "pending",
        "download_date": datetime.now().isoformat(),
        "description": (
            "Behavioral and phenotypic data "
            "for ADHD machine learning research."
        )
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"Metadata created: {METADATA_FILE}"
    )


def main():

    create_directories()

    create_metadata()

    print(
        """
Dataset acquisition pipeline initialized.

Next:
1. Add official dataset source.
2. Download raw files.
3. Verify integrity.
        """
    )


if __name__ == "__main__":
    main()