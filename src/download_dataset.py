"""
ADHD-200 Phenotypic Dataset Downloader

Downloads the official ADHD-200 phenotypic dataset.
"""

from pathlib import Path
import requests
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DIR / "adhd200_preprocessed_phenotypics.tsv"

METADATA_FILE = PROJECT_ROOT / "data" / "dataset_metadata.json"


DATASET_URL = (
    "https://www.nitrc.org"
    "/frs/download.php/9024/"
    "adhd200_preprocessed_phenotypics.tsv"
)


def create_directory():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def download_dataset():

    print("Downloading ADHD-200 phenotypic dataset...")

    response = requests.get(
        DATASET_URL,
        timeout=60
    )

    print(
        "HTTP status:",
        response.status_code
    )

    response.raise_for_status()


    with open(
        OUTPUT_FILE,
        "wb"
    ) as file:

        file.write(
            response.content
        )


    print("\nSaved:")
    print(OUTPUT_FILE)


def update_metadata():

    metadata = {

        "dataset":
        "ADHD-200 Consortium Phenotypic Data",

        "source":
        DATASET_URL,

        "file":
        str(OUTPUT_FILE),

        "download_date":
        datetime.now().isoformat(),

        "status":
        "downloaded"
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


def main():

    create_directory()

    download_dataset()

    update_metadata()


if __name__ == "__main__":

    main()