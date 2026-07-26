"""
ADHD Dataset Download Utility

Purpose:
Download and verify the raw ADHD research dataset.

Raw data location:
data/raw/
"""

from pathlib import Path
import requests
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

METADATA_FILE = PROJECT_ROOT / "data" / "dataset_metadata.json"


def create_raw_directory():

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Raw directory ready: {RAW_DATA_DIR}"
    )


def download_file(url, filename):

    destination = RAW_DATA_DIR / filename

    print(
        f"Downloading:\n{url}"
    )

    response = requests.get(
        url,
        stream=True
    )

    response.raise_for_status()

    with open(destination, "wb") as file:

        for chunk in response.iter_content(
            chunk_size=8192
        ):
            file.write(chunk)

    print(
        f"Saved: {destination}"
    )

    return destination


def update_metadata(source_url, filename):

    metadata = {
        "dataset": "ADHD-200 Consortium Phenotypic Data",
        "source": source_url,
        "file": filename,
        "download_date": datetime.now().isoformat(),
        "status": "downloaded"
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

    create_raw_directory()

    print(
        """
Dataset downloader ready.

Waiting for verified official dataset URL.

No download performed.
        """
    )


if __name__ == "__main__":

    main()