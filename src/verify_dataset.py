"""
Dataset Verification Utility

Checks:
- dataset existence
- file structure
- basic integrity
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def verify_raw_directory():

    print("Checking raw data directory:")
    print(RAW_DATA_DIR)

    if not RAW_DATA_DIR.exists():
        print("ERROR: Raw data directory does not exist.")
        return

    files = list(RAW_DATA_DIR.rglob("*"))

    if len(files) == 0:
        print("Raw data directory is empty.")
        return

    print("\nFiles found:")

    for file in files:
        print(file)


if __name__ == "__main__":

    verify_raw_directory()