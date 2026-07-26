"""
Dataset download utilities.

This script manages research dataset storage.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"


def create_data_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created {DATA_DIR}")


if __name__ == "__main__":
    create_data_directory()