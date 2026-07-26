"""
Download ADHD research dataset.

Raw data is stored locally and excluded from GitHub.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def setup_data_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")


def main():
    setup_data_directory()

    print(
        """
Dataset download step.

Next:
- identify official dataset source
- download files
- verify structure
        """
    )


if __name__ == "__main__":
    main()