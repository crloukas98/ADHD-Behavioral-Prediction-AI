from pathlib import Path
import pandas as pd

from config import RAW_DATA_DIR


def load_csv(filename):

    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}"
        )

    return pd.read_csv(filepath)