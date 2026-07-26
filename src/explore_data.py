from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw"


def inspect_data():

    print("Looking in:")
    print(RAW_DATA)

    if not RAW_DATA.exists():
        print("No dataset found yet.")
        return

    files = list(RAW_DATA.rglob("*"))

    if len(files) == 0:
        print("Dataset folder is empty.")
    else:
        for file in files:
            print(file)


if __name__ == "__main__":
    inspect_data()