"""Download and load the BANKING77 dataset.

BANKING77 (PolyAI, CC BY 4.0): 13,083 banking customer queries, 77 intents.
Source: https://github.com/PolyAI-LDN/task-specific-datasets
The CSVs are downloaded once into data/raw/ (git-ignored) and reused after that.
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_banking77(split: str) -> pd.DataFrame:
    """Return the 'train' or 'test' split as a DataFrame with columns text, intent."""
    path = RAW_DIR / f"{split}.csv"
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {split}.csv ...")
        urlretrieve(BASE_URL + f"{split}.csv", path)
    df = pd.read_csv(path)
    return df.rename(columns={"category": "intent"})


if __name__ == "__main__":
    train = load_banking77("train")
    test = load_banking77("test")
    print(f"train: {len(train)} rows, test: {len(test)} rows, total: {len(train) + len(test)}")
    print(f"number of intents: {train['intent'].nunique()}")
