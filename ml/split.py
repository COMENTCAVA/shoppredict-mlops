
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

V1_MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug"]
V2_MONTHS = V1_MONTHS + ["Sep", "Oct", "Nov"]


def main():
    source = DATA / "raw" / "online_shoppers_intention.csv"

    df = pd.read_csv(source)
    initial_count = len(df)
    df = df.drop_duplicates().copy()

    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)

    v1 = df[df["Month"].isin(V1_MONTHS)].copy()
    v2 = df[df["Month"].isin(V2_MONTHS)].copy()
    evaluation = df[df["Month"] == "Dec"].copy()

    if len(v2) + len(evaluation) != len(df):
        raise ValueError("Unexpected months in dataset")

    outputs = {
        DATA / "v1" / "train.csv": v1,
        DATA / "v2" / "train.csv": v2,
        DATA / "eval.csv": evaluation,
    }

    print(f"Duplicates removed: {initial_count - len(df)}")

    for path, dataset in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False, lineterminator="\n")

        purchase_rate = dataset["Revenue"].mean() * 100
        print(f"{path.relative_to(ROOT)}: {len(dataset)} rows, {purchase_rate:.2f}% purchases")


if __name__ == "__main__":
    main()
