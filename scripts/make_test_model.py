import shutil
import sys

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build(path):
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(
        {
            "Administrative": rng.integers(0, 10, n),
            "Administrative_Duration": rng.uniform(0, 300, n),
            "Informational": rng.integers(0, 5, n),
            "Informational_Duration": rng.uniform(0, 200, n),
            "ProductRelated": rng.integers(0, 100, n),
            "ProductRelated_Duration": rng.uniform(0, 3000, n),
            "BounceRates": rng.uniform(0, 0.2, n),
            "ExitRates": rng.uniform(0, 0.2, n),
            "PageValues": rng.uniform(0, 50, n),
            "SpecialDay": rng.choice([0.0, 0.2, 0.6, 1.0], n),
            "OperatingSystems": rng.integers(1, 5, n),
            "Browser": rng.integers(1, 8, n),
            "Region": rng.integers(1, 10, n),
            "TrafficType": rng.integers(1, 10, n),
            "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor", "Other"], n),
            "Weekend": rng.integers(0, 2, n),
        }
    )
    y = (X["PageValues"] > 25).astype(int)
    categorical = ["VisitorType"]
    numeric = [c for c in X.columns if c not in categorical]
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", StandardScaler(), numeric),
                    ]
                ),
            ),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    model.fit(X, y)
    shutil.rmtree(path, ignore_errors=True)
    mlflow.sklearn.save_model(model, str(path), pip_requirements=["scikit-learn", "pandas"])


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "test_model")
