
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.validate import FEATURES, TARGET, validate


def train(train_df, params):
    numeric_features = [
        column for column in FEATURES if column != "VisitorType"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["VisitorType"],
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    model_type = params.get("model", "logistic_regression")

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        )

    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
        )

    else:
        raise ValueError(f"Unknown model: {model_type}")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    model.fit(X_train, y_train)

    return model


def evaluate(model, eval_df):
    X_eval = eval_df[FEATURES]
    y_eval = eval_df[TARGET]

    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_eval, probabilities)),
        "f1": float(f1_score(y_eval, predictions, zero_division=0)),
        "average_precision": float(
            average_precision_score(y_eval, probabilities)
        ),
    }

def get_git_sha():
    return os.environ.get("GITHUB_SHA") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def get_dvc_md5(version):
    path = Path("data") / f"{version}.dvc"

    for line in path.read_text().splitlines():
        if line.strip().startswith("- md5:"):
            return line.split("md5:", 1)[1].strip()

    raise ValueError(f"DVC hash not found in {path}")


def log_experiment(model, train_df, eval_df, params, version):
    metrics = evaluate(model, eval_df)

    mlflow.set_tags(
        {
            "git_sha": get_git_sha(),
            "dataset_version": version,
            "dvc_md5": get_dvc_md5(version),
        }
    )

    classifier = model.named_steps["classifier"]

    mlflow.log_params(
        {
            **classifier.get_params(deep=False),
            **params,
        }
    )

    mlflow.log_metrics(metrics)

    probabilities = model.predict_proba(eval_df[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(
        eval_df[TARGET],
        predictions,
        labels=[0, 1],
    )

    mlflow.log_dict(
        {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
        },
        "confusion_matrix.json",
    )

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="shoppredict",
        serialization_format="cloudpickle",
        input_example=train_df[FEATURES].head(5),
    )

    return metrics, model_info


def run_experiment(version, params):
    if version not in ("v1", "v2"):
        raise ValueError(f"Unknown dataset version: {version}")

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise ValueError("MLFLOW_TRACKING_URI is not configured")

    train_df = pd.read_csv(f"data/{version}/train.csv")
    eval_df = pd.read_csv("data/eval.csv")

    validate(train_df, eval_df)

    mlflow.set_experiment("shoppredict")

    with mlflow.start_run(
        run_name=f"{version}-{params['model']}"
    ) as run:
        model = train(train_df, params)

        metrics, model_info = log_experiment(
            model,
            train_df,
            eval_df,
            params,
            version,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Model version: {model_info.registered_model_version}")
        print(metrics)

    return metrics
