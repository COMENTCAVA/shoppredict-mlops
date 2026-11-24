
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.policy import should_promote, should_retrain
from ml.train import (
    evaluate,
    get_dvc_md5,
    get_git_sha,
    log_experiment,
    train,
)
from ml.validate import TARGET, validate

MODEL_NAME = "shoppredict"
MODEL_ALIAS = "champion"

PARAMS = {"model": "random_forest"}


def get_champion(client):
    try:
        registered_model = client.get_registered_model(MODEL_NAME)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise

    version = registered_model.aliases.get(MODEL_ALIAS)

    if version is None:
        return None

    return client.get_model_version(MODEL_NAME, version)


def save_decision(decision, champion_version, reason):
    result = {
        "decision": decision,
        "champion_version": str(champion_version),
        "reason": reason,
    }

    Path("decision.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def run_pipeline(version):
    train_df = pd.read_csv(f"data/{version}/train.csv")
    eval_df = pd.read_csv("data/eval.csv")

    validate(train_df, eval_df)

    if train_df[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both classes")

    if eval_df[TARGET].nunique() != 2:
        raise ValueError("Evaluation data must contain both classes")

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise ValueError("MLFLOW_TRACKING_URI is not configured")

    client = MlflowClient()
    champion = get_champion(client)
    dvc_md5 = get_dvc_md5(version)

    mlflow.set_experiment("shoppredict")

    with mlflow.start_run(run_name=f"ct-{version}"):
        mlflow.set_tags(
            {
                "git_sha": get_git_sha(),
                "dataset_version": version,
                "dvc_md5": dvc_md5,
            }
        )

        champion_auc = None
        previous_version = ""

        if champion is not None:
            previous_version = str(champion.version)

            if champion.tags.get("approved") != "true":
                raise ValueError("Current champion is not approved")

            if "approved_auc" not in champion.tags:
                raise ValueError("Champion approved_auc is missing")

            if "dvc_md5" not in champion.tags:
                raise ValueError("Champion dvc_md5 is missing")

            champion_model = mlflow.sklearn.load_model(
                f"models:/{MODEL_NAME}/{previous_version}"
            )

            champion_metrics = evaluate(champion_model, eval_df)
            champion_auc = champion_metrics["roc_auc"]

            approved_auc = float(champion.tags["approved_auc"])
            new_data = champion.tags["dvc_md5"] != dvc_md5

            mlflow.log_metric("champion_auc", champion_auc)
            mlflow.set_tag("previous_champion_version", previous_version)

            retrain, reason = should_retrain(
                champion_auc,
                approved_auc,
                new_data,
            )

            mlflow.set_tag("retraining_reason", reason)

            if not retrain:
                mlflow.set_tags(
                    {
                        "decision": "retained",
                        "reason": reason,
                        "champion_version": previous_version,
                    }
                )

                return save_decision(
                    "retained",
                    previous_version,
                    reason,
                )

        candidate = train(train_df, PARAMS)

        metrics, model_info = log_experiment(
            candidate,
            train_df,
            eval_df,
            PARAMS,
            version,
        )

        candidate_auc = metrics["roc_auc"]
        candidate_version = str(
            model_info.registered_model_version
        )

        promote, reason = should_promote(
            candidate_auc,
            champion_auc,
        )

        mlflow.log_metric("candidate_auc", candidate_auc)
        mlflow.set_tag("candidate_version", candidate_version)

        version_tags = {
            "approved": str(promote).lower(),
            "dataset_version": version,
            "dvc_md5": dvc_md5,
        }

        if promote:
            version_tags["approved_auc"] = str(candidate_auc)

        for key, value in version_tags.items():
            client.set_model_version_tag(
                MODEL_NAME,
                candidate_version,
                key,
                value,
            )

        if promote:
            client.set_registered_model_alias(
                MODEL_NAME,
                MODEL_ALIAS,
                candidate_version,
            )

            decision = "bootstrap" if champion is None else "promoted"
            champion_version = candidate_version

        else:
            decision = "retained"
            champion_version = previous_version

        mlflow.set_tags(
            {
                "decision": decision,
                "reason": reason,
                "champion_version": champion_version,
            }
        )

        return save_decision(
            decision,
            champion_version,
            reason,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        required=True,
        choices=["v1", "v2"],
    )
    args = parser.parse_args()

    Path("decision.json").unlink(missing_ok=True)

    try:
        result = run_pipeline(args.version)
    except ValueError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result))

    return 0 if result["champion_version"] else 1


if __name__ == "__main__":
    sys.exit(main())
