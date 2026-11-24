
import argparse
import json
import os
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = os.getenv("MODEL_NAME", "shoppredict")
MODEL_ALIAS = "champion"


def rollback(version):
    client = MlflowClient()

    model_version = client.get_model_version(
        MODEL_NAME,
        version,
    )

    if model_version.tags.get("approved") != "true":
        raise ValueError(
            f"Model version {version} is not approved"
        )

    client.set_registered_model_alias(
        MODEL_NAME,
        MODEL_ALIAS,
        version,
    )

    return {
        "decision": "rollback",
        "champion_version": str(version),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--to-version",
        required=True,
    )

    args = parser.parse_args()

    try:
        result = rollback(args.to_version)
    except (ValueError, MlflowException) as exc:
        print(f"Rollback error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
