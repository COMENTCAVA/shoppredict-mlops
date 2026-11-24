import logging
import os

from mlflow import MlflowClient
from mlflow.sklearn import load_model

logger = logging.getLogger("shoppredict")


def load():
    name = os.getenv("MODEL_NAME", "shoppredict")
    alias = os.getenv("MODEL_ALIAS", "champion")
    try:
        path = os.getenv("MODEL_PATH")
        if path:
            version = "test"
            model = load_model(path)
        else:
            version = str(MlflowClient().get_model_version_by_alias(name, alias).version)
            model = load_model(f"models:/{name}/{version}")
        logger.info("Loaded %s version %s", name, version)
        return model, {"name": name, "alias": alias, "version": version}
    except Exception:
        logger.exception("Model loading failed")
        return None, None
