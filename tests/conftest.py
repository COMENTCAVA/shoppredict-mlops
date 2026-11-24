import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SCRIPT = Path(__file__).parent.parent / "scripts" / "make_test_model.py"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "shoppredict")
    monkeypatch.setenv("MODEL_ALIAS", "champion")
    monkeypatch.delenv("MODEL_PATH", raising=False)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "model"
    subprocess.run([sys.executable, str(SCRIPT), str(path)], check=True)
    return path


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_dir))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/does/not/exist")
    with TestClient(app) as test_client:
        yield test_client
