import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

PAGE = Path(__file__).parent.parent / "app" / "static" / "index.html"
EXAMPLES = json.loads(
    re.search(
        r'<script type="application/json" id="examples">(.*?)</script>',
        PAGE.read_text(encoding="utf-8"),
        re.DOTALL,
    ).group(1)
)
VALID = EXAMPLES["Likely buyer"]


def valid_with(**changes):
    return {**VALID, **changes}


MINIMUMS = {
    "Administrative": 0,
    "Administrative_Duration": 0,
    "Informational": 0,
    "Informational_Duration": 0,
    "ProductRelated": 0,
    "ProductRelated_Duration": 0,
    "BounceRates": 0,
    "ExitRates": 0,
    "PageValues": 0,
    "SpecialDay": 0,
    "OperatingSystems": 1,
    "Browser": 1,
    "Region": 1,
    "TrafficType": 1,
    "VisitorType": "Other",
    "Weekend": False,
}
MAXIMUMS = valid_with(
    Administrative_Duration=100000,
    ProductRelated_Duration=100000,
    BounceRates=1,
    ExitRates=1,
    PageValues=100000,
    SpecialDay=1,
    Region=9,
    VisitorType="New_Visitor",
    Weekend=True,
)


class TestPrediction:
    def test_valid_input_has_expected_keys(self, client):
        response = client.post("/predict", json=VALID)
        assert response.status_code == 200
        assert set(response.json()) == {"probability", "purchase", "model_version"}

    @pytest.mark.parametrize("payload", [VALID, MINIMUMS, MAXIMUMS])
    def test_probability_range_and_threshold(self, client, payload):
        body = client.post("/predict", json=payload).json()
        assert 0 <= body["probability"] <= 1
        assert round(body["probability"], 4) == body["probability"]
        assert body["purchase"] == (body["probability"] >= 0.5)

    @pytest.mark.parametrize("visitor", ["Returning_Visitor", "New_Visitor", "Other"])
    def test_every_visitor_type_is_accepted(self, client, visitor):
        assert client.post("/predict", json=valid_with(VisitorType=visitor)).status_code == 200

    def test_model_version_is_reported(self, client):
        assert client.post("/predict", json=VALID).json()["model_version"] == "test"

    @pytest.mark.parametrize("name", list(EXAMPLES))
    def test_page_examples_are_accepted(self, client, name):
        response = client.post("/predict", json=EXAMPLES[name])
        assert response.status_code == 200
        assert 0 <= response.json()["probability"] <= 1

    def test_buying_session_scores_higher_than_bouncing_session(self, client):
        buyer = client.post("/predict", json=EXAMPLES["Likely buyer"]).json()
        bouncer = client.post("/predict", json=EXAMPLES["Quick bounce"]).json()
        assert buyer["probability"] > bouncer["probability"]


class TestValidation:
    @pytest.mark.parametrize("field", list(VALID))
    def test_missing_field(self, client, field):
        body = {k: v for k, v in VALID.items() if k != field}
        assert client.post("/predict", json=body).status_code == 422

    @pytest.mark.parametrize(
        "field, value",
        [
            ("BounceRates", 1.5),
            ("BounceRates", -0.1),
            ("ExitRates", 1.01),
            ("SpecialDay", 2),
            ("ProductRelated_Duration", -3),
            ("Administrative", -1),
            ("PageValues", -0.5),
            ("Region", 0),
            ("Region", 10),
            ("OperatingSystems", 0),
            ("Browser", 0),
            ("TrafficType", 0),
            ("VisitorType", "Bot"),
            ("VisitorType", ""),
            ("Administrative", "abc"),
            ("Administrative", 2.5),
            ("BounceRates", "high"),
            ("Weekend", "maybe"),
            ("Browser", None),
        ],
    )
    def test_invalid_value(self, client, field, value):
        assert client.post("/predict", json=valid_with(**{field: value})).status_code == 422

    @pytest.mark.parametrize("extra", [{"Month": "Dec"}, {"Revenue": 1}, {"unknown": "x"}])
    def test_extra_field(self, client, extra):
        assert client.post("/predict", json={**VALID, **extra}).status_code == 422

    @pytest.mark.parametrize("body", [{}, [], "text"])
    def test_wrong_body_shape(self, client, body):
        assert client.post("/predict", json=body).status_code == 422

    def test_malformed_json(self, client):
        response = client.post(
            "/predict", content="{not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class StubModel:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]] * len(X))


class TestModelAndReadiness:
    def test_without_model_everything_is_unavailable(self, client_without_model):
        assert client_without_model.get("/ready").status_code == 503
        assert client_without_model.get("/model-info").status_code == 503
        assert client_without_model.post("/predict", json=VALID).status_code == 503

    def test_without_model_page_is_still_served(self, client_without_model):
        assert client_without_model.get("/").status_code == 200

    def test_with_model_ready_and_model_info(self, client):
        assert client.get("/ready").status_code == 200
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/model-info").json() == {
            "name": "shoppredict",
            "alias": "champion",
            "version": "test",
        }

    def test_registry_failure_keeps_app_up_but_not_ready(self):
        with (
            patch("app.model.MlflowClient", side_effect=RuntimeError("registry down")),
            TestClient(app) as test_client,
        ):
            assert test_client.get("/ready").status_code == 503
            assert test_client.get("/").status_code == 200

    def test_loads_champion_even_when_newer_candidate_exists(self):
        approved = SimpleNamespace(version="1", tags={"approved": "true"})
        candidate = SimpleNamespace(version="2", tags={"approved": "false"})
        registry = MagicMock()
        registry.get_model_version_by_alias.side_effect = lambda name, alias: {
            "champion": approved
        }[alias]
        registry.search_model_versions.return_value = [approved, candidate]
        with (
            patch("app.model.MlflowClient", return_value=registry),
            patch("app.model.load_model", return_value=StubModel()) as loader,
            TestClient(app) as test_client,
        ):
            info = test_client.get("/model-info").json()
            prediction = test_client.post("/predict", json=VALID).json()
        loader.assert_called_once_with("models:/shoppredict/1")
        assert info == {"name": "shoppredict", "alias": "champion", "version": "1"}
        assert prediction == {"probability": 0.7, "purchase": True, "model_version": "1"}

    def test_alias_is_resolved_with_environment_names(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "other")
        monkeypatch.setenv("MODEL_ALIAS", "staging")
        registry = MagicMock()
        registry.get_model_version_by_alias.return_value = SimpleNamespace(version=7)
        with (
            patch("app.model.MlflowClient", return_value=registry),
            patch("app.model.load_model", return_value=StubModel()) as loader,
            TestClient(app) as test_client,
        ):
            info = test_client.get("/model-info").json()
        registry.get_model_version_by_alias.assert_called_once_with("other", "staging")
        loader.assert_called_once_with("models:/other/7")
        assert info == {"name": "other", "alias": "staging", "version": "7"}
