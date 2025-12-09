import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference_service import app as inference_app
from src.inference_service.dummy_data import build_dummy_features
from src.inference_service.model_loader import DummyModel, load_model, predict
from src.inference_service.schemas import PredictionRequest


class _SentinelModel:
    pass


def test_load_model_success(monkeypatch):
    sentinel = _SentinelModel()

    def _fake_load_model(uri):
        return sentinel

    monkeypatch.setattr("mlflow.pyfunc.load_model", _fake_load_model)
    assert load_model("models:/anything") is sentinel


def test_load_model_fallback_to_dummy(monkeypatch):
    def _fake_load_model(uri):
        raise RuntimeError("boom")

    monkeypatch.setattr("mlflow.pyfunc.load_model", _fake_load_model)
    model = load_model("models:/anything")
    assert isinstance(model, DummyModel)


def test_predict_with_valid_payload():
    class _FakeModel:
        def __init__(self):
            self.metadata = types.SimpleNamespace(run_id="run-123")

        def predict(self, df):
            return [1.5]

    payload = PredictionRequest(
        home_team="KC", away_team="BUF", game_id="week1", location="KC", spread=-3.5
    )
    resp = predict(_FakeModel(), payload)
    assert resp.predicted_spread == pytest.approx(1.5)
    assert resp.model_version == "run-123"
    assert resp.input == payload


def test_build_dummy_features_structure():
    payload = PredictionRequest(
        home_team="KC", away_team="BUF", game_id="week1", location="KC", spread=-3.5
    )
    features = build_dummy_features(payload)
    # Expect a non-empty all-numeric mapping matching the booster feature list
    expected_keys = {
        "season_year",
        "avg_epa",
        "success_rate",
        "pct_drives_scored",
        "avg_drive_yards_penalized",
        "total_epa",
        "penalty_yards_total",
        "temp",
        "wind",
        "pass_epa_mean",
        "pass_yards_avg",
        "comp_pct",
        "rush_epa_mean",
        "rush_yards_avg",
        "td_rate_in_redzone",
        "pct_plays_over_15yds",
        "turnover_count",
        "third_down_success_rate",
        "is_home",
        "is_turf",
        "posteam_id",
        "roof_type_open",
        "roof_type_closed",
        "roof_type_dome",
        "roof_type_outdoors",
    }
    assert set(features.keys()) == expected_keys
    assert all(isinstance(v, (int, float)) for v in features.values())


def test_predict_endpoint_error_handling(monkeypatch):
    class _FailingModel:
        def predict(self, df):
            raise RuntimeError("bad model")

    inference_app.model = _FailingModel()
    client = TestClient(inference_app.app)
    payload = {
        "home_team": "KC",
        "away_team": "BUF",
        "game_id": "week1",
        "location": "KC",
        "spread": 1.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Prediction failed"
