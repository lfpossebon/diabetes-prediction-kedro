"""End-to-end tests for the API, driven through TestClient (no server needed)."""

import json

import pytest
from fastapi.testclient import TestClient

from diabetes.api import app
from diabetes.api.service import PROJECT_PATH, read_dataset

PATIENT = {
    "Pregnancies": 1,
    "Glucose": 89,
    "BloodPressure": 66,
    "SkinThickness": 23,
    "Insulin": 94,
    "BMI": 28.1,
    "DiabetesPedigreeFunction": 0.167,
    "Age": 21,
}

HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_NOT_FOUND = 404
HTTP_UNPROCESSABLE = 422

N_INSTANCES = 2


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_reports_ok_and_artefact_availability(self, client):
        response = client.get("/health")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["production_model_available"], bool)


class TestOnlineInference:
    def test_scores_every_instance(self, client):
        response = client.post("/inference", json={"instances": [PATIENT, PATIENT]})

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["n"] == N_INSTANCES
        assert [p["index"] for p in body["predictions"]] == [0, 1]
        for item in body["predictions"]:
            assert item["prediction"] in (0, 1)
            assert 0.0 <= item["probability"] <= 1.0

    def test_identical_inputs_score_identically(self, client):
        """Cheap determinism check: no state leaks between rows or requests."""
        body = client.post("/inference", json={"instances": [PATIENT, PATIENT]}).json()

        first, second = body["predictions"]
        assert first["probability"] == second["probability"]

    def test_high_glucose_and_bmi_score_higher(self, client):
        """The strongest signals in the EDA must survive the whole pipeline."""
        at_risk = {**PATIENT, "Glucose": 185, "BMI": 42.0, "Age": 55, "Pregnancies": 6}
        body = client.post("/inference", json={"instances": [PATIENT, at_risk]}).json()

        healthy_prob, at_risk_prob = (p["probability"] for p in body["predictions"])
        assert at_risk_prob > healthy_prob

    def test_missing_fields_and_zeros_are_imputed(self, client):
        """Unmeasured values (absent or 0) must be imputed, not crash the request."""
        partial = {"Glucose": 120, "BMI": 0, "Age": 35}
        response = client.post("/inference", json={"instances": [partial]})

        assert response.status_code == HTTP_OK
        assert 0.0 <= response.json()["predictions"][0]["probability"] <= 1.0

    def test_empty_payload_is_rejected(self, client):
        response = client.post("/inference", json={"instances": []})

        assert response.status_code == HTTP_UNPROCESSABLE

    def test_nothing_is_written_to_disk(self, client):
        """The whole point of the MemoryDataset overrides."""
        watched = [
            PROJECT_PATH
            / "data"
            / "02_intermediate"
            / "cleaned_diabetes-inference.csv",
            PROJECT_PATH / "data" / "03_primary" / "imputed_diabetes-inference.csv",
            PROJECT_PATH / "data" / "03_primary" / "capped_diabetes-inference.csv",
            PROJECT_PATH / "data" / "04_feature" / "featured_diabetes-inference.csv",
            PROJECT_PATH / "data" / "04_feature" / "encoded_diabetes-inference.csv",
            PROJECT_PATH / "data" / "05_model_input" / "scaled_diabetes-inference.csv",
            PROJECT_PATH / "data" / "07_model_output" / "inference_predictions.json",
        ]
        before = {
            path: path.stat().st_mtime_ns if path.exists() else None for path in watched
        }

        client.post("/inference", json={"instances": [PATIENT]})

        after = {
            path: path.stat().st_mtime_ns if path.exists() else None for path in watched
        }
        assert before == after


class TestTrainingRuns:
    def test_unknown_run_id_is_404(self, client):
        response = client.get("/train/does-not-exist")

        assert response.status_code == HTTP_NOT_FOUND

    def test_train_returns_a_run_id_immediately(self, client, mocker):
        """Must return without waiting for the grid search to finish.

        The worker is stubbed out: running the real pipelines here would write
        to data/ from a daemon thread that pytest kills on exit, which can
        leave half-written CSVs behind. What is under test is the HTTP
        contract, not the pipelines — those have their own tests.
        """
        worker = mocker.patch("diabetes.api.service._run_pipelines")

        response = client.post("/train")

        assert response.status_code == HTTP_ACCEPTED
        body = response.json()
        assert body["status"] == "running"
        assert body["pipelines"] == ["data_engineering", "modelling", "refit"]

        # The run is registered and queryable straight away.
        status_response = client.get(f"/train/{body['run_id']}")
        assert status_response.status_code == HTTP_OK
        assert status_response.json()["kind"] == "train"

        worker.assert_called_once()

    def test_batch_inference_starts_the_inference_pipeline(self, client, mocker):
        worker = mocker.patch("diabetes.api.service._run_pipelines")

        response = client.post("/batch-inference")

        assert response.status_code == HTTP_ACCEPTED
        assert response.json()["pipelines"] == ["inference"]
        worker.assert_called_once()


class TestDatasets:
    def test_predictions_serve_the_catalog_dataset(self, client):
        """Same content as the file the inference pipeline wrote."""
        path = PROJECT_PATH / "data" / "07_model_output" / "inference_predictions.json"
        on_disk = json.loads(path.read_text())

        response = client.get("/predictions")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["n"] == len(on_disk)
        assert body["predictions"] == on_disk

    def test_metrics_cover_every_split(self, client):
        response = client.get("/metrics/optimized")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert set(body) == {"train", "test", "validate"}
        cm = body["validate"]["confusion_matrix"]
        assert (
            cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == body["validate"]["n_samples"]
        )

    def test_unknown_model_is_rejected(self, client):
        response = client.get("/metrics/xgboost")

        assert response.status_code == HTTP_UNPROCESSABLE

    def test_dataset_not_written_yet_is_404(self, client, mocker):
        mocker.patch("diabetes.api.main.read_dataset", return_value=None)

        response = client.get("/predictions")

        assert response.status_code == HTTP_NOT_FOUND

    def test_only_whitelisted_datasets_are_served(self):
        """Pickled models and patient-level tables must never leave the server."""
        with pytest.raises(ValueError, match="not exposed"):
            read_dataset("production_model")
