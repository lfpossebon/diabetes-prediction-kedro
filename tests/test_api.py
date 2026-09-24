"""End-to-end tests for the API, driven through TestClient (no server needed)."""

import json

import pytest
from fastapi.testclient import TestClient

from diabetes.api import app, service
from diabetes.api.service import PROJECT_PATH, read_dataset

ELIGIBLE = {
    "Pregnant": False,
    "KnownDiabetes": False,
    "GlucoseAffectingMedication": False,
}

PATIENT = {
    "Sex": "female",
    **ELIGIBLE,
    "Pregnancies": 1,
    "Glucose": 89,
    "BloodPressure": 66,
    "SkinThickness": 23,
    "Insulin": 94,
    "BMI": 28.1,
    "Age": 21,
}

HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_UNPROCESSABLE = 422

N_INSTANCES = 2

DATA_DIR = PROJECT_PATH / "data"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def empty_run_registry():
    """Runs started with a stubbed worker never finish; do not let them leak."""
    service._runs.clear()
    yield
    service._runs.clear()


def _latest_version(dataset_file: str) -> str:
    """Contents of the newest saved version of a versioned catalog dataset."""
    versions = sorted((DATA_DIR / dataset_file).glob("*/*"))
    return versions[-1].read_text()


def _snapshot(root) -> dict:
    """Every file under ``root`` with its modification time."""
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


class TestHealth:
    def test_reports_ok_and_artefact_availability(self, client):
        response = client.get("/health")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["status"] == "ok"
        assert body["production_model_available"] is True


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
            assert item["decision_basis"] == "model"
            assert item["criteria_met"] == []

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

    def test_optional_fields_absent_null_or_zero_are_imputed(self, client):
        """Unmeasured values (absent, null or 0) are imputed, not rejected."""
        partial = {
            "Sex": "female",
            **ELIGIBLE,
            "Glucose": 120,
            "BMI": 30.5,
            "Age": 35,
            "Insulin": 0,
            "SkinThickness": None,
        }
        response = client.post("/inference", json={"instances": [partial]})

        assert response.status_code == HTTP_OK
        assert 0.0 <= response.json()["predictions"][0]["probability"] <= 1.0

    @pytest.mark.parametrize(
        "patient",
        [
            pytest.param({}, id="empty"),
            pytest.param({"glucose": 148, "bmi": 33.6, "age": 50}, id="typo-keys"),
            pytest.param({**PATIENT, "Cholesterol": 200}, id="unknown-field"),
            pytest.param({**PATIENT, "Glucose": "abc"}, id="non-numeric"),
            pytest.param({**PATIENT, "Glucose": None}, id="required-null"),
            pytest.param({**PATIENT, "Glucose": -50}, id="negative"),
            pytest.param({**PATIENT, "BMI": 0}, id="required-zero"),
            pytest.param({**PATIENT, "Age": 10}, id="below-training-ages"),
            pytest.param({**PATIENT, "Age": 82}, id="above-training-ages"),
            pytest.param({**PATIENT, "Sex": "male"}, id="not-a-woman"),
            pytest.param(
                {k: v for k, v in PATIENT.items() if k != "Sex"}, id="sex-missing"
            ),
            pytest.param({**PATIENT, "Insulin": 5000}, id="implausible"),
            pytest.param({**PATIENT, "HbA1c": 45}, id="hba1c-in-mmol-per-mol"),
            pytest.param(
                {**PATIENT, "DiabetesPedigreeFunction": 0.5},
                id="pedigree-function-is-not-an-input",
            ),
            *(
                pytest.param(
                    {k: v for k, v in PATIENT.items() if k != field},
                    id=f"{field}-unanswered",
                )
                for field in ELIGIBLE
            ),
        ],
    )
    def test_invalid_patients_are_rejected(self, client, patient):
        """Regression: these used to be imputed silently and scored — an empty
        record came back flagged as diabetic."""
        response = client.post("/inference", json={"instances": [patient]})

        assert response.status_code == HTTP_UNPROCESSABLE

    @pytest.mark.parametrize(
        ("field", "reason"),
        [
            ("Pregnant", "IADPSG"),
            ("KnownDiabetes", "do not have it"),
            ("GlucoseAffectingMedication", "glucocorticoids"),
        ],
    )
    def test_patients_outside_the_model_population_are_rejected(
        self, client, field, reason
    ):
        """The clinician gets the clinical reason, not just a type error."""
        response = client.post(
            "/inference", json={"instances": [{**PATIENT, field: True}]}
        )

        assert response.status_code == HTTP_UNPROCESSABLE
        assert reason in response.text

    @pytest.mark.parametrize(
        ("measured", "met"),
        [
            ({"Glucose": 230}, ["Glucose >= 200"]),
            ({"FastingGlucose": 131}, ["FastingGlucose >= 126"]),
            ({"HbA1c": 6.7}, ["HbA1c >= 6.5"]),
        ],
    )
    def test_diagnostic_values_are_reported_as_such_not_scored(
        self, client, measured, met
    ):
        """Any ADA criterion is diabetes by definition: no risk score, whatever
        the other measurements say, and the criterion to confirm is named."""
        diabetic = {**PATIENT, **measured}
        response = client.post("/inference", json={"instances": [diabetic, PATIENT]})

        assert response.status_code == HTTP_OK
        diagnosed, scored = response.json()["predictions"]
        assert diagnosed["decision_basis"] == "diagnostic_criterion"
        assert diagnosed["prediction"] == 1
        assert diagnosed["probability"] is None
        assert diagnosed["criteria_met"] == met
        assert scored["decision_basis"] == "model"

    def test_normal_fasting_glucose_and_hba1c_leave_the_model_in_charge(self, client):
        normal = {**PATIENT, "FastingGlucose": 92, "HbA1c": 5.4}
        (item,) = client.post("/inference", json={"instances": [normal]}).json()[
            "predictions"
        ]

        assert item["decision_basis"] == "model"
        assert item["probability"] is not None

    def test_impaired_glucose_tolerance_is_flagged_whatever_the_risk(self, client):
        """Young, lean and nulliparous, so the model alone would not flag her;
        a 2-hour glucose of 145 mg/dL is prediabetes, and the guideline does."""
        lean = {**PATIENT, "Glucose": 145, "BMI": 20.0, "Age": 21, "Pregnancies": 0}
        (item,) = client.post("/inference", json={"instances": [lean]}).json()[
            "predictions"
        ]

        assert item["prediction"] == 1
        assert item["decision_basis"] == "impaired_glucose_tolerance"
        assert item["probability"] is not None

    def test_empty_payload_is_rejected(self, client):
        response = client.post("/inference", json={"instances": []})

        assert response.status_code == HTTP_UNPROCESSABLE

    def test_missing_artefacts_is_409(self, client, mocker):
        mocker.patch(
            "diabetes.api.main.predict_online",
            side_effect=service.ArtefactsMissingError("missing"),
        )

        response = client.post("/inference", json={"instances": [PATIENT]})

        assert response.status_code == HTTP_CONFLICT

    def test_internal_errors_do_not_leak_details(self, client, mocker):
        mocker.patch(
            "diabetes.api.main.predict_online",
            side_effect=RuntimeError("/secret/path/production_model.pkl"),
        )

        response = client.post("/inference", json={"instances": [PATIENT]})

        assert "/secret/path" not in response.text

    def test_nothing_is_written_to_disk(self, client):
        """The whole point of the MemoryDataset overrides."""
        before = _snapshot(DATA_DIR)

        client.post("/inference", json={"instances": [PATIENT]})

        assert _snapshot(DATA_DIR) == before


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

    def test_second_training_run_is_refused_while_one_is_running(self, client, mocker):
        """Two concurrent refits would write the same production artefacts."""
        worker = mocker.patch("diabetes.api.service._run_pipelines")

        first = client.post("/train")
        second = client.post("/train")

        assert first.status_code == HTTP_ACCEPTED
        assert second.status_code == HTTP_CONFLICT
        assert first.json()["run_id"] in second.json()["detail"]
        worker.assert_called_once()

    def test_training_and_batch_inference_do_not_block_each_other(self, client, mocker):
        """Only runs of the same kind are exclusive; the artefact lock orders the rest."""
        mocker.patch("diabetes.api.service._run_pipelines")

        assert client.post("/train").status_code == HTTP_ACCEPTED
        assert client.post("/batch-inference").status_code == HTTP_ACCEPTED

    def test_batch_inference_starts_the_inference_pipeline(self, client, mocker):
        worker = mocker.patch("diabetes.api.service._run_pipelines")

        response = client.post("/batch-inference")

        assert response.status_code == HTTP_ACCEPTED
        assert response.json()["pipelines"] == ["inference"]
        worker.assert_called_once()


class TestRunWorker:
    """_run_pipelines itself, synchronously, with KedroSession stubbed out."""

    @pytest.fixture
    def sessions(self, mocker):
        """Record, for each pipeline run, whether the artefacts lock was held."""
        runs: list[tuple[str, bool]] = []

        def run(pipeline_name):
            runs.append((pipeline_name, service._artefacts_lock.locked()))

        session = mocker.MagicMock()
        session.run.side_effect = run
        create = mocker.patch("diabetes.api.service.KedroSession.create")
        create.return_value.__enter__.return_value = session
        return runs, session

    def test_artefact_pipelines_hold_the_lock(self, sessions):
        """Refit writes and inference reads the production artefacts; the
        others must not block online scoring."""
        runs, _ = sessions
        run_id = service._register_run("train", service.TRAINING_PIPELINES)

        service._run_pipelines(run_id, (*service.TRAINING_PIPELINES, "inference"))

        assert runs == [
            ("data_engineering", False),
            ("modelling", False),
            ("refit", True),
            ("inference", True),
        ]
        assert not service._artefacts_lock.locked()
        assert service.get_run(run_id)["status"] == "completed"

    def test_failure_is_recorded_and_releases_the_lock(self, sessions):
        _, session = sessions
        session.run.side_effect = RuntimeError("boom")
        run_id = service._register_run("train", ("refit",))

        service._run_pipelines(run_id, ("refit",))

        run = service.get_run(run_id)
        assert run["status"] == "failed"
        assert run["error"] == "RuntimeError: boom"
        assert not service._artefacts_lock.locked()


class TestDatasets:
    def test_predictions_serve_the_catalog_dataset(self, client):
        """Same content as the latest version the inference pipeline wrote."""
        on_disk = json.loads(
            _latest_version("07_model_output/inference_predictions.json")
        )

        response = client.get("/predictions")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["n"] == len(on_disk)
        assert body["predictions"] == on_disk

    def test_metrics_flag_in_sample_splits(self, client):
        """The tuned model is fitted on train + test: only validate is a holdout."""
        response = client.get("/metrics/optimized")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["train"]["in_sample"] is True
        assert body["validate"]["in_sample"] is False
        low, high = body["validate"]["roc_auc_ci"]
        assert low <= body["validate"]["roc_auc"] <= high
        cm = body["validate"]["confusion_matrix"]
        assert (
            cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == body["validate"]["n_samples"]
        )

    def test_unknown_model_is_rejected(self, client):
        response = client.get("/metrics/xgboost")

        assert response.status_code == HTTP_UNPROCESSABLE

    def test_champion_report_names_the_production_model(self, client):
        response = client.get("/reports/champion")

        assert response.status_code == HTTP_OK
        assert response.json()["champion"] in ("baseline", "optimized")

    def test_odds_ratios_describe_the_production_model(self, client):
        response = client.get("/reports/odds_ratios")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["model"] in ("LogisticRegression", "RandomForestClassifier")
        if body["odds_ratios"] is not None:
            assert all(r["odds_ratio_per_unit"] > 0 for r in body["odds_ratios"])

    def test_threshold_curve_contains_the_configured_threshold(self, client):
        response = client.get("/reports/threshold_curve")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["configured_threshold"]["threshold"] in {
            point["threshold"] for point in body["curve"]
        }

    def test_dataset_not_written_yet_is_404(self, client, mocker):
        mocker.patch("diabetes.api.main.read_dataset", return_value=None)

        response = client.get("/predictions")

        assert response.status_code == HTTP_NOT_FOUND

    def test_only_whitelisted_datasets_are_served(self):
        """Pickled models and patient-level tables must never leave the server."""
        with pytest.raises(ValueError, match="not exposed"):
            read_dataset("production_model")
