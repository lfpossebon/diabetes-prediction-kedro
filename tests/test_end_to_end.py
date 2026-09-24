"""End-to-end run of all four pipelines on the real raw data, entirely in memory.

The unit tests check each node and test_run.py checks the DAG wiring; this is
the one test that pushes the actual CSVs through every node, with the real
parameters. Every dataset except the two raw CSVs is swapped for a
MemoryDataset, so nothing under data/ is read or written, and the grid search
is shrunk to a single candidate to keep it fast.
"""

from pathlib import Path

import pandas as pd
import pytest
from kedro.framework.session import KedroSession
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner

from diabetes.pipeline_registry import register_pipelines

PROJECT_PATH = Path(__file__).parent.parent
RAW_DATASETS = {"raw_diabetes_data", "raw_inference_data"}


@pytest.fixture(scope="module")
def catalog():
    pipeline = register_pipelines()["__default__"]

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        catalog = context.catalog

        for name in pipeline.datasets():
            if not name.startswith("params:") and name not in RAW_DATASETS:
                catalog[name] = MemoryDataset()

        params = context.params
        catalog["params:modelling_optimization"] = MemoryDataset(
            {
                **params["modelling_optimization"],
                "param_grid": {"n_estimators": [50], "max_depth": [5]},
                "n_jobs": 1,
            }
        )
        catalog["params:evaluation"] = MemoryDataset(
            {**params["evaluation"], "n_bootstrap": 100}
        )

        SequentialRunner().run(pipeline, catalog)
        yield catalog


def test_every_inference_row_is_scored(catalog):
    raw = pd.read_csv(
        PROJECT_PATH / "data" / "01_raw" / "diabetes-dataset-inference.csv"
    )

    predictions = catalog.load("inference_predictions")

    assert len(predictions) == len(raw)
    assert all(0.0 <= p["probability"] <= 1.0 for p in predictions)


def test_champion_is_chosen_with_the_real_parameters(catalog):
    """select_champion raises on an in-sample split, so reaching here proves the
    configured comparison split is a holdout for both models."""
    report = catalog.load("champion_report")

    assert report["champion"] in ("baseline", "optimized")
    assert report["split"] == catalog.load("params:champion_selection")["split"]


def test_odds_ratios_describe_the_champion(catalog):
    """production_model itself is an intermediate the runner has released, so
    the champion report stands in for it."""
    report = catalog.load("production_odds_ratios")
    champion = catalog.load("champion_report")

    assert report["model"] == champion["estimator"]
    if champion["champion"] == "baseline":
        features = catalog.load("params:modelling_baseline")["features"]
        assert [r["feature"] for r in report["odds_ratios"]] == features


def test_every_impaired_glucose_tolerance_is_flagged(catalog):
    """The inference file comes from the same cohort: no 2-hour glucose >= 200,
    and no fasting glucose or HbA1c, so no row meets a diagnostic criterion.
    Every 2-hour glucose of 140-199 is flagged, by the model or the guideline."""
    raw = pd.read_csv(
        PROJECT_PATH / "data" / "01_raw" / "diabetes-dataset-inference.csv"
    )
    predictions = catalog.load("inference_predictions")

    assert "diagnostic_criterion" not in {p["decision_basis"] for p in predictions}
    impaired = raw["Glucose"].between(140, 199).to_numpy()
    assert impaired.any()
    assert all(p["prediction"] == 1 for p, igt in zip(predictions, impaired) if igt)


def test_threshold_curve_reports_the_configured_cut_off(catalog):
    curve = catalog.load("threshold_curve")
    threshold = catalog.load("params:decision")["threshold"]

    assert curve["configured_threshold"]["threshold"] == pytest.approx(threshold)
    served = curve["referral_strategies"]["model_or_guideline"]
    assert served["sensitivity"] >= curve["configured_threshold"]["sensitivity"]
    assert (
        served["sensitivity"]
        >= curve["referral_strategies"]["guideline"]["sensitivity"]
    )
