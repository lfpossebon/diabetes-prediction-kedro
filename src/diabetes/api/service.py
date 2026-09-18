"""Service layer: every interaction with Kedro lives here.

Three scopes, three lifetimes (the rule the API is built around):

* **Bootstrap** — once per process. Discovers ``settings.py``, the hooks and the
  pipeline registry.
* **Session** — once per pipeline run. Builds a *fresh* DataCatalog, so two
  concurrent requests can never see each other's data.
* **Context** — only when the catalog has to be mutated, which is just the
  online-inference path.

Sharing a session across requests would mean sharing one catalog: the overrides
below would leak into the next /train, and concurrent writes would corrupt it.
"""

import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from kedro.framework.project import configure_project, pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner

logger = logging.getLogger(__name__)

# api -> diabetes -> src -> project root
PROJECT_PATH = Path(__file__).resolve().parents[3]
PACKAGE_NAME = "diabetes"

TRAINING_PIPELINES = ("data_engineering", "modelling", "refit")
BATCH_INFERENCE_PIPELINES = ("inference",)

PRODUCTION_ARTEFACTS = (
    "production_model.pkl",
    "production_imputers.pkl",
    "production_outlier_caps.pkl",
    "production_encoders.pkl",
    "production_scalers.pkl",
)

# Every dataset the inference pipeline would otherwise persist. They are
# CSVDatasets pointing at fixed paths, so two concurrent requests would write
# over each other; in the online path they all become memory-only.
ONLINE_MEMORY_DATASETS = (
    "raw_inference_dataframe",
    "cleaned_inference_data",
    "imputed_inference_data",
    "capped_inference_data",
    "featured_inference_data",
    "encoded_inference_data",
    "scaled_inference_data",
    "inference_predictions",
)

# An Event rather than a bool: same double-checked locking, no global statement.
_bootstrapped = threading.Event()
_bootstrap_lock = threading.Lock()

# Shared mutable state -> always behind its lock.
_runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ensure_bootstrap() -> None:
    """Configure the Kedro project exactly once per process.

    Double-checked locking: the flag is read once outside the lock (the fast
    path taken by every request after the first) and again inside it, so two
    threads arriving together cannot both run the bootstrap.
    """
    if _bootstrapped.is_set():
        return

    with _bootstrap_lock:
        if _bootstrapped.is_set():
            return
        bootstrap_project(PROJECT_PATH)
        configure_project(PACKAGE_NAME)
        _bootstrapped.set()
        logger.info("Kedro project bootstrapped from %s", PROJECT_PATH)


def production_artefacts_available() -> bool:
    """Whether the refit pipeline has already produced what /inference needs."""
    models_dir = PROJECT_PATH / "data" / "06_models"
    return all((models_dir / name).exists() for name in PRODUCTION_ARTEFACTS)


# --------------------------------------------------------------- run registry


def _register_run(kind: str, pipeline_names: tuple[str, ...]) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _runs_lock:
        _runs[run_id] = {
            "run_id": run_id,
            "kind": kind,
            "status": "running",
            "pipelines": list(pipeline_names),
            "started_at": _now(),
            "finished_at": None,
            "error": None,
        }
    return run_id


def _finish_run(run_id: str, error: str | None = None) -> None:
    with _runs_lock:
        run = _runs.get(run_id)
        if run is None:
            return
        run["status"] = "failed" if error else "completed"
        run["finished_at"] = _now()
        run["error"] = error


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return a *copy* of the run record, so callers cannot mutate shared state."""
    with _runs_lock:
        run = _runs.get(run_id)
        return dict(run) if run is not None else None


# ------------------------------------------------------------ background jobs


def _run_pipelines(run_id: str, pipeline_names: tuple[str, ...]) -> None:
    """Run pipelines in order, one KedroSession each.

    A session can only run once, so a loop over sessions is the way to chain
    pipelines — not one session running several times.
    """
    try:
        ensure_bootstrap()
        for name in pipeline_names:
            with KedroSession.create(project_path=PROJECT_PATH) as session:
                session.run(pipeline_name=name)
            logger.info("run %s: pipeline '%s' completed", run_id, name)
    except Exception as exc:  # noqa: BLE001 - recorded in the run registry
        logger.exception("run %s failed", run_id)
        _finish_run(run_id, error=f"{type(exc).__name__}: {exc}")
    else:
        _finish_run(run_id)


def _start_background(kind: str, pipeline_names: tuple[str, ...]) -> dict[str, Any]:
    """Kick off a long job on a daemon thread and return immediately.

    Deliberately not a thread-pool worker: those exist to serve short requests,
    and a grid search would occupy one for minutes.
    """
    ensure_bootstrap()
    run_id = _register_run(kind, pipeline_names)
    thread = threading.Thread(
        target=_run_pipelines,
        args=(run_id, pipeline_names),
        daemon=True,
        name=f"{kind}-{run_id}",
    )
    thread.start()
    logger.info("run %s started (%s): %s", run_id, kind, ", ".join(pipeline_names))
    return {"run_id": run_id, "status": "running", "pipelines": list(pipeline_names)}


def start_training() -> dict[str, Any]:
    """Run data_engineering -> modelling -> refit in the background."""
    return _start_background("train", TRAINING_PIPELINES)


def start_batch_inference() -> dict[str, Any]:
    """Score the inference file declared in the catalog, in the background."""
    return _start_background("batch-inference", BATCH_INFERENCE_PIPELINES)


# --------------------------------------------------------------- online path


def predict_online(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score records that arrived over HTTP, touching no files.

    The request payload is injected into the catalog as a MemoryDataset and the
    outputs are captured the same way, so the *same* inference pipeline that
    reads a CSV in batch mode runs here unchanged. Only the production
    artefacts are still loaded from disk — which is exactly what we want.
    """
    ensure_bootstrap()

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = session.load_context().catalog

        catalog["raw_inference_data"] = MemoryDataset(data=pd.DataFrame(instances))
        for name in ONLINE_MEMORY_DATASETS:
            catalog[name] = MemoryDataset()

        SequentialRunner().run(pipelines["inference"], catalog)

        predictions = catalog.load("inference_predictions")

    logger.info("online inference scored %d instances", len(predictions))
    return predictions
