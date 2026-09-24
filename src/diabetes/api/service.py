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

The production artefacts on disk are shared, though, so one lock guards them:
the refit and batch-inference runs hold it while they write or read them, and
online inference holds it only while it loads them into memory. A request can
therefore never score with new imputers and an old model, or read a pickle
that is still being written. The lock is per process: it does not cover a
`kedro run` launched from a shell while the API is serving.
"""

import contextlib
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

# Catalog datasets the inference pipeline loads — and the refit pipeline writes.
PRODUCTION_DATASETS = (
    "production_imputers",
    "production_outlier_caps",
    "production_encoders",
    "production_scalers",
    "production_model",
)

# Pipelines that write or read PRODUCTION_DATASETS, and must hold the lock.
ARTEFACT_PIPELINES = frozenset({"refit", "inference"})

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

_artefacts_lock = threading.Lock()


class RunInProgressError(RuntimeError):
    """A run of the same kind is still going; starting another would race it."""


class ArtefactsMissingError(RuntimeError):
    """The refit pipeline has not produced the production artefacts yet."""


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
    """Whether the refit pipeline has already produced what /inference needs.

    Asked of the catalog, not of the file system, so it keeps working when
    ``catalog.yml`` versions the artefacts or moves them to object storage.
    """
    ensure_bootstrap()
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = session.load_context().catalog
        return all(catalog.exists(name) for name in PRODUCTION_DATASETS)


# ------------------------------------------------------- read-only datasets

# Catalog datasets the API may serve. A whitelist, never the catalog as a
# whole: it also holds pickled models and intermediate patient-level tables.
EXPOSED_DATASETS = frozenset(
    {
        "inference_predictions",
        "baseline_metrics",
        "optimized_metrics",
        "champion_report",
        "threshold_curve",
    }
)


def read_dataset(name: str) -> Any | None:
    """Load a whitelisted catalog dataset, or ``None`` if no run has written it yet.

    Goes through the catalog instead of opening the file, so where the dataset
    lives and in which format stays a ``catalog.yml`` concern — exactly as it
    is for the pipelines.
    """
    if name not in EXPOSED_DATASETS:
        raise ValueError(f"Dataset not exposed by the API: {name}")

    ensure_bootstrap()
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = session.load_context().catalog
        if not catalog.exists(name):
            return None
        return catalog.load(name)


# --------------------------------------------------------------- run registry


def _register_run(kind: str, pipeline_names: tuple[str, ...]) -> str:
    """Record a new run, refusing it while another run of the same kind is active.

    Check and insert happen under one lock acquisition, so two requests
    arriving together cannot both pass the check.
    """
    run_id = uuid.uuid4().hex[:12]
    with _runs_lock:
        active = next(
            (
                r["run_id"]
                for r in _runs.values()
                if r["kind"] == kind and r["status"] == "running"
            ),
            None,
        )
        if active is not None:
            raise RunInProgressError(f"A {kind} run is already in progress: {active}")
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
            guard = (
                _artefacts_lock
                if name in ARTEFACT_PIPELINES
                else contextlib.nullcontext()
            )
            with guard, KedroSession.create(project_path=PROJECT_PATH) as session:
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

    Raises:
        RunInProgressError: If a run of the same kind has not finished yet.
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
    reads a CSV in batch mode runs here unchanged.

    The production artefacts are loaded from the catalog up front, all under
    the artefacts lock, and injected as MemoryDatasets: the five of them come
    from one consistent refit even if a /train finishes mid-request, and the
    lock is held for a few milliseconds rather than the whole run.

    Raises:
        ArtefactsMissingError: If the refit pipeline has not run yet.
    """
    ensure_bootstrap()

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = session.load_context().catalog

        with _artefacts_lock:
            if not all(catalog.exists(name) for name in PRODUCTION_DATASETS):
                raise ArtefactsMissingError(
                    "Production artefacts are missing. Run POST /train first."
                )
            for name in PRODUCTION_DATASETS:
                catalog[name] = MemoryDataset(
                    data=catalog.load(name), copy_mode="assign"
                )

        catalog["raw_inference_data"] = MemoryDataset(data=pd.DataFrame(instances))
        for name in ONLINE_MEMORY_DATASETS:
            catalog[name] = MemoryDataset()

        SequentialRunner().run(pipelines["inference"], catalog)

        predictions = catalog.load("inference_predictions")

    logger.info("online inference scored %d instances", len(predictions))
    return predictions
