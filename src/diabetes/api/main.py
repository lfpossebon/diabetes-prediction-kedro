"""FastAPI app: thin controllers over the service layer.

Every handler is a plain ``def``, never ``async def``. FastAPI dispatches plain
functions to the anyio thread pool, which keeps the event loop free; an
``async def`` without an ``await`` would run the CPU-bound Kedro work *on* the
loop and freeze the whole server. That is the single most important line in
this file.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from diabetes import __version__

from .schemas import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    RunStartedResponse,
    RunStatusResponse,
)
from .service import (
    ensure_bootstrap,
    get_run,
    predict_online,
    production_artefacts_available,
    start_batch_inference,
    start_training,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pay the bootstrap cost once, at startup, instead of on the first request."""
    ensure_bootstrap()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="Diabetes API",
        version=__version__,
        description=(
            "HTTP interface to the Kedro diabetes pipelines. Training runs in the "
            "background; online scoring reuses the very same inference pipeline "
            "as the batch run, with the payload injected into the catalog."
        ),
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            production_model_available=production_artefacts_available(),
        )

    @app.post(
        "/train",
        response_model=RunStartedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["training"],
    )
    def train() -> RunStartedResponse:
        """Start data_engineering -> modelling -> refit and return a run id."""
        return RunStartedResponse(**start_training())

    @app.get("/train/{run_id}", response_model=RunStatusResponse, tags=["training"])
    def train_status(run_id: str) -> RunStatusResponse:
        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return RunStatusResponse(**run)

    @app.post("/inference", response_model=InferenceResponse, tags=["inference"])
    def inference(request: InferenceRequest) -> InferenceResponse:
        """Score records synchronously, without writing anything to disk."""
        if not production_artefacts_available():
            raise HTTPException(
                status_code=409,
                detail="Production artefacts are missing. Run POST /train first.",
            )
        try:
            predictions = predict_online(request.instances)
        except Exception as exc:
            logger.exception("online inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return InferenceResponse(n=len(predictions), predictions=predictions)

    @app.post(
        "/batch-inference",
        response_model=RunStartedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["inference"],
    )
    def batch_inference() -> RunStartedResponse:
        """Score the inference file declared in the catalog, in the background."""
        return RunStartedResponse(**start_batch_inference())

    return app


app = create_app()
