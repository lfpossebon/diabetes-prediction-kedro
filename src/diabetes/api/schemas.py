"""Pydantic contracts for the API.

These validate types and shapes only — no business rules. Keeping them here
means the request/response contract is readable in one place, and FastAPI
turns them into the OpenAPI docs served at /docs for free.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    production_model_available: bool = Field(
        description="Whether the refit pipeline has produced the production artefacts "
        "that /inference needs."
    )


class InferenceRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        min_length=1,
        description="One object per patient, using the raw column names "
        "(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
        "DiabetesPedigreeFunction, Age). Unknown columns are ignored; missing "
        "ones, and zeros in the clinical measurements, are imputed.",
        examples=[
            [
                {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                }
            ]
        ],
    )


class Prediction(BaseModel):
    index: int
    prediction: int = Field(
        description="1 = flagged for a confirmatory diabetes test, 0 = not flagged. "
        "A patient is flagged when probability >= decision.threshold in "
        "conf/base/parameters.yml."
    )
    probability: float = Field(
        description="Probability of the positive (diabetes) class."
    )


class InferenceResponse(BaseModel):
    n: int
    predictions: list[Prediction]


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int = Field(description="Diabetic patients the model did not flag.")
    tp: int


class SplitMetrics(BaseModel):
    """One split of ``baseline_metrics`` / ``optimized_metrics``."""

    threshold: float = Field(description="decision.threshold used for the labels.")
    accuracy: float
    roc_auc: float
    f1_macro: float
    recall: float
    precision: float
    confusion_matrix: ConfusionMatrix
    classification_report: dict[str, Any]
    n_samples: int


class RunStartedResponse(BaseModel):
    run_id: str
    status: Literal["running"]
    pipelines: list[str]


class RunStatusResponse(BaseModel):
    run_id: str
    kind: str
    status: Literal["running", "completed", "failed"]
    pipelines: list[str]
    started_at: str
    finished_at: str | None = None
    error: str | None = None
