"""Pydantic contracts for the API.

These validate types, shapes and plausible ranges — the business rules stay
in the pipelines. Keeping them here means the request/response contract is
readable in one place, and FastAPI turns them into the OpenAPI docs served at
/docs for free.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    production_model_available: bool = Field(
        description="Whether the refit pipeline has produced the production artefacts "
        "that /inference needs."
    )


class Patient(BaseModel):
    """One patient, with the raw column names of the training CSV.

    Glucose, BMI and Age are required: they are the model's strongest signals,
    and without them the score would be little more than the imputed median
    patient — who sits above the decision threshold and would be flagged.
    The other measurements may be omitted, sent as ``null`` or, as in the
    training data, as ``0`` to mean "not measured"; they are then imputed.

    Unknown fields are rejected, so a typo such as ``glucose`` fails loudly
    instead of being silently imputed.
    """

    model_config = ConfigDict(extra="forbid")

    Pregnancies: int | None = Field(default=None, ge=0, le=20)
    Glucose: float = Field(gt=0, le=600, description="2-hour plasma glucose, mg/dL.")
    BloodPressure: float | None = Field(
        default=None, ge=0, le=250, description="Diastolic, mm Hg. 0 = not measured."
    )
    SkinThickness: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Triceps skinfold, mm. 0 = not measured.",
    )
    Insulin: float | None = Field(
        default=None,
        ge=0,
        le=1000,
        description="2-hour serum insulin, mu U/ml. 0 = not measured.",
    )
    BMI: float = Field(gt=0, le=90, description="kg/m².")
    DiabetesPedigreeFunction: float | None = Field(default=None, ge=0, le=3)
    Age: int = Field(
        ge=21, le=120, description="Years. The training population is 21 or older."
    )


class InferenceRequest(BaseModel):
    instances: list[Patient] = Field(
        min_length=1,
        description="One object per patient. Out-of-range values, unknown "
        "fields and a missing Glucose, BMI or Age are rejected with 422.",
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

    in_sample: bool = Field(
        description="True when the model was fitted on this split: a training "
        "score, not a holdout one."
    )
    threshold: float = Field(description="decision.threshold used for the labels.")
    accuracy: float
    roc_auc: float
    roc_auc_ci: list[float] = Field(
        description="Bootstrap confidence interval for roc_auc, [low, high]."
    )
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
