"""Pydantic contracts for the API.

These validate types, shapes and plausible ranges — the business rules stay
in the pipelines. Keeping them here means the request/response contract is
readable in one place, and FastAPI turns them into the OpenAPI docs served at
/docs for free.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# Answering True to any of these puts the patient outside the population the
# model was developed on; the reason is returned with the 422.
EXCLUSIONS = {
    "Pregnant": "the model does not apply in pregnancy, when the OGTT is read "
    "with the gestational diabetes criteria (IADPSG: fasting >= 92, 1 h >= 180, "
    "2 h >= 153 mg/dL)",
    "KnownDiabetes": "the model forecasts diabetes in women who do not have it",
    "GlucoseAffectingMedication": "glucose-lowering drugs and systemic "
    "glucocorticoids change the OGTT result the model was trained on",
}


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    production_model_available: bool = Field(
        description="Whether the refit pipeline has produced the production artefacts "
        "that /inference needs."
    )


class Patient(BaseModel):
    """One patient, with the raw column names of the training CSV.

    The model answers one clinical question: the risk that a non-diabetic,
    non-pregnant woman aged 21 to 81, who has just had a 75 g oral glucose
    tolerance test, develops diabetes within five years. The schema holds
    requests to that population: ``Sex`` must be ``"female"``, since the
    training data holds women only, ``Age`` must lie within the training
    range, and ``Pregnant``, ``KnownDiabetes`` and
    ``GlucoseAffectingMedication`` must be answered, and answered ``false``.
    They are required on purpose: a default would assume the patient is
    eligible without anyone having asked.

    A 2-hour glucose of 200 mg/dL or more, a fasting glucose of 126 or more
    and an HbA1c of 6.5% or more are accepted, but each is diabetes by
    definition: the pipeline reports it as meeting a diagnostic criterion,
    not as a risk.

    Glucose, BMI and Age are required: they are the model's strongest signals,
    and without them the score would describe the imputed median patient
    rather than the woman being assessed.
    The other measurements may be omitted, sent as ``null`` or, as in the
    training data, as ``0`` to mean "not measured"; they are then imputed.
    ``DiabetesPedigreeFunction`` is not an input: the model does without it,
    since no clinic can compute it.

    Unknown fields are rejected, so a typo such as ``glucose`` fails loudly
    instead of being silently imputed.
    """

    model_config = ConfigDict(extra="forbid")

    Sex: Literal["female"] = Field(
        description="The model was developed on women only; any other value is "
        "rejected."
    )
    Pregnant: bool = Field(
        description="Whether the patient is pregnant. True is rejected: "
        + EXCLUSIONS["Pregnant"]
        + "."
    )
    KnownDiabetes: bool = Field(
        description="An earlier diagnosis of diabetes. True is rejected: "
        + EXCLUSIONS["KnownDiabetes"]
        + "."
    )
    GlucoseAffectingMedication: bool = Field(
        description="Taking a glucose-lowering drug (metformin included) or a "
        "systemic glucocorticoid. True is rejected: "
        + EXCLUSIONS["GlucoseAffectingMedication"]
        + "."
    )
    Pregnancies: int | None = Field(default=None, ge=0, le=20)
    Glucose: float = Field(
        gt=0,
        le=600,
        description="2-hour plasma glucose in a 75 g OGTT, mg/dL. At 200 or more "
        "the patient meets the diagnostic criterion and the model is not used.",
    )
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
    Age: int = Field(
        ge=21, le=81, description="Years. The training population is 21 to 81."
    )
    FastingGlucose: float | None = Field(
        default=None,
        gt=0,
        le=600,
        description="Fasting plasma glucose, mg/dL, if measured. Not a model "
        "input: at 126 or more the patient meets the diagnostic criterion.",
    )
    HbA1c: float | None = Field(
        default=None,
        ge=3,
        le=20,
        description="Glycated haemoglobin, %, if measured. Not a model input: "
        "at 6.5 or more the patient meets the diagnostic criterion.",
    )

    @field_validator(*EXCLUSIONS)
    @classmethod
    def _within_the_model_population(cls, value: bool, info: ValidationInfo) -> bool:
        if value:
            raise ValueError(f"{info.field_name} = true: {EXCLUSIONS[info.field_name]}")
        return value


class InferenceRequest(BaseModel):
    instances: list[Patient] = Field(
        min_length=1,
        description="One object per patient. Out-of-range values, unknown "
        "fields, a Sex other than female, a missing Glucose, BMI or Age, and a "
        "patient who is pregnant, has known diabetes or takes a medication "
        "that affects glucose are rejected with 422.",
        examples=[
            [
                {
                    "Sex": "female",
                    "Pregnant": False,
                    "KnownDiabetes": False,
                    "GlucoseAffectingMedication": False,
                    "Pregnancies": 6,
                    "Glucose": 128,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "Age": 50,
                    "FastingGlucose": 98,
                }
            ]
        ],
    )


class Prediction(BaseModel):
    index: int
    prediction: int = Field(
        description="1 = flagged, 0 = not flagged. With decision_basis 'model' "
        "or 'impaired_glucose_tolerance', flagged means refer for prevention "
        "and yearly follow-up. With 'diagnostic_criterion', the patient already "
        "meets a diabetes criterion: confirm the diagnosis and treat."
    )
    probability: float | None = Field(
        description="Model estimate of the risk of developing diabetes within "
        "five years, calibrated on the development cohort (Pima women, about "
        "35% of whom developed diabetes within five years). It ranks patients "
        "anywhere, but it is an absolute risk only in a population with that "
        "incidence: with a lower one it overstates the risk (0.30 here is "
        "about 0.08 at a 10% incidence) until the model is recalibrated on "
        "local data. None when decision_basis is 'diagnostic_criterion', "
        "since the model is not used then."
    )
    decision_basis: Literal[
        "model", "impaired_glucose_tolerance", "diagnostic_criterion"
    ] = Field(
        description="'model': the prediction applies decision.threshold to the "
        "probability. 'impaired_glucose_tolerance': 2-hour glucose 140-199 "
        "mg/dL with a risk below the threshold, flagged anyway because the "
        "guidelines refer every patient with impaired glucose tolerance. "
        "'diagnostic_criterion': at least one ADA diabetes criterion is met; "
        "see criteria_met."
    )
    criteria_met: list[str] = Field(
        default_factory=list,
        description="The diagnostic criteria the patient met, e.g. "
        "'HbA1c >= 6.5'. Empty unless decision_basis is 'diagnostic_criterion'.",
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
    recall: float = Field(description="Sensitivity.")
    specificity: float
    precision: float = Field(description="Positive predictive value (PPV).")
    npv: float = Field(description="Negative predictive value.")
    brier: float = Field(description="Brier score: mean squared error of the risk.")
    observed_expected: float = Field(
        description="Observed event rate over mean predicted risk; 1 = calibrated "
        "on average."
    )
    calibration_slope: float = Field(
        description="1 = ideal; below 1 means the predicted risks are too extreme."
    )
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
