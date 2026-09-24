"""Nodes for the 'modelling' pipeline.

The estimator is never hard-coded: ``class_path`` in ``parameters.yml`` is
resolved at runtime by :func:`_load_class`, so swapping LogisticRegression for
a RandomForest — or adding a third model — is a YAML change, not a code change.

``train_model`` and ``optimize_hyperparameters`` return the same artefact
shape, which is why a single ``evaluate_model`` serves both, and why
``select_champion`` can hand either one to the refit pipeline.

Metrics are reported in the terms a clinician reads: sensitivity (recall),
specificity, PPV (precision) and NPV, plus calibration, since a risk score is
only useful if its probabilities can be taken at face value.
"""

import importlib
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)

SPLIT_COLUMN = "split"


def _load_class(class_path: str) -> type:
    """Resolve a fully-qualified class path to the class object.

    Args:
        class_path: Dotted path such as
            ``"sklearn.linear_model.LogisticRegression"``.

    Returns:
        The class object.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def feature_columns(
    master_table: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
) -> list[str]:
    """The columns a model is fitted on.

    A model configured with a ``features`` list uses exactly those columns: the
    parsimonious baseline picks a handful of raw measurements. Without one, it
    uses every master-table column except the target and the split label.
    One-hot encoding makes those names depend on the categories seen at fit
    time, so the table, not a hand-kept list, is then the source of truth.

    Raises:
        KeyError: If a configured feature is not in the master table.
    """
    if features:
        missing = [c for c in features if c not in master_table.columns]
        if missing:
            raise KeyError(f"Configured features not in the master table: {missing}")
        return list(features)
    return [c for c in master_table.columns if c not in (target, SPLIT_COLUMN)]


def train_model(
    master_table: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Fit a classifier on the specified splits and return a model artifact.

    Args:
        master_table: Fully-processed DataFrame with a ``split`` column and
            all feature and target columns.
        params: Training configuration with keys:
            - ``class_path`` (str): Fully-qualified sklearn estimator class.
            - ``init_args`` (dict, optional): Estimator constructor arguments.
            - ``target_column`` (str): Name of the target column.
            - ``features`` (list[str], optional): Columns to fit on; every
              master-table column when absent.
            - ``train_splits`` (list[str]): Split labels used for fitting.
            - ``eval_splits`` (list[str], optional): Split labels for evaluation.

    Returns:
        Model artifact dict with keys ``estimator``, ``target_column``,
        ``features``, ``feature_columns``, ``train_splits`` and
        ``eval_splits``.
    """
    target = params["target_column"]
    train_df = master_table[master_table[SPLIT_COLUMN].isin(params["train_splits"])]
    feature_cols = feature_columns(master_table, target, params.get("features"))

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    cls = _load_class(params["class_path"])
    estimator = cls(**params.get("init_args", {}))
    estimator.fit(X_train, y_train)

    logger.info(
        "Trained %s on %d samples (%d features)",
        params["class_path"].rsplit(".", 1)[-1],
        len(X_train),
        X_train.shape[1],
    )

    return {
        "estimator": estimator,
        "target_column": target,
        "features": params.get("features"),
        "feature_columns": feature_cols,
        "train_splits": list(params["train_splits"]),
        "eval_splits": params.get("eval_splits", []),
    }


def optimize_hyperparameters(
    master_table: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run a cross-validated grid search and return the best model artifact.

    The returned artifact has the same structure as ``train_model``, so it
    can be passed directly to ``evaluate_model``.

    Args:
        master_table: Fully-processed DataFrame with a ``split`` column and
            all feature and target columns.
        params: Optimization configuration with keys:
            - ``class_path`` (str): Fully-qualified estimator class.
            - ``init_args`` (dict, optional): Base estimator constructor args.
            - ``param_grid`` (dict): Hyperparameter grid for ``GridSearchCV``.
            - ``cv`` (int): Number of cross-validation folds.
            - ``scoring`` (str): Scoring metric for the search.
            - ``n_jobs`` (int): Parallel jobs for the search (-1 = all cores).
            - ``target_column`` (str): Name of the target column.
            - ``features`` (list[str], optional): Columns to fit on; every
              master-table column when absent.
            - ``train_splits`` (list[str]): Split labels used for fitting.
            - ``eval_splits`` (list[str]): Split labels for evaluation.

    Returns:
        Model artifact dict with keys ``estimator``, ``target_column``,
        ``features``, ``feature_columns``, ``train_splits``, ``eval_splits``,
        ``best_params``, and ``cv_best_score``.
    """
    target = params["target_column"]
    train_df = master_table[master_table[SPLIT_COLUMN].isin(params["train_splits"])]
    feature_cols = feature_columns(master_table, target, params.get("features"))

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    cls = _load_class(params["class_path"])
    base_estimator = cls(**params.get("init_args", {}))

    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=params["param_grid"],
        cv=params.get("cv", 5),
        scoring=params.get("scoring", "roc_auc"),
        n_jobs=params.get("n_jobs", -1),
        verbose=0,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Grid search complete — best %s: %.4f, params: %s",
        params.get("scoring", "roc_auc"),
        search.best_score_,
        search.best_params_,
    )

    return {
        "estimator": search.best_estimator_,
        "target_column": target,
        "features": params.get("features"),
        "feature_columns": feature_cols,
        "train_splits": list(params["train_splits"]),
        "eval_splits": params["eval_splits"],
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
    }


def _bootstrap_ci(
    y_true: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    params: dict[str, Any],
) -> list[float]:
    """Percentile bootstrap confidence interval of ``statistic`` over patients.

    ``statistic`` receives the resampled row positions, so a paired statistic
    (two models scored on the *same* resampled patients) is as easy as a
    single one. Resamples that happen to contain a single class are skipped,
    since the AUC is undefined for them.
    """
    rng = np.random.default_rng(params["random_state"])
    n = len(y_true)
    scores = []

    for _ in range(params["n_bootstrap"]):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() != y_true[idx].max():
            scores.append(statistic(idx))

    tail = (1 - params["confidence"]) / 2
    low, high = np.quantile(scores, [tail, 1 - tail])
    return [float(low), float(high)]


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when nothing is in the denominator."""
    return float(numerator / denominator) if denominator else 0.0


def _calibration(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Calibration-in-the-large and calibration slope.

    ``observed_expected`` is the observed event rate over the mean predicted
    risk: 1 means right on average, above 1 means risk is underestimated. The
    slope comes from an unpenalised logistic regression of the outcome on the
    logit of the predicted risk: 1 is ideal, below 1 means the predictions are
    too extreme, the usual mark of overfitting.
    """
    p = np.clip(y_proba, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=np.inf).fit(logit, y_true).coef_[0, 0]
    return {
        "observed_expected": _ratio(y_true.mean(), p.mean()),
        "calibration_slope": float(slope),
    }


def evaluate_model(
    model_artifact: dict[str, Any],
    master_table: pd.DataFrame,
    decision: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a fitted model on each split listed in the artifact.

    ``roc_auc`` is computed from predicted probabilities, not from predicted
    labels — using labels would silently understate the metric. It comes with
    a bootstrap confidence interval: on a ~100-row holdout, a gap of 0.01
    between two models is well inside the noise.

    Discrimination is not enough for a risk score, so calibration is reported
    too: the Brier score, the observed/expected ratio and the calibration
    slope (see ``_calibration``).

    The label-based metrics use the business threshold from
    ``params:decision`` instead of ``estimator.predict``, whose 0.5 cut-off is
    arbitrary. They are therefore the numbers the inference pipeline will
    actually deliver.

    Every split is flagged ``in_sample`` when the model was fitted on it, so a
    training score can never be read as a holdout score.

    Args:
        model_artifact: Dict produced by ``train_model`` or
            ``optimize_hyperparameters``, containing ``estimator``,
            ``target_column``, ``feature_columns``, ``train_splits`` and
            ``eval_splits``.
        master_table: Fully-processed DataFrame with a ``split`` column and
            all feature and target columns.
        decision: Decision configuration with key ``threshold`` (float):
            minimum positive-class probability to predict 1.
        evaluation: Bootstrap configuration with keys ``n_bootstrap`` (int),
            ``confidence`` (float) and ``random_state`` (int).

    Returns:
        Dict keyed by split name. Each value contains ``in_sample``,
        ``threshold``, ``accuracy``, ``roc_auc``, ``roc_auc_ci``, ``f1_macro``,
        ``recall`` (sensitivity), ``specificity``, ``precision`` (PPV),
        ``npv``, ``brier``, ``observed_expected``, ``calibration_slope``,
        ``confusion_matrix``, ``classification_report``, and ``n_samples``.
    """
    estimator = model_artifact["estimator"]
    target = model_artifact["target_column"]
    feature_cols = model_artifact["feature_columns"]
    threshold = decision["threshold"]

    all_metrics: dict[str, Any] = {}

    for split_name in model_artifact["eval_splits"]:
        split_df = master_table[master_table[SPLIT_COLUMN] == split_name]
        X_split = split_df[feature_cols]
        y_split = split_df[target].to_numpy()

        y_proba = estimator.predict_proba(X_split)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_split, y_pred, labels=[0, 1]).ravel()

        split_metrics = {
            "in_sample": split_name in model_artifact["train_splits"],
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_split, y_pred)),
            "roc_auc": float(roc_auc_score(y_split, y_proba)),
            "roc_auc_ci": _bootstrap_ci(
                y_split,
                lambda idx: roc_auc_score(y_split[idx], y_proba[idx]),
                evaluation,
            ),
            "f1_macro": float(f1_score(y_split, y_pred, average="macro")),
            "recall": float(recall_score(y_split, y_pred, zero_division=0)),
            "specificity": _ratio(tn, tn + fp),
            "precision": float(precision_score(y_split, y_pred, zero_division=0)),
            "npv": _ratio(tn, tn + fn),
            "brier": float(brier_score_loss(y_split, y_proba)),
            **_calibration(y_split, y_proba),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "classification_report": classification_report(
                y_split, y_pred, output_dict=True, zero_division=0
            ),
            "n_samples": int(len(y_split)),
        }

        logger.info(
            "Evaluation '%s'%s @ %.2f — roc_auc: %.4f %s, recall: %.4f, precision: %.4f",
            split_name,
            " (in-sample)" if split_metrics["in_sample"] else "",
            threshold,
            split_metrics["roc_auc"],
            split_metrics["roc_auc_ci"],
            split_metrics["recall"],
            split_metrics["precision"],
        )

        all_metrics[split_name] = split_metrics

    return all_metrics


def select_champion(
    baseline_model: dict[str, Any],
    optimized_model: dict[str, Any],
    master_table: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the model the refit pipeline promotes to production.

    The tuned model is a challenger: it replaces the baseline only when it is
    better beyond the noise of a holdout neither model was fitted on. Both
    models are scored on the same patients, and the difference in ROC AUC
    (challenger minus baseline) is bootstrapped over those patients: the
    paired resampling cancels out how easy or hard each patient is, which two
    separate intervals cannot do. The challenger is promoted only when the
    lower bound of that interval exceeds ``min_improvement``. Otherwise the
    baseline stays: simpler, smaller and interpretable.

    Args:
        baseline_model: Artifact produced by ``train_model``.
        optimized_model: Artifact produced by ``optimize_hyperparameters``.
        master_table: Fully-processed DataFrame with a ``split`` column.
        params: Selection configuration with keys:
            - ``split`` (str): Holdout split the comparison is made on.
            - ``min_improvement`` (float): Margin the lower bound of the
              difference must exceed.
            - ``n_bootstrap`` (int), ``confidence`` (float) and
              ``random_state`` (int): the paired bootstrap.

    Returns:
        The champion artifact, and a report of the comparison.

    Raises:
        ValueError: If ``split`` was used to fit either model.
    """
    split = params["split"]
    if any(split in m["train_splits"] for m in (baseline_model, optimized_model)):
        raise ValueError(
            f"Champion selection needs a holdout split; '{split}' was used "
            "to fit at least one of the models."
        )

    rows = master_table[master_table[SPLIT_COLUMN] == split]
    y_true = rows[baseline_model["target_column"]].to_numpy()
    p_base, p_opt = (
        m["estimator"].predict_proba(rows[m["feature_columns"]])[:, 1]
        for m in (baseline_model, optimized_model)
    )

    def auc_gain(idx: np.ndarray) -> float:
        return roc_auc_score(y_true[idx], p_opt[idx]) - roc_auc_score(
            y_true[idx], p_base[idx]
        )

    baseline_score = float(roc_auc_score(y_true, p_base))
    optimized_score = float(roc_auc_score(y_true, p_opt))
    low, high = _bootstrap_ci(y_true, auc_gain, params)
    promoted = low > params["min_improvement"]
    champion = optimized_model if promoted else baseline_model

    report = {
        "champion": "optimized" if promoted else "baseline",
        "estimator": type(champion["estimator"]).__name__,
        "split": split,
        "n_samples": int(len(y_true)),
        "metric": "roc_auc",
        "min_improvement": float(params["min_improvement"]),
        "baseline": {
            "estimator": type(baseline_model["estimator"]).__name__,
            "n_features": len(baseline_model["feature_columns"]),
            "score": baseline_score,
        },
        "optimized": {
            "estimator": type(optimized_model["estimator"]).__name__,
            "n_features": len(optimized_model["feature_columns"]),
            "score": optimized_score,
        },
        "difference": {
            "estimate": optimized_score - baseline_score,
            "ci": [low, high],
            "confidence": float(params["confidence"]),
        },
    }

    logger.info(
        "Champion: %s (%s) — roc_auc on '%s': baseline %.4f vs optimized %.4f, "
        "difference %+.4f [%.4f, %.4f]",
        report["champion"],
        report["estimator"],
        split,
        baseline_score,
        optimized_score,
        optimized_score - baseline_score,
        low,
        high,
    )
    return champion, report


def _decision_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    prevalences: list[float],
) -> dict[str, Any]:
    """What a cut-off on the model's risk means in patients (see ``_referral_metrics``)."""
    return _referral_metrics(
        y_true, (y_proba >= threshold).astype(int), threshold, prevalences
    )


def _referral_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    prevalences: list[float],
) -> dict[str, Any]:
    """Who a referral strategy flags, and whether acting on it pays off.

    ``net_benefit`` is the decision-curve measure (Vickers & Elkin, 2006):
    true positives minus false positives weighted by the odds of the cut-off,
    per patient. Choosing a cut-off of 0.35 says that finding one future
    diabetic is worth 0.65 / 0.35 = 1.9 unnecessary referrals. A strategy
    pays off at that cut-off only if it beats both referring everyone
    (``net_benefit_all``) and referring no one (0). The strategy need not be
    the model: the guideline rule is scored with the same weight, so the two
    are compared at the same valuation of a missed case.

    PPV and NPV depend on prevalence, so ``at_prevalence`` recomputes them
    with Bayes' rule from the sensitivity and specificity.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    prevalence = y_true.mean()
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    harm_weight = threshold / (1 - threshold)

    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "flagged_rate": float(y_pred.mean()),
        "missed": int(fn),
        "false_alarms": int(fp),
        "net_benefit": float((tp - fp * harm_weight) / n),
        "net_benefit_all": float(prevalence - (1 - prevalence) * harm_weight),
        "at_prevalence": [
            {
                "prevalence": float(prev),
                "ppv": _ratio(
                    sensitivity * prev,
                    sensitivity * prev + (1 - specificity) * (1 - prev),
                ),
                "npv": _ratio(
                    specificity * (1 - prev),
                    specificity * (1 - prev) + (1 - sensitivity) * prev,
                ),
            }
            for prev in prevalences
        ],
    }


def analyse_thresholds(  # noqa: PLR0913 - Kedro passes each dataset and param block explicitly
    champion_model: dict[str, Any],
    master_table: pd.DataFrame,
    measured_data: pd.DataFrame,
    decision: dict[str, Any],
    guideline_referral: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Out-of-fold decision analysis behind the business cut-off.

    ``decision.threshold`` is a clinical choice, so it stays a parameter; this
    node records the evidence it rests on. A clone of the champion is
    cross-validated on the rows the champion was trained on — the validate
    split and the inference file are never touched — and each candidate
    cut-off is scored in patients missed, referrals made, net benefit and
    PPV / NPV at lower prevalences (see ``_referral_metrics``).

    Beating "refer everyone" is a low bar: without the model, a clinician
    already refers every woman with impaired glucose tolerance. So the same
    patients are also scored under that guideline rule alone, and under the
    rule the inference pipeline serves (model at the cut-off *or* guideline),
    all at the configured cut-off's valuation of a missed case.

    Args:
        champion_model: Artifact returned by ``select_champion``.
        master_table: Fully-processed DataFrame with a ``split`` column.
        measured_data: The same rows before imputation, with the ``split``
            column and the measured values the guideline rule reads (an
            unmeasured value is NaN and never meets it).
        decision: Decision configuration with key ``threshold`` (float).
        guideline_referral: Rule configuration with keys ``column`` (str) and
            ``threshold`` (float).
        params: Analysis configuration with keys ``cv`` (int),
            ``random_state`` (int), ``thresholds`` (list[float]) and
            ``reference_prevalences`` (list[float], optional).

    Returns:
        Dict with the model name, sample counts, the metrics at the configured
        threshold, the full ``curve`` and ``referral_strategies``.

    Raises:
        ValueError: If ``measured_data`` does not hold the same rows, in the
            same splits, as ``master_table``.
    """
    target = champion_model["target_column"]
    rows = master_table[master_table[SPLIT_COLUMN].isin(champion_model["train_splits"])]
    y_true = rows[target].to_numpy()
    prevalences = params.get("reference_prevalences", [])

    measured = measured_data.reindex(rows.index)
    if not measured[SPLIT_COLUMN].equals(rows[SPLIT_COLUMN]):
        raise ValueError("measured_data and master_table do not hold the same rows.")
    guideline = (
        measured[guideline_referral["column"]]
        .ge(guideline_referral["threshold"])
        .to_numpy()
        .astype(int)
    )

    folds = StratifiedKFold(
        n_splits=params["cv"], shuffle=True, random_state=params["random_state"]
    )
    y_proba = cross_val_predict(
        clone(champion_model["estimator"]),
        rows[champion_model["feature_columns"]],
        y_true,
        cv=folds,
        method="predict_proba",
    )[:, 1]

    threshold = decision["threshold"]
    at_threshold = _decision_metrics(y_true, y_proba, threshold, prevalences)
    flagged = (y_proba >= threshold).astype(int)
    strategies = {
        "guideline": _referral_metrics(y_true, guideline, threshold, prevalences),
        "model_or_guideline": _referral_metrics(
            y_true, flagged | guideline, threshold, prevalences
        ),
    }
    logger.info(
        "Out-of-fold @ %.2f — sensitivity: %.3f, specificity: %.3f, "
        "flagged: %.1f%%, net benefit: %.3f (refer all: %.3f, guideline: %.3f, "
        "model or guideline: %.3f)",
        threshold,
        at_threshold["sensitivity"],
        at_threshold["specificity"],
        100 * at_threshold["flagged_rate"],
        at_threshold["net_benefit"],
        at_threshold["net_benefit_all"],
        strategies["guideline"]["net_benefit"],
        strategies["model_or_guideline"]["net_benefit"],
    )

    return {
        "model": type(champion_model["estimator"]).__name__,
        "splits": list(champion_model["train_splits"]),
        "n_samples": int(len(y_true)),
        "positives": int(y_true.sum()),
        "configured_threshold": at_threshold,
        "referral_strategies": {
            "guideline_rule": (
                f"{guideline_referral['column']} >= {guideline_referral['threshold']:g}"
            ),
            **strategies,
        },
        "curve": [
            _decision_metrics(y_true, y_proba, t, prevalences)
            for t in params["thresholds"]
        ],
    }
