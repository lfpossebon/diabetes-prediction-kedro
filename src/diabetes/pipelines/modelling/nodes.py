"""Nodes for the 'modelling' pipeline.

The estimator is never hard-coded: ``class_path`` in ``parameters.yml`` is
resolved at runtime by :func:`_load_class`, so swapping LogisticRegression for
a RandomForest — or adding a third model — is a YAML change, not a code change.

``train_model`` and ``optimize_hyperparameters`` return the same artefact
shape, which is why a single ``evaluate_model`` serves both, and why
``select_champion`` can hand either one to the refit pipeline.
"""

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
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


def feature_columns(master_table: pd.DataFrame, target: str) -> list[str]:
    """Every master-table column except the target and the split label.

    The data_engineering pipeline builds the master table as exactly the model
    input, and one-hot encoding makes the column names depend on the
    categories seen at fit time — so the table, not a hand-kept list, is the
    source of truth.
    """
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
            - ``train_splits`` (list[str]): Split labels used for fitting.
            - ``eval_splits`` (list[str], optional): Split labels for evaluation.

    Returns:
        Model artifact dict with keys ``estimator``, ``target_column``,
        ``feature_columns``, ``train_splits`` and ``eval_splits``.
    """
    target = params["target_column"]
    train_df = master_table[master_table[SPLIT_COLUMN].isin(params["train_splits"])]
    feature_cols = feature_columns(master_table, target)

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
            - ``train_splits`` (list[str]): Split labels used for fitting.
            - ``eval_splits`` (list[str]): Split labels for evaluation.

    Returns:
        Model artifact dict with keys ``estimator``, ``target_column``,
        ``feature_columns``, ``train_splits``, ``eval_splits``,
        ``best_params``, and ``cv_best_score``.
    """
    target = params["target_column"]
    train_df = master_table[master_table[SPLIT_COLUMN].isin(params["train_splits"])]
    feature_cols = feature_columns(master_table, target)

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
        "feature_columns": feature_cols,
        "train_splits": list(params["train_splits"]),
        "eval_splits": params["eval_splits"],
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
    }


def _roc_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    params: dict[str, Any],
) -> list[float]:
    """Percentile bootstrap confidence interval for the ROC AUC.

    Resamples that happen to contain a single class are skipped, since the
    AUC is undefined for them.
    """
    rng = np.random.default_rng(params["random_state"])
    n = len(y_true)
    scores = []

    for _ in range(params["n_bootstrap"]):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() != y_true[idx].max():
            scores.append(roc_auc_score(y_true[idx], y_proba[idx]))

    tail = (1 - params["confidence"]) / 2
    low, high = np.quantile(scores, [tail, 1 - tail])
    return [float(low), float(high)]


def evaluate_model(
    model_artifact: dict[str, Any],
    master_table: pd.DataFrame,
    decision: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a fitted model on each split listed in the artifact.

    ``roc_auc`` is computed from predicted probabilities, not from predicted
    labels — using labels would silently understate the metric. It comes with
    a bootstrap confidence interval: on a ~90-row holdout, a gap of 0.01
    between two models is well inside the noise.

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
        ``recall``, ``precision``, ``confusion_matrix``,
        ``classification_report``, and ``n_samples``.
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
            "roc_auc_ci": _roc_auc_ci(y_split, y_proba, evaluation),
            "f1_macro": float(f1_score(y_split, y_pred, average="macro")),
            "recall": float(recall_score(y_split, y_pred, zero_division=0)),
            "precision": float(precision_score(y_split, y_pred, zero_division=0)),
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
    baseline_metrics: dict[str, Any],
    optimized_model: dict[str, Any],
    optimized_metrics: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the model the refit pipeline promotes to production.

    The tuned model is a challenger: it replaces the baseline only if it beats
    it by more than ``min_improvement`` on a split neither of them was fitted
    on. A tie keeps the baseline, which is simpler, smaller and interpretable.

    Args:
        baseline_model: Artifact produced by ``train_model``.
        baseline_metrics: Output of ``evaluate_model`` for the baseline.
        optimized_model: Artifact produced by ``optimize_hyperparameters``.
        optimized_metrics: Output of ``evaluate_model`` for the tuned model.
        params: Selection configuration with keys:
            - ``split`` (str): Holdout split the comparison is made on.
            - ``metric`` (str): Metric key in the metrics dicts, higher is better.
            - ``min_improvement`` (float): Margin the challenger must exceed.

    Returns:
        The champion artifact, and a report of the comparison.

    Raises:
        ValueError: If ``split`` is in-sample for either model.
    """
    split, metric = params["split"], params["metric"]
    baseline_split = baseline_metrics[split]
    optimized_split = optimized_metrics[split]

    if baseline_split["in_sample"] or optimized_split["in_sample"]:
        raise ValueError(
            f"Champion selection needs a holdout split; '{split}' was used "
            "to fit at least one of the models."
        )

    baseline_score = baseline_split[metric]
    optimized_score = optimized_split[metric]
    promoted = optimized_score - baseline_score > params["min_improvement"]
    champion = optimized_model if promoted else baseline_model

    report = {
        "champion": "optimized" if promoted else "baseline",
        "estimator": type(champion["estimator"]).__name__,
        "split": split,
        "metric": metric,
        "min_improvement": float(params["min_improvement"]),
        "baseline": {
            "estimator": type(baseline_model["estimator"]).__name__,
            "score": float(baseline_score),
            "ci": baseline_split.get(f"{metric}_ci"),
        },
        "optimized": {
            "estimator": type(optimized_model["estimator"]).__name__,
            "score": float(optimized_score),
            "ci": optimized_split.get(f"{metric}_ci"),
        },
    }

    logger.info(
        "Champion: %s (%s) — %s on '%s': baseline %.4f vs optimized %.4f",
        report["champion"],
        report["estimator"],
        metric,
        split,
        baseline_score,
        optimized_score,
    )
    return champion, report


def _decision_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """What a cut-off means in patients: missed cases versus tests ordered."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "flagged_rate": float(y_pred.mean()),
        "missed": int(fn),
        "false_alarms": int(fp),
    }


def analyse_thresholds(
    champion_model: dict[str, Any],
    master_table: pd.DataFrame,
    decision: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Out-of-fold trade-off curve behind the business cut-off.

    ``decision.threshold`` is a business choice, so it stays a parameter; this
    node records the evidence it rests on. A clone of the champion is
    cross-validated on the rows the champion was trained on — the validate
    split and the inference file are never touched — and each candidate
    cut-off is scored in patients missed and tests ordered.

    Args:
        champion_model: Artifact returned by ``select_champion``.
        master_table: Fully-processed DataFrame with a ``split`` column.
        decision: Decision configuration with key ``threshold`` (float).
        params: Analysis configuration with keys ``cv`` (int),
            ``random_state`` (int) and ``thresholds`` (list[float]).

    Returns:
        Dict with the model name, sample counts, the metrics at the configured
        threshold and the full ``curve``.
    """
    target = champion_model["target_column"]
    rows = master_table[master_table[SPLIT_COLUMN].isin(champion_model["train_splits"])]
    y_true = rows[target].to_numpy()

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

    at_threshold = _decision_metrics(y_true, y_proba, decision["threshold"])
    logger.info(
        "Out-of-fold @ %.2f — recall: %.3f, precision: %.3f, flagged: %.1f%%",
        decision["threshold"],
        at_threshold["recall"],
        at_threshold["precision"],
        100 * at_threshold["flagged_rate"],
    )

    return {
        "model": type(champion_model["estimator"]).__name__,
        "splits": list(champion_model["train_splits"]),
        "n_samples": int(len(y_true)),
        "positives": int(y_true.sum()),
        "configured_threshold": at_threshold,
        "curve": [_decision_metrics(y_true, y_proba, t) for t in params["thresholds"]],
    }
