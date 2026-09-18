"""Nodes for the 'modelling' pipeline.

The estimator is never hard-coded: ``class_path`` in ``parameters.yml`` is
resolved at runtime by :func:`_load_class`, so swapping LogisticRegression for
a RandomForest — or adding a third model — is a YAML change, not a code change.

``train_model`` and ``optimize_hyperparameters`` return the same artefact
shape, which is why a single ``evaluate_model`` serves both.
"""

import importlib
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


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


def train_model(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Fit a classifier on the specified splits and return a model artifact.

    Args:
        master_table: Fully-processed DataFrame with a ``split`` column and
            all feature and target columns.
        columns: Column groups dictionary with ``numerical`` and
            ``categorical`` keys defining the feature columns.
        params: Training configuration with keys:
            - ``class_path`` (str): Fully-qualified sklearn estimator class.
            - ``init_args`` (dict, optional): Estimator constructor arguments.
            - ``target_column`` (str): Name of the target column.
            - ``train_splits`` (list[str]): Split labels used for fitting.
            - ``eval_splits`` (list[str], optional): Split labels for evaluation.

    Returns:
        Model artifact dict with keys ``estimator``, ``target_column``,
        ``feature_columns``, and ``eval_splits``.
    """
    target = params["target_column"]
    train_df = master_table[master_table["split"].isin(params["train_splits"])]
    feature_cols = columns["numerical"] + columns["categorical"]

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
        "eval_splits": params.get("eval_splits", []),
    }


def optimize_hyperparameters(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run a cross-validated grid search and return the best model artifact.

    The returned artifact has the same structure as ``train_model``, so it
    can be passed directly to ``evaluate_model``.

    Args:
        master_table: Fully-processed DataFrame with a ``split`` column and
            all feature and target columns.
        columns: Column groups dictionary with ``numerical`` and
            ``categorical`` keys defining the feature columns.
        params: Optimization configuration with keys:
            - ``class_path`` (str): Fully-qualified estimator class.
            - ``init_args`` (dict, optional): Base estimator constructor args.
            - ``param_grid`` (dict): Hyperparameter grid for ``GridSearchCV``.
            - ``cv`` (int): Number of cross-validation folds.
            - ``scoring`` (str): Scoring metric for the search.
            - ``target_column`` (str): Name of the target column.
            - ``train_splits`` (list[str]): Split labels used for fitting.
            - ``eval_splits`` (list[str]): Split labels for evaluation.

    Returns:
        Model artifact dict with keys ``estimator``, ``target_column``,
        ``feature_columns``, ``eval_splits``, ``best_params``,
        and ``cv_best_score``.
    """
    target = params["target_column"]
    train_df = master_table[master_table["split"].isin(params["train_splits"])]
    feature_cols = columns["numerical"] + columns["categorical"]

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    cls = _load_class(params["class_path"])
    base_estimator = cls(**params.get("init_args", {}))

    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=params["param_grid"],
        cv=params.get("cv", 5),
        scoring=params.get("scoring", "roc_auc"),
        n_jobs=-1,
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
        "eval_splits": params["eval_splits"],
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
    }


def evaluate_model(
    model_artifact: dict[str, Any],
    master_table: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate a fitted model on each split listed in the artifact.

    ``roc_auc`` is computed from predicted probabilities, not from predicted
    labels — using labels would silently understate the metric.

    Args:
        model_artifact: Dict produced by ``train_model`` or
            ``optimize_hyperparameters``, containing ``estimator``,
            ``target_column``, ``feature_columns``, and ``eval_splits``.
        master_table: Fully-processed DataFrame with a ``split`` column and
            all feature and target columns.

    Returns:
        Dict keyed by split name. Each value contains ``accuracy``,
        ``roc_auc``, ``f1_macro``, ``classification_report``,
        and ``n_samples``.
    """
    estimator = model_artifact["estimator"]
    target = model_artifact["target_column"]
    feature_cols = model_artifact["feature_columns"]
    eval_splits = model_artifact["eval_splits"]

    all_metrics: dict[str, Any] = {}

    for split_name in eval_splits:
        split_df = master_table[master_table["split"] == split_name]
        X_split = split_df[feature_cols]
        y_split = split_df[target]

        y_pred = estimator.predict(X_split)
        y_proba = estimator.predict_proba(X_split)[:, 1]

        split_metrics = {
            "accuracy": float(accuracy_score(y_split, y_pred)),
            "roc_auc": float(roc_auc_score(y_split, y_proba)),
            "f1_macro": float(f1_score(y_split, y_pred, average="macro")),
            "classification_report": classification_report(
                y_split, y_pred, output_dict=True
            ),
            "n_samples": int(len(y_split)),
        }

        logger.info(
            "Evaluation '%s' — accuracy: %.4f, roc_auc: %.4f, f1_macro: %.4f",
            split_name,
            split_metrics["accuracy"],
            split_metrics["roc_auc"],
            split_metrics["f1_macro"],
        )

        all_metrics[split_name] = split_metrics

    return all_metrics
