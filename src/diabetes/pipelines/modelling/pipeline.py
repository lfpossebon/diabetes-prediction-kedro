"""Pipeline 'modelling': train and evaluate a baseline and an optimized model,
then pick the champion that the refit pipeline promotes to production.

The two training branches are independent — Kedro runs them in whatever order
the DAG allows — and both feed the same ``evaluate_model`` function.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    analyse_thresholds,
    evaluate_model,
    optimize_hyperparameters,
    select_champion,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_model,
                inputs=["master_table", "params:modelling_baseline"],
                outputs="baseline_model",
                name="train_baseline_model",
            ),
            Node(
                func=evaluate_model,
                inputs=[
                    "baseline_model",
                    "master_table",
                    "params:decision",
                    "params:evaluation",
                ],
                outputs="baseline_metrics",
                name="evaluate_baseline_model",
            ),
            Node(
                func=optimize_hyperparameters,
                inputs=["master_table", "params:modelling_optimization"],
                outputs="optimized_model",
                name="optimize_hyperparameters",
            ),
            Node(
                func=evaluate_model,
                inputs=[
                    "optimized_model",
                    "master_table",
                    "params:decision",
                    "params:evaluation",
                ],
                outputs="optimized_metrics",
                name="evaluate_optimized_model",
            ),
            Node(
                func=select_champion,
                inputs=[
                    "baseline_model",
                    "baseline_metrics",
                    "optimized_model",
                    "optimized_metrics",
                    "params:champion_selection",
                ],
                outputs=["champion_model", "champion_report"],
                name="select_champion",
            ),
            Node(
                func=analyse_thresholds,
                inputs=[
                    "champion_model",
                    "master_table",
                    "params:decision",
                    "params:threshold_analysis",
                ],
                outputs="threshold_curve",
                name="analyse_thresholds",
            ),
        ]
    )
