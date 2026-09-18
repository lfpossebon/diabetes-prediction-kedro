"""Pipeline 'modelling': train and evaluate a baseline and an optimized model.

The two branches are independent — Kedro runs them in whatever order the DAG
allows — and both feed the same ``evaluate_model`` function.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import evaluate_model, optimize_hyperparameters, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_model,
                inputs=["master_table", "params:columns", "params:modelling_baseline"],
                outputs="baseline_model",
                name="train_baseline_model",
            ),
            Node(
                func=evaluate_model,
                inputs=["baseline_model", "master_table"],
                outputs="baseline_metrics",
                name="evaluate_baseline_model",
            ),
            Node(
                func=optimize_hyperparameters,
                inputs=[
                    "master_table",
                    "params:columns",
                    "params:modelling_optimization",
                ],
                outputs="optimized_model",
                name="optimize_hyperparameters",
            ),
            Node(
                func=evaluate_model,
                inputs=["optimized_model", "master_table"],
                outputs="optimized_metrics",
                name="evaluate_optimized_model",
            ),
        ]
    )
