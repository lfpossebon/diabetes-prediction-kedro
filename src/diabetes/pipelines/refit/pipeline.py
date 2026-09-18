"""Pipeline 'refit': rebuild every artefact on the full dataset for production.

Same functions as data_engineering, different parameters — every fit step uses
train + test + validate instead of train alone.

The whole fit/transform chain is replayed, not just the fit nodes: each step is
fitted on the output of the previous one, so the production model must be
trained on data transformed by the *production* artefacts — exactly what the
inference pipeline will feed it. The intermediates are not in the catalog, so
Kedro keeps them in memory; only the artefacts and the final table are saved.
"""

from kedro.pipeline import Node, Pipeline

from diabetes.pipelines.data_engineering.nodes import (
    create_features,
    fit_encoders,
    fit_imputers,
    fit_outlier_caps,
    fit_scalers,
    transform_encoders,
    transform_imputers,
    transform_outlier_caps,
    transform_scalers,
)

from .nodes import refit_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=fit_imputers,
                inputs=[
                    "split_diabetes_data",
                    "params:columns",
                    "params:refit_imputers",
                ],
                outputs="production_imputers",
                name="refit_imputers",
            ),
            Node(
                func=transform_imputers,
                inputs=["split_diabetes_data", "production_imputers"],
                outputs="refit_imputed_data",
                name="impute_refit_data",
            ),
            Node(
                func=fit_outlier_caps,
                inputs=[
                    "refit_imputed_data",
                    "params:columns",
                    "params:refit_outlier_caps",
                    "params:outliers",
                ],
                outputs="production_outlier_caps",
                name="refit_outlier_caps",
            ),
            Node(
                func=transform_outlier_caps,
                inputs=["refit_imputed_data", "production_outlier_caps"],
                outputs="refit_capped_data",
                name="cap_refit_data",
            ),
            Node(
                func=create_features,
                inputs=["refit_capped_data", "params:feature_engineering"],
                outputs="refit_featured_data",
                name="create_refit_features",
            ),
            Node(
                func=fit_encoders,
                inputs=[
                    "refit_featured_data",
                    "params:columns",
                    "params:refit_encoders",
                ],
                outputs="production_encoders",
                name="refit_encoders",
            ),
            Node(
                func=transform_encoders,
                inputs=["refit_featured_data", "production_encoders"],
                outputs="refit_encoded_data",
                name="encode_refit_data",
            ),
            Node(
                func=fit_scalers,
                inputs=["refit_encoded_data", "params:columns", "params:refit_scalers"],
                outputs="production_scalers",
                name="refit_scalers",
            ),
            Node(
                func=transform_scalers,
                inputs=["refit_encoded_data", "production_scalers"],
                outputs="production_master_table",
                name="scale_refit_data",
            ),
            Node(
                func=refit_model,
                inputs=[
                    "production_master_table",
                    "params:columns",
                    "optimized_model",
                    "params:refit_model",
                ],
                outputs="production_model",
                name="refit_model",
            ),
        ]
    )
