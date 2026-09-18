"""Pipeline 'inference': score new data with the production artefacts.

Transform only — the imputers, outlier caps, encoders, scalers and model all
come from the refit pipeline. Six of the eight nodes are data_engineering
functions reused as-is.
"""

from kedro.pipeline import Node, Pipeline

from diabetes.pipelines.data_engineering.nodes import (
    clean_data,
    create_features,
    transform_encoders,
    transform_imputers,
    transform_outlier_caps,
    transform_scalers,
)

from .nodes import predict, to_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=to_dataframe,
                inputs="raw_inference_data",
                outputs="raw_inference_dataframe",
                name="to_dataframe",
            ),
            Node(
                func=clean_data,
                inputs=["raw_inference_dataframe", "params:columns"],
                outputs="cleaned_inference_data",
                name="clean_inference_data",
            ),
            Node(
                func=transform_imputers,
                inputs=["cleaned_inference_data", "production_imputers"],
                outputs="imputed_inference_data",
                name="impute_inference_data",
            ),
            Node(
                func=transform_outlier_caps,
                inputs=["imputed_inference_data", "production_outlier_caps"],
                outputs="capped_inference_data",
                name="cap_inference_data",
            ),
            Node(
                func=create_features,
                inputs=["capped_inference_data", "params:feature_engineering"],
                outputs="featured_inference_data",
                name="create_inference_features",
            ),
            Node(
                func=transform_encoders,
                inputs=["featured_inference_data", "production_encoders"],
                outputs="encoded_inference_data",
                name="encode_inference_data",
            ),
            Node(
                func=transform_scalers,
                inputs=["encoded_inference_data", "production_scalers"],
                outputs="scaled_inference_data",
                name="scale_inference_data",
            ),
            Node(
                func=predict,
                inputs=["production_model", "scaled_inference_data", "params:decision"],
                outputs="inference_predictions",
                name="predict",
            ),
        ]
    )
