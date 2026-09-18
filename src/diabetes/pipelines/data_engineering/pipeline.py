"""Pipeline 'data_engineering': raw CSV -> cleaned -> split -> imputed -> capped
-> featured -> encoded -> master table.

Also produces the ``modelling_*`` artefacts (imputers, outlier caps, encoders,
scalers), each fitted on the train split only.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    add_split_column,
    clean_data,
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


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=clean_data,
                inputs=["raw_diabetes_data", "params:columns"],
                outputs="cleaned_diabetes_data",
                name="clean_data",
            ),
            Node(
                func=add_split_column,
                inputs=["cleaned_diabetes_data", "params:split"],
                outputs="split_diabetes_data",
                name="add_split_column",
            ),
            Node(
                func=fit_imputers,
                inputs=[
                    "split_diabetes_data",
                    "params:columns",
                    "params:modelling_imputers",
                ],
                outputs="modelling_imputers",
                name="fit_imputers",
            ),
            Node(
                func=transform_imputers,
                inputs=["split_diabetes_data", "modelling_imputers"],
                outputs="imputed_diabetes_data",
                name="impute_missing_values",
            ),
            Node(
                func=fit_outlier_caps,
                inputs=[
                    "imputed_diabetes_data",
                    "params:columns",
                    "params:modelling_outlier_caps",
                    "params:outliers",
                ],
                outputs="modelling_outlier_caps",
                name="fit_outlier_caps",
            ),
            Node(
                func=transform_outlier_caps,
                inputs=["imputed_diabetes_data", "modelling_outlier_caps"],
                outputs="capped_diabetes_data",
                name="cap_outliers",
            ),
            Node(
                func=create_features,
                inputs=["capped_diabetes_data", "params:feature_engineering"],
                outputs="featured_diabetes_data",
                name="create_features",
            ),
            Node(
                func=fit_encoders,
                inputs=[
                    "featured_diabetes_data",
                    "params:columns",
                    "params:modelling_encoders",
                ],
                outputs="modelling_encoders",
                name="fit_encoders",
            ),
            Node(
                func=transform_encoders,
                inputs=["featured_diabetes_data", "modelling_encoders"],
                outputs="encoded_diabetes_data",
                name="encode_categorical_features",
            ),
            Node(
                func=fit_scalers,
                inputs=[
                    "encoded_diabetes_data",
                    "params:columns",
                    "params:modelling_scalers",
                ],
                outputs="modelling_scalers",
                name="fit_scalers",
            ),
            Node(
                func=transform_scalers,
                inputs=["encoded_diabetes_data", "modelling_scalers"],
                outputs="master_table",
                name="scale_numerical_features",
            ),
        ]
    )
