"""Project-level tests: pipeline registration and DAG wiring.

These are the integration layer of the test pyramid — they check that the four
pipelines are discoverable and that their inputs and outputs connect, without
actually running any data through them.
"""

from kedro.framework.project import find_pipelines

from diabetes.pipeline_registry import register_pipelines

EXPECTED_PIPELINES = {"data_engineering", "modelling", "refit", "inference"}

# 11 data_engineering + 6 modelling + 11 refit + 10 inference
EXPECTED_NODE_COUNT = 38

PRODUCTION_ARTEFACTS = {
    "production_imputers",
    "production_outlier_caps",
    "production_encoders",
    "production_scalers",
    "production_model",
}


class TestPipelineRegistry:
    def test_all_pipelines_are_discovered(self):
        pipelines = find_pipelines(raise_errors=True)

        assert EXPECTED_PIPELINES <= set(pipelines)

    def test_default_pipeline_contains_every_node(self):
        """`kedro run` with no --pipeline must cover all four pipelines."""
        default = register_pipelines()["__default__"]

        assert len(default.nodes) == EXPECTED_NODE_COUNT

    def test_production_artefacts_connect_refit_to_inference(self):
        pipelines = find_pipelines(raise_errors=True)
        # all_outputs(): refit also consumes its own artefacts (fit -> transform),
        # so outputs() alone would hide them.
        produced = pipelines["refit"].all_outputs()
        consumed = pipelines["inference"].inputs()

        assert produced >= PRODUCTION_ARTEFACTS
        assert consumed >= PRODUCTION_ARTEFACTS

    def test_master_table_connects_data_engineering_to_modelling(self):
        pipelines = find_pipelines(raise_errors=True)

        assert "master_table" in pipelines["data_engineering"].outputs()
        assert "master_table" in pipelines["modelling"].inputs()

    def test_inference_pipeline_fits_nothing(self):
        """No artefact may be *created* by the inference pipeline except predictions."""
        pipelines = find_pipelines(raise_errors=True)
        node_names = {node.name for node in pipelines["inference"].nodes}

        assert not any(name.startswith(("fit_", "refit_")) for name in node_names)

    def test_production_model_is_trained_on_production_transforms(self):
        """The refit model must see data transformed exactly as inference will."""
        pipelines = find_pipelines(raise_errors=True)
        refit_node = next(
            n for n in pipelines["refit"].nodes if n.name == "refit_model"
        )

        assert "production_master_table" in refit_node.inputs
        assert "master_table" not in refit_node.inputs

    def test_refit_promotes_the_champion_not_a_fixed_model(self):
        """Regression: refit always took optimized_model, whatever the metrics said."""
        pipelines = find_pipelines(raise_errors=True)
        refit_node = next(
            n for n in pipelines["refit"].nodes if n.name == "refit_model"
        )

        # all_outputs(): analyse_thresholds also consumes the champion inside
        # modelling, so outputs() alone would hide it.
        assert "champion_model" in pipelines["modelling"].all_outputs()
        assert "champion_model" in refit_node.inputs
        assert "optimized_model" not in refit_node.inputs

    def test_clinical_rules_read_the_input_as_received(self):
        """The rules must see the values as measured, before imputation or
        outlier capping could move them, and fasting glucose and HbA1c never
        reach the cleaned table at all."""
        pipelines = find_pipelines(raise_errors=True)
        nodes = {n.name: n for n in pipelines["inference"].nodes}
        referral = nodes["apply_guideline_referral"]
        diagnosis = nodes["apply_diagnostic_criteria"]

        assert "raw_inference_dataframe" in referral.inputs
        assert "raw_inference_dataframe" in diagnosis.inputs
        # The diagnosis comes last: it overrides a guideline referral.
        assert referral.outputs[0] in diagnosis.inputs
        assert diagnosis.outputs == ["inference_predictions"]

    def test_threshold_analysis_compares_the_model_with_the_guideline(self):
        pipelines = find_pipelines(raise_errors=True)
        analysis = next(
            n for n in pipelines["modelling"].nodes if n.name == "analyse_thresholds"
        )

        assert "params:guideline_referral" in analysis.inputs
        # The measured, unimputed glucose, not the scaled one in the master table.
        assert "split_diabetes_data" in analysis.inputs
