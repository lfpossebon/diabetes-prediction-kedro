"""Shared pytest fixtures.

``find_pipelines()`` and anything that touches the Data Catalog need the Kedro
project to be bootstrapped first, which is normally done by the `kedro` CLI.
"""

from pathlib import Path

import pytest
from kedro.framework.startup import bootstrap_project

PROJECT_PATH = Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def bootstrap_kedro_project():
    """Configure the project once per test session."""
    bootstrap_project(PROJECT_PATH)
