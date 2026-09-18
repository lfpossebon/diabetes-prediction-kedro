"""FastAPI layer that exposes the Kedro pipelines over HTTP.

Re-exported here so the app is importable as ``diabetes.api:app``, which is the
path used by the uvicorn command in the Dockerfile.
"""

from .main import app

__all__ = ["app"]
