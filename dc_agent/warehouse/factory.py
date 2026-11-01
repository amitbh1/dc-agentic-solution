from __future__ import annotations

import os

from .databricks import DatabricksExecutor
from .snowflake import SnowflakeExecutor


def build_raw_executors() -> dict:
    """Unwrapped executors from environment variables (used by the API and the drift job)."""
    executors: dict = {}
    if os.getenv("DATABRICKS_HOST"):
        executors["databricks"] = DatabricksExecutor(
            os.environ["DATABRICKS_HOST"], os.environ["DATABRICKS_HTTP_PATH"], os.environ["DATABRICKS_TOKEN"]
        )
    if os.getenv("SNOWFLAKE_ACCOUNT"):
        executors["snowflake"] = SnowflakeExecutor(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            role=os.environ["SNOWFLAKE_ROLE"],
        )
    return executors
