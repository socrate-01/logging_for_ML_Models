"""Filters for adding extra info to log records."""
import os
import logging
from typing import List


class EnvironmentInfoFilter(logging.Filter):
    """Logging filter that adds information to log records from environment variables."""

    def __init__(self, env_variables: List[str]):
        super().__init__()
        self._env_variables = env_variables

    def filter(self, record):
        for env_variable in self._env_variables:
            record.__setattr__(env_variable.lower(), os.environ.get(env_variable, "N/A"))
        return True
