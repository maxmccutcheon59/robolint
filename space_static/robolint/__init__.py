"""robolint: quality checks for robot learning datasets (LeRobot format)."""

__version__ = "0.1.0"

from .api import Result, lint  # noqa: E402
from .checks import Config  # noqa: E402
from .findings import Finding, Severity  # noqa: E402

__all__ = ["Config", "Finding", "Result", "Severity", "__version__", "lint"]
