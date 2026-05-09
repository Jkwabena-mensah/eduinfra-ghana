"""
src/__init__.py
===============
Public API for the EduInfra Ghana `src` package.

Import the pipeline from here:

    from src.pipeline import EduInfraPipeline

Or access config constants directly:

    from src.config import GhanaColors, THRESHOLD_CRITICAL
"""

from src.pipeline import EduInfraPipeline
from src.config import GhanaColors

__all__ = [
    "EduInfraPipeline",
    "GhanaColors",
]
