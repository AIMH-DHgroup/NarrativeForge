"""NarrativeForge NRS Analyzer.

A reusable analysis package for NarrativeForge benchmark exports. It loads ZIP
archives or extracted directories containing nrs_runs.csv files, computes
quality/efficiency summaries, generates charts, identifies Pareto-optimal
configurations, and writes Markdown/HTML reports.
"""

from .version import __version__
from .analysis import AnalysisOptions, run_analysis

__all__ = ["__version__", "AnalysisOptions", "run_analysis"]
