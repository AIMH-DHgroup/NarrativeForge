from pathlib import Path
import os

import pytest

from narrativeforge_nrs_analyzer import AnalysisOptions, run_analysis


def test_sample_zip_smoke(tmp_path):
    source = os.environ.get("NRS_TEST_ZIP")
    if not source or not Path(source).exists():
        pytest.skip("Set NRS_TEST_ZIP to run the integration smoke test.")
    result = run_analysis(source, tmp_path / "out", AnalysisOptions())
    assert (result.output_dir / "combined_nrs_runs.csv").exists()
    assert (result.output_dir / "narrativeforge_nrs_analysis_report.html").exists()
    assert len(result.figure_paths) >= 11
    assert (result.output_dir / "tables" / "model_strategy_matrix.csv").exists()
    assert (result.output_dir / "tables" / "model_family_summary.tex").exists()
    assert (result.output_dir / "tables" / "top5_configurations.csv").exists()
    assert result.bundle_path.exists()
