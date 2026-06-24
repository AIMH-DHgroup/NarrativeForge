from __future__ import annotations

import math

import pandas as pd

from narrativeforge_nrs_analyzer.analysis import AnalysisOptions, make_tables
from narrativeforge_nrs_analyzer.coverage import (
    CoverageOptions,
    apply_coverage_diagnostics,
    compute_coverage_for_row,
    detect_text_columns,
    split_sentences,
)


class FakeEmbedder:
    def encode(self, sentences):
        vectors = []
        for sentence in sentences:
            text = sentence.lower()
            if "alpha" in text:
                vectors.append([1.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.7, 0.7])
        return vectors


def test_sentence_splitting():
    assert split_sentences("Alpha one. Beta two? Gamma three!") == ["Alpha one.", "Beta two?", "Gamma three!"]


def test_text_column_detection():
    df = pd.DataFrame({"source_text": ["a"], "generated_text": ["b"]})
    assert detect_text_columns(df) == ("source_text", "generated_text")


def test_compression_and_coverage_with_mocked_embeddings():
    metrics = compute_coverage_for_row(
        "Alpha source sentence. Beta source sentence.",
        "Alpha generated sentence.",
        thresholds=[0.70, 0.75, 0.80],
        embedder=FakeEmbedder(),
    )
    assert metrics["source_word_count"] == 6
    assert metrics["generated_word_count"] == 3
    assert metrics["compression_ratio"] == 0.5
    assert metrics["source_coverage_075"] == 0.5
    assert metrics["generation_support_075"] == 1.0


def test_missing_text_columns_writes_unavailable_file(tmp_path):
    df = pd.DataFrame({"NRS": [80.0], "failed": [False]})
    result = apply_coverage_diagnostics(df, CoverageOptions(), tmp_path)
    assert not result.available
    assert (tmp_path / "coverage_diagnostics_unavailable.txt").exists()


def test_omission_risk_and_aggregation_tables(tmp_path):
    df = pd.DataFrame(
        {
            "source_text": ["Alpha source sentence. Beta source sentence."],
            "generated_text": ["Alpha generated sentence."],
            "NRS": [80.0],
            "bertscore_f1": [0.9],
            "semantic_similarity": [0.8],
            "runtime_seconds": [1.0],
            "failed": [False],
            "success": [True],
            "model": ["gemma3:4b"],
            "family": ["gemma3"],
            "input_strategy": ["brief"],
            "prompt_strategy": ["standard"],
            "parameters_b": [4.0],
            "case_id": ["case1"],
        }
    )
    result = apply_coverage_diagnostics(df, CoverageOptions(skip_entity_coverage=True, skip_keyphrase_coverage=True), tmp_path)
    # Semantic coverage is skipped without sentence-transformers, but arithmetic columns still exist.
    assert "compression_ratio" in result.df.columns
    result.df["source_coverage_075"] = 0.5
    result.df["coverage_adjusted_bertscore_075"] = result.df["bertscore_f1"] * result.df["source_coverage_075"]
    result.df["omission_risk_075"] = result.df["bertscore_f1"] - result.df["source_coverage_075"]
    tables = make_tables(result.df, AnalysisOptions())
    assert math.isclose(float(result.df["omission_risk_075"].iloc[0]), 0.4)
    assert not tables["coverage_by_input_strategy"].empty


def test_coverage_cache_reuse_without_recompute(tmp_path):
    df = pd.DataFrame(
        {
            "source_text": ["Alpha source sentence."],
            "generated_text": ["Alpha source sentence."],
            "NRS": [80.0],
            "bertscore_f1": [0.9],
            "semantic_similarity": [0.8],
            "runtime_seconds": [1.0],
            "failed": [False],
            "success": [True],
            "model": ["gemma3:4b"],
            "input_strategy": ["brief"],
            "prompt_strategy": ["standard"],
            "run": ["1"],
            "case_id": ["case1"],
        }
    )
    options = CoverageOptions(skip_entity_coverage=True, skip_keyphrase_coverage=True)
    first = apply_coverage_diagnostics(df, options, tmp_path / "tables", tmp_path)
    second = apply_coverage_diagnostics(df, options, tmp_path / "tables", tmp_path)
    assert (tmp_path / "coverage_metrics.csv").exists()
    assert first.computed_rows == 1
    assert second.computed_rows == 0
    assert second.cached_rows == 1
