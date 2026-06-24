from __future__ import annotations

import pandas as pd

from narrativeforge_nrs_analyzer.case_study import add_case_study_column, case_study_tables, detect_case_column


def _sample_df() -> pd.DataFrame:
    rows = []
    for case, base in [("case_a", 80), ("case_b", 70)]:
        for model, params in [("gemma3:4b", 4.0), ("qwen3:8b", 8.0)]:
            for input_strategy in ["auto", "brief", "rag", "full"]:
                for prompt_strategy in ["short", "standard", "detailed"]:
                    rows.append(
                        {
                            "case_id": case,
                            "model": model,
                            "input_strategy": input_strategy,
                            "prompt_strategy": prompt_strategy,
                            "parameters_b": params,
                            "NRS": base + (2 if input_strategy == "rag" else 0) + (1 if prompt_strategy == "detailed" else 0),
                            "runtime_seconds": 10 + params,
                            "failed": False,
                            "success": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_case_column_detection():
    assert detect_case_column(pd.DataFrame({"case_name": ["x"]})) == "case_name"


def test_add_case_study_column_missing():
    df, col = add_case_study_column(pd.DataFrame({"model": ["m"]}))
    assert col is None
    assert "case_study" not in df.columns


def test_case_summary_and_best_configuration():
    tables, col = case_study_tables(_sample_df())
    assert col == "case_id"
    assert not tables["case_study_summary"].empty
    assert not tables["best_configuration_by_case"].empty
    assert set(tables["best_configuration_by_case"]["best_input_strategy"]) == {"rag"}


def test_difficulty_and_delta_tables():
    tables, _ = case_study_tables(_sample_df())
    assert "difficulty_score" in tables["case_difficulty_ranking"].columns
    assert "rag_minus_brief_NRS" in tables["case_input_strategy_delta"].columns
    assert "detailed_minus_short_NRS" in tables["case_prompt_strategy_delta"].columns


def test_model_size_loss_table():
    tables, _ = case_study_tables(_sample_df())
    assert not tables["case_model_size_loss"].empty
    assert "NRS_loss_vs_case_best" in tables["case_model_size_loss"].columns
