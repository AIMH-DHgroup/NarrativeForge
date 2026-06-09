from __future__ import annotations

import unittest
from pathlib import Path

from eventweaver.benchmark import PairSpec, compute_case_method_summary, compute_model_overall_summary, evaluate_pair
from eventweaver.models import ALL_MODELS, OLLAMA_MODEL_METADATA, REQUIRED_METADATA_KEYS, get_model_metadata, resolve_num_ctx_for_model

class ModelMetadataTests(unittest.TestCase):
    def test_all_models_have_metadata(self) -> None:
        missing = sorted(set(ALL_MODELS) - set(OLLAMA_MODEL_METADATA))
        self.assertEqual(missing, [])
        for model, metadata in OLLAMA_MODEL_METADATA.items():
            self.assertEqual(set(metadata), REQUIRED_METADATA_KEYS)
            self.assertIsInstance(metadata["context_length"], int)

    def test_get_model_metadata_known_model(self) -> None:
        metadata = get_model_metadata("llama3.2:1b")
        self.assertEqual(
            metadata,
            {
                "dimension": 2048,
                "arch": "llama",
                "parameters": "1.24B",
                "quantization": "Q8_0",
                "context_length": 128000,
            },
        )

    def test_full_strategy_uses_model_context_length(self) -> None:
        self.assertEqual(resolve_num_ctx_for_model("llama3.2:1b", "full", None), 128000)
        self.assertEqual(resolve_num_ctx_for_model("llama3.2:1b", "brief", None), 8192)
        self.assertEqual(resolve_num_ctx_for_model("llama3.2:1b", "full", 16384), 16384)

    def test_benchmark_row_includes_model_metadata_with_tempdir(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.txt"
            output = tmp / "output.txt"
            source.write_text("A short source text for metadata testing.", encoding="utf-8")
            output.write_text("A short output text for metadata testing.", encoding="utf-8")

            pair = PairSpec(
                source_path=source,
                output_path=output,
                method="qwen3_8b__auto__standard",
                run="1",
                model="qwen3:8b",
                input_strategy="auto",
                prompt_kind="cultural-heritage",
                prompt_strategy="standard",
                source_type="docx",
            )

            row = evaluate_pair(pair)
            for key in ("dimension", "arch", "parameters", "quantization", "context_length"):
                self.assertIn(key, row)

            case_summary = compute_case_method_summary([row])
            summary = compute_model_overall_summary(case_summary)
            self.assertTrue(summary)
            for key in ("dimension", "arch", "parameters", "quantization", "context_length"):
                self.assertIn(key, summary[0])


if __name__ == "__main__":
    unittest.main()
