from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from eventweaver.benchmark import PairSpec, compute_case_method_summary, compute_model_overall_summary, evaluate_pair
from eventweaver.generation import generate_narratives
from eventweaver.models import ALL_MODELS, OLLAMA_MODEL_METADATA, REQUIRED_METADATA_KEYS, get_model_metadata, resolve_num_ctx_for_model
from eventweaver.ollama_client import OllamaResult
from eventweaver.source_record import SourceRecord
from eventweaver.utils import read_csv

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
        self.assertEqual(resolve_num_ctx_for_model("gemma2:9b", "full", None), 8000)
        self.assertEqual(resolve_num_ctx_for_model("qwen3:4b", "full", None), 256000)
        self.assertEqual(resolve_num_ctx_for_model("mistral-nemo:12b", "full", None), 1000000)
        self.assertEqual(resolve_num_ctx_for_model("llama3.1:8b", "full", None), 128000)
        self.assertEqual(resolve_num_ctx_for_model("llama3.2:1b", "brief", None), 8192)
        self.assertEqual(resolve_num_ctx_for_model("llama3.2:1b", "full", 16384), 16384)

    def test_generation_resolves_num_ctx_per_model_for_full_strategy(self) -> None:
        import tempfile

        models = ["gemma2:9b", "qwen3:4b", "mistral-nemo:12b", "llama3.1:8b"]
        captured_num_ctx: list[int] = []

        def fake_generate_ollama(*args, **kwargs):
            captured_num_ctx.append(kwargs["num_ctx"])
            return OllamaResult(text="Generated narrative text.", runtime_seconds=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / "outputs"
            source = SourceRecord(
                source_type="docx",
                source_path=tmp / "Case Study.docx",
                source_text="This is a short source text for a cultural heritage case study.",
                prompt_kind="cultural-heritage",
            )
            with patch("eventweaver.generation.iter_source_records", return_value=[source]), patch("eventweaver.generation.generate_ollama", side_effect=fake_generate_ollama):
                generate_narratives(
                    tmp / "unused.docx",
                    models,
                    runs=1,
                    output_dir=output_dir,
                    temperature=0.1,
                    num_ctx=None,
                    input_strategy="full",
                    skip_ollama_preflight=True,
                )

            self.assertEqual(captured_num_ctx, [8000, 256000, 1000000, 128000])
            rows = read_csv(output_dir / "generation_metadata.csv")
            self.assertEqual({row["model"]: int(row["num_ctx"]) for row in rows}, dict(zip(models, captured_num_ctx)))

    def test_generation_explicit_num_ctx_overrides_model_metadata(self) -> None:
        import tempfile

        captured_num_ctx: list[int] = []

        def fake_generate_ollama(*args, **kwargs):
            captured_num_ctx.append(kwargs["num_ctx"])
            return OllamaResult(text="Generated narrative text.", runtime_seconds=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = SourceRecord(
                source_type="docx",
                source_path=tmp / "Case Study.docx",
                source_text="This is a short source text for a cultural heritage case study.",
                prompt_kind="cultural-heritage",
            )
            with patch("eventweaver.generation.iter_source_records", return_value=[source]), patch("eventweaver.generation.generate_ollama", side_effect=fake_generate_ollama):
                generate_narratives(
                    tmp / "unused.docx",
                    ["gemma2:9b", "mistral-nemo:12b"],
                    runs=1,
                    output_dir=tmp / "outputs",
                    temperature=0.1,
                    num_ctx=256000,
                    input_strategy="full",
                    skip_ollama_preflight=True,
                )

            self.assertEqual(captured_num_ctx, [256000, 256000])

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
                num_ctx=256000,
            )

            row = evaluate_pair(pair)
            for key in ("dimension", "arch", "parameters", "quantization", "context_length"):
                self.assertIn(key, row)
            self.assertEqual(row["num_ctx"], 256000)

            case_summary = compute_case_method_summary([row])
            summary = compute_model_overall_summary(case_summary)
            self.assertTrue(summary)
            for key in ("dimension", "arch", "parameters", "quantization", "context_length"):
                self.assertIn(key, summary[0])
            self.assertEqual(case_summary[0]["num_ctx"], 256000)
            self.assertEqual(summary[0]["num_ctx"], 256000)


if __name__ == "__main__":
    unittest.main()
