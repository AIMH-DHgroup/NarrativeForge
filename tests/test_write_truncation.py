import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from eventweaver import prompts
from eventweaver.config import GenerateConfig
from eventweaver.generation import generate_narratives, parse_paragraphs, split_writing_batch, writing_sections
from eventweaver.llm import JsonLLM, LLMOutputTruncated
from eventweaver.providers.base import GenerationResult
from test_pipeline import FakeProvider


def truncated():
    return GenerationResult('{"paragraphs":[{"text":"unfinished', 0, error="length", truncated=True)


class WritingTruncationTests(unittest.TestCase):
    def test_retry_reduces_word_target_in_actual_data(self):
        class Provider:
            def __init__(self):
                self.targets = []
            def generate(self, prompt, **kwargs):
                data = json.loads(prompt.split("DATA_JSON:\n")[1])
                self.targets.append(data["max_words"])
                if len(self.targets) < 3:
                    return truncated()
                return GenerationResult('{"paragraphs":[{"text":"Complete sentence."}]}', 0)
        provider = Provider()
        llm = JsonLLM(provider, GenerateConfig(retries=2))
        original = {"max_words": 400, "chunks": []}
        result = llm.call(prompts.WRITE, original, parse_paragraphs, allow_narrative=True)
        self.assertEqual(provider.targets, [400, 200, 100])
        self.assertEqual(original["max_words"], 400)
        self.assertEqual(result, [{"text": "Complete sentence."}])

    def test_split_preserves_all_characters_ids_and_exact_offsets(self):
        for value in ("record with whitespace " * 30, "x" * 513, "漢字 archaeology " * 30):
            chunk = {"id": "C1", "text": value, "start": 45, "end": 45 + len(value)}
            left, right = split_writing_batch([chunk])
            self.assertEqual(left[0]["text"] + right[0]["text"], value)
            self.assertEqual(left[0]["end"], right[0]["start"])
            for part in (left[0], right[0]):
                self.assertEqual(part["id"], chunk["id"])
                self.assertEqual(part["text"], value[part["start"] - 45:part["end"] - 45])

    def test_pipeline_splits_truncated_batches_and_single_chunks_without_source_loss(self):
        class Provider(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: WRITE" in prompt:
                    self.prompts.append(prompt)
                    data = json.loads(prompt.split("DATA_JSON:\n")[1])
                    if len(data["chunks"]) > 1 or len(data["chunks"][0]["text"]) > 500:
                        return truncated()
                    return GenerationResult(json.dumps({"paragraphs": [{"text": "Complete narrative section."}]}), 0)
                return super().generate(prompt, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("The archive records the amphitheatre and its entrances. " * 40)
            provider = Provider()
            config = GenerateConfig(retries=0, narrative_titles=("The Amphitheatre",),
                                    chunk_chars=1000, overlap_chars=0)
            run = generate_narratives([source], output_dir=root / "out", provider=provider, config=config,
                                      progress=lambda message: None)
            with sqlite3.connect(run / "analysis.sqlite") as db:
                chunks = [json.loads(row[0]) for row in db.execute("SELECT body FROM chunks ORDER BY id")]
                paragraphs = [json.loads(row[0]) for row in db.execute("SELECT body FROM paragraphs ORDER BY rowid")]
                errors = db.execute("SELECT COUNT(*) FROM responses WHERE truncated=1").fetchone()[0]
            self.assertGreater(errors, 0)
            for chunk in chunks:
                spans = [span for para in paragraphs for span in para["source_spans"] if span["chunk_id"] == chunk["id"]]
                self.assertEqual(spans[0]["start"], chunk["start"])
                self.assertEqual(spans[-1]["end"], chunk["end"])
                self.assertTrue(all(a["end"] == b["start"] for a, b in zip(spans, spans[1:])))
            self.assertNotIn("unfinished", (run / "narrative_T01.md").read_text())
            writes = [json.loads(p.split("DATA_JSON:\n")[1]) for p in provider.prompts if "FASE: WRITE" in p]
            self.assertTrue(any(p["previous_text"] for p in writes))
            self.assertTrue(all(p["max_words"] <= 512 for p in writes))

    def test_persistent_truncation_stops_at_minimum_without_accepting_partial_text(self):
        class Provider:
            calls = 0
            def generate(self, prompt, **kwargs):
                self.calls += 1
                return truncated()
        provider = Provider()
        llm = JsonLLM(provider, GenerateConfig(retries=0))
        chunk = {"id": "C1", "text": "x" * 1000, "start": 0, "end": 1000}
        with self.assertRaisesRegex(LLMOutputTruncated, "sezione minima"):
            list(writing_sections(llm, [chunk], lambda rows: {"chunks": rows}))
        self.assertLess(provider.calls, 10)

    def test_transport_error_does_not_trigger_source_splitting(self):
        class Provider:
            calls = 0
            def generate(self, prompt, **kwargs):
                self.calls += 1
                return GenerationResult("", 0, error="connection refused")
        provider = Provider()
        llm = JsonLLM(provider, GenerateConfig(retries=0))
        with self.assertRaisesRegex(RuntimeError, "connection refused"):
            list(writing_sections(llm, [{"text": "x" * 1000}], lambda rows: {"chunks": rows}))
        self.assertEqual(provider.calls, 1)

    def test_complete_json_at_token_limit_remains_accepted(self):
        class Provider:
            def generate(self, prompt, **kwargs):
                return GenerationResult('{"paragraphs":[{"text":"Complete section."}]}', 0,
                                        error="length", truncated=True)
        llm = JsonLLM(Provider(), GenerateConfig(retries=0))
        result = llm.call(prompts.WRITE, {"chunks": []}, parse_paragraphs, allow_narrative=True)
        self.assertEqual(result[0]["text"], "Complete section.")
