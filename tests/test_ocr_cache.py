import importlib.util
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eventweaver.cli import build_parser, main
from eventweaver.config import GenerateConfig
from eventweaver.ingestion.pdf_loader import load_pdf
from eventweaver.generation import generate_narratives
from test_pipeline import FakeProvider


@unittest.skipUnless(importlib.util.find_spec("pymupdf"), "PyMuPDF non installato")
class OCRCacheTests(unittest.TestCase):
    text = "The Roman amphitheatre has an ancient arena and several entrances recorded in the historical archive."

    def setUp(self):
        import pymupdf
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "scan.pdf"
        self.cache = self.root / "cache"
        with pymupdf.open() as pdf:
            pdf.new_page()
            pdf.new_page()
            pdf.save(self.source)
        identity = patch("eventweaver.ingestion.cache.engine_identity", return_value={"version": "test-1"})
        self.engine = identity.start()
        self.addCleanup(identity.stop)

    def read(self, **kwargs):
        options = dict(document_id="D00001", cache_dir=self.cache, ocr_mode="always", ocr_language="eng")
        options.update(kwargs)
        return load_pdf(self.source, **options)

    def test_repeated_and_renamed_file_reuses_both_pages_and_current_ids(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text) as ocr:
            first = self.read()
            self.source = self.source.rename(self.root / "renamed.pdf")
            messages = []
            second = self.read(document_id="D00009", progress=messages.append)
        self.assertEqual(ocr.call_count, 2)
        self.assertEqual(first.pages, second.pages)
        self.assertEqual(second.document_id, "D00009")
        self.assertEqual(second.source_name, "renamed.pdf")
        self.assertEqual(sum("riutilizzata" in message for message in messages), 2)

    def test_content_language_dpi_mode_engine_and_algorithm_invalidate(self):
        import pymupdf
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text) as ocr:
            self.read()
            self.read(ocr_language="spa+eng")
            self.read(ocr_dpi=300)
            self.read(ocr_mode="auto")
            self.engine.return_value = {"version": "test-2"}
            self.read()
            with patch("eventweaver.ingestion.cache.EXTRACTION_VERSION", 99):
                self.read()
            with pymupdf.open(self.source) as pdf:
                pdf[0].insert_text((30, 30), "Updated content")
                changed = self.root / "changed.pdf"
                pdf.save(changed)
            self.source = changed
            self.read()
        self.assertEqual(ocr.call_count, 14)

    def test_interruption_resumes_from_first_missing_page(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=[self.text, RuntimeError("interrupted")]):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.read()
        self.assertEqual(len(list(self.cache.rglob("*.json"))), 1)
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text) as ocr:
            result = self.read()
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(len(result.pages), 2)

    def test_refresh_replaces_successful_pages(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text) as ocr:
            self.read()
            self.read(refresh_ocr=True)
            self.read()
        self.assertEqual(ocr.call_count, 4)

    def test_failed_refresh_keeps_previous_valid_page(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text):
            self.read()
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=RuntimeError("temporary failure")):
            with self.assertRaisesRegex(RuntimeError, "temporary failure"):
                self.read(refresh_ocr=True)
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=AssertionError("cached")):
            self.assertEqual(self.read().pages[0].text, self.text)

    def test_failed_atomic_write_keeps_previous_valid_page(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text):
            self.read()
            with patch("eventweaver.ingestion.cache.os.replace", side_effect=OSError("disk unavailable")):
                self.read(refresh_ocr=True)
        self.assertEqual(list(self.cache.rglob("*.tmp")), [])
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=AssertionError("cached")):
            self.assertEqual(self.read().pages[0].text, self.text)

    def test_disabled_cache_neither_reads_nor_writes(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text) as ocr:
            self.read()
            before = {p: p.read_bytes() for p in self.cache.rglob("*.json")}
            self.read(cache_dir=None)
            self.read(cache_dir=None)
        self.assertEqual(ocr.call_count, 6)
        self.assertEqual(before, {p: p.read_bytes() for p in self.cache.rglob("*.json")})

    def test_corrupt_page_is_recomputed_without_discarding_other_pages(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text) as ocr:
            self.read()
            page = next(self.cache.rglob("page_000001.json"))
            page.write_text('{"text":', encoding="utf-8")
            self.read()
            record = json.loads(page.read_text())
            record["text"] = "modified without matching checksum"
            page.write_text(json.dumps(record))
            self.read()
        self.assertEqual(ocr.call_count, 4)

    def test_completed_empty_page_cached_but_document_still_errors(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value="") as ocr:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "Nessun testo leggibile"):
                    self.read()
        self.assertEqual(ocr.call_count, 4)  # initial+retry, once per page

    def test_unwritable_cache_does_not_discard_recognized_text(self):
        self.cache.write_text("not a directory")
        messages = []
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text):
            result = self.read(progress=messages.append)
        self.assertEqual(len(result.pages), 2)
        self.assertTrue(any("impossibile salvare la cache" in m for m in messages))

    def test_native_only_does_not_require_tesseract(self):
        with patch("eventweaver.ingestion.cache.engine_identity", side_effect=AssertionError("engine not needed")), \
             patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=AssertionError("OCR disabled")):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "OCR disattivato"):
                    self.read(ocr_mode="never")

    def test_generate_reuses_cache_across_new_runs_and_narrative_settings(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=self.text) as ocr:
            for limit in (None, 100):
                config = GenerateConfig(cache_dir=str(self.cache), ocr_mode="always", ocr_language="eng",
                                        max_narrative_words=limit)
                run = generate_narratives([self.source], output_dir=self.root / "out", provider=FakeProvider(),
                                           config=config, progress=lambda message: None)
                saved = json.loads((run / "manifest.json").read_text())
                self.assertEqual(saved["config"]["cache_dir"], str(self.cache))
                self.assertTrue(saved["config"]["cache_enabled"])
        self.assertEqual(ocr.call_count, 2)


class CacheCLIAndConfigTests(unittest.TestCase):
    def test_flags_defaults_and_conflict(self):
        parser = build_parser()
        args = parser.parse_args(["generate", "input.pdf", "--model", "test"])
        self.assertEqual(args.cache_dir, ".eventweaver_cache")
        self.assertFalse(args.no_cache)
        args = parser.parse_args(["generate", "input.pdf", "--model", "test", "--refresh-ocr", "--cache-dir", "custom"])
        self.assertTrue(args.refresh_ocr)
        self.assertEqual(args.cache_dir, "custom")
        with self.assertRaises(ValueError):
            GenerateConfig(cache_enabled=False, refresh_ocr=True)

    def test_cli_passes_cache_controls_to_generation(self):
        with redirect_stdout(io.StringIO()), patch("eventweaver.cli.build_provider"), patch("eventweaver.cli.generate_narratives") as generate:
            code = main(["generate", "input.pdf", "--model", "test", "--no-cache", "--cache-dir", "custom"])
        self.assertEqual(code, 0)
        config = generate.call_args.kwargs["config"]
        self.assertFalse(config.cache_enabled)
        self.assertEqual(config.cache_dir, "custom")
