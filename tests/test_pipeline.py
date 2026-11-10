import importlib.util
import hashlib
import io
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import json
import sqlite3
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import ModuleType
from pathlib import Path

from eventweaver.cli import build_parser
from eventweaver.config import GenerateConfig
from eventweaver.generation import generate_narratives, parse_themes, aggregate_corpus, themes, word_count, compact_summary, parse_paragraphs, narrative_history
from eventweaver.ingestion.chunking import iter_chunks
from eventweaver.ingestion.loader import discover_documents, iter_documents
from eventweaver.ingestion.models import LoadedDocument, DocumentPage
from eventweaver.llm import JsonLLM, InvalidLLMResponse, LLMOutputTruncated, estimate_tokens, text
from eventweaver.providers.base import GenerationResult
from eventweaver.store import Store
from eventweaver.resume import resume_narratives, load_resume_config


class FakeProvider:
    model = "deterministic-test-double"

    def __init__(self, fault=None, diverse=False):
        self.prompts = []
        self.fault = fault
        self.diverse = diverse

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
        if "FASE: SUMMARIZE" in prompt:
            result = {"summary": "The documents describe historical places, collections and recorded activities, with their dates and contexts."}
        elif "FASE: THEMES" in prompt:
            result = {"themes": [{"title": f"Theme {i}", "description": "Documented activities."}
                                 for i in range(1, 6 if self.diverse else 2)]}
            if data.get("requested_titles"):
                result = {"themes": [{"title": title, "description": "Documented activities."}
                                     for title in data["requested_titles"]]}
                self.manual_titles = True
            if self.fault == "too_many":
                result["themes"] *= 6
            if self.fault == "no_topics":
                result = {"themes": []}
        elif "FASE: ROUTE" in prompt:
            result = {"assignments": [{"chunk_id": row["id"],
                                      "theme_ids": [data["themes"][i % len(data["themes"])]["id"]]}
                                     for i, row in enumerate(data["chunks"])]}
            if getattr(self, "manual_titles", False):
                for row in result["assignments"]:
                    row["theme_ids"] = [theme["id"] for theme in data["themes"]]
            if self.fault == "missing_route":
                result["assignments"] = []
            elif self.fault == "invalid_theme":
                result["assignments"][0]["theme_ids"] = ["T99"]
            elif self.fault == "unassigned" and len(result["assignments"]) > 1:
                result["assignments"][-1].update(theme_ids=[], reason="Fuori tema")
        elif "FASE: CONDENSE" in prompt:
            result = {"paragraphs": [{"text": "The museum preserves its historical collections and records."}]}
        elif "FASE: WRITE" in prompt:
            result = {"paragraphs": [{"text": "The documents describe the museum and its activities."}
                                      for row in data["chunks"]]}
        else:
            raise AssertionError(prompt)
        return GenerationResult(json.dumps(result), 0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        (self.docs / "uno.txt").write_text("Il laboratorio ha organizzato tre incontri nel 2023.", encoding="utf-8")
        (self.docs / "due.txt").write_text("Il museo ha digitalizzato 120 schede nel 2024.", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self, provider, **kwargs):
        return generate_narratives([self.docs], output_dir=self.root / "out", provider=provider,
                                   config=GenerateConfig(retries=0, **kwargs), progress=lambda message: None)

    def test_end_to_end_original_text_no_extraction_no_citations(self):
        provider = FakeProvider()
        run = self.run_pipeline(provider)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["documents"]), 2)
        self.assertEqual(manifest["counts"]["chunks"], 2)
        story = (run / "narrative_T01.md").read_text()
        self.assertIn("The documents describe", story)
        self.assertTrue(story.startswith("# Theme 1"))
        self.assertNotIn("Fonti documentali", story)
        self.assertNotIn("[E", story)
        self.assertFalse((run / "evidence.jsonl").exists())
        with sqlite3.connect(run / "analysis.sqlite") as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("evidence", tables)
        for prompt in provider.prompts:
            self.assertNotIn("FASE: EXTRACT", prompt)
            self.assertNotIn('"quote"', prompt)
            self.assertLessEqual(estimate_tokens(prompt) + 4096 + 512, 16384)
            self.assertNotIn("FASE: VERIFY", prompt)
            if "FASE: WRITE" in prompt:
                data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
                self.assertTrue(data["chunks"])
                for chunk in data["chunks"]:
                    self.assertEqual(chunk["text"], Path(chunk["source"]).read_text())

    def test_generation_has_no_document_verifier(self):
        provider = FakeProvider()
        run = self.run_pipeline(provider)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertFalse((run / "verification.jsonl").exists())
        self.assertTrue((run / "paragraphs.jsonl").exists())
        for prompt in provider.prompts:
            self.assertNotIn("FASE: VERIFY", prompt)
        for line in (run / "paragraphs.jsonl").read_text().splitlines():
            self.assertNotIn("verification", json.loads(line)["body"])

    def test_english_default_in_cli_config_and_generation_prompts(self):
        self.assertEqual(GenerateConfig().language, "English")
        args = build_parser().parse_args(["generate", "documenti", "--model", "test"])
        self.assertEqual(args.language, "English")
        provider = FakeProvider()
        self.run_pipeline(provider)
        for prompt in provider.prompts:
            if "FASE: THEMES" in prompt:
                self.assertIn("temi in English", prompt)
            if "FASE: WRITE" in prompt:
                data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
                self.assertEqual(data["language"], "English")
                self.assertIn("do not copy untranslated prose", prompt)

    def test_too_many_themes_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Lista assente o troppo lunga"):
            self.run_pipeline(FakeProvider("too_many"))
        with self.assertRaises(ValueError):
            GenerateConfig(max_themes=6)

    def test_no_topics_returns_diagnostic(self):
        with self.assertRaisesRegex(RuntimeError, "Nessun tema restituito"):
            self.run_pipeline(FakeProvider("no_topics"))
        run = next((self.root / "out").iterdir())
        self.assertEqual(len((run / "chunks.jsonl").read_text().splitlines()), 2)

    def test_recursive_discovery_deduplicates(self):
        (self.docs / "themes.txt").write_text("Questo è un documento.")
        self.assertEqual(len(discover_documents([self.docs, self.docs / "uno.txt"])), 3)

    def test_output_inside_input_rejected_before_llm(self):
        provider = FakeProvider()
        with self.assertRaisesRegex(ValueError, "esterna"):
            generate_narratives([self.docs], output_dir=self.docs / "out", provider=provider)
        self.assertEqual(provider.prompts, [])

    def test_chunking_covers_all_characters_including_long_tokens(self):
        source = "x" * 1037
        doc = LoadedDocument("D1", Path("x"), "x", "txt", [DocumentPage(1, source, "text")])
        positions = set()
        for chunk in iter_chunks(doc, chunk_chars=100, overlap_chars=20):
            positions.update(range(chunk["start"], chunk["end"]))
            self.assertLessEqual(len(chunk["text"]), 100)
        self.assertEqual(positions, set(range(len(source))))

    def test_budget_rejects_oversized_input_before_call(self):
        provider = FakeProvider()
        with self.assertRaises(ValueError):
            JsonLLM(provider, GenerateConfig()).call("task", "x" * 20000, lambda value: value)
        self.assertEqual(provider.prompts, [])

    def test_csv_preserves_columns_and_locations(self):
        path = self.docs / "tabella.csv"
        path.write_text('nome;valore\nAurora;"tre; incontri"\nArco;120\n', encoding="utf-8")
        document = next(iter_documents([path]))
        self.assertEqual([page.page_number for page in document.pages], [2, 3])
        self.assertIn("valore: tre; incontri", document.pages[0].text)

    def test_empty_document_fails(self):
        (self.docs / "vuoto.txt").write_text("")
        with self.assertRaisesRegex(ValueError, "vuoto"):
            self.run_pipeline(FakeProvider())

    def test_only_generate_command(self):
        parser = build_parser()
        commands = next(a for a in parser._actions if isinstance(getattr(a, "choices", None), dict))
        self.assertEqual(set(commands.choices), {"generate"})

    def test_hierarchical_reduction_on_72_documents(self):
        for i in range(70):
            (self.docs / f"document_{i:03d}.txt").write_text(f"Attività numero {i}: è registrato un evento documentato.")
        provider = FakeProvider(diverse=True)
        run = self.run_pipeline(provider)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(len(manifest["documents"]), 72)
        self.assertEqual(len(manifest["outputs"]), 5)
        self.assertEqual(len((run / "assignments.jsonl").read_text().splitlines()), 72)
        self.assertEqual(sum("FASE: THEMES" in p for p in provider.prompts), 1)
        self.assertGreater(sum("FASE: SUMMARIZE" in p for p in provider.prompts), 0)
        for prompt in provider.prompts:
            self.assertLessEqual(estimate_tokens(prompt) + 4096 + 512, 16384)

    def test_every_long_document_chunk_is_analysed_and_used(self):
        (self.docs / "lungo.txt").write_text("Un evento è documentato. " * 150)
        provider = FakeProvider()
        run = self.run_pipeline(provider, chunk_chars=200, overlap_chars=20)
        count = json.loads((run / "manifest.json").read_text())["counts"]["chunks"]
        self.assertEqual(sum("FASE: THEMES" in p for p in provider.prompts), 1)
        self.assertEqual(sum("FASE: ANALYZE" in p for p in provider.prompts), 0)
        written = set()
        for prompt in provider.prompts:
            if "FASE: WRITE" in prompt:
                data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
                written.update(chunk["id"] for chunk in data["chunks"])
        self.assertEqual(len(written), count)
        self.assertGreater(count, 10)

    def test_unassigned_chunks_are_recorded(self):
        run = self.run_pipeline(FakeProvider("unassigned"))
        self.assertEqual(json.loads((run / "manifest.json").read_text())["counts"]["unassigned"], 1)
        self.assertIn("Fuori tema", (run / "unassigned.jsonl").read_text())

    def test_missing_chunk_assignment_fails(self):
        with self.assertRaisesRegex(RuntimeError, "ogni blocco"):
            self.run_pipeline(FakeProvider("missing_route"))

    def test_unknown_theme_fails(self):
        with self.assertRaisesRegex(RuntimeError, "ID tema sconosciuto"):
            self.run_pipeline(FakeProvider("invalid_theme"))


class ReaderTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("docx"), "python-docx non installato")
    def test_docx_preserves_table_paragraph_order(self):
        from docx import Document
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "documento.docx"
            doc = Document()
            doc.add_paragraph("Prima della tabella.")
            table = doc.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Soggetto"
            table.cell(0, 1).text = "Aurora"
            doc.add_paragraph("Dopo la tabella.")
            doc.save(path)
            loaded = next(iter_documents([path]))
            self.assertEqual(loaded.text, "Prima della tabella.\nSoggetto | Aurora\nDopo la tabella.")

    @unittest.skipUnless(importlib.util.find_spec("pymupdf"), "PyMuPDF non installato")
    def test_pdf_retains_page_numbers(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "documento.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page().insert_text((72, 72), "Prima pagina: tre incontri nel 2023.")
                pdf.new_page().insert_text((72, 72), "Seconda pagina: 120 schede nel 2024.")
                pdf.save(path)
            loaded = next(iter_documents([path], ocr_mode="never"))
            self.assertEqual([page.page_number for page in loaded.pages], [1, 2])
            self.assertIn("tre incontri", loaded.pages[0].text)
            self.assertIn("120 schede", loaded.pages[1].text)


class ThemeValidationTests(unittest.TestCase):
    def test_description_is_preserved_and_title_whitespace_normalized(self):
        messages = []
        result = parse_themes({"themes": [{"title": "  Tema\n generale  ",
                                            "description": "Descrizione tematica molto lunga. " * 30}]},
                              5, report=messages.append)
        self.assertEqual(result[0]["title"], "Tema generale")
        self.assertEqual(result[0]["description"], "Descrizione tematica molto lunga. " * 30)
        self.assertTrue(any("Theme description preserved:" in message for message in messages))

    def test_missing_null_or_empty_description_uses_title(self):
        for description in (None, "", "  "):
            with self.subTest(description=description):
                result = parse_themes({"themes": [{"title": "Tema", "description": description}]}, 5)
                self.assertEqual(result[0]["description"], "Tema")
        self.assertEqual(parse_themes({"themes": [{"title": "Tema"}]}, 5)[0]["description"], "Tema")

    def test_wrong_description_type_names_field(self):
        with self.assertRaisesRegex(ValueError, r"themes\[0\].description: attesa stringa, ricevuto list"):
            parse_themes({"themes": [{"title": "Tema", "description": ["testo"]}]}, 5)

    def test_empty_title_still_fails(self):
        with self.assertRaisesRegex(ValueError, r"themes\[0\].title: testo vuoto"):
            parse_themes({"themes": [{"title": "", "description": "Tema"}]}, 5)

    def test_retry_uses_feedback_and_preserves_source(self):
        class CorrectionProvider:
            model = "test"
            def __init__(self):
                self.prompts = []
            def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                description = [] if len(self.prompts) == 1 else "Descrizione corretta"
                return GenerationResult(json.dumps({"themes": [{"title": "Tema", "description": description}]}), 0)
        provider = CorrectionProvider()
        config = GenerateConfig(retries=1)
        llm = JsonLLM(provider, config)
        data = {"text": "Documento originale integrale: attività 2024."}
        result = llm.call("FASE: ANALYZE", data, lambda obj: parse_themes(obj, 5))
        self.assertEqual(result[0]["description"], "Descrizione corretta")
        self.assertEqual(len(provider.prompts), 2)
        self.assertNotEqual(*provider.prompts)
        self.assertIn("themes[0].description", provider.prompts[1])
        for prompt in provider.prompts:
            self.assertEqual(json.loads(prompt.split("DATA_JSON:\n", 1)[1]), data)
            self.assertLessEqual(estimate_tokens(prompt) + config.max_output_tokens + config.context_safety_tokens,
                                 config.context_tokens)

    def test_complete_run_accepts_model_descriptions_over_300_characters(self):
        class LongDescriptionProvider(FakeProvider):
            def generate(self, prompt, **kwargs):
                response = super().generate(prompt, **kwargs)
                if "FASE: THEMES" in prompt:
                    obj = json.loads(response.text)
                    for theme in obj["themes"]:
                        theme["description"] = "Descrizione molto lunga delle attività documentate. " * 20
                    response.text = json.dumps(obj)
                return response
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "documento.txt"
            original = "Il museo ha digitalizzato 120 schede nel 2024."
            source.write_text(original)
            run = generate_narratives([source], output_dir=root / "out", provider=LongDescriptionProvider(),
                                      config=GenerateConfig(retries=0), progress=lambda message: None)
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "completed")
            self.assertIn("The documents describe", (run / "narrative_T01.md").read_text())
            self.assertEqual(json.loads((run / "chunks.jsonl").read_text())["body"]["text"], original)
            self.assertEqual(json.loads((run / "themes.json").read_text())[0]["description"],
                             "Descrizione molto lunga delle attività documentate. " * 20)


class NarrativeLengthTests(unittest.TestCase):
    def test_long_paragraphs_are_saved_intact_without_retry(self):
        for length in (1251, 1252, 1275, 6000):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as temp:
                content = ("The museum documents its collection and related activities. " * 120)[:length - 1] + "."
                class LongParagraphProvider(FakeProvider):
                    def generate(self, prompt, **kwargs):
                        if "FASE: WRITE" in prompt:
                            self.prompts.append(prompt)
                            return GenerationResult(json.dumps({"paragraphs": [{"text": content}]}), 0)
                        return super().generate(prompt, **kwargs)
                root = Path(temp)
                source = root / "documento.txt"
                source.write_text("Il museo ha digitalizzato 120 schede nel 2024.")
                provider = LongParagraphProvider()
                run = generate_narratives([source], output_dir=root / "out", provider=provider,
                                          config=GenerateConfig(), progress=lambda message: None)
                manifest = json.loads((run / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "completed")
                self.assertEqual(sum("FASE: WRITE" in prompt for prompt in provider.prompts), 1)
                self.assertEqual(json.loads((run / "paragraphs.jsonl").read_text())["body"]["text"], content)
                self.assertEqual((run / "narrative_T01.md").read_text(), "# Theme 1\n\n" + content + "\n\n")
                writing_prompt = next(p for p in provider.prompts if "FASE: WRITE" in p)
                self.assertNotIn("1200", writing_prompt)
                self.assertFalse((run / "verification.jsonl").exists())

    def test_empty_or_non_text_paragraph_still_rejected(self):
        for value in (None, [], 12, "", "   "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                text(value, field="paragraphs[0].text")

    def test_explicit_limits_for_other_fields_still_apply(self):
        with self.assertRaisesRegex(ValueError, "massimo 500"):
            text("x" * 501, limit=500, field="reason")


@unittest.skipUnless(importlib.util.find_spec("pymupdf"), "PyMuPDF non installato")
class OCRTests(unittest.TestCase):
    def setUp(self):
        import pymupdf
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "mixed.pdf"
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Museum Archive: the collection contains 120 records from 1997.")
            pdf.new_page()
            pdf.save(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_auto_ocr_retries_empty_page_and_retains_page_numbers(self):
        messages = []
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=["", ""]) as ocr:
            doc = next(iter_documents([self.path], progress=messages.append))
        self.assertEqual(ocr.call_count, 2)
        self.assertEqual(ocr.call_args.kwargs["dpi"], 300)
        self.assertEqual(ocr.call_args.kwargs["page_segmentation"], 6)
        self.assertEqual(doc.pages[1].extraction_method, "ocr-empty")
        self.assertEqual(doc.pages[1].page_number, 2)
        self.assertIn("120 records", doc.text)
        self.assertTrue(any("pagina saltata" in message for message in messages))

    def test_retry_recovers_originally_empty_page(self):
        recovered = "The scanned page describes the museum collection and its 120 records from 1997."
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=["", recovered]):
            doc = next(iter_documents([self.path]))
        self.assertEqual(doc.pages[1].text, recovered)
        self.assertEqual(doc.pages[1].extraction_method, "ocr-retry")

    def test_never_does_not_invoke_ocr_for_blank_pages(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page") as ocr:
            doc = next(iter_documents([self.path], ocr_mode="never"))
        ocr.assert_not_called()
        self.assertEqual(doc.pages[1].extraction_method, "native-empty")

    def test_all_empty_document_errors_after_ocr_attempts(self):
        import pymupdf
        path = self.root / "empty.pdf"
        with pymupdf.open() as pdf:
            pdf.new_page()
            pdf.save(path)
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value="") as ocr:
            with self.assertRaisesRegex(ValueError, "anche dopo i tentativi OCR"):
                next(iter_documents([path]))
        self.assertEqual(ocr.call_count, 2)

    def test_ocr_setup_error_identifies_document_and_page(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", side_effect=RuntimeError("Tesseract non trovato")):
            with self.assertRaisesRegex(RuntimeError, "mixed.pdf, pagina 2: Tesseract non trovato"):
                next(iter_documents([self.path]))

    def test_manifest_audits_blank_pages_without_aborting_generation(self):
        with patch("eventweaver.ingestion.pdf_loader.ocr_pdf_page", return_value=""):
            run = generate_narratives([self.path], output_dir=self.root / "out", provider=FakeProvider(),
                                      config=GenerateConfig(retries=0), progress=lambda message: None)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["documents"][0]["pages"][1],
                         {"number": 2, "method": "ocr-empty", "characters": 0, "skipped": True})
        self.assertEqual(manifest["counts"]["chunks"], 1)

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract non installato")
    def test_real_ocr_of_image_only_pdf_without_pytesseract(self):
        import pymupdf
        path = self.root / "scan.pdf"
        with pymupdf.open() as original:
            page = original.new_page()
            page.insert_text((60, 100), "Museum Archive 1997", fontsize=24)
            page.insert_text((60, 150), "The collection contains 120 records.", fontsize=20)
            image = page.get_pixmap(dpi=150).tobytes("png")
        with pymupdf.open() as scanned:
            page = scanned.new_page()
            page.insert_image(page.rect, stream=image)
            self.assertEqual(page.get_text().strip(), "")
            scanned.save(path)
        doc = next(iter_documents([path], ocr_language="eng"))
        self.assertIn("Museum", doc.text)
        self.assertIn("1997", doc.text)
        self.assertIn("120", doc.text)
        self.assertIn(doc.pages[0].extraction_method, ("ocr", "ocr-retry"))


class GlobalCorpusTests(unittest.TestCase):
    def test_first_llm_call_sees_all_documents_already_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "first.txt").write_text("The amphitheatre is an archaeological monument.")
            (docs / "last.txt").write_text("The amphitheatre hosts theatre performances today.")
            class AggregateProvider(FakeProvider):
                def generate(self, prompt, **kwargs):
                    run = next((root / "out").iterdir())
                    manifest = json.loads((run / "manifest.json").read_text())
                    self_test.assertTrue(manifest["ingestion_completed"])
                    self_test.assertEqual(len(manifest["documents"]), 2)
                    with sqlite3.connect(run / "analysis.sqlite") as db:
                        self_test.assertEqual(db.execute("SELECT count(*) FROM chunks").fetchone()[0], 2)
                    if "FASE: THEMES" in prompt:
                        data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
                        self_test.assertEqual(data["document_count"], 2)
                        self_test.assertEqual(data["representation"], "original_chunks")
                        self_test.assertIn("archaeological monument", str(data))
                        self_test.assertIn("theatre performances", str(data))
                    return super().generate(prompt, **kwargs)
            self_test = self
            provider = AggregateProvider()
            run = generate_narratives([docs], output_dir=root / "out", provider=provider,
                                      config=GenerateConfig(retries=0), progress=lambda message: None)
            self.assertEqual(sum("FASE: THEMES" in p for p in provider.prompts), 1)
            self.assertFalse(any("FASE: SUMMARIZE" in p or "FASE: ANALYZE" in p for p in provider.prompts))
            chunks = [json.loads(line) for line in (run / "chunks.jsonl").read_text().splitlines()]
            self.assertTrue(all("analysis" not in chunk for chunk in chunks))

    def test_failed_ingestion_makes_no_llm_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "a.txt").write_text("Some readable content.")
            (docs / "z.txt").write_text("")
            provider = FakeProvider()
            with self.assertRaises(ValueError):
                generate_narratives([docs], output_dir=root / "out", provider=provider,
                                    progress=lambda message: None)
            self.assertEqual(provider.prompts, [])

    def test_multilevel_summaries_cover_all_sources_before_one_global_selection(self):
        class SummaryProvider(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: SUMMARIZE" in prompt:
                    self.prompts.append(prompt)
                    return GenerationResult(json.dumps({"summary": "The historical sites and museum collections are documented. " * 18}), 0)
                return super().generate(prompt, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "corpus.sqlite")
            try:
                source_ids = set()
                for index in range(80):
                    chunk = {"id": f"D{index:05d}_C000001", "text": "Original corpus content. " * 110}
                    source_ids.add(chunk["id"])
                    store.db.execute("INSERT INTO chunks VALUES (?,?)", (chunk["id"], json.dumps(chunk)))
                store.db.commit()
                provider = SummaryProvider()
                config = GenerateConfig(retries=0)
                llm = JsonLLM(provider, config)
                aggregate = aggregate_corpus(llm, store, config, 80)
                self.assertEqual(aggregate["representation"], "aggregate_summaries")
                self.assertGreaterEqual(store.db.execute("SELECT max(level) FROM summaries").fetchone()[0], 1)
                self.assertFalse(any("FASE: THEMES" in p for p in provider.prompts))
                edges = {}
                for row in store.rows("SELECT * FROM summary_inputs"):
                    edges.setdefault(row["summary_id"], []).append(row["input_id"])
                def leaves(key):
                    if key in source_ids:
                        return {key}
                    result = set()
                    for child in edges[key]:
                        result.update(leaves(child))
                    return result
                covered = set()
                for item in aggregate["items"]:
                    covered.update(leaves(item["id"]))
                    self.assertNotIn("themes", item)
                self.assertEqual(covered, source_ids)
                themes(llm, aggregate, config)
                self.assertEqual(sum("FASE: THEMES" in p for p in provider.prompts), 1)
                for prompt in provider.prompts:
                    self.assertLessEqual(estimate_tokens(prompt) + 4096 + 512, 16384)
            finally:
                store.db.close()


class TitleAndLengthTests(unittest.TestCase):
    def test_cli_options_and_config(self):
        args = build_parser().parse_args(["generate", "documents", "--model", "test",
                                         "--titles", "Ancient Segobriga", "The Amphitheatre", "--max-words", "500"])
        self.assertEqual(args.titles, ["Ancient Segobriga", "The Amphitheatre"])
        self.assertEqual(args.max_words, 500)
        config = GenerateConfig(narrative_titles=args.titles, max_narrative_words=args.max_words)
        self.assertEqual(config.narrative_titles, tuple(args.titles))

    def test_invalid_titles_or_word_limits_are_rejected(self):
        for titles in (("",), ("same", "Same"), ("line\nbreak",), tuple(str(i) for i in range(6)), "one title"):
            with self.subTest(titles=titles), self.assertRaises(ValueError):
                GenerateConfig(narrative_titles=titles)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                GenerateConfig(max_narrative_words=limit)

    def test_manual_titles_preserved_in_order_and_global_selection(self):
        titles = ("Ancient Segobriga", ("The Amphitheatre #1 " + "and its history " * 9).strip())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "doc.txt"
            source.write_text("The amphitheatre is part of the ancient city of Segobriga.")
            provider = FakeProvider()
            run = generate_narratives([source], output_dir=root / "out", provider=provider,
                                      config=GenerateConfig(narrative_titles=titles, retries=0), progress=lambda message: None)
            selected = json.loads((run / "themes.json").read_text())
            self.assertEqual([row["title"] for row in selected], list(titles))
            for i, title in enumerate(titles, 1):
                self.assertTrue((run / f"narrative_T{i:02d}.md").read_text().startswith("# " + title + "\n"))
            calls = [p for p in provider.prompts if "FASE: THEMES" in p]
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0].split("DATA_JSON:\n", 1)[1])["requested_titles"], list(titles))

    def test_word_maximum_applies_separately_to_each_narrative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "doc.txt"
            source.write_text("The city has a museum and an amphitheatre.")
            provider = FakeProvider()
            run = generate_narratives([source], output_dir=root / "out", provider=provider,
                                      config=GenerateConfig(narrative_titles=("The City", "The Museum"), max_narrative_words=5,
                                                            retries=0), progress=lambda message: None)
            for file in sorted(run.glob("narrative_*.md")):
                body = file.read_text().split("\n", 1)[1]
                self.assertLessEqual(word_count(body), 5)
                self.assertGreater(word_count(body), 0)
            self.assertEqual(len(list(run.glob("narrative_*.md"))), 2)
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertTrue(all(row["words"] <= 5 for row in manifest["narratives"]))
            self.assertTrue(any("FASE: CONDENSE" in p for p in provider.prompts))
            self.assertFalse(any("FASE: VERIFY" in p for p in provider.prompts))

    def test_under_limit_keeps_original_without_editorial_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "doc.txt"
            source.write_text("The city has a museum.")
            provider = FakeProvider()
            run = generate_narratives([source], output_dir=root / "out", provider=provider,
                                      config=GenerateConfig(max_narrative_words=500, retries=0), progress=lambda message: None)
            self.assertFalse(any("FASE: CONDENSE" in p for p in provider.prompts))
            self.assertIn("The documents describe the museum and its activities.", (run / "narrative_T01.md").read_text())

    def test_all_parts_of_long_draft_enter_shortening_before_final_limit(self):
        class LongDraftProvider(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: WRITE" in prompt:
                    self.prompts.append(prompt)
                    return GenerationResult(json.dumps({"paragraphs": [{"text": "Draft narrative information. " * 600}]}), 0)
                return super().generate(prompt, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "doc.txt"
            source.write_text("The city has a museum.")
            provider = LongDraftProvider()
            run = generate_narratives([source], output_dir=root / "out", provider=provider,
                                      config=GenerateConfig(max_narrative_words=100, retries=0), progress=lambda message: None)
            seen_words = 0
            for prompt in provider.prompts:
                if "FASE: CONDENSE" in prompt:
                    data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
                    seen_words += sum(word_count(row["text"]) for row in data["paragraphs"])
                self.assertLessEqual(estimate_tokens(prompt) + 4096 + 512, 16384)
            self.assertEqual(seen_words, 1800)
            self.assertLessEqual(word_count((run / "narrative_T01.md").read_text().split("\n", 1)[1]), 100)


class AggregationRecoveryTests(unittest.TestCase):
    def test_verbose_summaries_converge_for_14_documents(self):
        class VerboseProvider(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: SUMMARIZE" in prompt:
                    self.prompts.append(prompt)
                    return GenerationResult(json.dumps({"summary": "The museum preserves historical material and records. " * 240}), 0)
                return super().generate(prompt, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            for i in range(14):
                (docs / f"source_{i:02d}.txt").write_text((f"Document {i} describes the museum collection and its historical context. " * 125))
            provider = VerboseProvider()
            run = generate_narratives([docs], output_dir=root / "out", provider=provider,
                                      config=GenerateConfig(retries=0), progress=lambda message: None)
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["documents"]), 14)
            self.assertEqual(sum("FASE: THEMES" in p for p in provider.prompts), 1)
            records = [json.loads(line) for line in (run / "summaries.jsonl").read_text().splitlines()]
            self.assertTrue(records)
            self.assertLess(max(row["level"] for row in records), 5)
            for row in records:
                self.assertLessEqual(estimate_tokens(json.dumps(row["body"]["text"], ensure_ascii=False)), row["byte_budget"])
                self.assertGreater(len(row["raw_summary"]), len(row["body"]["text"]))
            for prompt in provider.prompts:
                self.assertLessEqual(estimate_tokens(prompt) + 4096 + 512, 16384)
                self.assertNotIn("FASE: VERIFY", prompt)

    def test_compaction_honors_serialized_utf8_budget_and_samples_ends(self):
        original = "Beginning fact. " + "Middle material is described in detail. " * 40 + "Final fact."
        result = compact_summary(original, 140)
        self.assertIn("Beginning fact.", result)
        self.assertIn("Final fact.", result)
        for value in ('é漢字"\\' * 300, "x" * 10000):
            result = compact_summary(value, 128)
            self.assertTrue(result)
            self.assertLessEqual(estimate_tokens(json.dumps(result, ensure_ascii=False)), 128)

    def test_truncated_call_retries_with_larger_output_within_user_limit(self):
        class TruncatingProvider:
            model = "test"
            def __init__(self):
                self.calls = []
            def generate(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs["max_new_tokens"]))
                if len(self.calls) == 1:
                    return GenerationResult('{"summary":"unfinished', 0, error="length", truncated=True)
                return GenerationResult('{"summary":"A concise complete summary."}', 0)
        provider = TruncatingProvider()
        config = GenerateConfig(retries=1)
        result = JsonLLM(provider, config).call("FASE: SUMMARIZE", {"text": "original"},
                                               lambda obj: obj["summary"], output_tokens=256)
        self.assertEqual(result, "A concise complete summary.")
        self.assertEqual([tokens for _, tokens in provider.calls], [256, 512])
        self.assertIn("molto più breve", provider.calls[1][0])
        for prompt, tokens in provider.calls:
            self.assertLessEqual(estimate_tokens(prompt) + tokens + config.context_safety_tokens, config.context_tokens)
            self.assertEqual(json.loads(prompt.split("DATA_JSON:\n", 1)[1]), {"text": "original"})

    def test_complete_json_at_token_limit_is_accepted(self):
        class CompleteProvider:
            model = "test"
            def generate(self, prompt, **kwargs):
                return GenerationResult('{"summary":"Complete."}', 0, error="length", truncated=True)
        llm = JsonLLM(CompleteProvider(), GenerateConfig(retries=0))
        self.assertEqual(llm.call("FASE: SUMMARIZE", {}, lambda obj: obj["summary"]), "Complete.")
        self.assertEqual(llm.calls, 1)

    def test_ollama_retains_text_and_propagates_truncation(self):
        from eventweaver.ollama_client import generate_ollama
        from eventweaver.providers.ollama import OllamaProvider
        requests_module = ModuleType("requests")
        requests_module.post = MagicMock()
        requests_module.post.return_value.status_code = 200
        requests_module.post.return_value.json.return_value = {"done_reason": "length", "response": '{"summary":"partial'}
        with patch.dict("sys.modules", {"requests": requests_module}):
            result = generate_ollama("prompt", "test", max_new_tokens=256)
        self.assertTrue(result.truncated)
        self.assertEqual(result.text, '{"summary":"partial')
        with patch("eventweaver.providers.ollama.generate_ollama", return_value=result):
            provider = OllamaProvider("test", preflight=False)
            forwarded = provider.generate("prompt")
        self.assertTrue(forwarded.truncated)
        self.assertEqual(forwarded.text, result.text)

    def test_truncated_batches_split_and_merged_summaries_retain_input_lineage(self):
        class SplitProvider(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: SUMMARIZE" in prompt:
                    self.prompts.append(prompt)
                    data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
                    if len(data["items"]) > 1:
                        return GenerationResult('{"summary":"partial', 0, error="length", truncated=True)
                    return GenerationResult(json.dumps({"summary": "The museum collection and its history are documented. " * 22}), 0)
                return super().generate(prompt, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "corpus.sqlite")
            try:
                source_ids = set()
                for i in range(80):
                    chunk = {"id": f"D{i:05d}_C000001", "text": "Source material about the museum. " * 90}
                    source_ids.add(chunk["id"])
                    store.db.execute("INSERT INTO chunks VALUES (?,?)", (chunk["id"], json.dumps(chunk)))
                store.db.commit()
                provider = SplitProvider()
                config = GenerateConfig(retries=0)
                llm = JsonLLM(provider, config)
                aggregate = aggregate_corpus(llm, store, config, 80)
                self.assertTrue(store.db.execute("SELECT count(*) FROM summaries WHERE compression_method='bounded_merge'").fetchone()[0])
                edges = {}
                for row in store.rows("SELECT * FROM summary_inputs"):
                    edges.setdefault(row["summary_id"], []).append(row["input_id"])
                def leaves(key):
                    if key in source_ids:
                        return {key}
                    result = set()
                    for child in edges[key]:
                        result.update(leaves(child))
                    return result
                covered = set()
                for item in aggregate["items"]:
                    covered.update(leaves(item["id"]))
                self.assertEqual(covered, source_ids)
                self.assertFalse(any("FASE: THEMES" in p for p in provider.prompts))
                themes(llm, aggregate, config)
                self.assertEqual(sum("FASE: THEMES" in p for p in provider.prompts), 1)
            finally:
                store.db.close()


class NarrativeResponseTests(unittest.TestCase):
    def test_common_narrative_response_formats(self):
        values = ({"paragraphs": [{"text": "First."}, {"text": "Second."}]},
                  {"paragraphs": ["First.", "Second."]}, {"paragraphs": "First.\n\nSecond."},
                  {"text": "First.\n\nSecond."}, {"summary": "First.\n\nSecond."},
                  {"narrative": {"paragraphs": ["First.", "Second."]}},
                  ["First.", {"content": "Second."}], "First.\n\nSecond.")
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(parse_paragraphs(value), [{"text": "First."}, {"text": "Second."}])
        with self.assertRaisesRegex(ValueError, "ricevute chiavi: unexpected"):
            parse_paragraphs({"unexpected": "metadata"})

    def test_condense_accepts_json_string_root_list_and_plain_prose(self):
        for raw in ('{"text":"The museum preserves records."}',
                    '{"summary":"The museum preserves records."}',
                    '["The museum preserves records."]', '"The museum preserves records."',
                    'The museum preserves records.'):
            class AlternativeProvider:
                model = "test"
                def generate(self, prompt, **kwargs):
                    return GenerationResult(raw, 0)
            with self.subTest(raw=raw):
                llm = JsonLLM(AlternativeProvider(), GenerateConfig(retries=0))
                result = llm.call("FASE: CONDENSE", {}, parse_paragraphs, allow_narrative=True)
                self.assertEqual(result, [{"text": "The museum preserves records."}])
                self.assertEqual(llm.calls, 1)

    def test_incomplete_json_is_not_treated_as_plain_prose(self):
        for raw in ('{"text":"broken', '["broken', '"broken'):
            class BrokenProvider:
                model = "test"
                def generate(self, prompt, **kwargs):
                    return GenerationResult(raw, 0)
            with self.subTest(raw=raw), self.assertRaises(InvalidLLMResponse):
                JsonLLM(BrokenProvider(), GenerateConfig(retries=0)).call(
                    "FASE: CONDENSE", {}, parse_paragraphs, allow_narrative=True)

    def test_unusable_editorial_response_uses_draft_and_logs_exact_reply(self):
        class UnusableEditor(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: WRITE" in prompt:
                    self.prompts.append(prompt)
                    return GenerationResult(json.dumps({"text": ("Museum records document its collection. " * 900)}), 0)
                if "FASE: CONDENSE" in prompt:
                    self.prompts.append(prompt)
                    return GenerationResult('{"unexpected":"No paragraph array"}', 0)
                return super().generate(prompt, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "doc.txt"
            source.write_text("The museum documents its collection.")
            run = generate_narratives([source], output_dir=root / "out", provider=UnusableEditor(),
                                      config=GenerateConfig(retries=0, max_narrative_words=3000), progress=lambda message: None)
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertLessEqual(manifest["narratives"][0]["words"], 3000)
            self.assertGreater(manifest["narratives"][0]["words"], 0)
            records = [json.loads(row) for row in (run / "responses.jsonl").read_text().splitlines()]
            failed = [row for row in records if row["phase"] == "FASE: CONDENSE"]
            self.assertTrue(failed)
            self.assertEqual(failed[0]["response_text"], '{"unexpected":"No paragraph array"}')
            self.assertIn("ricevute chiavi: unexpected", failed[0]["error"])
            revisions = [json.loads(row) for row in (run / "revisions.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["body"]["editorial_method"] == "draft_word_cap" for row in revisions))

    def test_provider_failure_is_not_masked_by_editorial_fallback(self):
        class OfflineEditor(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: CONDENSE" in prompt:
                    return GenerationResult("", 0, error="Server unavailable")
                return super().generate(prompt, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "doc.txt"
            source.write_text("The museum documents its collection.")
            with self.assertRaisesRegex(RuntimeError, "Server unavailable"):
                generate_narratives([source], output_dir=root / "out", provider=OfflineEditor(),
                                    config=GenerateConfig(retries=0, max_narrative_words=1), progress=lambda message: None)


class ResumeTests(unittest.TestCase):
    def make_legacy_failure(self, root):
        class FullDraftProvider(FakeProvider):
            def generate(self, prompt, **kwargs):
                if "FASE: WRITE" in prompt:
                    return GenerationResult(json.dumps({"paragraphs": [{"text": " ".join(["Museum"] * 4063) + " records."}]}), 0)
                return super().generate(prompt, **kwargs)
        source = root / "doc.txt"
        source.write_text("The museum documents its collection.")
        run = generate_narratives([source], output_dir=root / "runs with spaces", provider=FullDraftProvider(),
                                  config=GenerateConfig(retries=0, context_tokens=32768), progress=lambda message: None)
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(status="failed", error="FASE: CONDENSE: risposta LLM non valida dopo 3 tentativi: Lista assente o troppo lunga")
        manifest["config"]["max_narrative_words"] = 3000
        manifest.pop("draft_completed", None)  # Compatibility with the actual 3.6 run.
        path.write_text(json.dumps(manifest))
        with sqlite3.connect(run / "analysis.sqlite") as database:
            database.execute("DROP TABLE responses")  # The 3.6 schema did not have response logging.
        return run

    def test_resume_legacy_condense_failure_skips_ocr_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_legacy_failure(Path(temp))
            before = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in source.iterdir() if file.is_file()}
            provider = FakeProvider()
            with patch("eventweaver.generation.iter_documents", side_effect=AssertionError("OCR repeated")), \
                 patch("eventweaver.generation.write_narratives", side_effect=AssertionError("draft repeated")):
                run = resume_narratives(source, provider=provider, progress=lambda message: None)
            self.assertNotEqual(run, source)
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["resumed_from"], str(source))
            self.assertEqual(manifest["config"]["max_narrative_words"], 3000)
            self.assertLessEqual(manifest["narratives"][0]["words"], 3000)
            self.assertTrue(provider.prompts)
            self.assertTrue(all("FASE: CONDENSE" in prompt for prompt in provider.prompts))
            self.assertEqual(before, {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in source.iterdir() if file.is_file()})

    def test_incomplete_draft_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_legacy_failure(Path(temp))
            path = source / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["error"] = "FASE: WRITE: risposta non valida"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "completato la bozza"):
                resume_narratives(source, provider=FakeProvider(), progress=lambda message: None)

    def test_resume_cli_reuses_saved_context_and_overrides_word_limit(self):
        from eventweaver.cli import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_legacy_failure(root)
            destination = root / "resumed"
            with patch("eventweaver.cli.build_provider", return_value=FakeProvider()) as build, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(["generate", "--resume-run", str(source), "--model", "test",
                             "--output-dir", str(destination), "--max-words", "250", "--max-new-tokens", "16384"])
            self.assertEqual(code, 0)
            self.assertEqual(build.call_args.kwargs["num_ctx"], 32768)
            run = next(destination.iterdir())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["config"]["max_output_tokens"], 16384)
            self.assertEqual(manifest["config"]["max_narrative_words"], 250)
            self.assertLessEqual(manifest["narratives"][0]["words"], 250)

    def test_resume_does_not_change_existing_draft_language(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_legacy_failure(Path(temp))
            with self.assertRaisesRegex(ValueError, "conserva lingua e titoli"):
                resume_narratives(source, provider=FakeProvider(), config=replace(load_resume_config(source), language="Italiano"))


class RepetitionControlTests(unittest.TestCase):
    def test_penalty_defaults_cli_and_invalid_values(self):
        default = GenerateConfig()
        self.assertEqual((default.frequency_penalty, default.presence_penalty), (None, None))
        args = build_parser().parse_args(["generate", "docs", "--model", "test",
                                         "--frequency-penalty", "1.2", "--presence-penalty", "1.0"])
        self.assertEqual((args.frequency_penalty, args.presence_penalty), (1.2, 1.0))
        for value in (-0.1, 2.1, float("nan"), float("inf"), True, "1.0"):
            for field in ("frequency_penalty", "presence_penalty"):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    GenerateConfig(**{field: value})
        self.assertEqual(GenerateConfig(frequency_penalty=0, presence_penalty=2).presence_penalty, 2)

    def test_explicit_ollama_penalties_are_rejected(self):
        from eventweaver.cli import main
        with redirect_stderr(io.StringIO()):
            status = main(["generate", "docs", "--model", "test", "--skip-ollama-preflight",
                           "--frequency-penalty", "1.2"])
        self.assertEqual(status, 1)

    def test_zero_penalties_override_provider_defaults(self):
        from eventweaver.providers.ollama import OllamaProvider
        with self.assertRaises(ValueError):
            OllamaProvider("test", preflight=False, frequency_penalty=0.0)

    def test_prompts_require_diversity_and_preserve_factual_identifiers(self):
        from eventweaver import prompts
        for prompt in (prompts.WRITE, prompts.CONDENSE):
            self.assertIn("Use lexical diversity", prompt)
            self.assertIn("Do not repeat phrases", prompt)
            self.assertIn("proper names, dates, numbers and technical terms consistent", prompt)
            self.assertIn("not an additional factual source", prompt)

    def test_previous_section_is_supplied_to_subsequent_writing_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "doc.txt"
            source.write_text("The museum documents the history of the city and its collection. " * 80)
            provider = FakeProvider()
            generate_narratives([source], output_dir=root / "out", provider=provider,
                                config=GenerateConfig(chunk_chars=500, overlap_chars=30, retries=0), progress=lambda message: None)
            payloads = [json.loads(p.split("DATA_JSON:\n", 1)[1]) for p in provider.prompts if "FASE: WRITE" in p]
            self.assertGreater(len(payloads), 1)
            self.assertEqual(payloads[0]["previous_text"], "")
            self.assertIn("The documents describe", payloads[1]["previous_text"])
            for payload in payloads:
                self.assertLessEqual(estimate_tokens(json.dumps(payload["previous_text"], ensure_ascii=False)), 1600)

    def test_history_respects_serialized_utf8_budget(self):
        for value in ("A sentence about the museum. " * 1000, 'é漢字"\\ ' * 1000):
            history = narrative_history(value)
            self.assertTrue(history)
            self.assertLessEqual(estimate_tokens(json.dumps(history, ensure_ascii=False)), 1600)
            self.assertGreater(len(value), len(history))

    def test_resume_shell_overrides_penalties_from_saved_settings(self):
        from eventweaver.cli import main
        with tempfile.TemporaryDirectory() as temp:
            source = ResumeTests().make_legacy_failure(Path(temp))
            # A pre-3.8 configuration has no penalty fields.
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["config"].pop("frequency_penalty")
            manifest["config"].pop("presence_penalty")
            manifest_path.write_text(json.dumps(manifest))
            self.assertIsNone(load_resume_config(source).frequency_penalty)
            with patch("eventweaver.cli.build_provider", return_value=FakeProvider()) as build, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(["generate", "--resume-run", str(source), "--model", "test",
                               "--frequency-penalty", "1.4", "--presence-penalty", "1.1"])
            self.assertEqual(status, 0)
            self.assertEqual(build.call_args.kwargs["frequency_penalty"], 1.4)
            self.assertEqual(build.call_args.kwargs["presence_penalty"], 1.1)

    def test_explicit_penalties_for_other_providers_are_not_silently_ignored(self):
        from eventweaver.cli import main
        with patch("eventweaver.cli.build_provider") as build, redirect_stderr(io.StringIO()):
            status = main(["generate", "docs", "--model", "test", "--provider", "huggingface-api",
                           "--frequency-penalty", "1.0"])
        self.assertEqual(status, 1)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
