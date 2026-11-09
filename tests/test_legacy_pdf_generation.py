from pathlib import Path
from unittest.mock import patch

from eventweaver.cli import build_parser
from eventweaver.generation import generate_narratives
from eventweaver.ingestion.models import DocumentPage, LoadedDocument
from eventweaver.ollama_client import OllamaResult
from eventweaver.source_record import SourceRecord, iter_source_records


def test_pdf_source_record_uses_corpus_loader_and_preserves_path(tmp_path):
    pdf = tmp_path / "heritage source.pdf"
    pdf.write_bytes(b"mock pdf")
    loaded = LoadedDocument(
        document_id="DOC0001",
        source_path=pdf,
        source_name=pdf.name,
        source_type="pdf",
        pages=[DocumentPage(1, "Extracted PDF heritage text.", "ocr")],
    )

    with patch("eventweaver.source_record.load_corpus", return_value=[loaded]) as load:
        records = iter_source_records(pdf, ocr_mode="always", ocr_language="eng", ocr_dpi=300)

    load.assert_called_once_with(pdf, ocr_mode="always", ocr_language="eng", ocr_dpi=300)
    assert len(records) == 1
    assert records[0].source_type == "pdf"
    assert records[0].source_path == pdf
    assert records[0].prompt_kind == "cultural-heritage"
    assert records[0].source_text == "Extracted PDF heritage text."


def test_pdf_generation_is_not_csv_generation(tmp_path):
    pdf = tmp_path / "heritage source.pdf"
    source_text = "A documented heritage route connects the historic sites."
    captured_prompts = []

    def fake_generate(prompt, *args, **kwargs):
        captured_prompts.append(prompt)
        return OllamaResult(text="Generated narrative.", runtime_seconds=0.01)

    source = SourceRecord(
        source_type="pdf",
        source_path=pdf,
        source_text=source_text,
        prompt_kind="cultural-heritage",
    )
    with patch("eventweaver.generation.iter_source_records", return_value=[source]), patch(
        "eventweaver.generation.generate_ollama", side_effect=fake_generate
    ):
        rows = generate_narratives(
            pdf,
            ["mock-model"],
            runs=1,
            output_dir=tmp_path / "output",
            temperature=0.1,
            num_ctx=1024,
            skip_ollama_preflight=True,
        )

    assert "CSV VALUE-CHAIN RECORD" not in captured_prompts[0]
    assert source_text in captured_prompts[0]
    assert rows[0]["source_type"] == "pdf"
    assert rows[0]["prompt_kind"] == "cultural-heritage"
    assert rows[0]["input_strategy"] == "auto"


def test_generate_and_all_accept_pdf_ocr_options():
    parser = build_parser()
    for command in ("generate", "all"):
        args = parser.parse_args(
            [command, "source.pdf", "--ocr", "always", "--ocr-language", "eng", "--ocr-dpi", "300"]
        )
        assert args.ocr == "always"
        assert args.ocr_language == "eng"
        assert args.ocr_dpi == 300
