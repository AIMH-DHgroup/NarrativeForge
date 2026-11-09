from __future__ import annotations

from pathlib import Path

from eventweaver.cli import build_parser
from eventweaver.ingestion.models import DocumentChunk
from eventweaver.narrative.grounding import repair_paragraph_prompt
from eventweaver.narrative.models import EventCandidate, EventVerification, NarrativeEvent, SentenceVerification
from eventweaver.narrative.prompts import narrative_synthesis_prompt


def test_corpus_cli_defaults_to_english():
    args = build_parser().parse_args(["corpus", "docs", "--model", "mock"])
    assert args.language == "English"


def test_corpus_cli_accepts_requested_language():
    args = build_parser().parse_args(["corpus", "docs", "--model", "mock", "--language", "Italian"])
    assert args.language == "Italian"


def test_synthesis_prompt_requires_requested_language():
    prompt = narrative_synthesis_prompt(
        [EventCandidate("EVT00001", "The collection opened in 1920.", ["DOC0001_C0001"], 1, "1920")],
        language="Italian",
    )
    assert "TARGET NARRATIVE LANGUAGE: Italian" in prompt
    assert "MUST be written in Italian" in prompt
    assert "150-250 words per paragraph" in prompt
    assert "different source documents" in prompt
    assert "every evidence_id" in prompt


def test_repair_prompt_preserves_requested_language():
    chunk = DocumentChunk(
        chunk_id="DOC0001_C0001",
        document_id="DOC0001",
        source_path=Path("source.txt"),
        source_name="source.txt",
        source_type="txt",
        page_start=1,
        page_end=1,
        text="La collezione apri nel 1920.",
        order=1,
        extraction_method="text",
    )
    event = NarrativeEvent(1, "La collezione apri nel 1920.", [chunk.chunk_id])
    verification = EventVerification(
        event_id=1,
        paragraph_supported=False,
        sentences=[
            SentenceVerification(
                sentence_index=1,
                sentence=event.paragraph,
                status="partial",
                supporting_evidence_ids=[chunk.chunk_id],
                reason="Needs repair.",
            )
        ],
    )
    prompt = repair_paragraph_prompt(event, verification, [chunk], language="Italian")
    assert "MUST be written in Italian" in prompt
    assert "Do not switch language during repair" in prompt
