from __future__ import annotations

import json
from pathlib import Path

from eventweaver.ingestion.models import DocumentChunk
from eventweaver.narrative.grounding import repair_event, split_sentences, verify_event
from eventweaver.narrative.models import NarrativeEvent
from eventweaver.providers.base import GenerationResult


class StaticProvider:
    model = "mock"

    def __init__(self, payload: dict):
        self.payload = payload

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        return GenerationResult(text=json.dumps(self.payload), runtime_seconds=0.01)


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="DOC0001_C0001",
        document_id="DOC0001",
        source_path=Path("source.txt"),
        source_name="source.txt",
        source_type="txt",
        page_start=1,
        page_end=1,
        text="In 1920 the collection opened to the public.",
        order=1,
        extraction_method="text",
    )


def test_split_sentences_keeps_numbered_claim_units():
    assert split_sentences("One fact. Another fact! Final fact?") == [
        "One fact.",
        "Another fact!",
        "Final fact?",
    ]


def test_verify_event_returns_sentence_level_grounding():
    chunk = _chunk()
    event = NarrativeEvent(
        event_id=1,
        paragraph="In 1920 the collection opened to the public. It was immediately famous.",
        evidence_ids=[chunk.chunk_id],
    )
    verifier = StaticProvider(
        {
            "sentences": [
                {
                    "sentence_index": 1,
                    "status": "supported",
                    "supporting_evidence_ids": [chunk.chunk_id],
                    "reason": "The source states the opening and year.",
                },
                {
                    "sentence_index": 2,
                    "status": "unsupported",
                    "supporting_evidence_ids": [],
                    "reason": "The source does not state fame.",
                },
            ]
        }
    )
    result = verify_event(event, {chunk.chunk_id: chunk}, verifier)
    assert not result.paragraph_supported
    assert [item.status for item in result.sentences] == ["supported", "unsupported"]


def test_repair_event_cannot_invent_new_evidence_ids():
    chunk = _chunk()
    event = NarrativeEvent(1, "Unsupported claim.", [chunk.chunk_id])
    verification = verify_event(
        event,
        {chunk.chunk_id: chunk},
        StaticProvider(
            {
                "sentences": [
                    {
                        "sentence_index": 1,
                        "status": "unsupported",
                        "supporting_evidence_ids": [],
                        "reason": "Not in source.",
                    }
                ]
            }
        ),
    )
    repaired = repair_event(
        event,
        verification,
        {chunk.chunk_id: chunk},
        StaticProvider(
            {
                "paragraph": "In 1920 the collection opened to the public.",
                "evidence_ids": [chunk.chunk_id, "FAKE_ID"],
            }
        ),
    )
    assert repaired.evidence_ids == [chunk.chunk_id]
