from __future__ import annotations

import csv
import json
import re

from eventweaver.narrative.models import EventCandidate
from eventweaver.narrative.pipeline import _plan_themes, _split_even, generate_corpus_narrative
from eventweaver.providers.base import GenerationResult


class MockProvider:
    model = "mock"

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        if "extracting factual narrative events" in prompt:
            evidence_ids = re.findall(r"\[EVIDENCE ([A-Z0-9_]+)", prompt)
            payload = {
                "events": [
                    {
                        "summary": f"Grounded event from {evidence_id}",
                        "evidence_ids": [evidence_id],
                        "date_hint": "",
                    }
                    for evidence_id in evidence_ids
                ]
            }
        elif "strict semantic grounding verifier" in prompt:
            evidence_ids = re.findall(r"\[EVIDENCE ([A-Z0-9_]+)", prompt)
            sentence_indexes = [int(value) for value in re.findall(r"^\[(\d+)\] ", prompt, flags=re.MULTILINE)]
            evidence_id = evidence_ids[0]
            payload = {
                "sentences": [
                    {
                        "sentence_index": index,
                        "status": "supported",
                        "supporting_evidence_ids": [evidence_id],
                        "reason": "The cited source supports this sentence.",
                    }
                    for index in sentence_indexes
                ]
            }
        elif "Revise one narrative paragraph" in prompt:
            evidence_ids = re.findall(r"\[EVIDENCE ([A-Z0-9_]+)", prompt)
            payload = {
                "paragraph": "Repaired grounded paragraph.",
                "evidence_ids": evidence_ids[:1],
            }
        else:
            evidence_ids = []
            for match in re.findall(r"evidence=([^\]]+)", prompt):
                for item in match.split(","):
                    item = item.strip()
                    if item and item not in evidence_ids:
                        evidence_ids.append(item)
            payload = {
                "events": [
                    {
                        "paragraph": f"Narrative paragraph grounded in {evidence_id}.",
                        "evidence_ids": [evidence_id],
                    }
                    for evidence_id in evidence_ids
                ]
            }
        return GenerationResult(text=json.dumps(payload), runtime_seconds=0.01)


def test_multi_document_pipeline_writes_grounded_outputs(tmp_path, monkeypatch):
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("In 1920 the documented collection opened to the public.", encoding="utf-8")
    (source_dir / "b.md").write_text("In 1925 the documented collection acquired additional material.", encoding="utf-8")

    monkeypatch.setattr("eventweaver.narrative.pipeline.build_provider", lambda *args, **kwargs: MockProvider())

    output_dir = tmp_path / "out"
    events = generate_corpus_narrative(
        source_dir,
        output_dir=output_dir,
        provider_name="ollama",
        model="mock",
        ocr_mode="never",
        chunk_words=100,
        chunk_overlap=10,
        extraction_batch_chunks=2,
    )

    assert len(events) == 2
    assert (output_dir / "narrative_001.txt").exists()
    assert (output_dir / "narratives_manifest.csv").exists()
    assert (output_dir / "events.json").exists()
    assert (output_dir / "verification.json").exists()
    assert (output_dir / "grounding_report.csv").exists()
    assert (output_dir / "provenance.csv").exists()
    assert (output_dir / "provenance_detailed.csv").exists()
    assert (output_dir / "documents.json").exists()

    payload = json.loads((output_dir / "events.json").read_text(encoding="utf-8"))
    known_ids = {row["chunk_id"] for row in payload["evidence"]}
    assert known_ids
    assert len(payload["verification"]) == len(events)
    assert all(item["paragraph_supported"] for item in payload["verification"])
    for event in payload["events"]:
        assert event["evidence_ids"]
        assert set(event["evidence_ids"]).issubset(known_ids)

    provenance = (output_dir / "provenance.csv").read_text(encoding="utf-8-sig")
    assert "a.txt" in provenance
    assert "b.md" in provenance


def test_pipeline_can_skip_grounding_for_textual_narrative(tmp_path, monkeypatch):
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("The collection opened in 1920.", encoding="utf-8")
    monkeypatch.setattr("eventweaver.narrative.pipeline.build_provider", lambda *args, **kwargs: MockProvider())

    events = generate_corpus_narrative(
        source_dir,
        output_dir=tmp_path / "out",
        provider_name="ollama",
        model="mock",
        ocr_mode="never",
        skip_verification=True,
        events_per_narrative=1,
    )

    assert events
    output_dir = tmp_path / "out"
    with (output_dir / "narrative_001.txt").open(encoding="utf-8", newline="") as handle:
        narrative = handle.read()
    # The text stays clean for entity extraction; provenance lives in the sidecar CSV.
    assert "[Event" not in narrative and "a.txt" not in narrative
    with (output_dir / "narrative_001_events.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_files"] == "a.txt"
    assert rows[0]["evidence_ids"] == "DOC0001_C0001"
    assert narrative[int(rows[0]["char_start"]) : int(rows[0]["char_end"])] == events[0].paragraph


class ThemeProvider:
    model = "mock"

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        if prompt.startswith("Propose"):
            return GenerationResult(text="THEME|1|Theatre\nTHEME|2|Forum", runtime_seconds=0.01)
        ids = re.findall(r"^(EVT\d+):", prompt, flags=re.M)
        return GenerationResult(
            text="\n".join(f"ASSIGN|{candidate_id}|{1 + index % 2}" for index, candidate_id in enumerate(ids)),
            runtime_seconds=0.01,
        )


def test_theme_planning_mixes_sources_and_respects_limits():
    candidates = [
        EventCandidate(f"EVT{index:05d}", f"Fact {index}", [f"DOC000{index % 3 + 1}_C0001"], index)
        for index in range(1, 61)
    ]
    themes = _plan_themes(candidates, ThemeProvider(), max_themes=5, narrative_guidance="", temperature=0.0, max_new_tokens=100)

    assert [title for title, _ in themes] == ["Theatre", "Forum"]
    assert sum(len(group) for _, group in themes) == 60
    assert len({c.evidence_ids[0][:7] for c in themes[0][1]}) == 3
    assert [len(group) for group in _split_even(themes[0][1], 25)] == [2] * 5 + [1] * 20

class RepairingMockProvider:
    model = "mock"

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        evidence_ids = re.findall(r"\[EVIDENCE ([A-Z0-9_]+)", prompt)
        if "extracting factual narrative events" in prompt:
            payload = {
                "events": [
                    {
                        "summary": "The collection opened in 1920.",
                        "evidence_ids": evidence_ids[:1],
                        "date_hint": "1920",
                    }
                ]
            }
        elif "Write narrative paragraphs" in prompt:
            candidate_ids = re.findall(r"evidence=([^\]]+)", prompt)
            evidence_id = candidate_ids[0].split(",")[0].strip()
            payload = {
                "events": [
                    {
                        "paragraph": "The collection opened in 1920. It immediately became world famous.",
                        "evidence_ids": [evidence_id],
                    }
                ]
            }
        elif "strict semantic grounding verifier" in prompt:
            sentence_indexes = [int(value) for value in re.findall(r"^\[(\d+)\] ", prompt, flags=re.MULTILINE)]
            cited = evidence_ids[0]
            if "world famous" in prompt:
                payload = {
                    "sentences": [
                        {
                            "sentence_index": 1,
                            "status": "supported",
                            "supporting_evidence_ids": [cited],
                            "reason": "Opening and date are in the source.",
                        },
                        {
                            "sentence_index": 2,
                            "status": "unsupported",
                            "supporting_evidence_ids": [],
                            "reason": "Fame is not stated in the source.",
                        },
                    ]
                }
            else:
                payload = {
                    "sentences": [
                        {
                            "sentence_index": index,
                            "status": "supported",
                            "supporting_evidence_ids": [cited],
                            "reason": "The repaired sentence is supported.",
                        }
                        for index in sentence_indexes
                    ]
                }
        elif "Revise one narrative paragraph" in prompt:
            payload = {
                "paragraph": "The collection opened in 1920.",
                "evidence_ids": evidence_ids[:1],
            }
        else:
            raise AssertionError(f"Unexpected prompt: {prompt[:120]}")
        return GenerationResult(text=json.dumps(payload), runtime_seconds=0.01)


def test_pipeline_repairs_unsupported_sentence_and_reverifies(tmp_path, monkeypatch):
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("The collection opened in 1920.", encoding="utf-8")
    monkeypatch.setattr(
        "eventweaver.narrative.pipeline.build_provider",
        lambda *args, **kwargs: RepairingMockProvider(),
    )

    output_dir = tmp_path / "out"
    events = generate_corpus_narrative(
        source_dir,
        output_dir=output_dir,
        provider_name="ollama",
        model="generator",
        verifier_model="verifier",
        ocr_mode="never",
        verification_repair_attempts=1,
    )

    assert events[0].paragraph == "The collection opened in 1920."
    verification = json.loads((output_dir / "verification.json").read_text(encoding="utf-8"))
    assert verification["events"][0]["paragraph_supported"] is True
    assert verification["events"][0]["repair_attempts"] == 1


class UserThemeProvider:
    model = "mock"

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        assert not prompt.startswith("Propose"), "user themes must skip model theme planning"
        ids = re.findall(r"^(EVT\d+):", prompt, flags=re.M)
        # EVT 1-2 -> theme 1, EVT 3 -> theme 2, the rest matches no theme.
        mapping = {"EVT00001": 1, "EVT00002": 1, "EVT00003": 2}
        return GenerationResult(
            text="\n".join(f"ASSIGN|{i}|{mapping.get(i, 0)}" for i in ids), runtime_seconds=0.01
        )


def test_user_themes_are_used_and_unmatched_facts_are_left_out():
    candidates = [EventCandidate(f"EVT{i:05d}", f"Fact {i}", [f"DOC000{i}_C0001"], i) for i in range(1, 6)]
    themes = _plan_themes(
        candidates, UserThemeProvider(), max_themes=5, narrative_guidance="", temperature=0.0,
        max_new_tokens=100, user_themes=["Theatre", "Forum"],
    )
    assert [(title, [c.candidate_id for c in group]) for title, group in themes] == [
        ("Theatre", ["EVT00001", "EVT00002"]),
        ("Forum", ["EVT00003"]),
    ]


def test_themes_file_writes_one_narrative_per_theme(tmp_path, monkeypatch):
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("The theatre opened. The forum was paved.", encoding="utf-8")
    (source_dir / "themes.txt").write_text("Theatre\n\nForum\n", encoding="utf-8")

    class Provider(MockProvider):
        def generate(self, prompt, **kwargs):
            if prompt.startswith("Assign"):
                return GenerationResult(text="ASSIGN|EVT00001|1", runtime_seconds=0.01)
            return super().generate(prompt, **kwargs)

    monkeypatch.setattr("eventweaver.narrative.pipeline.build_provider", lambda *a, **k: Provider())
    generate_corpus_narrative(
        source_dir, output_dir=tmp_path / "out", provider_name="ollama", model="mock",
        ocr_mode="never", skip_verification=True,
    )
    assert (tmp_path / "out" / "narrative_001.txt").exists()
    # themes.txt is configuration, not a source document
    assert "themes.txt" not in (tmp_path / "out" / "documents.json").read_text(encoding="utf-8")


def test_too_many_user_themes_are_rejected(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    themes = tmp_path / "t.txt"
    themes.write_text("\n".join(f"T{i}" for i in range(6)), encoding="utf-8")
    try:
        generate_corpus_narrative(tmp_path, output_dir=tmp_path / "o", provider_name="ollama", model="m", themes_file=themes)
    except ValueError as exc:
        assert "at most 5" in str(exc)
    else:
        raise AssertionError("expected ValueError")
