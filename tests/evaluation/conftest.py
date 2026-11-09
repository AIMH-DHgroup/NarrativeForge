import json

import pytest

from eventweaver.evaluation.sources import load_inputs


@pytest.fixture
def evaluation_case(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The church was built in 1200.", encoding="utf-8")
    narrative = tmp_path / "story.md"
    narrative.write_text("# History\n\nThe church was built in 1200.", encoding="utf-8")
    inputs = load_inputs(narrative, [source], original_prompt="Describe the church")
    doc = inputs.corpus["documents"][0]
    quote = doc["pages"][0]["text"]
    evidence = {"document_id": doc["id"], "page_index": 0, "start": 0,
                "end": len(quote), "quote": quote}
    offset = inputs.narrative["text"].index("The church")
    span = {"start": offset, "end": offset + len(quote), "quote": quote}

    def save(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    annotations = {"narrative_sha256": inputs.narrative["sha256"],
                   "corpus_sha256": inputs.corpus["sha256"], "annotator": "reviewer",
                   "annotated_at": "2026-11-09T10:00:00Z"}
    return inputs, source, narrative, evidence, span, annotations, save
