from eventweaver.providers.base import GenerationResult
from eventweaver import simple_kb


class FakeProvider:
    model = "fake"

    def generate(self, prompt, **kwargs):
        if "KEYWORDS:" in prompt:
            return GenerationResult("KEYWORDS: museum, restoration", 0)
        return GenerationResult("The museum's restoration shaped the collection.", 0)


def test_simple_kb_add_search_story_and_incremental(tmp_path, monkeypatch):
    source = tmp_path / "one.txt"
    source.write_text("Restoration of the museum collection began in 1920.\n\nThe museum reopened.", encoding="utf-8")
    db = tmp_path / "simple.sqlite"
    monkeypatch.setattr(simple_kb, "build_provider", lambda *args, **kwargs: FakeProvider())

    assert len(simple_kb.add_documents(db, [source], provider_name="ollama", model="fake")) == 1
    assert simple_kb.add_documents(db, [source], provider_name="ollama", model="fake") == []
    assert len(simple_kb.search(db, "restoration")) == 1

    output = tmp_path / "story.txt"
    simple_kb.story(db, "restoration", output, provider_name="ollama", model="fake")
    text = output.read_text(encoding="utf-8")
    assert "The museum's restoration" in text
    assert text.endswith("SOURCES\n1. one.txt, pages unknown, source events " + simple_kb.status(db)[0]["document_id"] + "E0001V1\n")
    assert "[END OF NARRATIVE]" in text


def test_simple_kb_multiple_documents(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("A Roman inscription records a dedication.", encoding="utf-8")
    second.write_text("A medieval restoration preserved the wall.", encoding="utf-8")
    db = tmp_path / "simple.sqlite"

    assert len(simple_kb.add_documents(db, [first, second])) == 2
    assert len(simple_kb.search(db, "restoration")) == 1
    assert len(simple_kb.status(db)) == 2
