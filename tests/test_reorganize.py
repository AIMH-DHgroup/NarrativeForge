from pathlib import Path

from eventweaver.reorganize import reorganize_markdown
from eventweaver.providers.base import GenerationResult


class StubProvider:
    model = "stub"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 4096):
        return GenerationResult(text=self.text, runtime_seconds=0)


def test_reorganize_preserves_title_and_word_count(tmp_path: Path) -> None:
    source = tmp_path / "story.md"
    source.write_text("# Story\n\nFirst fact.\n\nSecond fact.\n\nThird fact.", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Rewrite:\n{document}", encoding="utf-8")

    output = reorganize_markdown(
        source,
        prompt,
        StubProvider("First fact. Third fact.\n\nSecond fact."),
    )

    assert output.read_text(encoding="utf-8") == "# Story\n\nFirst fact. Third fact.\n\nSecond fact.\n"
    assert "First fact. Third fact." in output.read_text(encoding="utf-8")


def test_reorganize_accepts_changed_word_count(tmp_path: Path) -> None:
    source = tmp_path / "story.md"
    source.write_text("# Story\n\nFirst fact.", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{document}", encoding="utf-8")

    output = reorganize_markdown(source, prompt, StubProvider("Changed fact entirely."))
    assert output.exists()
