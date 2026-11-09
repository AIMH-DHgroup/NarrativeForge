from pathlib import Path
from unittest.mock import patch

from eventweaver.reorganize import reorganize_markdown
from eventweaver.providers.base import GenerationResult
from eventweaver.providers.huggingface import HuggingFaceApiProvider, HuggingFaceLocalProvider
from eventweaver.providers.ollama import OllamaProvider


class StubProvider:
    model = "stub"

    def __init__(self, text: str) -> None:
        self.text = text
        self.temperatures: list[float] = []

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 4096):
        self.temperatures.append(temperature)
        return GenerationResult(text=self.text, runtime_seconds=0)

    def validate_temperature(self, temperature: float) -> None:
        return None


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


def test_reorganize_passes_default_and_custom_temperature(tmp_path: Path) -> None:
    source = tmp_path / "story.md"
    source.write_text("# Story\n\nFirst fact.", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{document}", encoding="utf-8")

    default_provider = StubProvider("Rewritten fact.")
    reorganize_markdown(source, prompt, default_provider)
    custom_provider = StubProvider("Rewritten fact.")
    reorganize_markdown(source, prompt, custom_provider, temperature=0.6)

    assert default_provider.temperatures == [0.1]
    assert custom_provider.temperatures == [0.6]


def test_provider_temperature_ranges() -> None:
    OllamaProvider.validate_temperature(object.__new__(OllamaProvider), 0.0)
    HuggingFaceLocalProvider.validate_temperature(object.__new__(HuggingFaceLocalProvider), 0.0)
    HuggingFaceApiProvider.validate_temperature(object.__new__(HuggingFaceApiProvider), 0.1)

    with patch.object(OllamaProvider, "validate_temperature", wraps=OllamaProvider.validate_temperature) as validate:
        try:
            validate(object.__new__(OllamaProvider), -0.1)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("Expected invalid Ollama temperature")

    try:
        HuggingFaceApiProvider.validate_temperature(object.__new__(HuggingFaceApiProvider), 0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected invalid Hugging Face API temperature")


def test_reorganize_cli_propagates_temperature(tmp_path: Path) -> None:
    from eventweaver import cli

    class Provider:
        model = "stub"

        def __init__(self) -> None:
            self.value = None

        def validate_temperature(self, value: float) -> None:
            self.value = value

    provider = Provider()
    source = tmp_path / "story.md"
    source.write_text("# Story\n\nFirst fact.", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{document}", encoding="utf-8")
    with patch.object(cli, "build_provider", return_value=provider), patch.object(
        cli, "reorganize_markdown", return_value=tmp_path / "out.md"
    ):
        assert cli.main([
            "reorganize", str(source), "--prompt-file", str(prompt), "--model", "stub",
            "--temperature", "0.6",
        ]) == 0
    assert provider.value == 0.6
