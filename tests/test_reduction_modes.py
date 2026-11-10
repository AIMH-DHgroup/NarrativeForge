import ast
import hashlib
import inspect
import json

import pytest

from eventweaver import cli, generation, prompts
from eventweaver.config import GenerateConfig
from eventweaver.llm import JsonLLM
from eventweaver.providers.base import GenerationResult
from eventweaver.resume import load_resume_config
from eventweaver.store import Store


@pytest.mark.parametrize("function,expected", [
    (generation.cap_paragraphs, "89381105c4e6f7cac3fb45458f3d3a87f89c8570661d0551418cc9a6c8353ecc"),
    (generation.editorial_items, "33b27b1cd4685b553b4f13e09c867cf2692587cb3d87e7c3ff3614676c25b484"),
    (generation.condense_narratives, "8498a5b3074ee00247a18d1e39646d6e83d3b53caf7c5cac6a3afd502e06d7c0"),
    (generation.summarize_narratives, "8c7fae445232900133c6770b5083e4cf074c4e2db3f5e10a060925b065473bc5"),
])
def test_reduction_bodies_match_historical_implementations(function, expected):
    # CONDENSE: 8133045^; SUMMARIZE: 8133045. Ignore names/comments/formatting, not logic.
    function_node = ast.parse(inspect.getsource(function)).body[0]
    body = ast.dump(ast.Module(body=function_node.body, type_ignores=[]), include_attributes=False)
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == expected


def test_both_reduction_prompts_are_unchanged():
    assert hashlib.sha256(prompts.CONDENSE.encode("utf-8")).hexdigest() == "986aa869a8d760aaf5dc06480ae99bcedd6cff366fc32b5f6244ae095b8cacf2"
    assert hashlib.sha256(prompts.SUMMARIZE_NARRATIVE.encode("utf-8")).hexdigest() == "48e248b8e4d887062ff887ab7ebfb809256193b3b9591165c959726f007f5ccf"


class Provider:
    model = "test"

    def __init__(self, malformed=False):
        self.calls = []
        self.malformed = malformed

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        if prompt.startswith("PHASE: SUMMARIZE_NARRATIVE"):
            return GenerationResult("Retained facts.", 0)
        data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
        if "PHASE: CONDENSE" in prompt:
            result = {"unexpected": "invalid response"} if self.malformed else {"paragraphs": [{"text": "Retained facts."}]}
        elif "PHASE: THEMES" in prompt:
            result = {"themes": [{"title": "History", "description": "Detailed theme information. " * 50}]}
        elif "PHASE: ROUTE" in prompt:
            result = {"assignments": [{"chunk_id": row["id"], "theme_ids": ["T01"]} for row in data["chunks"]]}
        elif "PHASE: WRITE" in prompt:
            result = {"paragraphs": [{"text": "A complete narrative with several important documented facts."}]}
        else:
            raise AssertionError("Generation pipeline unexpectedly changed")
        return GenerationResult(json.dumps(result), 0)


@pytest.mark.parametrize("arguments,mode", [([], "condense"), (["--reduction-mode", "condense"], "condense"),
                                           (["--reduction-mode", "summarize"], "summarize")])
def test_cli_selects_and_persists_reduction_mode(tmp_path, monkeypatch, capsys, arguments, mode):
    source = tmp_path / "source.txt"
    source.write_text("Source information.")
    provider = Provider()
    monkeypatch.setattr(cli, "build_provider", lambda *args, **kwargs: provider)
    assert cli.main(["generate", str(source), "--model", "test", "--max-words", "3", "--json-retries", "0", "--no-cache",
                     "--output-dir", str(tmp_path / "out"), *arguments]) == 0
    assert f"Reduction mode: {mode.upper()}" in capsys.readouterr().err
    run = next((tmp_path / "out").iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["config"]["reduction_mode"] == mode
    assert manifest["narratives"][0]["words"] <= 3
    assert any("PHASE: CONDENSE" in prompt for prompt in provider.calls) == (mode == "condense")
    assert any(prompt.startswith("PHASE: SUMMARIZE_NARRATIVE") for prompt in provider.calls) == (mode == "summarize")
    assert len(json.loads((run / "themes.json").read_text())[0]["description"]) > 300


@pytest.mark.parametrize("mode", ["condense", "summarize"])
def test_no_max_words_skips_both_algorithms(tmp_path, monkeypatch, mode):
    config = GenerateConfig(reduction_mode=mode)
    monkeypatch.setattr(generation, "condense_narratives", lambda *a: pytest.fail("CONDENSE should not execute"))
    monkeypatch.setattr(generation, "summarize_narratives", lambda *a: pytest.fail("SUMMARIZE should not execute"))
    generation.limit_narratives(None, None, [], config)


def test_invalid_mode_is_rejected_by_cli_and_configuration(capsys):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["generate", "docs", "--model", "test", "--reduction-mode", "unknown"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    with pytest.raises(ValueError, match="reduction_mode"):
        GenerateConfig(reduction_mode="unknown")


def test_condense_handles_large_drafts_in_context_compatible_batches(tmp_path):
    config = GenerateConfig(max_narrative_words=50, retries=0, chunk_chars=1000, overlap_chars=0)
    original = "Important source information and context. " * 600
    store = Store(tmp_path / "analysis.sqlite")
    try:
        store.db.execute("INSERT INTO paragraphs VALUES (?,?)", ("T01", json.dumps({"text": original})))
        store.db.commit()
        provider = Provider()
        llm = JsonLLM(provider, config)
        generation.limit_narratives(llm, store, [{"id": "T01", "title": "History"}], config)
        assert len(provider.calls) > 1
        observed_words = 0
        for prompt in provider.calls:
            assert "PHASE: CONDENSE" in prompt
            data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
            observed_words += sum(generation.word_count(row["text"]) for row in data["paragraphs"])
            assert len(prompt.encode("utf-8")) + config.max_output_tokens + config.context_safety_tokens <= config.context_tokens
        assert observed_words == generation.word_count(original)
        assert sum(generation.word_count(row["text"]) for row in store.narrative_paragraphs("T01")) <= 50
    finally:
        store.db.close()


def test_condense_restores_draft_fallback_and_sentence_aware_cap(tmp_path):
    assert generation.cap_paragraphs([{"text": "First complete sentence. A substantially longer following sentence."}], 6) == [{"text": "First complete sentence."}]
    assert generation.cap_paragraphs([{"text": "First complete sentence."}], 1) == [{"text": "First."}]
    config = GenerateConfig(max_narrative_words=3, retries=0)
    store = Store(tmp_path / "analysis.sqlite")
    try:
        store.db.execute("INSERT INTO paragraphs VALUES (?,?)", ("T01", json.dumps({"text": "First complete sentence. A substantially longer following sentence."})))
        store.db.commit()
        generation.limit_narratives(JsonLLM(Provider(malformed=True), config), store, [{"id": "T01", "title": "History"}], config)
        assert list(store.narrative_paragraphs("T01")) == [{"text": "First complete sentence.", "editorial_method": "draft_word_cap"}]
    finally:
        store.db.close()


def test_old_manifest_defaults_to_condense_and_resume_accepts_override(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("Source information.")
    original = generation.generate_narratives([source], provider=Provider(), output_dir=tmp_path / "out",
                                              config=GenerateConfig(cache_enabled=False), progress=lambda message: None)
    path = original / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"].pop("reduction_mode")
    path.write_text(json.dumps(manifest))
    assert load_resume_config(original).reduction_mode == "condense"
    before = (original / "narrative_T01.md").read_bytes()
    provider = Provider()
    monkeypatch.setattr(cli, "build_provider", lambda *args, **kwargs: provider)
    assert cli.main(["generate", "--resume-run", str(original), "--model", "test", "--max-words", "3",
                     "--reduction-mode", "summarize", "--output-dir", str(tmp_path / "resumed")]) == 0
    resumed = next((tmp_path / "resumed").iterdir())
    assert json.loads((resumed / "manifest.json").read_text())["config"]["reduction_mode"] == "summarize"
    assert (original / "narrative_T01.md").read_bytes() == before
    assert any(prompt.startswith("PHASE: SUMMARIZE_NARRATIVE") for prompt in provider.calls)


@pytest.mark.parametrize("saved_mode", ["condense", "summarize"])
def test_resume_preserves_saved_mode_when_cli_option_is_absent(tmp_path, monkeypatch, saved_mode):
    source = tmp_path / "source.txt"
    source.write_text("Source information.")
    original = generation.generate_narratives([source], provider=Provider(), output_dir=tmp_path / "out",
                                              config=GenerateConfig(cache_enabled=False, reduction_mode=saved_mode),
                                              progress=lambda message: None)
    provider = Provider()
    monkeypatch.setattr(cli, "build_provider", lambda *args, **kwargs: provider)
    assert cli.main(["generate", "--resume-run", str(original), "--model", "test", "--max-words", "3",
                     "--output-dir", str(tmp_path / "resumed")]) == 0
    resumed = next((tmp_path / "resumed").iterdir())
    assert json.loads((resumed / "manifest.json").read_text())["config"]["reduction_mode"] == saved_mode
    assert any("PHASE: CONDENSE" in prompt for prompt in provider.calls) == (saved_mode == "condense")
    assert any(prompt.startswith("PHASE: SUMMARIZE_NARRATIVE") for prompt in provider.calls) == (saved_mode == "summarize")
