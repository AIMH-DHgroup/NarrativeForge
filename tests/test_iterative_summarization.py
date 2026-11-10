import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from eventweaver.config import GenerateConfig
from eventweaver.generation import generate_narratives, limit_narratives, publish, word_count
from eventweaver.llm import JsonLLM
from eventweaver.ollama_client import OllamaResult
from eventweaver.providers.base import GenerationResult
from eventweaver.providers.ollama import OllamaProvider
from eventweaver.store import Store


class ScriptedProvider:
    model = "test-model"

    def __init__(self, *summaries):
        self.summaries = iter(summaries)
        self.calls = []

    def generate(self, prompt, *, temperature=0.1, max_new_tokens=2048):
        self.calls.append((prompt, temperature, max_new_tokens))
        result = next(self.summaries)
        return result if isinstance(result, GenerationResult) else GenerationResult(result, 0)


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "draft.sqlite")
    yield value
    value.db.close()


def draft(store, *paragraphs, theme_id="T01"):
    for paragraph in paragraphs:
        store.db.execute("INSERT INTO paragraphs VALUES (?,?)", (theme_id, json.dumps({"text": paragraph, "chunk_ids": ["C1"]})))
    store.db.commit()


THEMES = [{"id": "T01", "title": "History"}]


@pytest.mark.parametrize("maximum", [None, 4, 10])
def test_within_limit_or_unlimited_is_completely_unchanged(store, maximum):
    draft(store, "One fact.", "Second fact.")
    original = list(store.narrative_paragraphs("T01"))
    config = GenerateConfig(max_narrative_words=maximum)
    provider = ScriptedProvider()
    limit_narratives(JsonLLM(provider, config), store, THEMES, config)
    assert provider.calls == []
    assert list(store.narrative_paragraphs("T01")) == original
    assert store.db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 0


def test_entire_narrative_including_last_paragraph_is_summarized(store):
    draft(store, "One two three.", "Four five six.")
    config = GenerateConfig(max_narrative_words=3)
    provider = ScriptedProvider("Two retained facts.")
    events, logs = [], []
    llm = JsonLLM(provider, config, logs.append, audit=events.append)
    limit_narratives(llm, store, THEMES, config)
    prompt, temperature, tokens = provider.calls[0]
    assert prompt.endswith("Narrative:\nOne two three.\n\nFour five six.\n")
    assert "no more than 3 words" in prompt
    assert temperature == 0.0 and tokens == config.max_output_tokens
    assert [row["text"] for row in store.narrative_paragraphs("T01")] == ["Two retained facts."]
    assert len(events) == llm.calls == 1
    assert any("6 words before" in line for line in logs)
    assert any("3 words after" in line for line in logs)


def test_each_attempt_uses_previous_result_not_original(store):
    original = "One two three four five six seven eight."
    summaries = ["One two three four five six.", "One two three four.", "One two three."]
    draft(store, original)
    config = GenerateConfig(max_narrative_words=3)
    provider = ScriptedProvider(*summaries)
    limit_narratives(JsonLLM(provider, config), store, THEMES, config)
    for call, previous in zip(provider.calls, [original] + summaries[:-1]):
        assert call[0].endswith(f"Narrative:\n{previous}\n")
    assert store.db.execute("SELECT count(DISTINCT level) FROM revisions").fetchone()[0] == 1
    assert list(store.narrative_paragraphs("T01"))[0]["text"] == summaries[-1]


def test_nonreducing_summary_stops_after_ten_attempts_without_word_cap(store):
    draft(store, "Six original words remain completely unchanged.")
    config = GenerateConfig(max_narrative_words=2)
    provider = ScriptedProvider(*["Still too many words."] * 10)
    with pytest.raises(ValueError, match="after 10 summarization attempts"):
        limit_narratives(JsonLLM(provider, config), store, THEMES, config)
    assert len(provider.calls) == 10
    assert store.db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 0
    assert list(store.narrative_paragraphs("T01"))[0]["text"] == "Six original words remain completely unchanged."


@pytest.mark.parametrize("bad", [GenerationResult("unfinished", 0, truncated=True),
                                 GenerationResult("", 0, error="Server unavailable"), " \n"])
def test_failed_or_incomplete_summary_never_replaces_previous_success(store, bad):
    draft(store, "Original source narrative with many words.")
    prior = {"text": "Previously saved complete summary."}
    store.db.execute("INSERT INTO revisions VALUES (?,?,?,?)", ("T01", 0, 1, json.dumps(prior)))
    store.db.commit()
    config = GenerateConfig(max_narrative_words=2)
    provider = ScriptedProvider(bad)
    with pytest.raises(RuntimeError):
        limit_narratives(JsonLLM(provider, config), store, THEMES, config)
    assert list(store.narrative_paragraphs("T01")) == [prior]
    assert store.db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 1


def test_whole_narrative_that_cannot_fit_context_is_not_split_or_truncated(store):
    draft(store, "Important fact. " * 2000)
    config = GenerateConfig(max_narrative_words=100)
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match="Whole narrative.*context budget"):
        limit_narratives(JsonLLM(provider, config), store, THEMES, config)
    assert provider.calls == []
    assert store.db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 0


def test_limits_are_applied_per_narrative_and_saved_in_existing_format(store, tmp_path):
    draft(store, "First story with several documented facts.")
    draft(store, "Second story with distinct documented facts.", theme_id="T02")
    themes = THEMES + [{"id": "T02", "title": "Second history"}]
    config = GenerateConfig(max_narrative_words=3)
    provider = ScriptedProvider("First retained facts.", "Second retained facts.")
    limit_narratives(JsonLLM(provider, config), store, themes, config)
    publish(store, themes, tmp_path)
    for theme in themes:
        output = (tmp_path / f"narrative_{theme['id']}.md").read_text()
        assert output.startswith(f"# {theme['title']}\n\n")
        assert word_count(output.split("\n", 1)[1]) <= 3


def test_plain_text_override_does_not_change_ollama_json_generation_defaults():
    provider = OllamaProvider("test", preflight=False, json_mode=True)
    with patch("eventweaver.providers.ollama.generate_ollama", return_value=OllamaResult("Two facts.", 0)) as generate:
        llm = JsonLLM(provider, GenerateConfig())
        assert llm.summarize_narrative("A longer narrative.", 2, 1) == "Two facts."
        assert generate.call_args.kwargs["json_mode"] is False
        provider.generate("Initial JSON request")
        assert generate.call_args.kwargs["json_mode"] is True
    assert provider.json_mode is True


class PipelineProvider(ScriptedProvider):
    def generate(self, prompt, *, temperature=0.1, max_new_tokens=2048):
        if prompt.startswith("PHASE: SUMMARIZE_NARRATIVE"):
            return super().generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
        if "PHASE: THEMES" in prompt:
            result = {"themes": [{"title": "History", "description": "Documented history"}]}
        elif "PHASE: ROUTE" in prompt:
            result = {"assignments": [{"chunk_id": item["id"], "theme_ids": ["T01"]} for item in data["chunks"]]}
        elif "PHASE: WRITE" in prompt:
            result = {"paragraphs": [{"text": "The church was built in 1200 and restored in 1950."}]}
        else:
            raise AssertionError("Initial generation flow changed")
        return GenerationResult(json.dumps(result), 0)


def test_generate_publishes_only_after_limit_is_met(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The church was built in 1200 and restored in 1950.")
    config = GenerateConfig(max_narrative_words=5, retries=0, cache_enabled=False)
    run = generate_narratives([source], output_dir=tmp_path / "out", provider=PipelineProvider("Built 1200; restored 1950."),
                              config=config, progress=lambda message: None)
    body = (run / "narrative_T01.md").read_text().split("\n", 1)[1]
    assert word_count(body) == 4
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["narratives"][0]["words"] == 4


def test_failed_loop_does_not_publish_overlimit_markdown(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The church was built in 1200 and restored in 1950.")
    provider = PipelineProvider(*["The church was built in 1200 and restored in 1950."] * 10)
    config = GenerateConfig(max_narrative_words=5, retries=0, cache_enabled=False)
    with pytest.raises(ValueError, match="after 10"):
        generate_narratives([source], output_dir=tmp_path / "out", provider=provider, config=config, progress=lambda message: None)
    run = next((tmp_path / "out").iterdir())
    assert not list(run.glob("narrative_*.md"))
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["draft_completed"]


def test_resume_uses_same_loop_without_touching_successful_source_run(tmp_path):
    from eventweaver.resume import resume_narratives

    source = tmp_path / "source.txt"
    source.write_text("The church was built in 1200 and restored in 1950.")
    config = GenerateConfig(retries=0, cache_enabled=False)
    original = generate_narratives([source], output_dir=tmp_path / "out", provider=PipelineProvider(),
                                   config=config, progress=lambda message: None)
    original_text = (original / "narrative_T01.md").read_bytes()
    result = resume_narratives(original, provider=ScriptedProvider("Built 1200; restored 1950."),
                               config=replace(config, max_narrative_words=5), progress=lambda message: None)
    assert word_count((result / "narrative_T01.md").read_text().split("\n", 1)[1]) == 4
    assert (original / "narrative_T01.md").read_bytes() == original_text
