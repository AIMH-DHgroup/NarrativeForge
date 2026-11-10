import json

import pytest

from eventweaver import prompts
from eventweaver.config import GenerateConfig
from eventweaver.generation import generate_narratives, parse_themes, themes
from eventweaver.llm import JsonLLM
from eventweaver.providers.base import GenerationResult


class ThemeProvider:
    model = "fixture"

    def __init__(self, description):
        self.description = description
        self.requests = []

    def generate(self, prompt, **kwargs):
        data = json.loads(prompt.split("DATA_JSON:\n", 1)[1])
        self.requests.append((prompt, data))
        if "PHASE: THEMES" in prompt:
            titles = data.get("requested_titles") or ["Church history"]
            result = {"themes": [{"title": title, "description": self.description} for title in titles]}
        elif "PHASE: ROUTE" in prompt:
            result = {"assignments": [{"chunk_id": row["id"], "theme_ids": ["T01"]} for row in data["chunks"]]}
        elif "PHASE: WRITE" in prompt:
            result = {"paragraphs": [{"text": "The church was restored in 1950."}]}
        else:
            raise AssertionError("Unexpected pipeline change")
        return GenerationResult(json.dumps(result), 0)


@pytest.mark.parametrize("length", [301, 544, 1000, 2000])
def test_descriptions_are_preserved_verbatim_without_length_logs(length):
    description = ("  Detailed theme\nwith multiple  spaces. " * 100)[:length]
    assert len(description) == length
    messages = []
    result = parse_themes({"themes": [{"title": "History", "description": description}]}, 5, report=messages.append)
    assert result[0]["description"] == description
    assert messages == [f"themes[0].description: Theme description preserved: {length} characters"]


@pytest.mark.parametrize("manual_title", [False, True])
def test_full_description_reaches_route_write_and_themes_json(tmp_path, manual_title):
    source = tmp_path / "church.txt"
    source.write_text("The church was restored in 1950.", encoding="utf-8")
    description = ("Detailed church history.\n" * 100)[:1000]
    provider = ThemeProvider(description)
    config = GenerateConfig(narrative_titles=("Requested history",) if manual_title else (),
                            retries=0, cache_enabled=False)
    run = generate_narratives([source], provider=provider, output_dir=tmp_path / "out",
                              config=config, progress=lambda message: None)
    stored = json.loads((run / "themes.json").read_text())
    assert stored[0]["description"] == description
    assert stored[0]["id"] == "T01"
    routed = [data for prompt, data in provider.requests if "PHASE: ROUTE" in prompt]
    written = [data for prompt, data in provider.requests if "PHASE: WRITE" in prompt]
    assert routed and written
    assert all(data["themes"][0]["description"] == description for data in routed)
    assert all(data["theme"]["description"] == description for data in written)


@pytest.mark.parametrize("description", [None, "", "  \n\t"])
def test_blank_descriptions_keep_title_fallback(description):
    result = parse_themes({"themes": [{"title": "History", "description": description}]}, 5)
    assert result[0]["description"] == "History"


def test_missing_description_uses_title():
    assert parse_themes({"themes": [{"title": "History"}]}, 5)[0]["description"] == "History"


@pytest.mark.parametrize("description", [[], {}, 0, False])
def test_wrong_types_still_raise_a_clear_validation_error(description):
    with pytest.raises(ValueError, match=r"themes\[0\].description: expected string"):
        parse_themes({"themes": [{"title": "History", "description": description}]}, 5)


def test_title_constraints_are_unchanged():
    row = parse_themes({"themes": [{"title": "Long theme title " * 30, "description": "Full description."}]}, 5)[0]
    assert len(row["title"]) <= 100
    assert row["title"].endswith("…")
    with pytest.raises(ValueError, match="empty text"):
        parse_themes({"themes": [{"title": " \n", "description": "Description"}]}, 5)
    with pytest.raises(ValueError, match="Duplicate themes"):
        parse_themes({"themes": [{"title": "History", "description": "One"}, {"title": "history", "description": "Two"}]}, 5)


def test_manual_title_fallback_is_not_truncated_and_ids_stay_sequential():
    title = "A complete requested narrative title " * 12
    config = GenerateConfig(narrative_titles=(title.strip(), "Second requested narrative"), retries=0)
    provider = ThemeProvider(None)
    result = themes(JsonLLM(provider, config), {"items": []}, config)
    assert result[0]["description"] == title.strip()
    assert len(result[0]["description"]) > 300
    assert [item["id"] for item in result] == ["T01", "T02"]


def test_theme_prompt_no_longer_requests_short_descriptions():
    assert "maximum 300 characters" not in prompts.THEMES
    assert "Do not apply an arbitrary character limit" in prompts.THEMES
    assert '"themes"' in prompts.THEMES and '"description"' in prompts.THEMES
