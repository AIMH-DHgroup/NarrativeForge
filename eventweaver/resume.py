"""Finish an existing draft in a new run without repeating OCR or narrative generation."""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from .config import GenerateConfig
from .generation import limit_narratives, publish, save_json, word_count
from .llm import JsonLLM, text
from .store import Store


def load_resume_config(run_dir: Path) -> GenerateConfig:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    saved = manifest.get("config")
    if not isinstance(saved, dict):
        raise ValueError("The run manifest does not contain a valid configuration")
    names = {field.name for field in fields(GenerateConfig)}
    return GenerateConfig(**{key: value for key, value in saved.items() if key in names})


def resume_narratives(run_dir: Path, *, provider, output_dir: Path | None = None,
                      config: GenerateConfig | None = None, progress=print) -> Path:
    source = Path(run_dir).resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    saved = load_resume_config(source)
    config = config or saved
    for field in ("language", "narrative_titles", "max_themes"):
        if getattr(config, field) != getattr(saved, field):
            raise ValueError("Resume preserves the draft language and titles; start a new generate to change them")
    error = str(manifest.get("error", ""))
    draft_ready = (manifest.get("draft_completed") is True or manifest.get("status") == "completed"
                   or error.startswith("FASE: CONDENSE:")
                   or error.startswith("Riduzione della narrazione non convergente:"))
    if not manifest.get("ingestion_completed") or not draft_ready:
        raise ValueError("Resume unavailable: the run must have completed the draft and reached final reduction")
    selected = json.loads((source / "themes.json").read_text(encoding="utf-8"))
    if not isinstance(selected, list) or not 1 <= len(selected) <= 5:
        raise ValueError("The run contains no valid themes for resume")
    for index, theme in enumerate(selected, 1):
        if not isinstance(theme, dict) or theme.get("id") != f"T{index:02d}":
            raise ValueError("The run's theme IDs are invalid")
        text(theme.get("title"), field="theme.title")
    database = source / "analysis.sqlite"
    if not database.is_file():
        raise FileNotFoundError(f"Draft database not found: {database}")
    # The source is opened read-only and remains untouched, including old partial revisions.
    previous = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        for theme in selected:
            count = previous.execute("SELECT count(*) FROM paragraphs WHERE theme_id=?", (theme["id"],)).fetchone()[0]
            if not count:
                raise ValueError(f"Draft missing for theme {theme['id']}")
        parent = Path(output_dir).resolve() if output_dir is not None else source.parent
        parent.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
        copy = sqlite3.connect(destination / "analysis.sqlite")
        try:
            previous.backup(copy)
        finally:
            copy.close()
    finally:
        previous.close()
    for filename in ("themes.json", "aggregate_context.json"):
        if (source / filename).is_file():
            shutil.copy2(source / filename, destination / filename)
    old_calls = manifest.get("llm_calls", 0)
    draft_model = manifest.get("draft_model", manifest.get("model"))
    manifest.update(status="running", resumed_from=str(source), draft_completed=True,
                    draft_model=draft_model, model=getattr(provider, "model", "unknown"),
                    config=asdict(config), previous_llm_calls=old_calls, outputs=[],
                    started_at=datetime.now(timezone.utc).isoformat())
    for key in ("error", "finished_at", "narratives"):
        manifest.pop(key, None)
    save_json(destination / "manifest.json", manifest)
    store = Store(destination / "analysis.sqlite")
    llm = JsonLLM(provider, config, progress, audit=store.record_response)
    try:
        # Restart editing from the complete draft, never from an incomplete old reduction.
        store.db.execute("DELETE FROM revisions")
        store.db.commit()
        progress(f"Resuming from {source}: reusing the complete draft without repeating OCR or generation")
        limit_narratives(llm, store, selected, config)
        manifest["narratives"] = [{"id": theme["id"], "title": theme["title"],
                                   "words": sum(word_count(row["text"]) for row in store.narrative_paragraphs(theme["id"])),
                                   "max_words": config.max_narrative_words} for theme in selected]
        store.export(destination)
        manifest["outputs"] = publish(store, selected, destination)
        manifest["status"] = "completed"
        return destination
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        store.db.commit()
        store.export(destination)
        progress(f"Resume failed; details in {destination}")
        raise
    finally:
        manifest["llm_calls"] = llm.calls
        manifest["counts"] = {name: store.db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                              for name in ("chunks", "unassigned", "paragraphs")}
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(destination / "manifest.json", manifest)
        store.db.close()
