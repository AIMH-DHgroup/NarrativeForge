from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import GenerateConfig
from .ingestion.chunking import iter_chunks
from .ingestion.loader import discover_documents, iter_documents
from .llm import JsonLLM, InvalidLLMResponse, LLMOutputTruncated, RETRY_FEEDBACK_BYTES, array, text, estimate_tokens
from . import prompts
from .store import Store


def save_json(path, value):
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.replace(path)


def theme_metadata(value, *, field, report, limit=None, fallback=None):
    # Only titles have a length limit; descriptions are preserved verbatim.
    if value is None and fallback is not None:
        report(f"{field}: description missing, using the theme title")
        value = fallback
    if not isinstance(value, str):
            raise ValueError(f"{field}: expected string, received {type(value).__name__}")
    if limit is not None:
        value = re.sub(r"\s+", " ", value).strip()
    if not value.strip():
        if fallback is None:
            raise ValueError(f"{field}: empty text")
        report(f"{field}: description missing, using the theme title")
        return fallback
    if limit is None:
        report(f"{field}: Theme description preserved: {len(value)} characters")
    if limit is not None and len(value) > limit:
        report(f"{field}: metadata shortened from {len(value)} to at most {limit} characters")
        prefix = value[:limit - 1]
        boundary = prefix.rfind(" ")
        if boundary > 0:
            prefix = prefix[:boundary]
        value = prefix.rstrip() + "…"
    return value


def parse_themes(obj, maximum, *, allow_empty=False, report=lambda message: None):
    result = []
    for index, row in enumerate(array(obj.get("themes"), maximum=maximum)):
        if not isinstance(row, dict):
            raise ValueError(f"themes[{index}]: expected an object with title and description")
        title = theme_metadata(row.get("title"), field=f"themes[{index}].title",
                               limit=100, report=report)
        description = theme_metadata(row.get("description"), field=f"themes[{index}].description",
                                     fallback=title, report=report)
        result.append({"title": title, "description": description})
    if not result and not allow_empty:
        raise ValueError("No themes returned")
    if len({row["title"].casefold() for row in result}) != len(result):
        raise ValueError("Duplicate themes: generate distinct titles within 100 characters")
    return result


def compact_summary(value, byte_budget):
    """Bound intermediate context only; prefer sentences distributed across the summary."""
    size = lambda content: estimate_tokens(json.dumps(content, ensure_ascii=False))
    if size(value) <= byte_budget:
        return value
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", value) if part.strip()]

    def select(units):
        order = []
        queue = [(0, len(units) - 1)]
        while queue:
            left, right = queue.pop(0)
            if left > right:
                continue
            if not order:
                order.extend([left] if left == right else [left, right])
                queue.append((left + 1, right - 1))
            else:
                middle = (left + right) // 2
                order.append(middle)
                queue.extend([(left, middle - 1), (middle + 1, right)])
        chosen = []
        result = ""
        for index in order:
            candidate_indices = sorted(chosen + [index])
            candidate = ""
            previous = None
            for selected in candidate_indices:
                if previous is not None:
                    candidate += " " if selected == previous + 1 else " … "
                candidate += units[selected]
                previous = selected
            if size(candidate) <= byte_budget:
                chosen.append(index)
                result = candidate
        return result

    result = select(parts)
    if result:
        return result
    # A long sentence is split into word groups, retaining a spread across the text.
    words = value.split()
    result = select([" ".join(words[i:i + 8]) for i in range(0, len(words), 8)])
    if result:
        return result
    # Extremely long unbroken tokens: honor the byte budget without invalid UTF-8.
    lower, upper = 0, len(value)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if size(value[:middle] + "…") <= byte_budget:
            lower = middle
        else:
            upper = middle - 1
    return value[:lower].rstrip() + "…"


def aggregate_corpus(llm, store, config, document_count):
    """Read every source into a bounded, global context without local theme selection."""
    task = prompts.THEMES.replace("MAX_THEMES", str(config.max_themes)).replace("OUTPUT_LANGUAGE", config.language)
    summary_task = prompts.SUMMARIZE.replace("OUTPUT_LANGUAGE", config.language)
    empty = {"document_count": document_count, "representation": "aggregate_summaries", "items": [],
             "requested_titles": list(config.narrative_titles)}
    available = (config.context_tokens - config.max_output_tokens - config.context_safety_tokens
                 - (RETRY_FEEDBACK_BYTES if config.retries else 0) - estimate_tokens(llm.prompt(task, empty)))
    if available < 128:
        raise ValueError("Insufficient context for theme metadata: increase --num-ctx or reduce --max-new-tokens")
    byte_budget = max(32, min(1536, available // 4))
    level = -1
    while True:
        def items():
            if level == -1:
                yield from store.chunks()
            else:
                for row in store.rows("SELECT body FROM summaries WHERE level=? ORDER BY id", (level,)):
                    yield json.loads(row["body"])

        representation = "original_chunks" if level == -1 else "aggregate_summaries"
        wrap = lambda rows: {"document_count": document_count, "representation": representation, "items": rows}
        context = []
        all_fit = True
        for item in items():
            if not llm.fits(task, dict(wrap(context + [item]), requested_titles=list(config.narrative_titles))):
                all_fit = False
                break
            context.append(item)
        if all_fit:
            if not context:
                raise ValueError("No content available in the aggregated corpus")
            return wrap(context)

        count_before = (store.db.execute("SELECT count(*) FROM chunks").fetchone()[0] if level == -1 else
                        store.db.execute("SELECT count(*) FROM summaries WHERE level=?", (level,)).fetchone()[0])
        llm.progress(f"Corpus aggregation: level {level + 1}, {count_before} items, "
                     f"budget {byte_budget} bytes per summary, without theme selection")
        summary_wrap = lambda rows: dict(wrap(rows), max_summary_bytes=byte_budget,
                                        max_summary_words=max(8, byte_budget // 10))

        def summarize(batch):
            try:
                value = llm.call(summary_task, summary_wrap(batch),
                                 lambda obj: text(obj.get("summary"), field="summary"),
                                 output_tokens=min(config.max_output_tokens, max(256, byte_budget // 2 + 128)))
                yield value, batch, "model"
            except LLMOutputTruncated:
                if level >= 0:
                    llm.progress("Output still truncated: compact merge of already available summaries")
                    yield " ".join(item["text"] for item in batch), batch, "bounded_merge"
                elif len(batch) > 1:
                    llm.progress(f"Output still truncated: splitting the batch of {len(batch)} chunks")
                    middle = len(batch) // 2
                    yield from summarize(batch[:middle])
                    yield from summarize(batch[middle:])
                else:
                    raise

        index = 0
        for batch in llm.batches(summary_task, items(), summary_wrap):
            for raw_value, inputs, method in summarize(batch):
                value = compact_summary(raw_value, byte_budget)
                if value != raw_value:
                    llm.progress(f"Intermediate summary automatically reduced to {byte_budget} bytes")
                index += 1
                summary = {"id": f"S{level + 1:03d}_{index:08d}", "text": value}
                store.db.execute("INSERT INTO summaries VALUES (?,?,?,?,?,?)",
                                 (summary["id"], level + 1, json.dumps(summary, ensure_ascii=False),
                                  raw_value, byte_budget, method))
                store.db.executemany("INSERT INTO summary_inputs VALUES (?,?)",
                                     [(summary["id"], item["id"]) for item in inputs])
                store.db.commit()
        llm.progress(f"Aggregation level {level + 1}: {count_before} → {index} summaries")
        # Tighten intermediate context if unusually small windows prevent grouping.
        if level >= 0 and index >= count_before:
            byte_budget = max(16, byte_budget // 2)
            if byte_budget == 16 and index >= count_before:
                raise ValueError("Context too small even for minimum summaries: increase --num-ctx")
        level += 1


def themes(llm, aggregate, config):
    task = prompts.THEMES.replace("MAX_THEMES", str(config.max_themes)).replace("OUTPUT_LANGUAGE", config.language)
    def validate(obj):
        if not config.narrative_titles:
            return parse_themes(obj, config.max_themes, report=llm.progress)
        rows = array(obj.get("themes"), maximum=len(config.narrative_titles))
        if len(rows) != len(config.narrative_titles):
            raise ValueError("Return one entry for each requested title")
        result = []
        for index, (row, title) in enumerate(zip(rows, config.narrative_titles)):
            if not isinstance(row, dict) or row.get("title") != title:
                raise ValueError(f"themes[{index}].title must be identical to the requested title: {title}")
            description = theme_metadata(row.get("description"), field=f"themes[{index}].description",
                                         fallback=title, report=llm.progress)
            result.append({"title": title, "description": description})
        return result
    result = llm.call(task, dict(aggregate, requested_titles=list(config.narrative_titles)), validate)
    return [dict(theme, id=f"T{i:02d}") for i, theme in enumerate(result, 1)]


def route(llm, store, selected):
    valid_themes = {row["id"] for row in selected}
    wrap = lambda rows: {"themes": selected, "chunks": rows}
    for batch in llm.batches(prompts.ROUTE, store.chunks(), wrap):
        ids = {row["id"] for row in batch}

        def validate(obj):
            rows = array(obj["assignments"])
            returned = [row["chunk_id"] for row in rows]
            if len(returned) != len(ids) or set(returned) != ids:
                raise ValueError("Routing must include every chunk exactly once")
            for row in rows:
                values = array(row["theme_ids"], maximum=len(selected))
                if any(not isinstance(value, str) or value not in valid_themes for value in values):
                    raise ValueError("Unknown theme ID")
                if len(set(values)) != len(values):
                    raise ValueError("Duplicate theme ID")
                if not values:
                    text(row.get("reason"), limit=500)
            return rows

        rows = llm.call(prompts.ROUTE, wrap(batch), validate)
        for row in rows:
            if row["theme_ids"]:
                store.db.executemany("INSERT INTO assignments VALUES (?,?)",
                                     [(row["chunk_id"], value) for value in row["theme_ids"]])
            else:
                store.db.execute("INSERT INTO unassigned VALUES (?,?)", (row["chunk_id"], row["reason"]))
        store.db.commit()


def parse_paragraphs(obj):
    """Normalize common narrative response shapes without imposing length limits."""
    def normalize(value, field="response", depth=0):
        if depth > 6:
            raise ValueError(f"{field}: narrative response nested too deeply")
        if isinstance(value, str):
            content = text(value, field=field)
            return [{"text": part.strip()} for part in re.split(r"\n\s*\n", content) if part.strip()]
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{field}: no paragraphs generated")
            result = []
            for index, row in enumerate(value):
                result.extend(normalize(row, f"{field}[{index}]", depth + 1))
            return result
        if isinstance(value, dict):
            for key in ("paragraphs", "text", "narrative", "summary", "content", "story"):
                if key in value and value[key] is not None:
                    return normalize(value[key], f"{field}.{key}", depth + 1)
            keys = ", ".join(str(key) for key in list(value)[:10]) or "none"
            raise ValueError(f"{field}: narrative text missing; expected paragraphs or text, received keys: {keys}")
        raise ValueError(f"{field}: expected text, paragraph list or narrative object; received {type(value).__name__}")
    return normalize(obj)


def word_count(value):
    return len(value.split())


def cap_paragraphs(paragraphs, maximum):
    """Last-resort word cap after editorial rewriting; prefer complete sentences."""
    result = []
    remaining = maximum
    for paragraph in paragraphs:
        value = paragraph["text"]
        words = list(re.finditer(r"\S+", value))
        if len(words) <= remaining:
            result.append({"text": value})
            remaining -= len(words)
        elif remaining:
            prefix = value[:words[remaining - 1].end()]
            if not re.search(r"[.!?][\"'’”)]*$", prefix):
                endings = list(re.finditer(r"[.!?](?:[\"'’”)]+)?(?=\s|$)", prefix))
                if endings:
                    prefix = prefix[:endings[-1].end()]
                else:
                    prefix = prefix.rstrip(",;:-") + "."
            result.append({"text": prefix})
            break
        else:
            break
    return result


def narrative_history(value, byte_budget=1600):
    """Keep a bounded tail for continuity; never shorten the saved narrative itself."""
    value = re.sub(r"\s+", " ", value).strip()
    if estimate_tokens(json.dumps(value, ensure_ascii=False)) <= byte_budget:
        return value
    lower, upper = 0, min(len(value), byte_budget)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if estimate_tokens(json.dumps(value[-middle:], ensure_ascii=False)) <= byte_budget:
            lower = middle
        else:
            upper = middle - 1
    tail = value[-lower:] if lower else ""
    if " " in tail:
        tail = tail.split(" ", 1)[1]
    return tail


def editorial_items(store, theme_id, level, chunk_chars):
    """Split long draft paragraphs at word boundaries, without omitting any words."""
    for paragraph in store.narrative_paragraphs(theme_id, level):
        value = paragraph["text"]
        start = None
        end = 0
        for match in re.finditer(r"\S+", value):
            if start is not None and match.end() - start > chunk_chars:
                yield {"text": value[start:end]}
                start = None
            if start is None:
                start = match.start()
            end = match.end()
        if start is not None:
            yield {"text": value[start:end]}


def condense_narratives(llm, store, selected, config):
    maximum = config.max_narrative_words
    if maximum is None:
        return
    for theme in selected:
        level = -1
        while True:
            total = sum(word_count(row["text"]) for row in store.narrative_paragraphs(theme["id"], level))
            if total <= maximum:
                break
            llm.progress(f"Reduction {theme['id']}: {total} words, maximum {maximum}")
            # Every draft section enters editorial processing before the global cap is met.
            history = ""
            wrap = lambda rows: {"theme": theme, "language": config.language, "previous_text": history,
                                 "max_words": maximum, "paragraphs": [{"text": row["text"]} for row in rows]}
            position = 0
            next_total = 0
            for batch in llm.batches(prompts.CONDENSE, editorial_items(store, theme["id"], level, config.chunk_chars), wrap):
                budget = max(1, maximum * sum(word_count(row["text"]) for row in batch) // total)
                data = dict(wrap(batch), max_words=budget)
                method = "model"
                try:
                    rewritten = llm.call(prompts.CONDENSE, data, parse_paragraphs, allow_narrative=True)
                except InvalidLLMResponse as exc:
                    llm.progress(f"{theme['id']}: unusable editorial reduction; applying the limit to the available draft ({exc})")
                    rewritten = [{"text": row["text"]} for row in batch]
                    method = "draft_word_cap"
                if sum(word_count(row["text"]) for row in rewritten) > budget:
                    llm.progress(f"{theme['id']}: applying the {budget}-word limit to the rewritten section")
                for paragraph in cap_paragraphs(rewritten, budget):
                    paragraph["editorial_method"] = method
                    history = narrative_history(history + " " + paragraph["text"])
                    position += 1
                    next_total += word_count(paragraph["text"])
                    store.db.execute("INSERT INTO revisions VALUES (?,?,?,?)",
                                     (theme["id"], level + 1, position, json.dumps(paragraph, ensure_ascii=False)))
                store.db.commit()
            if next_total >= total:
                raise ValueError("Narrative reduction did not converge: increase --num-ctx")
            level += 1


def limit_narratives(llm, store, selected, config):
    """Select only the post-WRITE reduction step; generation is shared."""
    if config.max_narrative_words is None:
        return
    llm.progress(f"Reduction mode: {config.reduction_mode.upper()}")
    if config.reduction_mode == "condense":
        llm.progress("Starting original condensation procedure...")
        condense_narratives(llm, store, selected, config)
    else:
        summarize_narratives(llm, store, selected, config)


def summarize_narratives(llm, store, selected, config):
    """Summarize each complete narrative until it fits, at most ten times."""
    maximum = config.max_narrative_words
    if maximum is None:
        return
    for theme in selected:
        narrative = "\n\n".join(row["text"] for row in store.narrative_paragraphs(theme["id"]))
        total = word_count(narrative)
        llm.progress(f"Narrative generated {theme['id']}: {total} words; maximum allowed: {maximum}")
        if total <= maximum:
            continue
        for attempt in range(1, 11):
            llm.progress(f"SUMMARIZE {theme['id']} attempt {attempt}: {total} words before summarization")
            narrative = llm.summarize_narrative(narrative, maximum, attempt)
            total = word_count(narrative)
            llm.progress(f"SUMMARIZE {theme['id']} attempt {attempt}: {total} words after summarization")
            if total <= maximum:
                break
        else:
            raise ValueError(
                f"Narrative {theme['id']} still has {total} words after 10 summarization attempts "
                f"(maximum {maximum}). No over-limit narrative was published."
            )
        level = store.db.execute("SELECT max(level) FROM revisions WHERE theme_id=?", (theme["id"],)).fetchone()[0]
        level = 0 if level is None else level + 1
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", narrative) if part.strip()]
        with store.db:
            for position, paragraph in enumerate(paragraphs, 1):
                body = {"text": paragraph, "editorial_method": "iterative_summarization"}
                store.db.execute("INSERT INTO revisions VALUES (?,?,?,?)",
                                 (theme["id"], level, position, json.dumps(body, ensure_ascii=False)))
        llm.progress(f"Saving final narrative {theme['id']}: {total} words")


def split_writing_batch(batch):
    """Reduce a writing request without dropping source characters or changing source IDs."""
    if len(batch) > 1:
        middle = len(batch) // 2
        return [batch[:middle], batch[middle:]]
    chunk = batch[0]
    value = chunk["text"]
    if len(value) < 256:
        return None
    middle = len(value) // 2
    boundaries = [match.end() for match in re.finditer(r"\s+", value)
                  if len(value) // 4 <= match.end() <= 3 * len(value) // 4]
    cut = min(boundaries, key=lambda offset: abs(offset - middle)) if boundaries else middle
    start = chunk.get("start", 0)
    return [[dict(chunk, text=value[:cut], end=start + cut)],
            [dict(chunk, text=value[cut:], start=start + cut)]]


def writing_sections(llm, batch, wrap, write_prompt, *, depth=0):
    """Retry truncated sections with smaller source requests; never publish incomplete JSON."""
    try:
        paragraphs = llm.call(write_prompt, wrap(batch), parse_paragraphs, allow_narrative=True)
    except LLMOutputTruncated as exc:
        smaller = split_writing_batch(batch) if depth < 8 else None
        if smaller is None:
            raise LLMOutputTruncated(
                "PHASE: WRITE: the model continues truncating even a minimal section, "
                "despite reducing the word budget. Increase --max-new-tokens "
                "together with --num-ctx or try another model. No incomplete text was accepted."
            ) from exc
        llm.progress("Writing: response still truncated; splitting the text into two smaller sections")
        for part in smaller:
            yield from writing_sections(llm, part, wrap, write_prompt, depth=depth + 1)
    else:
        yield paragraphs, batch


def write_narratives(llm, store, selected, config):
    write_prompt = prompts.get_prompt(config.write_prompt_name)
    for theme in selected:
        history = ""
        theme_write_prompt = write_prompt.replace("{title}", str(theme.get("title", "")))
        def wrap(rows):
            # Conservative token/word allowance leaves room for JSON syntax and escaping.
            budget = min(600, max(8, config.max_output_tokens // 8),
                         max(8, sum(word_count(row["text"]) for row in rows)))
            return {"theme": theme, "language": config.language, "previous_text": history,
                    "max_words": budget, "chunks": rows}
        count = 0
        for batch in llm.batches(theme_write_prompt, store.chunks(theme["id"]), wrap, max_items=2):
            for paragraphs, sources in writing_sections(llm, batch, wrap, theme_write_prompt):
                for paragraph in paragraphs:
                    history = narrative_history(history + " " + paragraph["text"])
                    paragraph["chunk_ids"] = [chunk["id"] for chunk in sources]
                    paragraph["source_spans"] = [{"chunk_id": chunk["id"], "start": chunk["start"],
                                                   "end": chunk["end"]} for chunk in sources]
                    store.db.execute("INSERT INTO paragraphs VALUES (?,?)",
                                     (theme["id"], json.dumps(paragraph, ensure_ascii=False)))
                    count += 1
                store.db.commit()
        if not count:
            raise RuntimeError(f"Theme {theme['id']} has no relevant chunks; no narrative produced")


def publish(store, selected, run_dir):
    staged = run_dir / "drafts"
    staged.mkdir()
    outputs = []
    for theme in selected:
        name = f"narrative_{theme['id']}.md"
        with (staged / name).open("w", encoding="utf-8") as handle:
            title = theme["title"].replace("\n", " ")
            handle.write(f"# {title}\n\n")
            for paragraph in store.narrative_paragraphs(theme["id"]):
                handle.write(paragraph["text"] + "\n\n")
        outputs.append(name)
    for name in outputs:
        (staged / name).replace(run_dir / name)
    staged.rmdir()
    return outputs


def generate_narratives(inputs: list[Path], *, output_dir: Path, provider,
                        config: GenerateConfig | None = None, progress=print) -> Path:
    config = config or GenerateConfig()
    inputs = [Path(path) for path in inputs]
    output_dir = Path(output_dir).resolve()
    for source in inputs:
        if source.is_dir() and output_dir.is_relative_to(source.resolve()):
            raise ValueError("The output directory must be outside the input directories")
    paths = discover_documents(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=output_dir))
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config), "model": getattr(provider, "model", "unknown"),
                "documents": [], "outputs": [], "ingestion_completed": False}
    save_json(run_dir / "manifest.json", manifest)
    store = Store(run_dir / "analysis.sqlite")
    llm = JsonLLM(provider, config, progress, audit=store.record_response)
    try:
        for doc in iter_documents(paths, ocr_mode=config.ocr_mode,
                                  ocr_language=config.ocr_language, ocr_dpi=config.ocr_dpi,
                                  cache_dir=Path(config.cache_dir).resolve() if config.cache_enabled else None,
                                  refresh_ocr=config.refresh_ocr, progress=progress):
            progress(f"Reading {doc.document_id}: {doc.source_name}")
            manifest["documents"].append({"id": doc.document_id, "path": str(doc.source_path),
                                           "pages_or_rows": len(doc.pages),
                                           "pages": [{"number": page.page_number,
                                                      "method": page.extraction_method,
                                                      "characters": len(page.text),
                                                      "skipped": not bool(page.text.strip())}
                                                     for page in doc.pages]})
            for chunk in iter_chunks(doc, chunk_chars=config.chunk_chars, overlap_chars=config.overlap_chars):
                store.db.execute("INSERT INTO chunks VALUES (?,?)",
                                 (chunk["id"], json.dumps(chunk, ensure_ascii=False)))
            store.db.commit()
            save_json(run_dir / "manifest.json", manifest)
        manifest["ingestion_completed"] = True
        save_json(run_dir / "manifest.json", manifest)
        progress(f"Reading complete: {len(manifest['documents'])} documents. Aggregating the corpus")
        aggregate = aggregate_corpus(llm, store, config, len(manifest["documents"]))
        save_json(run_dir / "aggregate_context.json", aggregate)
        progress("Extracting global themes from the entire aggregated corpus (maximum 5)")
        selected = themes(llm, aggregate, config)
        save_json(run_dir / "themes.json", selected)
        progress(f"Associating documents with {len(selected)} themes")
        route(llm, store, selected)
        progress("Writing narratives from the original text")
        write_narratives(llm, store, selected, config)
        manifest["draft_completed"] = True
        save_json(run_dir / "manifest.json", manifest)
        limit_narratives(llm, store, selected, config)
        manifest["narratives"] = [{"id": theme["id"], "title": theme["title"],
                                   "words": sum(word_count(row["text"]) for row in store.narrative_paragraphs(theme["id"])),
                                   "max_words": config.max_narrative_words} for theme in selected]
        store.export(run_dir)
        manifest["outputs"] = publish(store, selected, run_dir)
        manifest["status"] = "completed"
        return run_dir
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        store.db.commit()
        store.export(run_dir)
        progress(f"Execution failed; details in {run_dir}")
        raise
    finally:
        manifest["llm_calls"] = llm.calls
        manifest["counts"] = {name: store.db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                              for name in ("chunks", "unassigned", "paragraphs")}
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(run_dir / "manifest.json", manifest)
        store.db.close()
