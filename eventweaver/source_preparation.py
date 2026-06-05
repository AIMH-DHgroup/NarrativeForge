from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass

from .text_metrics import word_count


RETRIEVAL_QUERY = (
    "itinerary concept, geographic context, historical period, main places, route, musical genre, instruments, "
    "performers, repertoire, cultural significance, social significance, religious significance, acoustic dimension, "
    "soundscape, digital technology, immersive technology, VR, AR, XR, auralization, accessibility, public engagement, "
    "education, promotion, objectives"
)

SECTION_ORDER = [
    "title / concept",
    "description of itinerary",
    "geographic context",
    "historical period",
    "main itinerary and places",
    "musical dimension",
    "cultural significance",
    "objects and sources",
    "accessibility",
    "technological and immersive layer",
    "objectives",
]

SECTION_LABELS: dict[str, tuple[str, ...]] = {
    "title / concept": ("title", "concept", "itinerary name", "tour name", "project name", "heading"),
    "description of itinerary": ("description", "summary", "overview", "presentation"),
    "geographic context": ("city", "region", "territory", "location", "geographic", "context", "area"),
    "historical period": ("historical period", "period", "chronology", "date", "era", "century"),
    "main itinerary and places": ("place", "site", "sites", "building", "buildings", "route", "itinerary", "landscape"),
    "musical dimension": ("music", "musical", "genre", "instrument", "instruments", "performer", "performers", "repertoire", "song", "sound"),
    "cultural significance": ("cultural", "social", "religious", "significance", "heritage", "meaning", "memory"),
    "objects and sources": ("object", "objects", "document", "documents", "source", "sources", "archive", "archives", "musical source"),
    "accessibility": ("access", "accessible", "accessibility", "preserve", "preservation", "transform", "transformation"),
    "technological and immersive layer": ("digital", "immersive", "technology", "technological", "virtual", "vr", "ar", "xr", "auralization", "soundscape"),
    "objectives": ("objective", "objectives", "aim", "aims", "goal", "goals", "promotion", "promote", "audience", "education", "collaboration", "collaborations", "strategy"),
}

LABEL_TO_SECTION = {
    label: section
    for section, labels in SECTION_LABELS.items()
    for label in labels
}

SHORT_DOC_WORD_LIMIT = 3500
RAG_MIN_WORDS = 180


@dataclass
class PreparedSource:
    text: str
    strategy_used: str
    original_source_word_count: int
    prepared_source_word_count: int
    selected_chunk_count: int


@dataclass
class SourceChunk:
    section: str
    text: str
    order: int


def prepare_source_text(source_text: str, strategy: str, max_source_words: int, chunk_words: int, chunk_overlap: int, top_k: int) -> str:
    return prepare_source_context(source_text, strategy, max_source_words, chunk_words, chunk_overlap, top_k).text


def prepare_source_context(source_text: str, strategy: str, max_source_words: int, chunk_words: int, chunk_overlap: int, top_k: int) -> PreparedSource:
    normalized = _normalize_source(source_text)
    original_words = word_count(normalized)
    strategy = (strategy or "auto").strip().lower()

    if not normalized:
        return PreparedSource(text="", strategy_used="full", original_source_word_count=0, prepared_source_word_count=0, selected_chunk_count=0)

    if strategy == "auto":
        strategy = "brief" if original_words > max_source_words else "full"

    if strategy == "full":
        if original_words > max_source_words:
            warnings.warn(
                f"Source text is {original_words} words, which may exceed the model context window in full mode.",
                stacklevel=2,
            )
        return PreparedSource(text=normalized, strategy_used="full", original_source_word_count=original_words, prepared_source_word_count=original_words, selected_chunk_count=0)

    if strategy == "rag":
        prepared = _prepare_rag_source(normalized, chunk_words, chunk_overlap, top_k)
        prepared_words = word_count(prepared.text)
        if prepared_words < RAG_MIN_WORDS:
            warnings.warn("RAG returned too little text; falling back to a compact brief.", stacklevel=2)
            prepared = _prepare_brief_source(normalized, max_source_words)
        return PreparedSource(
            text=prepared.text,
            strategy_used="rag",
            original_source_word_count=original_words,
            prepared_source_word_count=word_count(prepared.text),
            selected_chunk_count=prepared.selected_chunk_count,
        )

    prepared = _prepare_brief_source(normalized, max_source_words)
    return PreparedSource(
        text=prepared.text,
        strategy_used="brief",
        original_source_word_count=original_words,
        prepared_source_word_count=word_count(prepared.text),
        selected_chunk_count=prepared.selected_chunk_count,
    )


def _normalize_source(source_text: str) -> str:
    replacements = {
        "\u2011": "-",
        "\u2010": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    text = source_text
    for old, new in replacements.items():
        text = text.replace(old, new)
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _prepare_brief_source(source_text: str, max_source_words: int) -> PreparedSource:
    sections = _collect_sections(source_text)
    if not sections:
        brief = _truncate_words(source_text, max_source_words)
        wrapped = _wrap_brief(brief)
        return PreparedSource(text=wrapped, strategy_used="brief", original_source_word_count=word_count(source_text), prepared_source_word_count=word_count(wrapped), selected_chunk_count=0)

    selected: list[tuple[int, str, str]] = []
    for section_name in SECTION_ORDER:
        section_blocks = sections.get(section_name, [])
        if not section_blocks:
            continue
        section_text = "\n\n".join(section_blocks)
        sentences = _extract_informative_sentences(section_text)
        if not sentences:
            continue
        selected_text = _truncate_words(" ".join(sentences[:3]), max(80, max_source_words // 8))
        selected.append((SECTION_ORDER.index(section_name), section_name, selected_text))

    if not selected:
        selected = [(0, "description of itinerary", _truncate_words(source_text, max_source_words))]

    selected.sort(key=lambda item: item[0])
    body_parts = ["CASE STUDY BRIEF", ""]
    for _, section_name, text in selected:
        if not text.strip():
            continue
        body_parts.append(f"{_brief_heading(section_name)}:")
        body_parts.append(text.strip())
        body_parts.append("")

    brief = "\n".join(body_parts).strip()
    brief = _truncate_words(brief, max_source_words)
    return PreparedSource(text=brief, strategy_used="brief", original_source_word_count=word_count(source_text), prepared_source_word_count=word_count(brief), selected_chunk_count=len(selected))


def _prepare_rag_source(source_text: str, chunk_words: int, chunk_overlap: int, top_k: int) -> PreparedSource:
    chunks = _build_section_chunks(source_text, chunk_words, chunk_overlap)
    if not chunks:
        wrapped = _wrap_brief(_truncate_words(source_text, SHORT_DOC_WORD_LIMIT))
        return PreparedSource(text=wrapped, strategy_used="brief", original_source_word_count=word_count(source_text), prepared_source_word_count=word_count(wrapped), selected_chunk_count=0)

    scores = _score_chunks(chunks, RETRIEVAL_QUERY)
    ranked = sorted(scores, key=lambda item: item[0], reverse=True)[: max(1, top_k)]
    selected_chunks = [chunk for _, chunk in sorted(ranked, key=lambda item: item[1].order)]

    if not selected_chunks:
        wrapped = _wrap_brief(_truncate_words(source_text, SHORT_DOC_WORD_LIMIT))
        return PreparedSource(text=wrapped, strategy_used="brief", original_source_word_count=word_count(source_text), prepared_source_word_count=word_count(wrapped), selected_chunk_count=0)

    parts = ["RETRIEVED SOURCE EXCERPTS", ""]
    for chunk in selected_chunks:
        parts.append(f"[Section: {chunk.section}]")
        parts.append(chunk.text.strip())
        parts.append("")

    text = "\n".join(parts).strip()
    return PreparedSource(text=text, strategy_used="rag", original_source_word_count=word_count(source_text), prepared_source_word_count=word_count(text), selected_chunk_count=len(selected_chunks))


def _collect_sections(source_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {section: [] for section in SECTION_ORDER}
    blocks = [block.strip() for block in re.split(r"\n\s*\n", source_text) if block.strip()]
    for block in blocks:
        section = _section_for_block(block)
        sections.setdefault(section, []).append(block)
    return sections


def _section_for_block(block: str) -> str:
    text = block.lower()
    best_section = "description of itinerary"
    best_score = 0
    for section, labels in SECTION_LABELS.items():
        score = 0
        for label in labels:
            if re.search(rf"\b{re.escape(label)}\b", text):
                score += 2
        if score > best_score:
            best_section = section
            best_score = score
    return best_section


def _extract_informative_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []
    scored = [(_score_sentence(sentence), idx, sentence) for idx, sentence in enumerate(sentences)]
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:3]
    return [sentence for _, _, sentence in sorted(top, key=lambda item: item[1])]


def _score_sentence(sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    for section, labels in SECTION_LABELS.items():
        for label in labels:
            if label in lowered:
                score += 2
    if re.search(r"\b\d{3,4}\b", sentence):
        score += 2
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sentence):
        score += 2
    if re.search(r"\b(technology|digital|immersive|music|heritage|accessibility|objective|promotion|route|place|site)\b", lowered):
        score += 1
    return score + min(len(sentence.split()) // 8, 3)


def _build_section_chunks(source_text: str, chunk_words: int, chunk_overlap: int) -> list[SourceChunk]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", source_text) if block.strip()]
    chunks: list[SourceChunk] = []
    order = 0
    for block in blocks:
        section = _section_for_block(block)
        block_words = word_count(block)
        if block_words <= chunk_words:
            chunks.append(SourceChunk(section=section, text=block, order=order))
            order += 1
            continue
        words = re.findall(r"\S+", block)
        start = 0
        step = max(1, chunk_words - chunk_overlap)
        while start < len(words):
            excerpt = " ".join(words[start : start + chunk_words]).strip()
            if excerpt:
                chunks.append(SourceChunk(section=section, text=excerpt, order=order))
                order += 1
            if start + chunk_words >= len(words):
                break
            start += step
    return chunks


def _score_chunks(chunks: list[SourceChunk], query: str) -> list[tuple[float, SourceChunk]]:
    backend = os.getenv("EVENTWEAVER_RAG_BACKEND", "tfidf").strip().lower()
    if backend in {"auto", "embeddings", "sentence-transformers", "sbert"}:
        try:
            return _score_chunks_with_embeddings(chunks, query)
        except Exception:
            warnings.warn("sentence-transformers retrieval is unavailable; falling back to TF-IDF.", stacklevel=2)
    return _score_chunks_with_tfidf(chunks, query)


def _score_chunks_with_tfidf(chunks: list[SourceChunk], query: str) -> list[tuple[float, SourceChunk]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return _score_chunks_with_overlap(chunks, query)

    corpus = [query] + [chunk.text for chunk in chunks]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
    query_vector = matrix[0:1]
    chunk_matrix = matrix[1:]
    similarities = cosine_similarity(chunk_matrix, query_vector).ravel()
    return [(float(score), chunk) for score, chunk in zip(similarities, chunks)]


def _score_chunks_with_overlap(chunks: list[SourceChunk], query: str) -> list[tuple[float, SourceChunk]]:
    query_terms = set(re.findall(r"\w+", query.lower()))
    scores: list[tuple[float, SourceChunk]] = []
    for chunk in chunks:
        terms = set(re.findall(r"\w+", chunk.text.lower()))
        score = 0.0 if not terms or not query_terms else len(terms & query_terms) / len(terms | query_terms)
        scores.append((score, chunk))
    return scores


def _score_chunks_with_embeddings(chunks: list[SourceChunk], query: str) -> list[tuple[float, SourceChunk]]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode([query] + [chunk.text for chunk in chunks], normalize_embeddings=True)
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    scores = np.dot(chunk_embeddings, query_embedding)
    return [(float(score), chunk) for score, chunk in zip(scores, chunks)]


def _brief_heading(section_name: str) -> str:
    mapping = {
        "title / concept": "Title / concept",
        "description of itinerary": "Description of itinerary",
        "geographic context": "Geographic context",
        "historical period": "Historical period",
        "main itinerary and places": "Main itinerary and places",
        "musical dimension": "Musical dimension",
        "cultural significance": "Cultural significance",
        "objects and sources": "Objects and sources",
        "accessibility": "Accessibility",
        "technological and immersive layer": "Technological and immersive layer",
        "objectives": "Objectives",
    }
    return mapping.get(section_name, section_name.title())


def _wrap_brief(text: str) -> str:
    return f"CASE STUDY BRIEF\n\n{text.strip()}".strip()


def _truncate_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()
