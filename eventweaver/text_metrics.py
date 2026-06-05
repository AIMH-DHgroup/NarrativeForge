from __future__ import annotations

import re
from functools import lru_cache

from .utils import normalize_text


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:[-']\w+)?\b", text, flags=re.UNICODE))


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    return paragraphs


def broken_sentence_count(text: str) -> int:
    endings = (".", "!", "?", ".”", "!”", "?”", "’", '"')
    dangling = (",", ";", ":", "-", "—")
    broken = 0
    for paragraph in split_paragraphs(text):
        cleaned = paragraph.rstrip()
        if not cleaned:
            broken += 1
            continue
        if cleaned.endswith(dangling):
            broken += 1
            continue
        if cleaned.endswith(endings):
            continue
        if re.search(r"\b(and|or|but|because|although|however)\s*$", cleaned, flags=re.IGNORECASE):
            broken += 1
            continue
        broken += 1
    return broken


def forbidden_formatting_count(text: str) -> int:
    patterns = [
        r"^\s*#{1,6}\s+",
        r"^\s*[-*•]\s+",
        r"^\s*\d+[\).]\s+",
        r"^\s*[\{\}\[\]]\s*$",
        r"^\s*[\{\[]",
        r"\|",
        r"```",
        r"^\s*(Title|Case-study form|Case study form|Narrative|Event\s*\d+|Paragraph\s*\d+)\s*:",
    ]
    count = 0
    for line in text.splitlines():
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
            count += 1
    return count


def is_failed_output(text: str) -> bool:
    return (
        not text.strip()
        or word_count(text) < 250
        or word_count(text) > 900
        or len(split_paragraphs(text)) < 2
        or forbidden_formatting_count(text) > 0
        or broken_sentence_count(text) > 0
    )


def semantic_similarity_with_method(source_text: str, output_text: str, method: str = "sentence-transformers") -> tuple[float, str]:
    source = normalize_text(source_text)
    output = normalize_text(output_text)

    if method in {"sentence-transformers", "auto", "sbert"}:
        try:
            return _sentence_transformer_similarity(source, output), "sentence-transformers/all-MiniLM-L6-v2"
        except Exception:
            if method != "auto":
                pass

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        mat = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([source, output])
        return max(0.0, min(1.0, float(cosine_similarity(mat[0:1], mat[1:2])[0][0]))), "tfidf"
    except Exception:
        return _token_overlap_similarity(source, output), "token_overlap"


def semantic_similarity(source_text: str, output_text: str, method: str = "sentence-transformers") -> float:
    return semantic_similarity_with_method(source_text, output_text, method)[0]


def bertscore_f1(source_text: str, output_text: str) -> float | None:
    try:
        from bert_score import score as bert_score
    except Exception:
        return None

    try:
        preds = [normalize_text(output_text)]
        refs = [normalize_text(source_text)]
        _, _, f1 = bert_score(preds, refs, lang="en", verbose=False, rescale_with_baseline=False)
        value = float(f1[0].item() if hasattr(f1[0], "item") else f1[0])
        return max(0.0, min(1.0, value))
    except Exception:
        return None


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _sentence_transformer_similarity(source: str, output: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    import numpy as np

    model = _load_sentence_transformer(model_name)
    emb = model.encode([source, output], normalize_embeddings=True)
    return max(0.0, min(1.0, float(np.dot(emb[0], emb[1]))))


def _token_overlap_similarity(source: str, output: str) -> float:
    source_terms = set(re.findall(r"\w+", source.lower()))
    output_terms = set(re.findall(r"\w+", output.lower()))
    if not source_terms and not output_terms:
        return 1.0
    if not source_terms or not output_terms:
        return 0.0
    return len(source_terms & output_terms) / len(source_terms | output_terms)
