"""Lossless, source-referenced context packaging, with no metric reasoning."""

from .models import content_hash


def source_contexts(corpus, chunk_chars=4000):
    if type(chunk_chars) is not int or chunk_chars < 1:
        raise ValueError("chunk_chars must be a positive integer")
    contexts = []
    for document in corpus["documents"]:
        for page in document["pages"]:
            start = 0
            while start < len(page["text"]):
                end = min(start + chunk_chars, len(page["text"]))
                while end < len(page["text"]) and not page["text"][end - 1].isspace():
                    end += 1
                item = {"document_id": document["id"], "page_index": page["index"],
                        "start": start, "end": end, "text": page["text"][start:end]}
                contexts.append(dict(item, id="P-" + content_hash(item)))
                start = end
    if not contexts:
        raise ValueError("No documentary contexts available")
    return contexts


def positive_integer(config, key, default):
    value = config.get(key, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value
