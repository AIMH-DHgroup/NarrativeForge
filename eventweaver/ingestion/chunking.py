from __future__ import annotations

import re
from typing import Iterator
from .models import LoadedDocument


def iter_chunks(document: LoadedDocument, *, chunk_chars: int = 5000,
                overlap_chars: int = 400) -> Iterator[dict]:
    """Bound even single unbroken tokens, retaining exact normalized source offsets."""
    if chunk_chars < 64 or not 0 <= overlap_chars < chunk_chars:
        raise ValueError("Richiesti chunk_chars >= 64 e 0 <= overlap_chars < chunk_chars")
    index = 0
    for page in document.pages:
        text = re.sub(r"\s+", " ", page.text).strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_chars // 2, end)
                if boundary != -1:
                    end = boundary
            index += 1
            yield {"id": f"{document.document_id}_C{index:06d}",
                   "document_id": document.document_id, "source": str(document.source_path),
                   "page_or_row": page.page_number, "method": page.extraction_method,
                   "start": start, "end": end, "text": text[start:end]}
            if end == len(text):
                break
            start = max(start + 1, end - overlap_chars)
