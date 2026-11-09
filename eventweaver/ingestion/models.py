from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentPage:
    page_number: int | None
    text: str
    extraction_method: str


@dataclass
class LoadedDocument:
    document_id: str
    source_path: Path
    source_name: str
    source_type: str
    pages: list[DocumentPage] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text.strip() for page in self.pages if page.text.strip()).strip()

