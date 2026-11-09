"""Content-addressed, atomic page cache, independent of narrative runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .models import DocumentPage
from .ocr import resolve_tesseract

# Bump when extraction, quality heuristics or retry policy changes.
EXTRACTION_VERSION = 1
METHODS = {"native", "native-empty", "ocr", "ocr-retry", "native-fallback", "ocr-empty"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_identity() -> dict:
    """Identify the installed engine without running recognition."""
    try:
        executable = resolve_tesseract()
        stat = Path(executable).stat()
        result = subprocess.run([executable, "--version"], capture_output=True,
                                timeout=10, check=False)
        version = (result.stdout + result.stderr).decode("utf-8", errors="replace").strip()
        return {"version": version, "executable": str(Path(executable).resolve()),
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "tessdata_prefix": os.environ.get("TESSDATA_PREFIX", "")}
    except (OSError, RuntimeError, subprocess.SubprocessError):
        # Native extraction must remain usable without an installed OCR engine.
        return {"version": "unavailable"}


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PageCache:
    def __init__(self, directory: Path, source: Path, *, mode: str, language: str,
                 dpi: int, pdf_version: str, progress=lambda message: None):
        self.progress = progress
        self.identity = {"schema": EXTRACTION_VERSION, "document_sha256": file_digest(source),
                         "mode": mode, "language": language, "dpi": dpi,
                         "pymupdf": pdf_version,
                         "engine": engine_identity() if mode != "never" else None}
        key = _checksum(json.dumps(self.identity, sort_keys=True, ensure_ascii=True))
        self.directory = Path(directory) / "pdf" / key

    def read(self, number: int) -> DocumentPage | None:
        try:
            record = json.loads((self.directory / f"page_{number:06d}.json").read_text(encoding="utf-8"))
            value, method = record["text"], record["method"]
            if (record["identity"] != self.identity or record["page"] != number
                    or not isinstance(value, str) or method not in METHODS
                    or record["text_sha256"] != _checksum(value)):
                raise ValueError("incompatible content")
            return DocumentPage(number, value, method)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError):
            self.progress(f"Page cache {number} is invalid or unreadable; extracting again")
            return None

    def write(self, page: DocumentPage):
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            record = {"identity": self.identity, "page": page.page_number,
                      "text": page.text, "method": page.extraction_method,
                      "text_sha256": _checksum(page.text)}
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.directory / f"page_{page.page_number:06d}.json")
        except OSError as exc:
            self.progress(f"Warning: could not save the cache for page {page.page_number}: {exc}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
