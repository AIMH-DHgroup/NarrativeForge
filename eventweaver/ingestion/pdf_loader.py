from __future__ import annotations

from pathlib import Path

from .cache import PageCache
from .models import DocumentPage, LoadedDocument
from .ocr import needs_ocr, ocr_pdf_page, text_quality_score


def load_pdf(path: Path, *, document_id: str, ocr_mode: str = "auto",
             ocr_language: str = "ita+eng", ocr_dpi: int = 200,
             cache_dir: Path | None = None, refresh_ocr: bool = False, progress=lambda message: None) -> LoadedDocument:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF support requires PyMuPDF. Install it with 'pip install PyMuPDF'.") from exc
    mode = ocr_mode.strip().lower()
    if mode not in {"auto", "always", "never"}:
        raise ValueError("ocr_mode must be one of: auto, always, never")
    cache = (PageCache(cache_dir, path, mode=mode, language=ocr_language, dpi=ocr_dpi,
                       pdf_version=str(getattr(fitz, "VersionBind", "unknown")), progress=progress)
             if cache_dir is not None else None)
    pages = []
    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf, 1):
            cached = cache.read(index) if cache is not None and not refresh_ocr else None
            if cached is not None:
                progress(f"Cached text {path.name}, page {index}: reused ({cached.extraction_method})")
                pages.append(cached)
                continue
            native = page.get_text("text").strip()
            value, method = native, "native" if native else "native-empty"
            use_ocr = mode == "always" or (mode == "auto" and needs_ocr(native))
            if use_ocr:
                progress(f"OCR {path.name}, page {index}, {ocr_dpi} DPI ({ocr_language})")
                try:
                    recognized = ocr_pdf_page(page, language=ocr_language, dpi=ocr_dpi)
                    method = "ocr"
                    if needs_ocr(recognized):
                        retry_dpi = max(300, ocr_dpi)
                        progress(f"OCR retry {path.name}, page {index}, {retry_dpi} DPI, PSM 6")
                        retry = ocr_pdf_page(page, language=ocr_language, dpi=retry_dpi, page_segmentation=6)
                        if text_quality_score(retry) > text_quality_score(recognized):
                            recognized, method = retry, "ocr-retry"
                except RuntimeError as exc:
                    raise RuntimeError(f"{path}, page {index}: {exc}") from exc
                if native and text_quality_score(native) > text_quality_score(recognized):
                    value, method = native, "native-fallback"
                else:
                    value = recognized.strip()
                    if not value:
                        method = "ocr-empty"
            if not value:
                progress(f"Warning: {path.name}, page {index} has no text ({method}); page skipped")
            extracted = DocumentPage(index, value, method)
            if cache is not None:
                cache.write(extracted)
            pages.append(extracted)
    document = LoadedDocument(document_id, path, path.name, "pdf", pages)
    if not document.text:
        diagnostic = "even after OCR attempts" if mode != "never" else "with OCR disabled; retry with --ocr always"
        raise ValueError(f"No readable text in {path} {diagnostic}. Check the scan, OCR languages and content.")
    return document
