from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


def text_quality_score(text: str) -> float:
    """Simple quality heuristic; OCR output is not semantically verified."""
    value = (text or "").strip()
    if not value:
        return 0.0
    chars = len(value)
    if chars < 20:
        return min(0.2, chars / 100.0)
    alnum = sum(ch.isalnum() for ch in value)
    printable = sum(ch.isprintable() for ch in value)
    words = re.findall(r"\b\w+\b", value, flags=re.UNICODE)
    return max(0.0, min(1.0, 0.45 * alnum / chars + 0.25 * printable / chars
                        + 0.30 * min(1.0, len(words) / 25.0)))


def needs_ocr(text: str, threshold: float = 0.55) -> bool:
    return text_quality_score(text) < threshold


def resolve_tesseract() -> str:
    explicit = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        if Path(explicit).is_file():
            return explicit
        raise RuntimeError(f"TESSERACT_CMD does not identify an executable: {explicit}")
    executable = shutil.which("tesseract")
    if executable:
        return executable
    if os.name == "nt":
        for directory in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                          os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            candidate = Path(directory) / "Tesseract-OCR" / "tesseract.exe"
            if candidate.is_file():
                return str(candidate)
    raise RuntimeError("OCR is required but Tesseract was not found. Install Tesseract OCR, "
                       "add it to PATH, or set TESSERACT_CMD to the tesseract.exe path. "
                       "Check languages with 'tesseract --list-langs'.")


def ocr_pdf_page(page, language: str = "ita+eng", dpi: int = 200,
                 page_segmentation: int = 3) -> str:
    """Render and run Tesseract directly, without pytesseract/Pillow dependencies."""
    executable = resolve_tesseract()
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    command = [executable, "stdin", "stdout", "-l", language, "--dpi", str(dpi),
               "--psm", str(page_segmentation)]
    try:
        result = subprocess.run(command, input=pix.tobytes("png"), capture_output=True,
                                timeout=180, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Tesseract exceeded the 180-second timeout for a page") from exc
    except OSError as exc:
        raise RuntimeError(f"Impossibile avviare Tesseract: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract OCR failed (languages {language}): {detail}. "
                           "Check language packages with 'tesseract --list-langs'.")
    return result.stdout.decode("utf-8", errors="replace").strip()
