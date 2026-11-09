from __future__ import annotations

import re
from pathlib import Path

from .providers.base import LLMProvider


def split_markdown_document(text: str) -> tuple[str, list[str]]:
    """Return the first Markdown heading and the document paragraphs."""

    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    title = blocks.pop(0) if blocks and blocks[0].lstrip().startswith("#") else ""
    if not blocks:
        raise ValueError("Markdown document must contain at least one paragraph")
    return title, blocks


def clean_rewritten_markdown(response: str) -> str:
    """Remove optional Markdown fences around an LLM response."""

    cleaned = response.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def reorganize_markdown(
    input_path: Path,
    prompt_path: Path,
    provider: LLMProvider,
    output_path: Path | None = None,
    max_new_tokens: int = 12000,
    prompt_text: str | None = None,
) -> Path:
    """Rewrite and regroup a Markdown story."""

    source = input_path.read_text(encoding="utf-8", errors="replace")
    title, paragraphs = split_markdown_document(source)
    prompt = prompt_text if prompt_text is not None else prompt_path.read_text(encoding="utf-8")
    numbered = "\n\n".join(f"[{index}] {paragraph}" for index, paragraph in enumerate(paragraphs, 1))
    response = provider.generate(
        prompt.replace("{document}", numbered),
        temperature=0.1,
        max_new_tokens=max_new_tokens,
    )
    if response.error:
        raise RuntimeError(response.error)
    rewritten = clean_rewritten_markdown(response.text)
    if not rewritten:
        raise ValueError("LLM returned an empty rewritten document")

    destination = output_path or input_path.with_name(f"{input_path.stem}_reorganized.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(([title] if title else []) + [rewritten]) + "\n", encoding="utf-8")
    return destination
