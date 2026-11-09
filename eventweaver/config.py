from dataclasses import dataclass
import math


def validate_penalty(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError(f"{name} must be a finite number between 0 and 2")
    return float(value)


@dataclass(frozen=True)
class GenerateConfig:
    max_themes: int = 5
    language: str = "English"
    narrative_titles: tuple[str, ...] = ()
    max_narrative_words: int | None = None
    context_tokens: int = 16384
    max_output_tokens: int = 4096
    context_safety_tokens: int = 512
    chunk_chars: int = 3000
    overlap_chars: int = 300
    retries: int = 2
    frequency_penalty: float = 1.0
    presence_penalty: float = 0.8
    ocr_mode: str = "auto"
    ocr_language: str = "ita+eng"
    ocr_dpi: int = 200
    cache_enabled: bool = True
    cache_dir: str = ".eventweaver_cache"
    refresh_ocr: bool = False
    write_prompt_name: str = "WRITE"

    def __post_init__(self):
        if not isinstance(self.cache_dir, str) or not self.cache_dir.strip():
            raise ValueError("--cache-dir must be a non-empty path")
        if type(self.cache_enabled) is not bool or type(self.refresh_ocr) is not bool:
            raise ValueError("Cache options must be booleans")
        if self.refresh_ocr and not self.cache_enabled:
            raise ValueError("--refresh-ocr and --no-cache cannot be used together")
        validate_penalty(self.frequency_penalty, "--frequency-penalty")
        validate_penalty(self.presence_penalty, "--presence-penalty")
        if not 1 <= self.max_themes <= 5:
            raise ValueError("max_themes must be between 1 and 5")
        if isinstance(self.narrative_titles, str):
            raise ValueError("narrative_titles must be a list of titles")
        titles = tuple(self.narrative_titles)
        if len(titles) > self.max_themes:
            raise ValueError("The number of titles cannot exceed --max-themes (maximum 5)")
        if any(not isinstance(title, str) or not title.strip() or "\n" in title or "\r" in title for title in titles):
            raise ValueError("Each title must be non-empty text on one line")
        titles = tuple(title.strip() for title in titles)
        if len({title.casefold() for title in titles}) != len(titles):
            raise ValueError("Titles must be distinct")
        object.__setattr__(self, "narrative_titles", titles)
        if self.max_narrative_words is not None and (type(self.max_narrative_words) is not int or self.max_narrative_words <= 0):
            raise ValueError("--max-words must be an integer greater than zero")
        if self.chunk_chars < 64 or not 0 <= self.overlap_chars < self.chunk_chars:
            raise ValueError("chunk_chars >= 64 e 0 <= overlap_chars < chunk_chars")
        if self.max_output_tokens < 256 or self.context_safety_tokens < 0:
            raise ValueError("Output budget >= 256; context margin >= 0")
        if self.context_tokens - self.max_output_tokens - self.context_safety_tokens < 2048:
            raise ValueError("Insufficient context: increase --num-ctx or reduce --max-new-tokens")
        if self.retries < 0 or self.ocr_dpi <= 0 or self.ocr_mode not in {"auto", "always", "never"}:
            raise ValueError("Invalid retry/OCR configuration")
        if not self.language.strip() or len(self.language) > 80:
            raise ValueError("Specify a language (maximum 80 characters)")
