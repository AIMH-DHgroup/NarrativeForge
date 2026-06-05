from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class OllamaResult:
    text: str
    runtime_seconds: float
    error: str | None = None


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def generate_ollama(prompt: str, model: str, temperature: float = 0.1, num_ctx: int = 8192, timeout: int = 900) -> OllamaResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    started = time.perf_counter()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        if "deepseek" in model.lower():
            text = remove_think_blocks(text)
        return OllamaResult(text=text, runtime_seconds=time.perf_counter() - started)
    except Exception as exc:
        return OllamaResult(text="", runtime_seconds=time.perf_counter() - started, error=str(exc))
