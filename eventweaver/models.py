from __future__ import annotations

"""
Model configuration for EventWeaver.

Constraints:
- Only include local Ollama models that are expected to fit within a 32 GB model-size budget.
- User-provided model names are still allowed through the CLI, even if not listed here.
- Presets are organized by practical usage for cultural-heritage narrative generation,
  long-document handling, and robustness experiments.
"""

# ---------------------------------------------------------------------
# Core default set
# ---------------------------------------------------------------------
# Balanced default for routine experiments:
# - general-purpose models
# - multilingual/document-capable models
# - one reasoning-style model
# - no very large model by default
DEFAULT_MODELS: list[str] = [
    "llama3.1:8b",
    "qwen3:8b",
    "gemma3:12b",
    "mistral-nemo:12b",
    "granite3.3:8b",
    "cogito:8b",
    "deepseek-r1:7b",
]

# ---------------------------------------------------------------------
# Feasible models under 32 GB
# ---------------------------------------------------------------------
# This list intentionally excludes models whose Ollama default size is
# above 32 GB, such as llama3.1:70b and qwen3:235b.
ALL_MODELS: list[str] = [
    # Llama
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.1:8b",

    # Qwen 2.5
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",

    # Qwen 3
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:30b",
    "qwen3:32b",

    # Qwen 3.5
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "qwen3.5:27b",
    "qwen3.5:35b",

    # Gemma
    "gemma3:1b",
    "gemma3:4b",
    "gemma3:12b",
    "gemma3:27b",
    "gemma2:2b",
    "gemma2:9b",
    "gemma2:27b",

    # Gemma 4
    "gemma4:e2b",
    "gemma4:e4b",
    "gemma4:12b",
    "gemma4:26b",
    "gemma4:31b",

    # Mistral
    "mistral:7b",
    "mistral-nemo:12b",
    "mistral-small:24b",
    "mistral-small3.2:24b",

    # Phi
    "phi3.5:3.8b",
    "phi4-mini:3.8b",
    "phi4:14b",

    # DeepSeek
    "deepseek-v2:16b",
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",

    # Granite
    "granite3.3:2b",
    "granite3.3:8b",

    # Cogito
    "cogito:3b",
    "cogito:8b",
    "cogito:14b",
    "cogito:32b",
]


# ---------------------------------------------------------------------
# Optional exclusion list
# ---------------------------------------------------------------------
# These are not in ALL_MODELS because they exceed the 32 GB model-size rule
# or are cloud/very-large tags.
EXCLUDED_OVER_32GB: list[str] = [
    "llama3.1:70b",
    "llama3.1:405b",
    "qwen3:235b",
    "qwen3.5:122b",
    "cogito:70b",
]


# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------
MODEL_PRESETS: dict[str, list[str]] = {
    # Same as DEFAULT_MODELS.
    "default": DEFAULT_MODELS.copy(),

    # Very light tests, useful for smoke tests and weaker machines.
    "tiny": [
        "llama3.2:1b",
        "qwen3:0.6b",
        "qwen3:1.7b",
        "gemma3:1b",
        "granite3.3:2b",
    ],

    # Fast models for quick experiments.
    "small": [
        "llama3.2:3b",
        "qwen3:4b",
        "qwen3.5:4b",
        "gemma3:4b",
        "phi3.5:3.8b",
        "phi4-mini:3.8b",
        "mistral:7b",
        "deepseek-r1:7b",
        "cogito:3b",
    ],

    # Good routine set for most machines.
    "balanced": [
        "llama3.1:8b",
        "qwen3:8b",
        "qwen3.5:9b",
        "gemma2:9b",
        "gemma3:12b",
        "mistral:7b",
        "mistral-nemo:12b",
        "granite3.3:8b",
        "cogito:8b",
        "deepseek-r1:7b",
    ],

    # Mid-size models, generally stronger but still practical.
    "medium": [
        "llama3.1:8b",
        "qwen2.5:14b",
        "qwen3:14b",
        "gemma3:12b",
        "gemma4:12b",
        "mistral-nemo:12b",
        "phi4:14b",
        "deepseek-r1:14b",
        "cogito:14b",
    ],

    # High-capability local models under 32 GB.
    "large": [
        "qwen2.5:32b",
        "qwen3:30b",
        "qwen3:32b",
        "qwen3.5:27b",
        "qwen3.5:35b",
        "gemma3:27b",
        "gemma2:27b",
        "gemma4:26b",
        "gemma4:31b",
        "mistral-small:24b",
        "mistral-small3.2:24b",
        "deepseek-r1:32b",
        "cogito:32b",
    ],

    # Long-document candidates, chosen for large context windows and document tasks.
    "long-context": [
        "llama3.1:8b",
        "gemma3:12b",
        "gemma3:27b",
        "gemma4:12b",
        "gemma4:26b",
        "gemma4:31b",
        "granite3.3:8b",
        "cogito:8b",
        "cogito:14b",
        "cogito:32b",
        "qwen3.5:9b",
        "qwen3.5:27b",
        "qwen3.5:35b",
    ],

    # Models worth testing specifically on narrative rewriting and summarization.
    "narrative": [
        "llama3.1:8b",
        "qwen3:8b",
        "qwen3.5:9b",
        "gemma3:12b",
        "gemma4:12b",
        "mistral:7b",
        "mistral-nemo:12b",
        "granite3.3:8b",
        "cogito:8b",
    ],

    # Reasoning-style models. May need stricter prompting and post-processing.
    "reasoning": [
        "deepseek-r1:7b",
        "deepseek-r1:14b",
        "deepseek-r1:32b",
        "qwen3:14b",
        "qwen3:32b",
        "qwen3.5:27b",
        "qwen3.5:35b",
        "cogito:14b",
        "cogito:32b",
    ],

    # Strong but still below 32 GB.
    "high": [
        "qwen3:32b",
        "qwen3.5:35b",
        "gemma4:31b",
        "gemma4:26b",
        "mistral-small3.2:24b",
        "deepseek-r1:32b",
        "cogito:32b",
    ],

    # Full feasible benchmark set.
    "all": ALL_MODELS.copy(),
    "mixed": ALL_MODELS.copy(),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def list_available_models() -> list[str]:
    """Return the curated list of feasible Ollama models."""
    return ALL_MODELS.copy()


def list_presets() -> list[str]:
    """Return available preset names."""
    return sorted(MODEL_PRESETS.keys())


def resolve_models(models: list[str] | None = None, preset: str | None = None) -> list[str]:
    """
    Resolve model names from explicit CLI input or a preset.

    Explicit models always win. This allows users to run custom local Ollama
    models even when they are not present in ALL_MODELS.
    """
    if models:
        return list(dict.fromkeys(models))

    if preset:
        selected = MODEL_PRESETS.get(preset)
        if selected is not None:
            return selected.copy()

    return DEFAULT_MODELS.copy()