from .evidence import read_json
from pathlib import Path

METRIC_IDS = ("hanna", "factscore", "ragas_faithfulness", "alignscore", "source_coverage")

ALLOWED_KEYS = {
    "hanna": {"annotations", "rubric_version", "agreement_annotations"},
    "factscore": {"annotations", "gamma", "mode", "model", "base_url", "api_key_env",
                  "max_tokens", "max_input_chars", "timeout_seconds"},
    "source_coverage": {"inventory", "annotations", "mode", "model", "base_url", "api_key_env",
                        "max_tokens", "max_input_chars", "timeout_seconds"},
    "ragas_faithfulness": {"model", "base_url", "api_key_env", "user_input", "chunk_chars",
                           "max_input_chars", "max_tokens", "timeout_seconds"},
    "alignscore": {"python", "checkpoint", "checkpoint_sha256", "backbone", "device", "batch_size",
                   "chunk_chars", "max_input_chars", "timeout_seconds"},
}


def validate_options(name, values):
    if name not in ALLOWED_KEYS or not isinstance(values, dict) or set(values) - ALLOWED_KEYS[name]:
        raise ValueError(f"Invalid configuration for {name}")
    if name in {"factscore", "source_coverage"} and values.get("mode", "human") == "human":
        judge_options = {"model", "base_url", "api_key_env", "max_tokens", "max_input_chars", "timeout_seconds"}
        if set(values) & judge_options:
            raise ValueError(f"{name} judge settings require mode='automatic'; human mode will not silently ignore them")


def load_config(path=None):
    config = read_json(str(path)) if path else {}
    if not isinstance(config, dict) or set(config) - set(METRIC_IDS):
        raise ValueError("Evaluation config must map known metric IDs to configuration objects")
    if any(not isinstance(value, dict) for value in config.values()):
        raise ValueError("Each metric requires its own configuration object")
    defaults = read_json(str(Path(__file__).with_name("defaults.json")))["metrics"]
    for name, values in config.items():
        validate_options(name, values)
        defaults[name].update(values)
    return defaults
