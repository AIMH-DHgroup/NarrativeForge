from .evidence import read_json
from pathlib import Path

METRIC_IDS = ("hanna", "factscore", "ragas_faithfulness", "alignscore", "source_coverage")

ALLOWED_KEYS = {
    "hanna": {"annotations", "rubric_version"},
    "factscore": {"annotations", "gamma"},
    "source_coverage": {"inventory", "annotations"},
    "ragas_faithfulness": set(), "alignscore": set(),
}


def load_config(path=None):
    config = read_json(str(path)) if path else {}
    if not isinstance(config, dict) or set(config) - set(METRIC_IDS):
        raise ValueError("Evaluation config must map known metric IDs to configuration objects")
    if any(not isinstance(value, dict) for value in config.values()):
        raise ValueError("Each metric requires its own configuration object")
    defaults = read_json(str(Path(__file__).with_name("defaults.json")))["metrics"]
    for name, values in config.items():
        if set(values) - ALLOWED_KEYS[name]:
            raise ValueError(f"Unsupported configuration fields for {name}")
        defaults[name].update(values)
    return defaults
