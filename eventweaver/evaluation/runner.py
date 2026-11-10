from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from . import IMPLEMENTATION_VERSION
from .config import METRIC_IDS, load_config, validate_options
from .evidence import parse_json
from .metrics.alignscore import AlignScoreEvaluator
from .metrics.factscore import FactScoreEvaluator
from .metrics.hanna import HannaEvaluator
from .metrics.ragas_faithfulness import RagasFaithfulnessEvaluator
from .metrics.source_coverage import SourceCoverageEvaluator
from .sources import load_inputs

EVALUATORS = {
    "hanna": HannaEvaluator, "factscore": FactScoreEvaluator,
    "ragas_faithfulness": RagasFaithfulnessEvaluator, "alignscore": AlignScoreEvaluator,
    "source_coverage": SourceCoverageEvaluator,
}


def write_json(path: Path, value) -> None:
    # Exclusive creation protects previous results and prevents accidental source overwrites.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def run_evaluation(narrative_path, source_paths, output_dir, metrics, *, config=None,
                   narrative_id=None, original_prompt=None):
    metrics = list(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(METRIC_IDS):
        raise ValueError(f"Select distinct known metric IDs: {', '.join(METRIC_IDS)}")
    effective_config = load_config()
    for name, values in (config or {}).items():
        validate_options(name, values)
        effective_config[name].update(values)
    config = effective_config
    inputs = load_inputs(narrative_path, source_paths, narrative_id=narrative_id,
                         original_prompt=original_prompt)
    destination = Path(output_dir)
    # The output must be new, including when an earlier run failed.
    destination.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now(timezone.utc).isoformat()
    versions = {}
    for package in ("eventweaver", "python-docx", "PyMuPDF", "ragas", "alignscore", "openai", "instructor"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    implementation_files = sorted(
        file for file in Path(__file__).parent.rglob("*") if file.suffix in {".py", ".json"}
    )
    implementation_hash = hashlib.sha256(b"".join(
        file.relative_to(Path(__file__).parent).as_posix().encode() + b"\0" + file.read_bytes()
        for file in implementation_files
    )).hexdigest()
    metadata = {"experiment_id": str(uuid.uuid4()), "implementation_version": IMPLEMENTATION_VERSION,
                "implementation_sha256": implementation_hash, "timestamp": timestamp,
                "python": platform.python_version(), "dependencies": versions,
                "narrative": inputs.narrative, "corpus_id": inputs.corpus["id"],
                "corpus_sha256": inputs.corpus["sha256"], "configuration": config,
                "original_prompt": original_prompt, "metrics": metrics,
                "config_schema_version": "evaluation-config-v1"}
    write_json(destination / "evaluation_metadata.json", metadata)
    write_json(destination / "sources.json", inputs.corpus)
    statuses = {}
    for name in metrics:
        evaluator_type = EVALUATORS[name]
        metric_config = deepcopy(config.get(name, {}))
        result = {"metric": name, "implementation_version": IMPLEMENTATION_VERSION,
                  "implementation_sha256": implementation_hash,
                  "methodological_reference": evaluator_type.reference, "method": evaluator_type.method,
                  "experiment_id": metadata["experiment_id"], "timestamp": timestamp,
                  "configuration": metric_config, "narrative_id": inputs.narrative["id"],
                  "narrative_sha256": inputs.narrative["sha256"],
                  "corpus_id": inputs.corpus["id"], "corpus_sha256": inputs.corpus["sha256"],
                  "score": None, "warnings": [], "dependencies": versions}
        try:
            evaluator = evaluator_type()
            # Every evaluator gets a private snapshot; no metric results are supplied.
            metric_inputs = deepcopy(inputs)
            snapshots = {}
            for field in ("annotations", "inventory", "agreement_annotations"):
                path = metric_config.get(field)
                if path is None:
                    continue
                if not isinstance(path, str) or not path.strip():
                    raise ValueError(f"{field} must be a non-empty file path string")
                raw = Path(path).read_bytes()
                data = parse_json(raw)
                metric_inputs.artifacts[path] = data
                snapshots[field] = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "data": deepcopy(data)}
            result["input_artifacts"] = snapshots
            outcome = evaluator.evaluate(metric_inputs, metric_config)
            if outcome.get("status") not in {"completed", "partial", "failed", "unsupported", "uncertain"}:
                raise ValueError("Evaluator returned an invalid status")
            json.dumps(outcome, allow_nan=False)
            result.update(outcome)
        except Exception as exc:
            result.update(status="failed", score=None, error=f"{type(exc).__name__}: {exc}")
        write_json(destination / f"{name}.json", result)
        statuses[name] = result["status"]
    return statuses
