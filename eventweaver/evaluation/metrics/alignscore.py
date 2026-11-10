import hashlib
import json
import math
import subprocess
from pathlib import Path

from ..base import unavailable
from ..contexts import positive_integer, source_contexts

FACTIVERSE_REVISION = "247c7f8549694111155b6b13b55bc39103280ddc"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AlignScoreEvaluator:
    reference = "https://github.com/factiverse/alignscore/tree/247c7f8549694111155b6b13b55bc39103280ddc"
    method = "Factiverse nli_sp; explicit entailment component matching original mean-over-sentences/max-over-source-chunks"

    def evaluate(self, inputs, config):
        if not config.get("checkpoint") or not config.get("python") or not config.get("backbone"):
            return unavailable("Configure isolated python, a trusted AlignScore checkpoint, and a local backbone directory")
        checkpoint, executable, backbone = (Path(config[key]).resolve() for key in ("checkpoint", "python", "backbone"))
        if not checkpoint.is_file():
            return unavailable(f"Missing AlignScore checkpoint: {checkpoint}")
        if not executable.is_file() or not backbone.is_dir():
            return unavailable("Isolated Python executable or local backbone assets are missing")
        digest = file_hash(checkpoint)
        if config.get("checkpoint_sha256") and config["checkpoint_sha256"] != digest:
            raise ValueError("Checkpoint SHA-256 does not match the configured identity")
        contexts = source_contexts(inputs.corpus, positive_integer(config, "chunk_chars", 4000))
        if not inputs.narrative["body"].strip():
            raise ValueError("AlignScore narrative must be non-empty")
        text = "\n\n".join(item["text"] for item in contexts)
        if len(text) + len(inputs.narrative["body"]) > positive_integer(config, "max_input_chars", 200000):
            return unavailable("Full AlignScore input exceeds max_input_chars; no documents were silently discarded")
        job = {"context": text, "narrative": inputs.narrative["body"], "checkpoint": str(checkpoint),
               "backbone": str(backbone), "batch_size": positive_integer(config, "batch_size", 16),
               "device": config.get("device", "cpu")}
        worker = Path(__file__).resolve().parents[1] / "alignscore_worker.py"
        try:
            completed = subprocess.run(
                [str(executable), "-B", str(worker)], input=json.dumps(job), text=True,
                encoding="utf-8", capture_output=True, check=False,
                timeout=positive_integer(config, "timeout_seconds", 600),
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed", "score": None, "error": "Isolated AlignScore process timed out", "checkpoint_sha256": digest}
        if completed.returncode != 0:
            return {"status": "failed", "score": None, "error": "AlignScore subprocess failed: " + completed.stderr[-4000:]}
        output = json.loads(completed.stdout)
        if output.get("error"):
            return unavailable("Isolated AlignScore environment/resource error: " + output["error"])
        if output.get("implementation_revision") != FACTIVERSE_REVISION:
            raise ValueError("AlignScore implementation revision mismatch")
        values = output.get("raw_scores")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], list) or len(values[0]) != 3:
            raise ValueError("Expected one Factiverse nli_sp three-class result; scalar output is not silently reinterpreted")
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values[0]):
            raise ValueError("Invalid AlignScore output values")
        return {"status": "completed", "score": values[0][0], "raw_class_scores": values[0],
                "entailment_class_index": 0, "checkpoint_sha256": digest,
                "backbone_files": {path.relative_to(backbone).as_posix(): file_hash(path)
                                   for path in sorted(backbone.rglob("*")) if path.is_file()},
                "contexts_used": contexts, "worker_output": output,
                "worker_sha256": file_hash(worker), "seed": 0,
                "warnings": ["Factiverse nli_sp returns three class-wise scores; only class 0 is entailment, as in original AlignScore.",
                             "Corpus concatenation is a project adaptation. All source chunks are compared to every narrative sentence.",
                             "Reference chunking estimates source groups from whitespace words // 350 + 1, groups consecutive NLTK sentences, zero overlap; it is not strict subword chunking.",
                             "Reference pair tokenization may truncate long pairs to the backbone window (usually 512). Scores are sentence-weighted consistency, not atomic precision.",
                             "Checkpoint uses pickle-based loading; load only trusted artifacts. Deterministic algorithms may fail on unsupported devices."]}
