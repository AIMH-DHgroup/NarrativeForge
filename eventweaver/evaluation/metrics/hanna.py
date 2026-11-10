from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..base import unavailable
from ..evidence import artifact, load_annotations
from ..models import EvaluationInput


class HannaEvaluator:
    reference = "https://arxiv.org/html/2208.11646v4#A2.T7"
    method = "HANNA-derived human assessment; per-dimension means of supplied ratings"

    def evaluate(self, inputs: EvaluationInput, config: dict[str, Any]) -> dict[str, Any]:
        rubric = json.loads(
            (Path(__file__).resolve().parents[1] / "rubrics" / "hanna-v1.json").read_text(
                encoding="utf-8"
            )
        )
        if config.get("rubric_version", "hanna-2022-table7-v1") != rubric["version"]:
            raise ValueError("Unsupported HANNA rubric_version")
        dimensions = tuple(rubric["dimensions"])
        warnings = [
            "HANNA-derived human assessment, not scores from the original HANNA dataset. "
            "No automated HANNA scores or aggregate score are computed.",
            "Human-story calibration and the original worker screening and quality controls "
            "are not verified; this is not an exact replication of the HANNA protocol.",
        ]
        result = {
            "score": None,
            "dimension_scores": dict.fromkeys(dimensions),
            "dimension_counts": dict.fromkeys(dimensions, 0),
            "annotations": None,
            "rubric": rubric,
            "rubric_version": rubric["version"],
            "warnings": warnings,
        }
        if config.get("agreement_annotations"):
            from .hanna_agreement import agreement_report
            result["agreement"] = agreement_report(artifact(inputs, config["agreement_annotations"]), inputs, rubric)
        if not config.get("annotations"):
            result.update(unavailable("HANNA requires supplied human annotations."))
            if "agreement" in result:
                result["status"] = "partial"
            result["warnings"].extend(warnings)
            return result

        annotations = load_annotations(inputs, config)
        if annotations.get("rubric_version") != rubric["version"]:
            raise ValueError("HANNA annotations require the matching rubric_version")
        if inputs.original_prompt is not None:
            prompt_sha256 = hashlib.sha256(inputs.original_prompt.encode("utf-8")).hexdigest()
            if annotations.get("prompt_sha256") != prompt_sha256:
                raise ValueError("HANNA annotations belong to a different or unbound original prompt")

        ratings = annotations.get("ratings")
        if not isinstance(ratings, list):
            raise ValueError("HANNA annotations require a ratings list")
        values: dict[str, list[int]] = {dimension: [] for dimension in dimensions}
        raters: dict[str, set[str]] = {dimension: set() for dimension in dimensions}
        seen_raters = set()
        for rating in ratings:
            required = {"annotator", "annotated_at", "scores"}
            if (not isinstance(rating, dict) or not required <= rating.keys()
                    or rating.keys() - required - {"explanations"}):
                raise ValueError("Each HANNA rating requires annotator, annotated_at and scores; "
                                 "only explanations is optional")
            for field in ("annotator", "annotated_at"):
                if not isinstance(rating[field], str) or not rating[field].strip():
                    raise ValueError(f"Each HANNA rating requires nonempty {field} provenance")
            if rating["annotator"] in seen_raters:
                raise ValueError("Each HANNA annotator may contribute only one rating per narrative")
            seen_raters.add(rating["annotator"])
            scores = rating["scores"]
            if not isinstance(scores, dict) or set(scores) != set(dimensions):
                raise ValueError("HANNA scores must contain exactly the six rubric dimensions")
            for dimension, value in scores.items():
                if value is not None and (type(value) is not int or not 1 <= value <= 5):
                    raise ValueError(f"HANNA {dimension} must be an integer from 1 to 5 or null")
                if dimension == "relevance" and inputs.original_prompt is None and value is not None:
                    raise ValueError("HANNA relevance must be null without an original prompt")
                if value is not None:
                    values[dimension].append(value)
                    raters[dimension].add(rating["annotator"])

        result["annotations"] = annotations
        result["dimension_scores"] = {
            dimension: sum(scores) / len(scores) if scores else None
            for dimension, scores in values.items()
        }
        result["dimension_counts"] = {
            dimension: len(scores) for dimension, scores in values.items()
        }
        result["status"] = "completed" if all(values.values()) else "partial"
        if any(len(raters[dimension]) != 3 or len(values[dimension]) != 3
               for dimension in dimensions):
            warnings.append("The original HANNA protocol used three distinct human raters per "
                            "story and dimension; these annotations do not match that coverage.")
        if inputs.original_prompt is None:
            warnings.append("No original prompt supplied; relevance is unavailable and remains null.")
        return result
