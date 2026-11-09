import math

from ..base import unavailable
from ..evidence import check_evidence, check_span, load_annotations


class FactScoreEvaluator:
    reference = "https://github.com/shmsw25/FActScore/tree/f28272deffcf33efc1f1117d5479c10bb75221a9"
    method = "Human atomic-fact precision against project corpus; explicit FActScore adaptation"

    def evaluate(self, inputs, config):
        if not config.get("annotations"):
            return unavailable("Human-reviewed atomic claims and evidence required; automatic decomposition is not implemented")
        annotations = load_annotations(inputs, config)
        claims = annotations.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError("Provide a non-empty atomic claim inventory; no-claim precision is undefined")
        gamma = config.get("gamma", 10)
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma < 0:
            raise ValueError("FActScore gamma must be finite and nonnegative")
        identifiers = set()
        claim_spans = set()
        counts = dict.fromkeys(("supported", "unsupported", "unverifiable"), 0)
        for claim in claims:
            if not claim.get("id") or claim["id"] in identifiers or not claim.get("text", "").strip():
                raise ValueError("Claims require unique IDs and non-empty atomic text")
            identifiers.add(claim["id"])
            check_span(inputs.narrative["text"], claim["narrative_span"])
            identity = (" ".join(claim["text"].split()), claim["narrative_span"]["start"],
                        claim["narrative_span"]["end"])
            if identity in claim_spans:
                raise ValueError("Duplicate atomic claim at the same narrative span")
            claim_spans.add(identity)
            verdict = claim.get("verdict")
            if verdict not in counts:
                raise ValueError("Claim verdict must be supported, unsupported or unverifiable")
            evidence = claim.get("evidence", [])
            if verdict == "supported" and not evidence:
                raise ValueError("Supported claims require documentary evidence")
            if not claim.get("explanation"):
                raise ValueError("Each claim needs an explanation, including searches without evidence")
            for item in evidence:
                check_evidence(inputs.corpus, item)
            counts[verdict] += 1
        total = len(claims)
        penalty = math.exp(1 - gamma / total) if total < gamma else 1.0
        lower = counts["supported"] / total
        upper = (counts["supported"] + counts["unverifiable"]) / total
        reviewed = annotations.get("decomposition_reviewed") is True
        status = "partial" if not reviewed else "uncertain" if counts["unverifiable"] else "completed"
        return {
            "status": status,
            "score": lower * penalty if status == "completed" else None,
            "raw_precision": lower if status == "completed" else None,
            "precision_bounds": [lower, upper] if reviewed else None,
            "length_penalty": penalty, "gamma": gamma, "counts": counts,
            "denominator": total, "annotations": annotations,
            "warnings": [
                "Human corpus-relative adaptation, not the reference automatic biography estimator.",
                "Unsupported means contradicted or not supported by this corpus, not globally false.",
                "Unverifiable claims remain in the denominator; only bounds are reported when present.",
                "Completeness and atomicity of decomposition require human review; irrelevant claims are not silently excluded.",
            ],
        }
