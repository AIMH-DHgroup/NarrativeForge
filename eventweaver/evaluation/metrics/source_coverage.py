from ..base import unavailable
from ..evidence import artifact, check_span, load_annotations
from ..inventory import validate_inventory


class SourceCoverageEvaluator:
    reference = "NarrativeForge source-coverage-v1; project-specific metric, not a standardized benchmark"
    method = "Human matching against a frozen, source-only, reviewed relevant-information inventory"

    def evaluate(self, inputs, config):
        if not config.get("inventory"):
            return unavailable("Requires an independently extracted, reviewed and frozen source inventory")
        inventory = artifact(inputs, config["inventory"])
        validate_inventory(inventory, inputs.corpus)
        if not config.get("annotations"):
            return dict(unavailable("Frozen inventory available; human matching decisions required"), inventory=inventory)
        annotations = load_annotations(inputs, config)
        if annotations.get("inventory_sha256") != inventory["sha256"]:
            raise ValueError("Coverage decisions belong to another frozen inventory")
        units = {unit["id"] for unit in inventory["units"]}
        decisions = annotations.get("decisions", [])
        seen = set()
        counts = {"covered": 0, "not_covered": 0, "uncertain": 0}
        for decision in decisions:
            unit_id, verdict = decision.get("unit_id"), decision.get("verdict")
            if unit_id not in units or unit_id in seen or verdict not in counts:
                raise ValueError("Unknown/duplicate coverage unit or invalid verdict")
            seen.add(unit_id)
            spans = decision.get("narrative_evidence", [])
            if verdict == "covered" and not spans:
                raise ValueError("Covered information units require narrative evidence")
            for span in spans:
                check_span(inputs.narrative["text"], span)
            if not decision.get("explanation"):
                raise ValueError("Coverage decisions require explanations")
            counts[verdict] += 1
        total = len(units)
        lower = counts["covered"] / total
        upper = (counts["covered"] + counts["uncertain"] + total - len(seen)) / total
        complete = seen == units
        return {"status": "partial" if not complete else "uncertain" if counts["uncertain"] else "completed",
                "score": lower if complete else None, "coverage_bounds": [lower, upper],
                "counts": counts, "denominator": total, "unreviewed": total - len(seen),
                "inventory": inventory, "annotations": annotations,
                "warnings": ["Project-specific human metric. Uncertain units count as not covered in the conservative score; the upper bound counts them as covered.",
                             "Inventory relevance and completeness require review; no narrative-informed inventory generation is performed."]}
