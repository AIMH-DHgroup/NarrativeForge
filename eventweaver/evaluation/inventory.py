"""Freeze human-extracted source units independently of any evaluated narrative."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evidence import check_evidence, read_json, require_text_fields
from .models import content_hash


def freeze_inventory(draft: dict, corpus: dict) -> dict:
    require_text_fields(draft, ("relevance_criteria", "reviewed_by", "reviewed_at"))
    version = draft.get("version", "source-inventory-v1")
    if version not in {"source-inventory-v1", "source-inventory-v2"}:
        raise ValueError("Unsupported source inventory version")
    if draft.get("corpus_sha256", corpus["sha256"]) != corpus["sha256"]:
        raise ValueError("Draft belongs to a different source corpus")
    units = draft.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("Inventory needs non-empty independently extracted source units")
    frozen_units = []
    seen_texts = set()
    for unit in units:
        if version == "source-inventory-v1" and "relevance" in unit:
            raise ValueError("Explicit relevance requires source-inventory-v2; version 1 must not silently discard it")
        if not isinstance(unit.get("text"), str) or not unit["text"].strip() or not unit.get("evidence"):
            raise ValueError("Each source information unit requires text and evidence")
        for evidence in unit["evidence"]:
            check_evidence(corpus, evidence)
        value = {"text": unit["text"], "evidence": unit["evidence"]}
        unit_id = "U-" + content_hash(value)
        if unit.get("id", unit_id) != unit_id:
            raise ValueError("Information unit ID is inconsistent with its text and evidence")
        frozen_unit = dict(value, id=unit_id)
        if version == "source-inventory-v2":
            if unit.get("relevance") not in {"relevant", "not_relevant"}:
                raise ValueError("Version 2 inventory requires reviewed relevance for every unit; uncertain proposals cannot be frozen")
            normalized = " ".join(unit["text"].split()).casefold()
            if normalized in seen_texts:
                raise ValueError("Duplicate information statement; merge its source evidence before freezing")
            seen_texts.add(normalized)
            frozen_unit["relevance"] = unit["relevance"]
        frozen_units.append(frozen_unit)
    if len({unit["id"] for unit in frozen_units}) != len(frozen_units):
        raise ValueError("Duplicate source information units")
    result = {"version": version, "corpus_sha256": corpus["sha256"],
              "relevance_criteria": draft["relevance_criteria"], "reviewed_by": draft["reviewed_by"],
              "reviewed_at": draft["reviewed_at"], "units": frozen_units, "frozen": True}
    if "preparation" in draft:
        result["preparation"] = draft["preparation"]
    return dict(result, sha256=content_hash(result))


def validate_inventory(inventory: dict, corpus: dict) -> None:
    unsigned = {key: value for key, value in inventory.items() if key != "sha256"}
    if inventory.get("sha256") != content_hash(unsigned) or inventory.get("frozen") is not True:
        raise ValueError("Inventory is not frozen or its hash has changed")
    if inventory.get("corpus_sha256") != corpus["sha256"]:
        raise ValueError("Inventory belongs to another source corpus")
    rebuilt = freeze_inventory(inventory, corpus)
    if rebuilt != inventory:
        raise ValueError("Inventory schema or stable unit IDs do not match")


def main(argv=None):
    from .runner import write_json

    parser = argparse.ArgumentParser(description="Freeze a reviewed source-only inventory against a corpus snapshot")
    parser.add_argument("draft", nargs="?", type=Path)
    parser.add_argument("--corpus", required=True, type=Path, help="sources.json from an evaluation run")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extract-config", type=Path, help="Propose a source-only v2 inventory using this judge configuration")
    parser.add_argument("--relevance-criteria", help="Required relevance policy for source-only extraction")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; frozen inventories are never silently regenerated")
    if args.extract_config:
        from .metrics.source_coverage_automatic import prepare_inventory
        if args.draft:
            parser.error("Do not supply a reviewed draft with --extract-config")
        inventory = prepare_inventory(read_json(str(args.corpus)), args.relevance_criteria, read_json(str(args.extract_config)))
    else:
        if args.relevance_criteria:
            parser.error("--relevance-criteria is only used with --extract-config; edit reviewed draft criteria before freezing")
        if not args.draft:
            parser.error("Supply a reviewed draft, or --extract-config to prepare a proposal")
        inventory = freeze_inventory(read_json(str(args.draft)), read_json(str(args.corpus)))
    write_json(args.output, inventory)


if __name__ == "__main__":
    main()
