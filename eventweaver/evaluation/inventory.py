"""Freeze human-extracted source units independently of any evaluated narrative."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evidence import check_evidence, read_json, require_text_fields
from .models import content_hash


def freeze_inventory(draft: dict, corpus: dict) -> dict:
    require_text_fields(draft, ("relevance_criteria", "reviewed_by", "reviewed_at"))
    units = draft.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("Inventory needs non-empty independently extracted source units")
    frozen_units = []
    for unit in units:
        if not isinstance(unit.get("text"), str) or not unit["text"].strip() or not unit.get("evidence"):
            raise ValueError("Each source information unit requires text and evidence")
        for evidence in unit["evidence"]:
            check_evidence(corpus, evidence)
        value = {"text": unit["text"], "evidence": unit["evidence"]}
        frozen_units.append(dict(value, id="U-" + content_hash(value)))
    if len({unit["id"] for unit in frozen_units}) != len(frozen_units):
        raise ValueError("Duplicate source information units")
    result = {"version": "source-inventory-v1", "corpus_sha256": corpus["sha256"],
              "relevance_criteria": draft["relevance_criteria"], "reviewed_by": draft["reviewed_by"],
              "reviewed_at": draft["reviewed_at"], "units": frozen_units, "frozen": True}
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
    parser.add_argument("draft", type=Path)
    parser.add_argument("--corpus", required=True, type=Path, help="sources.json from an evaluation run")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inventory = freeze_inventory(read_json(str(args.draft)), read_json(str(args.corpus)))
    write_json(args.output, inventory)


if __name__ == "__main__":
    main()
