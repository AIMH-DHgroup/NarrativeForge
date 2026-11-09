from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class EvaluationInput:
    narrative: dict[str, Any]
    corpus: dict[str, Any]
    original_prompt: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
