"""Executed by an isolated Python; no imports of the generation pipeline."""
import contextlib
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

FACTIVERSE_REVISION = "247c7f8549694111155b6b13b55bc39103280ddc"


def make_scorer(job):
    from alignscore import AlignScore
    from alignscore import inference
    from alignscore.model import BERTAlignModel

    backbone = job["backbone"]
    config = json.loads((Path(backbone) / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "roberta" or "roberta" not in backbone or "muppet" in backbone.lower():
        raise ValueError("Reference base/large checkpoints require local RoBERTa assets in a path containing 'roberta' (not 'muppet')")

    class LocalCheckpointLoader:
        def __init__(self, model):
            if model != backbone:
                raise ValueError("Unexpected backbone override")

        def load_from_checkpoint(self, checkpoint_path, strict):
            # Override saved Hub identifiers; initialize from local config, then require all checkpoint weights.
            return BERTAlignModel.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=True,
                                                       model=backbone, using_pretrained=False)

    with patch.object(inference, "BERTAlignModel", LocalCheckpointLoader):
        return AlignScore(model=backbone, batch_size=job["batch_size"], device=job["device"],
                          ckpt_path=job["checkpoint"], evaluation_mode="nli_sp", verbose=False)


def score_job(job):
    distribution = importlib.metadata.distribution("alignscore")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if distribution.version != "0.1.5" or origin.get("vcs_info", {}).get("commit_id") != FACTIVERSE_REVISION:
        raise RuntimeError("Install Factiverse AlignScore 0.1.5 from the documented pinned Git revision")
    import nltk
    import spacy
    import torch

    nltk.sent_tokenize("Check sentence resources.")
    spacy.load("en_core_web_sm")
    random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    scorer = make_scorer(job)
    raw = scorer.score(contexts=[job["context"]], claims=[job["narrative"]])
    return {"raw_scores": raw, "implementation_revision": FACTIVERSE_REVISION,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("alignscore", "torch", "transformers", "nltk", "spacy")},
            "source_sentence_count": len(nltk.sent_tokenize(job["context"])),
            "narrative_sentence_count": len(nltk.sent_tokenize(job["narrative"]))}


def main():
    # Prevent implicit model downloads; the caller must provision all assets beforehand.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        job = json.load(sys.stdin)
        if not Path(job["checkpoint"]).is_file():
            raise ValueError("AlignScore checkpoint is missing")
        with contextlib.redirect_stdout(sys.stderr):
            result = score_job(job)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
