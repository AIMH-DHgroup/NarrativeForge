# Evaluation Runtime Extensions

This document extends [the human evaluation protocol](evaluation.md). Generation,
reorganization, their prompts, and their inference defaults are unchanged.

## RAGAS Faithfulness

The adapter uses **RAGAS 0.4.3** `ragas.metrics.collections.Faithfulness.ascore`.
Its two official steps extract self-contained statements and judge support
against documentary context. The returned score is the official supported / all
verdicts ratio. Trace hooks preserve statements and verdicts without changing
the library prompts or scoring. Missing, reordered or invalid verdicts cause a
failure instead of an inflated score. No-claim results are undefined (`uncertain`,
null score). This is independent of FActScore extraction and judgments.

Install in an evaluation environment (not an AlignScore environment):

```bash
python -m pip install -e '.[evaluation-ragas,documents]'
```

Example JSON passed to the existing `--config` option:

```json
{
  "ragas_faithfulness": {
    "model": "your-structured-output-judge",
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "http://localhost:11434/v1",
    "max_input_chars": 60000,
    "chunk_chars": 4000,
    "max_tokens": 4096,
    "timeout_seconds": 300
  }
}
```

The OpenAI-compatible judge must support Instructor JSON structured outputs.
For local servers set the key variable to a placeholder; for hosted APIs supply
credentials through the environment, never through configuration JSON. Hosted
judges receive the narrative and sources. No requests are made without an
explicit model configuration. The original prompt is required (`--original-prompt`)
unless a genuine evaluation question is supplied as metric-specific `user_input`.

All source pages are split consecutively into character-target chunks, extending
to a whitespace boundary to avoid splitting words, with zero overlap;
every character remains available. RAGAS joins the chunks in its official NLI
step. There is **no retrieval, ranking, evidence selection or silent truncation**.
The full input must fit `max_input_chars`; otherwise status is `unsupported`.
This character limit is an operational budget, not a tokenizer guarantee. The
judge context/output limits still need configuration. Supporting evidence is
the full set of source-linked contexts; RAGAS verdicts do not locate exact spans.

The adapter records the pinned library version, judge model, sampling settings,
source-linked contexts, extracted statements and verdict reasons. Judge outputs
are not deterministic guarantees or scientific validation of a domain. Repeat
controlled supported/unsupported/mixed/negation cases before interpreting results.
Requested settings are distinguished from `effective_model_parameters`: RAGAS
can force temperature 1 and remove top-p for some OpenAI reasoning models. Those
remappings are recorded, not mislabeled as temperature-zero execution. Failures
after statement extraction preserve that intermediate trace with a null score.

Reference inspected: https://github.com/vibrantlabsai/ragas/blob/v0.4.3/src/ragas/metrics/collections/faithfulness/metric.py

Offline tests mock the runtime boundary. The opt-in integration test uses
`EVALUATION_RAGAS_MODEL`, optional `EVALUATION_RAGAS_BASE_URL`, and
`OPENAI_API_KEY`; it makes actual inference calls and may incur costs.

## AlignScore: isolated reference runtime

Install separately (Python 3.10 recommended; Python 3.12/ARM64 not runtime-tested):

```bash
python3.10 -m venv .alignscore-venv
.alignscore-venv/bin/python -m pip install 'git+https://github.com/factiverse/alignscore.git@247c7f8549694111155b6b13b55bc39103280ddc'
.alignscore-venv/bin/python -m spacy download en_core_web_sm
.alignscore-venv/bin/python -m nltk.downloader punkt punkt_tab
```

Provision the trusted [trained checkpoint](https://huggingface.co/yzha/AlignScore)
and matching RoBERTa-base or RoBERTa-large backbone/tokenizer in a local directory
before evaluation. No model is downloaded by the worker. RAGAS requires
`datasets>=4`, while this AlignScore fork requires `datasets<3`; do not install
both in one environment. The fork also requires Transformers <5 and NumPy <2.

```json
{
  "alignscore": {
    "python": "/absolute/path/.alignscore-venv/bin/python",
    "checkpoint": "/absolute/path/AlignScore-base.ckpt",
    "backbone": "/absolute/path/roberta-base",
    "device": "cpu",
    "batch_size": 16,
    "timeout_seconds": 600,
    "max_input_chars": 200000
  }
}
```

The worker checks the installed distribution's `direct_url.json` Git commit,
forces offline model loading, seeds Torch/Python and requests deterministic
algorithms. The checkpoint and backbone file hashes, worker hash and runtime
versions are recorded. Pickle checkpoints are executable inputs: use trusted
weights only. Optional `checkpoint_sha256` pins an expected checkpoint hash.
The local backbone must be RoBERTa and its path must contain `roberta` (and not
`muppet`), because the pinned reference implementation chooses its architecture
from the model path. A worker-local loader override supplies this local path
explicitly to Lightning checkpoint reconstruction, initializes only from the
local config (`using_pretrained=False`), and requires a strict full weight load.
This avoids silently restoring a saved Hub model identifier or initializing
missing heads randomly; incompatible checkpoints fail explicitly.

All source contexts are concatenated in stable corpus order and scored against
the entire narrative. This is a corpus-concatenation adaptation of document-level
AlignScore, not a maximum of unrelated document scores. The official `nli_sp`
algorithm groups consecutive source sentences using whitespace word count
`//350+1`, then computes the mean over narrative sentences of the maximum
entailment across source groups. There is no overlap or retrieval. Its tokenizer
can truncate long pairs (usually 512 subword tokens); this reference limitation
remains explicitly reported. `max_input_chars` rejects oversized jobs rather
than silently selecting sources; `batch_size` does not bound every host-memory
allocation in the reference implementation.

The pinned Factiverse fork returns three per-class values. Their sum need not
be 1. The adapter retains all three and explicitly selects index 0, matching the
entailment head chosen by [original upstream AlignScore](https://github.com/yuh-zha/AlignScore/blob/main/src/alignscore/inference.py).
A scalar or unexpected vector is rejected, not silently averaged. This output
compatibility interpretation is documented, but awaits a real checkpoint test.
Missing dependencies, resources, incompatible revisions or checkpoints return
an unavailable result; subprocess failures/timeouts are explicit failures.

Opt-in integration variables: `EVALUATION_ALIGNSCORE_PYTHON`,
`EVALUATION_ALIGNSCORE_CHECKPOINT`, `EVALUATION_ALIGNSCORE_BACKBONE`. The integration
fixture checks repeatability, not published benchmark accuracy or domain validity.

## Automatic FActScore corpus adaptation

Human annotation mode is unchanged. Automatic mode is explicitly a new
`atomic-corpus-v1` adaptation, not the released biography-specific estimator.
It uses the same two-stage concept (atomic decomposition, then documentary
support) and the existing precision and gamma penalty formula. It does not use
RAGAS, AlignScore, coverage units, embeddings or keyword overlap as judgments.

```json
{
  "factscore": {
    "mode": "automatic",
    "model": "your-judge-model",
    "base_url": "http://127.0.0.1:11434/v1",
    "gamma": 10,
    "max_input_chars": 60000,
    "max_tokens": 4096,
    "timeout_seconds": 300
  }
}
```

This opt-in mode needs an OpenAI-compatible `/chat/completions` endpoint with
JSON-object output. Optional `api_key_env` names a credential environment
variable. Each metric owns its judge and trace. Evaluation-specific temperature
is zero and no frequency/presence penalties are injected; generation defaults
are not touched. The complete narrative is decomposed without silently cropping
it. Every claim is verified separately against **all** supplied source contexts;
oversized input fails clearly, rather than treating missing retrieved evidence
as proof of falsehood. There is no retrieval approximation in this increment.

The extractor must quote its narrative span; the verifier must quote supporting
or contradicting evidence and identify its source passage. Exact quotation,
offset and ID checks prevent fabricated evidence references; they cannot prove
that a cited passage actually entails the claim. Full prompts and raw responses,
claim IDs derived from narrative hash/text/span, decisions, explanations and
evidence are retained. Exact normalized duplicate claims are removed and recorded.
Empty factual sets yield undefined precision, not a perfect score. Ambiguous
claims remain in the denominator and produce uncertainty bounds and null point
scores, as in human mode. A software-completed automatic run remains unreviewed
and is not a scientific validation of extraction completeness or judge accuracy.

The released reference scorer was inspected at
https://github.com/shmsw25/FActScore/blob/f28272deffcf33efc1f1117d5479c10bb75221a9/factscore/factscorer.py
It uses biography demonstrations, a Wikipedia retrieval pipeline and binary
support decisions. This adaptation replaces those data and judge components and
adds an explicit unverifiable category. Validate against independently reviewed
claims (including contradictions, qualifications and negation) before comparisons.

## Source Coverage inventory v2 and automatic matching

Legacy frozen `source-inventory-v1` artifacts remain accepted unchanged: all their
reviewed units are implicitly relevant. New `source-inventory-v2` units require
`relevance: "relevant"` or `"not_relevant"` before freezing. Uncertain or missing
relevance cannot define an approved denominator. Normalized duplicate statements,
invalid spans, incorrect IDs and changed corpus hashes are rejected. Merge
duplicate statements' evidence when they occur in several sources. Relevant
units are unweighted; excluded units remain visible in the frozen inventory.

Generate a **proposal**, not a frozen inventory, using only a source snapshot:

```bash
python -m eventweaver.evaluation.inventory --corpus sources.json \
  --extract-config coverage-judge.json --relevance-criteria 'Construction history and conservation' \
  --output source-proposal.json
```

`coverage-judge.json` holds only the generic judge transport configuration, e.g.
`{"model":"judge","base_url":"http://localhost:11434/v1"}`. The source-only
extractor has no narrative argument. It proposes statements with exact evidence,
records all requests/responses, and leaves relevance `uncertain` and `frozen:false`.
Review and deduplicate the proposal, assign relevance, and supply `reviewed_by`
and `reviewed_at`. Then use the existing freeze command:

```bash
python -m eventweaver.evaluation.inventory source-proposal.json \
  --corpus sources.json --output approved-inventory.json
```

The command refuses existing output paths before making model requests. Never
regenerate an approved inventory per narrative. Reuse that exact file/hash for
every comparison; edits produce a new version of the experimental denominator.

Matching can still use human annotations, or opt into an independent judge:

```json
{
  "source_coverage": {
    "inventory": "approved-inventory.json",
    "mode": "automatic",
    "model": "your-judge-model",
    "base_url": "http://127.0.0.1:11434/v1"
  }
}
```

Each relevant unit is compared semantically to the complete narrative, without
reusing FActScore/RAGAS claims or decisions. Covered decisions require exact
narrative citations; partial meaning is `uncertain`, not automatically covered.
The denominator includes only approved relevant units. With all decisions,
score = covered/relevant; uncertain counts zero in the conservative score and
one in the reported upper bound. Missing decisions produce `partial` and a null
score. Zero relevant units produce an undefined score, never a perfect result.
The model trace, inventory, decisions and matching evidence are preserved.

This is a project-specific adapted metric, not a standardized benchmark.
Inventory completeness/relevance and automatic semantic matching need human
validation; copied quotations alone do not establish full meaning coverage.

## HANNA: human agreement

Human rubric, comments, provenance and independent per-dimension means remain
unchanged; no automatic HANNA ratings or global narrative-quality score are added.
Optional `hanna.agreement_annotations` references a separate JSON panel:

```json
{
  "annotator": "study-coordinator",
  "annotated_at": "2026-11-09T10:00:00Z",
  "corpus_sha256": "replace-with-corpus-hash",
  "rubric_version": "hanna-2022-table7-v1",
  "items": [
    {
      "narrative_sha256": "replace-with-narrative-hash",
      "ratings": [
        {"annotator":"r1", "annotated_at":"2026-11-09", "scores":{"coherence":4,"relevance":4,"empathy":3,"surprise":2,"engagement":4,"complexity":4}},
        {"annotator":"r2", "annotated_at":"2026-11-09", "scores":{"coherence":4,"relevance":3,"empathy":3,"surprise":2,"engagement":5,"complexity":4}}
      ]
    }
  ]
}
```

Add more independent narratives to the panel. Every narrative hash must be
unique, the current evaluated narrative must be present, and exactly two fixed
raters are supported. Scores must be 1..5 or null. The panel records hashes
and human assertions; it does not authenticate raters or verify other narratives'
contents or their prompt-relevance judgments.

Agreement is **quadratic weighted Cohen's kappa**, independently for each
dimension across narratives. With disagreement weights `(i-j)^2/16`, kappa is
`1 - observed_disagreement/expected_disagreement`, where expected disagreement
uses the two raters' empirical marginal distributions. Quadratic distance treats
larger ordinal-category disagreements more severely; the underlying quality is
not asserted to be interval-scaled. The statistic assumes two fixed raters and
independent narrative units. It is an additional analysis, not a replacement
for original HANNA's ICC-based study protocol (Cohen, 1968, weighted kappa;
https://doi.org/10.1037/h0026256).

Missing pairs are excluded separately per dimension, with counts reported.
Less than two paired narratives or zero expected disagreement gives null, not
perfect agreement. More than two raters is rejected rather than silently
selecting a pair; multi-rater Krippendorff alpha is not implemented. Six dimensions
of one narrative are never treated as six interchangeable items. No average of
agreement values is calculated. A panel-only invocation reports `partial` because
it does not supply the current narrative's dimension ratings.

## Implementation and validation status (0.2.0)

| Metric | Implemented | Software tests | Methodological validation | Remaining limits |
| --- | --- | --- | --- | --- |
| RAGAS Faithfulness | Official pinned adapter with traced extraction/verdicts | Mocked runtime and parameter mapping; opt-in real judge fixture | Reference procedure inspected; actual inference/domain calibration not run here | Structured-output judge, explicit context capacity, optional dependencies |
| AlignScore | Isolated pinned worker, local checkpoint override and explicit entailment aggregation | Subprocess contracts, local-loading override, shape/resources/budgets; opt-in repeatability fixture | Reference aggregation/source inspected; real checkpoint and benchmark comparison not run here | Separate environment and trusted weights, reference pair truncation, platform compatibility |
| FActScore | Existing human mode plus automatic two-stage corpus adaptation | Claims, duplicate suppression, evidence integrity, uncertainty, scoring | Formula/gamma checked against source; automatic judge accuracy/completeness not domain-validated | Not the reference biography estimator; opt-in supported fixture only |
| Source Coverage | Human or proposed v2 reviewed inventories; human or automatic matching | Relevance, freeze/reuse, modified corpus, exact evidence and conservative bounds | Custom metric definitions tested arithmetically, not validated as a published metric | Human inventory approval; automatic semantic judgments require calibration |
| HANNA | Human rubric and dimension scores; two-rater ordinal agreement panels | Valid/missing/invalid ratings, means and known kappa contingency tables | Original rubric preserved; kappa analytic examples checked; no original-study replication | No automatic HANNA, no multi-rater alpha, panel provenance not authenticated |

The installed test environment has no RAGAS, OpenAI SDK or AlignScore runtime.
No dependency installation or model download is performed by unit tests.
Real integrations are opt-in and must be run in provisioned environments before
claiming model-backed operational or scientific validation. Agreement arithmetic
tests do not validate a human annotation study. No combined metric is produced.

Run unit and optional integration suites:

```bash
python -m pytest -q tests/evaluation
python -m pytest -q tests/evaluation/integration
```

The second command skips unless the documented environment variables are set.
For automatic FActScore/Coverage fixtures set `EVALUATION_JSON_JUDGE_MODEL`,
optional `EVALUATION_JSON_JUDGE_BASE_URL`, and optional
`EVALUATION_JSON_JUDGE_KEY_ENV` (the name of a credential environment variable).
