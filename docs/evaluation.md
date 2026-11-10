# Independent Narrative Evaluation

## Scope

`eventweaver evaluate` evaluates an existing Markdown narrative against an explicit source corpus without generating or reorganizing it. Evaluation implementation version **0.2.0** preserves the human workflows described below and adds opt-in model-backed adapters and ordinal agreement. See [runtime extensions](evaluation-runtime.md) for current automatic-mode configuration, dependency isolation, assumptions, integration tests and scientific-validation limitations.

The metrics answer different questions and remain separate:

| Metric ID | Question | Implemented method |
| --- | --- | --- |
| `hanna` | How do humans judge the story's qualities? | Means of supplied human ratings, separately for six HANNA-derived dimensions |
| `factscore` | What fraction of narrative atomic claims does this corpus support? | Human-reviewed claim inventory and documentary evidence, with an explicit length penalty |
| `source_coverage` | How much independently identified relevant source information appears in the narrative? | Human matching against a reviewed, frozen source-only inventory |
| `ragas_faithfulness` | Are the response's claims inferable from documentary context? | Optional official RAGAS 0.4.3 adapter; independent statement/verdict trace |
| `alignscore` | What factual-consistency score does the reference alignment model produce? | Isolated pinned Factiverse runtime with explicit original entailment component |

There is no cross-metric aggregate, weighted quality index, or invented substitute for an unavailable metric. In particular, HANNA has no composite score. Precision and coverage are not interchangeable: a narrative can accurately state a small subset of the sources and still omit most relevant information.

## Architecture

The entry point is `eventweaver/evaluation/cli.py`, registered by the main application CLI. The implementation is divided into:

- `sources.py`: load and snapshot the narrative and complete extracted corpus, with content identities.
- `models.py`: `EvaluationInput` and deterministic JSON content hashing.
- `config.py` and `defaults.json`: known metric IDs, configuration allowlists, and defaults.
- `runner.py`: create a new run directory, record provenance, isolate evaluator failures, and write separate JSON results.
- `evidence.py`: validate annotation bindings and exact text spans.
- `inventory.py`: freeze and validate a human-extracted source-information inventory independently of narrative matching.
- `rubrics/hanna-v1.json`: versioned HANNA anchors and protocol provenance.
- `metrics/`: independent evaluators following the `Evaluator` protocol in `base.py`.

Each evaluator receives a deep copy of the same `EvaluationInput` and its own configuration. No metric receives another metric's result. FActScore claims are not supplied to RAGAS, and coverage units are not derived from narrative claims. These are software isolation properties, not proof that human reviewers worked independently.

## Running Evaluation

From the project environment, use either the installed `eventweaver` command or `python -m eventweaver`:

```console
python -m eventweaver evaluate story.md --sources corpus --metrics hanna,factscore,source_coverage --config evaluation-config.json --original-prompt prompt.txt --narrative-id story-a --output-dir evaluations/review-001
```

`--sources` accepts one or more supported files or directories, recursively. `--metrics` is a comma-separated list of distinct IDs, with no spaces between IDs. Omitting it selects all five metrics; each must have its required configuration, annotations or runtime assets for a fully completed run.

The configuration file is a JSON object keyed directly by metric ID, not the internal defaults file's `schema_version`/`metrics` wrapper:

```json
{
  "hanna": {
    "annotations": "annotations/hanna.json",
    "rubric_version": "hanna-2022-table7-v1"
  },
  "factscore": {
    "annotations": "annotations/claims.json",
    "gamma": 10
  },
  "source_coverage": {
    "inventory": "frozen.json",
    "annotations": "annotations/coverage.json"
  },
  "ragas_faithfulness": {},
  "alignscore": {}
}
```

All relative paths, including annotation and inventory paths inside the configuration, resolve from the process's **current working directory**, not from the configuration file's directory. `--original-prompt` names a UTF-8 text file, not a literal prompt. `--narrative-id` is an optional label; otherwise the Markdown filename stem is used. Annotation binding uses content hashes, not that label.

Unknown metric IDs, duplicate selections, unknown configuration fields, and non-object metric configurations are rejected. The example above describes human-mode configuration; [runtime extensions](evaluation-runtime.md) documents additional allowed settings. The rubric version and `gamma` default to the values shown. Human mode rejects judge settings instead of silently ignoring them.

### Statuses and Exit Codes

The result vocabulary is exactly `completed`, `partial`, `uncertain`, `unsupported`, and `failed`. There is no `incomplete` or `unavailable` status string. Incomplete or unavailable work is represented by the applicable statuses below, never by a fabricated zero.

| Metric | Exact status rule | Numeric output |
| --- | --- | --- |
| HANNA | `unsupported` without an annotations path; `completed` if every dimension has at least one non-null rating; otherwise `partial` | Per-dimension means/counts; top-level `score` always `null` |
| FActScore adaptation | `unsupported` without an annotations path; `partial` unless `decomposition_reviewed` is exactly `true`; then `uncertain` if any claim is `unverifiable`, otherwise `completed` | `score` and `raw_precision` only for `completed`; reviewed decompositions have precision bounds |
| Source coverage | `unsupported` without an inventory or matching annotations; `partial` if any unit has no decision; otherwise `uncertain` if any decision is uncertain, else `completed` | `score` is `null` while partial; with all decisions present it is conservative coverage, even when uncertain; bounds are reported |
| RAGAS Faithfulness | `unsupported` without configured runtime; `completed` on valid official results; `uncertain` for no claims; explicit failures otherwise | Official verdict ratio or null |
| AlignScore | `unsupported` without compatible environment/assets; `completed` on validated reference output; explicit failures otherwise | Entailment component or null |
| Any evaluator | An exception, including invalid supplied annotations, becomes `failed` in the runner | `score: null`, with an `error` string |

`completed` is an operational status, not certification of scientific protocol compliance. For example, one HANNA rater can supply all six dimensions and produce `completed`, with a warning about missing three-rater coverage.

The CLI prints one status per requested metric and the output directory. It returns `0` only if **every requested metric** is `completed`; otherwise it returns `1`. Ordinary input/configuration errors handled by the CLI also return `1`. Argument-parser usage errors follow argparse's normal behavior. Per-metric failures do not prevent the other evaluators from running. Errors before evaluation, such as unreadable sources, can stop the run before any output directory is created; file-writing failures can leave an incomplete directory.

Every run requires a **new output directory**, even after a failed run. Existing directories are rejected, and JSON files are created exclusively rather than overwritten. Use a new directory for each review iteration.

## Source and Evidence Contract

The narrative must be an existing `.md` file with a nonempty body. It is decoded as UTF-8 with an optional BOM removed (`utf-8-sig`). If the first line is an ATX Markdown heading, that line is excluded from the stored `body`, but the complete decoded `text`, `body_start`, and raw-file SHA-256 are retained. Evidence always addresses the full `text`, including the heading, not the body alone.

Source discovery supports PDF, DOCX, TXT, MD, and CSV. Missing explicit paths and unsupported explicit file formats are errors. Directory discovery includes supported extensions and skips temporary names starting with `~$`. Overlapping paths are deduplicated; separate files with identical raw content are rejected as duplicate source content. The evaluated narrative cannot be in its own source corpus, including through recursive directory selection.

Evaluation reuses the document loaders with **OCR disabled and no OCR cache**. It retains all extracted pages or rows, with no retrieval or context truncation. A document with no extracted pages, or with any extracted page containing only whitespace, is rejected; provide separately reviewed extracted text when necessary. This strict check does not prove extraction quality. Native PDF text can still be wrong, and text embedded only in images is not recovered.

The snapshot reflects the loader's representation, not a lossless transcription of every original layout feature:

- PDF uses native extracted text, one entry per PDF page.
- DOCX preserves body paragraph/table order as text in a single entry, not physical Word pagination.
- TXT and MD use UTF-8-sig decoded text in a single entry.
- CSV uses the header to label each retained nonblank data row; blank rows are skipped by the loader. Each entry records its original row number.

`sources.json` records every extracted document and page entry, including `path`, `format`, `index`, `page_or_row`, `text`, and `extraction_method`. It is a complete snapshot of the **parsed corpus used by evaluation**, not a copy of the original binary files. Preserve the original files separately for re-extraction.

Identity and evidence rules are:

- Narrative `sha256` hashes the original Markdown bytes, including any BOM and original line endings.
- A document ID is `D-` plus its raw-file SHA-256. Its `text_sha256` hashes its parsed page records.
- Documents are sorted by ID. Corpus `sha256` hashes the list of document IDs and parsed-text hashes; corpus ID is `C-` plus that hash. Renaming unchanged files does not change corpus identity; changing raw bytes or extraction results does.
- Structured hashes use `content_hash`: SHA-256 of UTF-8 encoded `json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)` with Python's default separators. Do not hash pretty-printed JSON as a replacement.
- A source evidence object requires `document_id`, zero-based `page_index`, `start`, `end`, and `quote`. `page_index` indexes `pages`, not the human-facing PDF page or CSV row number.
- A narrative span requires `start`, `end`, and `quote` against the complete stored narrative `text`.
- Offsets count **Unicode code points**, with an inclusive start and exclusive end. They are not UTF-8 byte offsets or UTF-16 code-unit offsets. For extracted sources, offsets address the stored page text, not the original PDF/DOCX/CSV bytes.
- Bounds must satisfy `0 <= start < end <= len(text)`, and `text[start:end]` must equal `quote` exactly. No whitespace normalization or fuzzy matching occurs.

Use the snapshots as the authority for offsets, especially when decoding, line endings, non-ASCII characters, or document extraction affect the text. Raw-byte hashes and parsed-text hashes serve different purposes.

## Human Annotation Files

All three implemented human workflows require a JSON annotation object bound to both `narrative_sha256` and `corpus_sha256`, plus nonempty `annotator` and `annotated_at` provenance. JSON non-finite constants are rejected. Provenance fields record assertions by the submitter; timestamps and reviewer identities are not authenticated.

The examples below are complete JSON objects for each artifact type, but use **short illustrative hash placeholders** such as `narrative-hash`, `corpus-hash`, `D-source-hash`, and `U-unit-hash`. Replace them with the full hashes/IDs from the actual run or frozen inventory. They are not runnable hash values. No extra ellipses or omitted required fields are hidden in the examples.

For the span examples, assume the narrative text is exactly `# History\n\nThe church was built in 1200.` and source page 0 is exactly `The church was built in 1200.`, with no trailing newline. The narrative claim occupies `[11, 40)` and source evidence `[0, 29)`. Recalculate spans for your real snapshots.

### HANNA Ratings

`annotations/hanna.json`:

```json
{
  "narrative_sha256": "narrative-hash",
  "corpus_sha256": "corpus-hash",
  "annotator": "study-curator",
  "annotated_at": "2026-11-09T10:00:00Z",
  "rubric_version": "hanna-2022-table7-v1",
  "prompt_sha256": "prompt-hash",
  "ratings": [
    {
      "annotator": "rater-1",
      "annotated_at": "2026-11-09T09:00:00Z",
      "scores": {
        "coherence": 4,
        "relevance": 5,
        "empathy": 2,
        "surprise": 2,
        "engagement": 3,
        "complexity": 2
      },
      "explanations": {
        "coherence": "The chronology is understandable."
      }
    }
  ]
}
```

The annotation must declare the matching `rubric_version`. Each rating must contain exactly `annotator`, `annotated_at`, and `scores`, with only `explanations` optional. Both per-rating provenance values must be nonempty strings. Each rater may contribute only once. `scores` must contain exactly the six dimension keys shown. Each value must be an integer from 1 through 5 or `null`; booleans are not ratings. Add separate rating objects for additional human raters. The example deliberately has one rater and will trigger the three-rater-coverage warning.

With an original prompt supplied, `prompt_sha256` must equal SHA-256 of the UTF-8 encoded prompt string loaded by the CLI. This binds ratings to the prompt actually passed to evaluation, not merely a filename. Without an original prompt, every relevance rating must be `null`; missing relevance then makes HANNA `partial`. Other dimensions can still be reported. Prompt relevance is not source coverage.

The canonical rubric is [`eventweaver/evaluation/rubrics/hanna-v1.json`](../eventweaver/evaluation/rubrics/hanna-v1.json), version **`hanna-2022-table7-v1`**. It contains the exact, verbatim Table 7 anchor text for all five Likert levels in each dimension, without numeric labels in the anchor strings and with original punctuation retained. Read those anchors rather than replacing them with generic labels such as "poor" and "excellent". The entire rubric is embedded in the HANNA result.

The dimensions assess whether the story makes sense (`coherence`), matches its original prompt (`relevance`), conveys characters' emotions (`empathy`), has a surprising ending (`surprise`), engages the reader (`engagement`), and has an elaborate setting/story (`complexity`). Arithmetic means and counts are computed **separately per dimension**, ignoring null ratings for that dimension. A dimension with no ratings remains null. There is no average across the six dimensions, no imputation, and no automatic HANNA judge.

This is a **HANNA-derived human adaptation**, not scores from the original HANNA dataset or an exact replication of its study. The original protocol used three distinct raters per story/dimension and showed a human story alongside the subject story to calibrate judgment. It also specified worker screening and quality controls. The bundled rubric records those details and the instructions that prompt mismatch affects only relevance and that an abruptly cut final sentence should be ignored when judging the story's ending. This implementation warns about calibration/screening not being verified and about deviations from three distinct ratings per dimension; it does not enforce the original recruitment, timing, calibration, or quality-control protocol.

### Atomic Claims

`annotations/claims.json`:

```json
{
  "narrative_sha256": "narrative-hash",
  "corpus_sha256": "corpus-hash",
  "annotator": "fact-reviewer",
  "annotated_at": "2026-11-09T10:00:00Z",
  "decomposition_reviewed": true,
  "claims": [
    {
      "id": "claim-1",
      "text": "The church was built in 1200.",
      "narrative_span": {
        "start": 11,
        "end": 40,
        "quote": "The church was built in 1200."
      },
      "verdict": "supported",
      "evidence": [
        {
          "document_id": "D-source-hash",
          "page_index": 0,
          "start": 0,
          "end": 29,
          "quote": "The church was built in 1200."
        }
      ],
      "explanation": "The source explicitly gives this construction date."
    }
  ]
}
```

Humans must decompose the narrative into a nonempty list of atomic claims and review completeness and atomicity. Every claim requires a unique nonempty `id`, nonempty `text`, valid `narrative_span`, a verdict, and a nonempty explanation. Verdicts are exactly `supported`, `unsupported`, or `unverifiable`. Supported claims require at least one valid documentary evidence span. Other verdicts may use an empty evidence list, but explanations must describe the contradiction, missing support/search, or uncertainty. Every supplied source span is validated regardless of verdict. A valid span does not itself establish entailment.

The adaptation is relative to the supplied project corpus, unlike the reference automatic biography estimator and its retrieval/model pipeline. `unsupported` means contradicted or not supported by this corpus, not globally false. In human mode, extraction and judgments come from the reviewed ledger. The opt-in automatic extractor and verifier are documented in the runtime guide; neither mode can prove decomposition completeness. Irrelevant claims are not silently removed from the denominator.

For `n` total claims and `s` supported claims:

```text
raw_precision = s / n
length_penalty = exp(1 - gamma / n)  if n < gamma
                 1                 otherwise
score = raw_precision * length_penalty
```

The default is `gamma = 10`, matching the pinned reference scorer's default. `gamma` must be finite and nonnegative; `gamma = 0` disables the penalty. At `n = gamma`, the penalty is 1. For example, one supported claim with `gamma = 10` has raw precision 1 but penalized score `exp(-9)`. An empty claim list is invalid: no-claim precision is undefined, not zero or perfect.

In human mode, `decomposition_reviewed` must be exactly `true` for a reviewed result. Otherwise status is `partial`, and `score`, `raw_precision`, and `precision_bounds` are null. With reviewed decomposition and `u` unverifiable claims, bounds are `[s/n, (s+u)/n]`; these are **raw precision bounds**, not penalty-adjusted scores. If `u > 0`, status is `uncertain`, both point estimates remain null, and unverifiable claims stay in the denominator. A completed result has no unverifiable claims and reports raw precision, penalty, and penalized score separately, alongside verdict counts and the full annotations. Automatic mode uses the same arithmetic on machine judgments but remains explicitly unreviewed.

### Source Inventory

Coverage starts with **independent human source extraction**, not the narrative being scored. Define relevance criteria, identify source information units, attach exact source evidence, review relevance/completeness, and freeze the inventory before matching narratives. Freeze/review is a required human workflow, not automated extraction.

`draft.json`:

```json
{
  "relevance_criteria": "Include source facts about church construction and dates.",
  "reviewed_by": "source-curator",
  "reviewed_at": "2026-11-09T08:00:00Z",
  "units": [
    {
      "text": "The church was built in 1200.",
      "evidence": [
        {
          "document_id": "D-source-hash",
          "page_index": 0,
          "start": 0,
          "end": 29,
          "quote": "The church was built in 1200."
        }
      ]
    }
  ]
}
```

Obtain a corpus snapshot with a preliminary evaluation, even before annotations exist:

```console
python -m eventweaver evaluate story.md --sources corpus --metrics source_coverage --output-dir evaluations/snapshot-001
```

That run writes `evaluations/snapshot-001/sources.json`, reports `unsupported`, and returns `1` because the inventory is not supplied. Its narrative is not an input to inventory freezing; reviewers should work only from the sources and predetermined relevance criteria. Use the resulting snapshot as `sources.json` in the following command, or substitute its actual path:

```console
python -m eventweaver.evaluation.inventory draft.json --corpus sources.json --output frozen.json
```

The freeze command validates source spans, requires nonempty relevance/review provenance and a nonempty unit list, rejects duplicate content-identical units, assigns content-derived IDs, and writes a new file exclusively. Its output parent directory must already exist.

The full frozen artifact shape is:

```json
{
  "version": "source-inventory-v1",
  "corpus_sha256": "corpus-hash",
  "relevance_criteria": "Include source facts about church construction and dates.",
  "reviewed_by": "source-curator",
  "reviewed_at": "2026-11-09T08:00:00Z",
  "units": [
    {
      "text": "The church was built in 1200.",
      "evidence": [
        {
          "document_id": "D-source-hash",
          "page_index": 0,
          "start": 0,
          "end": 29,
          "quote": "The church was built in 1200."
        }
      ],
      "id": "U-unit-hash"
    }
  ],
  "frozen": true,
  "sha256": "inventory-hash"
}
```

A unit ID is `U-` plus `content_hash({"text": ..., "evidence": ...})`. Inventory `sha256` hashes the complete frozen object excluding its own `sha256` field, including review metadata, corpus binding, unit order, and `frozen: true`. Validation checks the hash and corpus, then rebuilds the frozen artifact and compares it exactly, including version/schema and unit IDs. Treat the frozen artifact as immutable; a changed inventory is a new experimental denominator, not an edit to an existing comparison.

Reuse the **same frozen inventory** across narratives evaluated against the same corpus. Unit IDs do not depend on the narrative or source paths; unchanged text/evidence retain their IDs. Changing evidence text, offsets, or evidence order changes a unit ID; changing inventory metadata/order can change the inventory hash without changing unit IDs. Hashes detect changes relative to an expected artifact, not malicious or mistaken review: someone can create a different valid frozen artifact. Neither hashing nor a `reviewed_by` field proves independence, relevance, completeness, or correctness.

### Coverage Decisions

`annotations/coverage.json`:

```json
{
  "narrative_sha256": "narrative-hash",
  "corpus_sha256": "corpus-hash",
  "annotator": "coverage-reviewer",
  "annotated_at": "2026-11-09T10:00:00Z",
  "inventory_sha256": "inventory-hash",
  "decisions": [
    {
      "unit_id": "U-unit-hash",
      "verdict": "covered",
      "narrative_evidence": [
        {
          "start": 11,
          "end": 40,
          "quote": "The church was built in 1200."
        }
      ],
      "explanation": "The narrative retains the construction date."
    }
  ]
}
```

The annotation must bind the frozen inventory's exact `sha256`. Each decision references a known, unique unit ID, uses `covered`, `not_covered`, or `uncertain`, and includes an explanation. `covered` requires at least one narrative evidence span. Other verdicts can omit spans or supply an empty list. Every supplied narrative span is validated.

Let `N` be all frozen units, `c` covered units, `q` uncertain units, and `m` units without decisions. Coverage bounds are `[c/N, (c+q+m)/N]`. Missing decisions produce `partial` and `score: null`. With every unit decided, the score is `c/N`; any uncertain decision produces `uncertain` but retains that conservative score and the bounds. Otherwise the result is `completed`. Counts, denominator, unreviewed count, full inventory, and decisions are retained. This is project-specific source coverage, not a standardized benchmark or an automatic semantic-matching metric.

## Reference Automated Metrics

### RAGAS Faithfulness

The methodological reference is RAGAS, *Automated Evaluation of Retrieval Augmented Generation*, arXiv `2309.15217`, EACL 2024. The official [v0.4.3 Faithfulness documentation](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/faithfulness/) defines:

```text
faithfulness = number of response claims supported by retrieved context
               / total number of response claims
```

That method independently decomposes the response into claims and checks whether each can be inferred from the supplied context. The documented collections API takes `user_input`, `response`, and `retrieved_contexts`, with an evaluator LLM. It is not equivalent to reusing the human FActScore verdict list merely because both involve a precision-like fraction.

The optional adapter runs the official **0.4.3** collections implementation, independently of FActScore. It preserves statement/verdict traces and records requested versus effective provider settings. Missing prerequisites report `unsupported`; no proxy scores are computed. Real inference remains an opt-in integration test, not a claim of validated cultural-heritage accuracy.

The adapter uses every source context and applies an explicit full-input character budget, rejecting oversized cases without dropping evidence. This operational budget is not a model-tokenizer guarantee. Runtime configuration and limitations are documented in the extensions guide.

### AlignScore

The original method is *AlignScore: Evaluating Factual Consistency with a Unified Alignment Function*, ACL 2023, [paper](https://aclanthology.org/2023.acl-long.634/) and [original repository](https://github.com/yuh-zha/AlignScore). The original scalar scoring contract must be distinguished from the [Factiverse fork pinned at `247c7f8549694111155b6b13b55bc39103280ddc`](https://github.com/factiverse/alignscore/tree/247c7f8549694111155b6b13b55bc39103280ddc), which is the local adapter's reference.

At that fork revision, `nli_sp` retains all three NLI class scores and aggregates them into a three-element output per context/claim pair, unlike the original scalar entailment/alignment contract. A return annotation suggesting `List[float]` does not resolve that behavior. Selecting a class, averaging a vector, or replacing it with embedding similarity would be an additional methodological choice, not a faithful scalar adapter by default.

The runtime dependencies also conflict: the pinned fork requires `datasets>=2.3.2,<3`, whereas [RAGAS v0.4.3](https://github.com/vibrantlabsai/ragas/blob/v0.4.3/pyproject.toml) requires `datasets>=4.0.0`. These requirements cannot be satisfied in a single environment as declared. They are research findings, not dependencies installed by this evaluation increment.

AlignScore runs in a controlled isolated subprocess with preprovisioned checkpoint and local RoBERTa assets. It retains the fork's full three-class output and explicitly selects index 0 to match original entailment aggregation. Missing assets or incompatible environments report `unsupported`; no embedding substitute or automatic checkpoint download is used. This adapter awaits real checkpoint integration validation.

## Artifacts and Reproducibility

A normal run writes:

```text
evaluations/review-001/
  evaluation_metadata.json
  sources.json
  hanna.json
  factscore.json
  source_coverage.json
```

Only requested metric result files are written. Selecting the unavailable metrics additionally produces their separate JSON files. Metadata and corpus snapshots are shared artifacts, not aggregate scores.

`evaluation_metadata.json` records an experiment UUID, UTC timestamp, implementation version and SHA-256, Python version, recorded dependency versions, full narrative snapshot, corpus identity, effective configuration, original prompt, requested metric IDs, and `config_schema_version: "evaluation-config-v1"`. Versions include `eventweaver`, `python-docx`, `PyMuPDF`, `ragas`, `alignscore`, `openai` and `instructor`; absent distributions have null versions. Isolated AlignScore worker versions are recorded separately. This is not a complete environment lockfile, and an installed package version does not mean its metric ran.

The implementation hash covers sorted `.py` and `.json` files under `eventweaver/evaluation`, combining each relative path, a null separator, and raw contents. It does not cover the entire repository or shared ingestion implementation. Each selected metric records `input_artifacts` with paths, raw-byte hashes and decoded annotation/inventory snapshots. Each artifact is read once for both hashing and evaluation, so concurrent edits cannot change the consumed snapshot. Missing or malformed artifacts fail only their metric. Unselected metrics' artifacts are not opened. Effective configuration is embedded rather than represented by a raw config-file hash.

Each metric result carries its method/reference, experiment identity, timestamp, implementation identity, narrative/corpus hashes and IDs, metric configuration, dependency versions, score/status/warnings, and metric-specific details. Evaluator exceptions are recorded as `failed` with their type/message. Input snapshots and embedded accepted annotations allow inspection of exactly what was evaluated, but the runner is not a transactional or authenticated evidence store. Keep inputs unchanged during execution and preserve outputs and original source files together.

For comparisons, retain the same parsed corpus and frozen coverage inventory, track prompts/rubric/gamma/reviewer procedures, and preserve the environment separately if exact re-extraction matters. Parser upgrades can change parsed hashes and invalidate existing annotations. Hashes bind artifacts to bytes/parsed content and reveal mismatches; they cannot prove factual correctness, honest reviewers, atomic decomposition completeness, calibration, or independence of source extraction from a narrative.

**Data sensitivity:** outputs contain the complete decoded narrative, full extracted source text, source paths, original prompt, annotation explanations, and reviewer provenance. They can expose confidential or personal information even when metric scores are null or execution returns `1`. Restrict access and retention accordingly. This increment does not automatically redact or encrypt snapshots, and raw hashes are not anonymization.

## Scientific References

- Chhun, Colombo, Suchanek, and Clavel. *Of Human Criteria and Automatic Metrics: A Benchmark of the Evaluation of Story Generation*. COLING 2022, [arXiv:2208.11646](https://arxiv.org/abs/2208.11646). Anchor source: [v4, Table 7](https://arxiv.org/html/2208.11646v4#A2.T7). Original [HANNA repository at `bf4a171e01224390d0fecf96b829fd1baa5da1f4`](https://github.com/dig-team/hanna-benchmark-asg/tree/bf4a171e01224390d0fecf96b829fd1baa5da1f4).
- Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. [arXiv:2305.14251](https://arxiv.org/abs/2305.14251). Reference [repository at `f28272deffcf33efc1f1117d5479c10bb75221a9`](https://github.com/shmsw25/FActScore/tree/f28272deffcf33efc1f1117d5479c10bb75221a9), specifically `factscore/factscorer.py` for the default gamma and exponential penalty. The local implementation adapts the scoring arithmetic to supplied human corpus judgments; it does not reproduce the reference estimator.
- Es, James, Espinosa-Anke, and Schockaert. *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. EACL 2024, [arXiv:2309.15217](https://arxiv.org/abs/2309.15217). Operational research reference: [official Faithfulness documentation v0.4.3](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/faithfulness/).
- Zha, Yang, Li, and Hu. *AlignScore: Evaluating Factual Consistency with a Unified Alignment Function*. ACL 2023, [ACL Anthology 2023.acl-long.634](https://aclanthology.org/2023.acl-long.634/). Fork behavior/dependency research: [Factiverse revision `247c7f8549694111155b6b13b55bc39103280ddc`](https://github.com/factiverse/alignscore/tree/247c7f8549694111155b6b13b55bc39103280ddc), especially `src/alignscore/inference.py` and `pyproject.toml`.
