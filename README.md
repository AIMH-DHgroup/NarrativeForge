# EventWeaver 3.10.0 - narratives directly from documents

One command, `generate`, reads the documents, automatically identifies **up to 5 themes**, and produces one narrative per theme using the original text. Titles, themes, and narratives are requested in **English** by default, even when the documents use other languages.

There is no textual-evidence extraction phase. The system does not build collections of quotations or atomic facts for writing. Final narratives contain a title and prose, without notes or source lists added by the program.

## Getting Started

Python 3.10 or later is required. From the project directory:

```bash
pip install -r requirements.txt
python -m eventweaver generate examples/documents \
  --model installed-model-name --output-dir outputs
```

Replace `installed-model-name` with a model already available in Ollama (`ollama list`); the server must be running. The model must support the configured context window and respond with JSON.

The package can also be installed with:

```bash
pip install -e '.[documents]'
eventweaver generate ./documents --model installed-model-name --output-dir ./results
```

To reorganize and rewrite an already generated Markdown story:

```powershell
python -m eventweaver reorganize `
  "./results/story.md" `
  --prompt-file "./eventweaver/reorganize_prompt.txt" `
  --model "gemma3:12b" `
  --output "./results/story_reorganized.md"
```

The command preserves the Markdown title and requests an English response.

## Local Hemmingway

`HemmingwayAI/Hemmingway-1.5` can be used as a local Hugging Face provider for
both `generate` and `reorganize`. The first run downloads the model to the
Hugging Face cache. It has roughly 27 billion parameters and needs more than
50 GB for the weights, so Transformers uses automatic GPU/CPU offload.

Install the local provider dependencies in the active environment:

```powershell
pip install -e ".[hf-local]"
```

Test generation:

```powershell
python -m eventweaver generate `
  "./documents" `
  --provider huggingface-local `
  --model "HemmingwayAI/Hemmingway-1.5" `
  --output-dir "./results_hf"
```

Test reorganization:

```powershell
python -m eventweaver reorganize `
  "./results/story.md" `
  --provider huggingface-local `
  --model "HemmingwayAI/Hemmingway-1.5" `
  --prompt-file "./eventweaver/reorganize_prompt.txt" `
  --output "./results/story_hemmingway.md"
```

Custom prompts can be added as uppercase string constants in
`eventweaver/prompts.py` and selected by name. For example, add `MY_WRITE = "..."`
and run:

```powershell
python -m eventweaver generate `
  "./documents" `
  --model "HemmingwayAI/Hemmingway-1.5" `
  --provider huggingface-local `
  --write-prompt MY_WRITE `
  --output-dir "./results_hf"
```

The same mechanism is available for `reorganize` with `--prompt-name`; use
`--prompt-file` when the prompt should remain outside the Python module.
The bundled `WRITE_PLACEMUS` prompt can be selected directly with
`--write-prompt WRITE_PLACEMUS`.

Output directories must be outside input directories so later runs do not
reread narratives as documents.

## Titles And Word Limits

Options are optional:

- `--titles "Title one" "Title two"`: 1 to 5 distinct quoted titles. The number cannot exceed `--max-themes`. Titles retain their order without shortening or automatic translation; write them in English if English titles are desired.
- `--max-words 800`: at most 800 words **per narrative**, excluding the title. It must be a positive integer. Without this option there is no overall narrative word limit.

All documents are still read and aggregated before narratives are defined. Supplied titles define the relevant content from the aggregated information and do not add automatic titles. A title with no relevant chunks produces the same error as a theme with no usable material. Without `--titles`, global themes are selected automatically.

The limit is a maximum, not an exact length. Counting uses whitespace-separated sequences (`str.split()`), so a hyphenated word counts as one word. Markdown titles and formatting whitespace are not counted. The final count is recorded in `manifest.json` under `narratives`.

The first draft considers all assigned chunks. If it exceeds the maximum, an editorial rewrite makes it more concise; long drafts are processed in context-compatible batches. If the model still exceeds the assigned budget, the code applies the limit while preferring complete sentences. Very small limits may therefore produce terse text. Drafts and revisions remain available in the run records.

`--max-new-tokens` limits each model response in every phase; `--max-words` limits the body of the final narrative. They are not equivalent.

## Ollama Repetition Controls

Two shell options are used during writing and editorial reduction:

| Option | Default | Effect |
| --- | --- | --- |
| `--frequency-penalty` | `1.0` | Penalizes frequent recurrence of tokens |
| `--presence-penalty` | `0.8` | Penalizes reuse of tokens already present in the model window |

Both options accept finite numbers from **0 to 2**. `0` disables the corresponding penalty. Values are sent in `options.frequency_penalty` and `options.presence_penalty` in the native `/api/generate` request, not merely written into the prompt.

Narrative prompts require lexical diversity, consistent names, dates, numbers, and technical terms, and no invented facts. Separate sections receive a short `previous_text` context, up to 1600 JSON/UTF-8 bytes, to reduce repetition. The saved section itself is not shortened.

Penalties apply to `WRITE` and `CONDENSE`. Structured summarization, theme selection, and routing send zero so necessary JSON keys and identifiers are not penalized. Higher values do not guarantee removal of repetition and may reduce fluency. These options are available with `--provider ollama`; other providers report an explicit error instead of ignoring them.

## Resuming A Run

If a run completed its draft and stopped in `FASE: CONDENSE`, the completed draft can be reused:

```bash
python -m eventweaver generate --resume-run "./results/run_alpcqzip" --model "mistral-small:24b" --max-words 3000
```

The program creates a new `run_*` directory, copies the database with SQLite, and resumes final reduction. The previous run is read-only and is not modified. OCR, document reading, aggregation, theme selection, and draft generation are not repeated. The saved text and draft are reused, so the original input files are not required.

Resume requires `manifest.json`, `themes.json`, and `analysis.sqlite`, with a draft for every theme and completed ingestion. It supports version 3.6 runs stopped with `FASE: CONDENSE`, including runs without the `draft_completed` checkpoint. A run stopped during initial writing cannot be resumed because its draft may be incomplete. Titles, language, and core parameters come from the manifest; supported reduction parameters can be overridden as documented by the CLI.

## Narrative Response Formats

Writing and reduction accept `paragraphs` as a list of objects, a list of strings, or a single string, as well as text fields such as `text`, `narrative`, `summary`, `content`, and `story`. Narrative phases also accept a root JSON list, a JSON string, or plain prose. An visibly incomplete JSON response is not treated as prose, and structural checks for themes and assignments remain active.

If an editorial response remains unusable after the configured attempts, the available draft batch is used and capped at the word maximum, preferring complete sentences. The method is recorded in `revisions.jsonl` as `editorial_method: draft_word_cap`. LLM service errors, such as an unreachable server, are still reported.

`responses.jsonl` and the SQLite `responses` table retain actual model responses, phase, attempt, parse errors, and truncation status. These records may contain document text and must be handled with the same confidentiality as the sources.

## Pipeline

1. **Reading.** Files and directories are discovered recursively, duplicate paths are removed, and every document is read. Pages or rows are segmented into overlapping chunks while retaining all text.
2. **Full-corpus aggregation.** All documents, including required OCR, are read before the first model call. If the corpus fits, texts are presented together. Otherwise neutral summaries are prepared in batches and recursively integrated into one context. No themes are selected during this preparation.
3. **Global theme extraction.** One selection phase receives the aggregated corpus and identifies 1 to 5 themes, considering connections and differences across documents. Manual `--titles` instead define the requested thematic focus. No per-document or per-chunk themes are generated.
4. **Routing.** The model rereads original chunks and associates each with relevant themes. A chunk may belong to multiple themes or none.
5. **Narrative.** For each theme, the model receives relevant original chunks in limited batches and writes narrative sections. Sections form a draft; `--max-words` triggers editorial reduction when needed.

No later factual or semantic verification is performed. Technical checks for JSON, field formats, theme counts, assignments, and context budgets remain active.

## Documents And Context

Theme selection occurs only after all documents have been read. Large corpora use multiple neutral summarization calls before one global theme-selection call. No preliminary `top-k` filter excludes documents. Text is loaded one document at a time and stored in SQLite; later calls receive only required batches.

```bash
python -m eventweaver generate ./archive ./reports document.csv \
  --model installed-model-name \
  --max-themes 5 \
  --language English \
  --num-ctx 16384 \
  --max-new-tokens 4096 \
  --chunk-chars 3000 \
  --overlap-chars 300 \
  --output-dir ./results
```

`--max-themes` accepts 1 to 5 and is a maximum; fewer themes may be produced. `--max-themes 1` requests one general theme and narrative. The default language is `English`; `--language` changes it explicitly. Replace legacy `--language Italiano` with `--language English`.

Every call checks its budget before execution:

```text
input_estimate + max_new_tokens + safety_margin <= num_ctx
```

The estimate uses UTF-8 prompt bytes as a conservative approximation, not the model tokenizer count. The default margin is 512 tokens. Writing batches contain at most two chunks within the available budget, and paragraphs are retained in full.

If an original chunk does not fit, the program reports an error rather than truncating documents: reduce `--chunk-chars` or increase `--num-ctx`. Intermediate summaries receive calculated word and byte budgets and may be compacted automatically. Full model output remains in `summaries.jsonl` as `raw_summary`; `body.text` contains the version used by later levels.

When Ollama reaches the output limit, the client retains the text and distinguishes truncation from other errors. Complete valid JSON is accepted even when the server reports the token limit. Incomplete batches are split recursively; if the model still cannot produce valid JSON for a minimal chunk, the run stops explicitly.

## Thematic Descriptions And Validation

Title and description limits are organizational metadata limits: 100 and 300 characters respectively. Missing descriptions fall back to the theme title. Empty titles, invalid field types, duplicate themes, and more than five themes remain errors. These rules do not shorten document text or narrative paragraphs.

## Formats And OCR

| Format | Reading |
| --- | --- |
| PDF | Native text per page, with automatic, forced, or disabled OCR |
| DOCX | Main paragraphs and tables in body order |
| TXT / MD | UTF-8 text, including BOM |
| CSV | All non-empty columns and rows; common delimiters detected automatically |

For scanned PDFs:

```bash
pip install -e '.[documents,ocr]'
python -m eventweaver generate ./scans --model installed-model-name \
  --ocr auto --ocr-language ita+eng --ocr-dpi 200 --output-dir ./results
```

OCR uses Tesseract directly and images rendered by PyMuPDF. Tesseract and the required language packages must be installed. `auto` activates OCR for pages without usable native text; `always` requests recognition on every page; `never` disables it. Empty pages are recorded as `skipped: true` and are never replaced with invented text.

The OCR language applies to input documents, not the English narrative language. For Spanish documents use `--ocr-language spa+eng` after installing both packages. OCR errors may appear in narratives. On Windows, set `TESSERACT_CMD` when Tesseract is not in PATH.

## Persistent PDF Text Cache

The cache is enabled by default in `.eventweaver_cache`, relative to the command directory. It stores final PDF page text, extraction method, and extraction parameters separately from `run_*` directories.

| Option | Effect |
| --- | --- |
| none | Read and update the cache automatically |
| `--cache-dir "./cache_ocr"` | Select the persistent cache directory |
| `--refresh-ocr` | Ignore saved text, re-extract pages, and update the cache |
| `--no-cache` | Do not read or write the cache |

`--refresh-ocr` and `--no-cache` are incompatible. Completed pages are written atomically; interrupted runs retain previous pages. The cache contains document text in plain form and can be deleted to reclaim space. `--resume-run` uses the saved draft without reading PDFs and does not accept the cache options.

## Providers

Ollama is the default. `--ollama-host` or `OLLAMA_HOST` selects another server; `--skip-ollama-preflight` skips only the initial installed-model check.

Optional Hugging Face providers:

```bash
pip install -e '.[documents,hf-api]'
# Set HF_TOKEN in the environment.
python -m eventweaver generate ./documents --provider huggingface-api \
  --model organization/model --output-dir ./results
```

```bash
pip install -e '.[documents,hf-local]'
python -m eventweaver generate ./documents --provider huggingface-local \
  --model model-path-or-identifier --output-dir ./results
```

OCR and Hugging Face dependencies are optional. Remote providers receive document text; Ollama and local Hugging Face process it locally.

## Output

Each execution creates a new `run_*` directory.

| File | Content |
| --- | --- |
| `narrative_T01.md` ... `narrative_T05.md` | Narrative title and prose |
| `themes.json` | Global themes with ID, title, and description |
| `manifest.json` | Status, model, configuration, documents, extraction details, counts, calls, and errors |
| `analysis.sqlite` | Original chunks, aggregate summaries, source relationships, assignments, and paragraphs |
| `chunks.jsonl` | Original chunk text, without local themes |
| `aggregate_context.json` | Context presented to global theme selection |
| `summaries.jsonl` | Intermediate summaries, full model text, compact version, budget, and method |
| `summary_inputs.jsonl` | Relationships between summaries and their inputs |
| `assignments.jsonl` | Chunk-to-theme assignments |
| `unassigned.jsonl` | Unassigned chunks and reasons |
| `paragraphs.jsonl` | Initial draft paragraphs and internal chunk IDs |
| `revisions.jsonl` | Editorial revisions and word-limit method |
| `responses.jsonl` | Original model responses, phase, attempt, and parse errors |

Use only runs with `manifest.status == "completed"`. Failed runs retain intermediate data for inspection.

## Limits

- Reading every chunk does not guarantee recognition of every theme or reporting of every fact; analysis, routing, and writing remain LLM decisions.
- Hierarchical summaries may omit details or give less space to rare information. Full source text is retained and reread for routing and writing.
- Chunks irrelevant to global themes do not enter narratives; they are recorded in `unassigned.jsonl`.
- Batch writing can contain repetition and have weaker cohesion than writing with all sources at once. There is no final whole-corpus summary.
- The model is instructed to use only supplied documents. No later factual check is performed.
- Costs and runtime increase with documents, chunks, themes, and paragraphs.

## Structure And API

`cli.py` exposes `generate` and `reorganize`; `config.py` validates parameters; `ingestion/` reads and segments documents; `llm.py` handles batches, budgets, and JSON; `prompts.py` defines phases; `generation.py` orchestrates the flow; `resume.py` reuses a complete draft in a new run; `store.py` stores text and metadata; `providers/` and `ollama_client.py` manage models.

```python
from pathlib import Path
from eventweaver.config import GenerateConfig
from eventweaver.generation import generate_narratives
from eventweaver.providers import build_provider

config = GenerateConfig(max_themes=5, context_tokens=16384,
                        narrative_titles=("Segobriga Through Time", "The Roman Amphitheatre"),
                        max_narrative_words=800, frequency_penalty=1.2, presence_penalty=1.0)
provider = build_provider("ollama", "installed-model-name", num_ctx=config.context_tokens, json_mode=True,
                          frequency_penalty=config.frequency_penalty, presence_penalty=config.presence_penalty)
run = generate_narratives([Path("documents")], output_dir=Path("results"), provider=provider, config=config)
print(run)
```

## Verification

```bash
python -m unittest discover -s tests -v
```

The test suite covers truncated WRITE responses, adaptive splitting without source loss, retry limits, OCR caching and interruption recovery, atomic writes, CLI validation, provider payloads, resume behavior, response formats, manual titles, narrative word limits, complete ingestion before the first LLM call, global theme selection, multi-level summaries, and OCR integration when Tesseract is available.
