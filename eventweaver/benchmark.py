from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path

from .csv_metrics import compute_field_coverage, compute_format_score, compute_q_score, csv_failed_output
from .csv_reader import load_csv_row
from .docx_reader import read_document
from .nrs import nrs, nrs_no_r, robustness_from_runs
from .source_record import row_to_source_text
from .text_metrics import bertscore_f1, broken_sentence_count, forbidden_formatting_count, is_failed_output, semantic_similarity_with_method, split_paragraphs, word_count
from .utils import case_id_from_path, read_csv, slugify, write_csv


@dataclass
class PairSpec:
    source_path: Path
    output_path: Path
    method: str
    run: str
    source_type: str = "docx"
    row_index: int | None = None
    row_id: str = ""
    row_title: str = ""
    runtime_seconds: float | None = None


def discover_sources(sources_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.docx", "*.txt", "*.md"):
        files.extend(sources_dir.rglob(ext))
    return sorted(p for p in files if p.is_file() and not p.name.startswith("~$"))


def discover_outputs(outputs_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.txt", "*.docx", "*.md"):
        files.extend(outputs_dir.rglob(ext))
    return sorted(p for p in files if p.is_file() and not p.name.startswith("~$"))


def infer_run_and_method(output_path: Path) -> tuple[str, str]:
    stem = output_path.stem
    run = "1"
    import re

    match = re.search(r"(?:^|[_\- ])(?:run|r)[_\- ]?(\d+)$", stem, flags=re.I)
    if match:
        run = match.group(1)
        stem = re.sub(r"[_\- ]?(?:run|r)[_\- ]?\d+$", "", stem, flags=re.I)
    if re.search(r"_(?:full|brief|rag)$", stem, flags=re.I):
        stem = re.sub(r"_(?:full|brief|rag)$", "", stem, flags=re.I)
    if "_narrative_" in stem:
        method = stem.split("_narrative_", 1)[1]
    elif "_output_" in stem:
        method = stem.split("_output_", 1)[1]
    elif "_generated_" in stem:
        method = stem.split("_generated_", 1)[1]
    else:
        method = stem
    return method.strip("_-") or "unknown", run


def infer_row_index_from_output(output_path: Path) -> int | None:
    import re

    match = re.search(r"_row(\d{3,})_", output_path.stem, flags=re.I)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _source_keys(source: Path) -> list[str]:
    return sorted({slugify(source.stem), case_id_from_path(source)}, key=len, reverse=True)


def match_outputs_to_sources(sources: list[Path], outputs: list[Path]) -> list[PairSpec]:
    pairs: list[PairSpec] = []
    source_index = [(src, _source_keys(src)) for src in sources]
    unmatched: list[Path] = []

    for out in outputs:
        out_slug = slugify(out.stem)
        best: tuple[int, Path] | None = None
        for src, keys in source_index:
            for key in keys:
                if key and key in out_slug:
                    score = len(key)
                    if best is None or score > best[0]:
                        best = (score, src)
                    break
        if best is None:
            unmatched.append(out)
            continue
        method, run = infer_run_and_method(out)
        source_path = best[1]
        source_type = "csv" if source_path.suffix.lower() == ".csv" else "docx"
        row_index = infer_row_index_from_output(out) if source_type == "csv" else None
        pairs.append(PairSpec(source_path, out, method, run, source_type=source_type, row_index=row_index))

    if unmatched:
        warnings.warn(f"Skipped {len(unmatched)} unmatched output file(s).", stacklevel=2)
    return pairs


def load_manifest(path: Path) -> list[PairSpec]:
    rows = read_csv(path)
    pairs: list[PairSpec] = []
    for row in rows:
        runtime = str(row.get("runtime_seconds", "")).strip()
        source_type = str(row.get("source_type", "docx")).strip().lower() or "docx"
        row_index_value = str(row.get("row_index", "")).strip()
        row_index = int(row_index_value) if row_index_value.isdigit() else None
        source_value = row.get("source") or row.get("source_file") or ""
        if not source_value:
            raise ValueError("Manifest row is missing a source/source_file value")
        pairs.append(
            PairSpec(
                source_path=Path(source_value),
                output_path=Path(row["output"]),
                method=row.get("method", "unknown"),
                run=str(row.get("run", "1")),
                source_type=source_type,
                row_index=row_index,
                row_id=str(row.get("row_id", "")).strip(),
                row_title=str(row.get("row_title", "")).strip(),
                runtime_seconds=float(runtime) if runtime else None,
            )
        )
    return pairs


def _load_runtime_manifest(output_dir: Path) -> dict[str, float]:
    runtime_map: dict[str, float] = {}
    for candidate in (output_dir / "generation_metadata.csv", output_dir / "generation_runs.csv", output_dir / "runs.csv", output_dir / "manifest.csv"):
        if not candidate.exists():
            continue
        for row in read_csv(candidate):
            output_value = str(row.get("output", "")).strip()
            runtime_value = str(row.get("runtime_seconds", "")).strip()
            if not output_value or not runtime_value:
                continue
            try:
                runtime = float(runtime_value)
            except ValueError:
                continue
            runtime_map[output_value] = runtime
            runtime_map[Path(output_value).name] = runtime
            runtime_map[str(Path(output_value).resolve())] = runtime
    return runtime_map


def enrich_pairs_with_runtime(pairs: list[PairSpec], outputs_dir: Path | None) -> list[PairSpec]:
    if outputs_dir is None:
        return pairs
    runtime_map = _load_runtime_manifest(outputs_dir)
    if not runtime_map:
        return pairs

    enriched: list[PairSpec] = []
    for pair in pairs:
        runtime = pair.runtime_seconds
        if runtime is None:
            for key in (str(pair.output_path), pair.output_path.name, str(pair.output_path.resolve())):
                if key in runtime_map:
                    runtime = runtime_map[key]
                    break
        enriched.append(
            PairSpec(
                pair.source_path,
                pair.output_path,
                pair.method,
                pair.run,
                source_type=pair.source_type,
                row_index=pair.row_index,
                row_id=pair.row_id,
                row_title=pair.row_title,
                runtime_seconds=runtime,
            )
        )
    return enriched


def write_manifest_template(path: Path) -> None:
    write_csv(
        path,
        [
            {"source_type": "docx", "source": "case_studies/Form - Case Study_ValdeLoire.docx", "output": "outputs/ValdeLoire_qwen_run1.txt", "method": "qwen2.5:7b", "run": "1", "runtime_seconds": ""},
            {"source_type": "csv", "source": "MOVING_VCs_DATASET_FINAL_V2.csv", "row_index": "1", "row_id": "VC_01_AT", "output": "outputs/MOVING_VCs_DATASET_FINAL_V2_row001_VC_01_AT_narrative_qwen3_8b_run1.txt", "method": "qwen3:8b", "run": "1", "runtime_seconds": ""},
        ],
    )


def _read_source(path: Path) -> str:
    return read_document(path)


def _read_csv_source(path: Path, row_index: int | None, row_id: str) -> tuple[str, dict[str, str]]:
    row = load_csv_row(path, row_index=row_index, row_id=row_id or None)
    return row_to_source_text(row), row


def _safe_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_pair(pair: PairSpec, semantic_method: str = "sentence-transformers") -> dict:
    source_type = (pair.source_type or ("csv" if pair.source_path.suffix.lower() == ".csv" else "docx")).lower()
    row: dict[str, str] | None = None
    if source_type == "csv":
        source_text, row = _read_csv_source(pair.source_path, pair.row_index, pair.row_id)
    else:
        source_text = _read_source(pair.source_path)
    output_text = _read_source(pair.output_path)

    bscore = bertscore_f1(source_text, output_text)
    semantic, semantic_method_label = semantic_similarity_with_method(source_text, output_text, semantic_method)
    paragraph_count = len(split_paragraphs(output_text))
    field_coverage = compute_field_coverage(row or {}, output_text) if source_type == "csv" else None
    format_score = compute_format_score(output_text) if source_type == "csv" else None
    q_score = compute_q_score(bscore, semantic, field_coverage or 0.0, format_score or 0.0) if source_type == "csv" else None
    failed = csv_failed_output(output_text) if source_type == "csv" else is_failed_output(output_text)

    row = {
        "source_type": source_type,
        "source_file": str(pair.source_path),
        "source_path": str(pair.source_path),
        "row_index": pair.row_index if pair.row_index is not None else "",
        "row_id": pair.row_id,
        "row_title": pair.row_title,
        "case_id": case_id_from_path(pair.source_path) if source_type != "csv" else "",
        "output_path": str(pair.output_path),
        "method": pair.method,
        "run": str(pair.run),
        "runtime_seconds": None if pair.runtime_seconds is None else round(pair.runtime_seconds, 3),
        "word_count": word_count(output_text),
        "paragraph_count": paragraph_count,
        "broken_sentence_count": broken_sentence_count(output_text),
        "forbidden_formatting_count": forbidden_formatting_count(output_text),
        "failed": failed,
        "bertscore_f1": None if bscore is None else round(max(0.0, min(1.0, bscore)), 6),
        "semantic_similarity": round(max(0.0, min(1.0, semantic)), 6),
        "semantic_similarity_method": semantic_method_label,
    }
    row["NRS_no_R"] = round(nrs_no_r(bscore, semantic), 3)
    if source_type == "csv":
        row["field_coverage"] = None if field_coverage is None else round(field_coverage, 6)
        row["format_score"] = None if format_score is None else round(format_score, 6)
        row["Q"] = None if q_score is None else round(q_score, 6)
        row["CSV_NRS"] = None
        row["NRS"] = round(nrs(bscore, semantic, 0.0), 3)
    else:
        row["field_coverage"] = None
        row["format_score"] = None
        row["Q"] = None
        row["CSV_NRS"] = None
        row["NRS"] = None
    row["robustness_available"] = False
    row["R"] = None
    return row


def compute_case_method_summary(run_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in run_rows:
        source_type = str(row.get("source_type", "docx")).strip().lower() or "docx"
        if source_type == "csv":
            source_key = f"{row.get('source_file', '')}#{row.get('row_index', '') or row.get('row_id', '')}"
        else:
            source_key = str(row.get("case_id", ""))
        groups.setdefault((source_type, source_key, str(row.get("method", ""))), []).append(row)

    summaries: list[dict] = []
    for (source_type, source_key, method), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: str(r.get("run", "")))
        robust = robustness_from_runs(rows)
        runtime_values = [value for r in rows if (value := _safe_float(r.get("runtime_seconds"))) is not None]
        mean_runtime_seconds = statistics.mean(runtime_values) if runtime_values else 0.0
        word_values = [value for r in rows if (value := _safe_float(r.get("word_count"))) is not None]
        paragraph_values = [value for r in rows if (value := _safe_float(r.get("paragraph_count"))) is not None]
        mean_word_count = statistics.mean(word_values) if word_values else 0.0
        mean_paragraph_count = statistics.mean(paragraph_values) if paragraph_values else 0.0
        total_broken_sentences = sum(int(r.get("broken_sentence_count", 0) or 0) for r in rows)
        total_forbidden_formatting = sum(int(r.get("forbidden_formatting_count", 0) or 0) for r in rows)
        failed_runs = sum(1 for r in rows if bool(r.get("failed")))
        b_values = [value for r in rows if (value := _safe_float(r.get("bertscore_f1"))) is not None]
        mean_bertscore_f1 = statistics.mean(b_values) if b_values else None
        semantic_values = [value for r in rows if (value := _safe_float(r.get("semantic_similarity"))) is not None]
        mean_semantic_similarity = statistics.mean(semantic_values) if semantic_values else 0.0
        methods = sorted({str(r.get("semantic_similarity_method", "")).strip() for r in rows if str(r.get("semantic_similarity_method", "")).strip()})
        semantic_similarity_method = "; ".join(methods)
        nrs_no_r_values = [value for r in rows if (value := _safe_float(r.get("NRS_no_R"))) is not None]
        mean_nrs_no_r = statistics.mean(nrs_no_r_values) if nrs_no_r_values else None

        if source_type == "csv":
            q_values = [value for r in rows if (value := _safe_float(r.get("Q"))) is not None]
            mean_q = statistics.mean(q_values) if q_values else None
            field_values = [value for r in rows if (value := _safe_float(r.get("field_coverage"))) is not None]
            format_values = [value for r in rows if (value := _safe_float(r.get("format_score"))) is not None]
            mean_field_coverage = statistics.mean(field_values) if field_values else None
            mean_format_score = statistics.mean(format_values) if format_values else None
            if robust["robustness_available"] and mean_q is not None:
                csv_nrs = 100.0 * (0.70 * mean_q + 0.30 * (robust["R"] or 0.0))
            elif mean_q is not None:
                csv_nrs = 100.0 * mean_q
            else:
                csv_nrs = None
            comparison_nrs = nrs(mean_bertscore_f1, mean_semantic_similarity, robust["R"] or 0.0) if robust["robustness_available"] else nrs_no_r(mean_bertscore_f1, mean_semantic_similarity)
            summaries.append(
                {
                    "source_type": "csv",
                    "source_file": str(rows[0].get("source_file", "")),
                    "row_index": rows[0].get("row_index", ""),
                    "row_id": rows[0].get("row_id", ""),
                    "row_title": rows[0].get("row_title", ""),
                    "method": method,
                    "number_of_runs": len(rows),
                    "mean_runtime_seconds": round(mean_runtime_seconds, 3),
                    "mean_word_count": round(mean_word_count, 2),
                    "mean_paragraph_count": round(mean_paragraph_count, 2),
                    "total_broken_sentences": total_broken_sentences,
                    "total_forbidden_formatting": total_forbidden_formatting,
                    "failed_runs": failed_runs,
                    "mean_bertscore_f1": None if mean_bertscore_f1 is None else round(mean_bertscore_f1, 6),
                    "mean_semantic_similarity": round(mean_semantic_similarity, 6),
                    "semantic_similarity_method": semantic_similarity_method,
                    "robustness_available": robust["robustness_available"],
                    "R_stab": None if robust["R_stab"] is None else round(robust["R_stab"], 6),
                    "R_struct": None if robust["R_struct"] is None else round(robust["R_struct"], 6),
                    "R_fail": None if robust["R_fail"] is None else round(robust["R_fail"], 6),
                    "R": None if robust["R"] is None else round(robust["R"], 6),
                    "mean_Q": None if mean_q is None else round(mean_q, 6),
                    "mean_field_coverage": None if mean_field_coverage is None else round(mean_field_coverage, 6),
                    "mean_format_score": None if mean_format_score is None else round(mean_format_score, 6),
                    "CSV_NRS": None if csv_nrs is None else round(csv_nrs, 3),
                    "NRS": round(comparison_nrs, 3),
                    "case_id": "",
                }
            )
            continue

        if robust["robustness_available"]:
            clean = nrs(mean_bertscore_f1, mean_semantic_similarity, robust["R"] or 0.0)
        else:
            clean = nrs_no_r(mean_bertscore_f1, mean_semantic_similarity)

        summaries.append(
            {
                "source_type": "docx",
                "case_id": source_key,
                "source_file": str(rows[0].get("source_file", rows[0].get("source_path", ""))),
                "row_index": "",
                "row_id": "",
                "row_title": "",
                "method": method,
                "number_of_runs": len(rows),
                "mean_runtime_seconds": round(mean_runtime_seconds, 3),
                "mean_word_count": round(mean_word_count, 2),
                "mean_paragraph_count": round(mean_paragraph_count, 2),
                "total_broken_sentences": total_broken_sentences,
                "total_forbidden_formatting": total_forbidden_formatting,
                "failed_runs": failed_runs,
                "mean_bertscore_f1": None if mean_bertscore_f1 is None else round(mean_bertscore_f1, 6),
                "mean_semantic_similarity": round(mean_semantic_similarity, 6),
                "semantic_similarity_method": semantic_similarity_method,
                "robustness_available": robust["robustness_available"],
                "R_stab": None if robust["R_stab"] is None else round(robust["R_stab"], 6),
                "R_struct": None if robust["R_struct"] is None else round(robust["R_struct"], 6),
                "R_fail": None if robust["R_fail"] is None else round(robust["R_fail"], 6),
                "R": None if robust["R"] is None else round(robust["R"], 6),
                "NRS_no_R": round(mean_nrs_no_r, 3) if mean_nrs_no_r is not None else None,
                "NRS": round(clean, 3),
                "mean_Q": None,
                "mean_field_coverage": None,
                "mean_format_score": None,
                "CSV_NRS": None,
            }
        )
    return summaries


def compute_model_overall_summary(case_summaries: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in case_summaries:
        source_type = str(row.get("source_type", "docx")).strip().lower() or "docx"
        groups.setdefault((source_type, row["method"]), []).append(row)

    summaries: list[dict] = []
    for (source_type, method), rows in sorted(groups.items()):
        nrs_values = [value for r in rows if (value := _safe_float(r.get("NRS"))) is not None]
        csv_nrs_values = [value for r in rows if (value := _safe_float(r.get("CSV_NRS"))) is not None]
        q_values = [value for r in rows if (value := _safe_float(r.get("mean_Q"))) is not None]
        r_values = [value for r in rows if (value := _safe_float(r.get("R"))) is not None]
        b_values = [value for r in rows if (value := _safe_float(r.get("mean_bertscore_f1"))) is not None]
        semantic_values = [value for r in rows if (value := _safe_float(r.get("mean_semantic_similarity"))) is not None]
        total_runs = sum(int(r["number_of_runs"]) for r in rows)
        failed_runs = sum(int(r["failed_runs"]) for r in rows)
        mean_runtime_values = [value for r in rows if (value := _safe_float(r.get("mean_runtime_seconds"))) is not None]
        mean_runtime_seconds = statistics.mean(mean_runtime_values) if mean_runtime_values else 0.0
        mean_word_values = [value for r in rows if (value := _safe_float(r.get("mean_word_count"))) is not None]
        mean_paragraph_values = [value for r in rows if (value := _safe_float(r.get("mean_paragraph_count"))) is not None]
        mean_word_count = statistics.mean(mean_word_values) if mean_word_values else 0.0
        mean_paragraph_count = statistics.mean(mean_paragraph_values) if mean_paragraph_values else 0.0

        summary = {
            "source_type": source_type,
            "method": method,
            "cases_count": len(rows),
            "total_runs": total_runs,
            "mean_runtime_seconds": round(mean_runtime_seconds, 3),
            "mean_word_count": round(mean_word_count, 2),
            "mean_paragraph_count": round(mean_paragraph_count, 2),
            "failed_rate": round(failed_runs / max(total_runs, 1), 6),
            "mean_bertscore_f1": None if not b_values else round(statistics.mean(b_values), 6),
            "mean_semantic_similarity": round(statistics.mean(semantic_values), 6),
            "mean_R": None if not r_values else round(statistics.mean(r_values), 6),
            "mean_NRS": None if not nrs_values else round(statistics.mean(nrs_values), 3),
            "mean_CSV_NRS": None if not csv_nrs_values else round(statistics.mean(csv_nrs_values), 3),
            "mean_Q": None if not q_values else round(statistics.mean(q_values), 6),
            "std_NRS": None if len(nrs_values) < 2 else round(statistics.pstdev(nrs_values), 3),
            "rank": None,
        }
        summaries.append(summary)

    ranked = sorted([row for row in summaries if row["mean_NRS"] is not None], key=lambda row: row["mean_NRS"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    summaries.sort(key=lambda row: (row["source_type"], row["rank"] is None, row["rank"] or 9999, -(row["mean_NRS"] or 0)))
    return summaries


def _load_run_rows_from_csv(runs_csv: Path) -> list[dict]:
    rows = read_csv(runs_csv)
    run_rows: list[dict] = []
    for row in rows:
        runtime_value = str(row.get("runtime_seconds", "")).strip()
        runtime = _safe_float(runtime_value)
        source_type = str(row.get("source_type", "docx")).strip().lower() or "docx"
        row_index_value = str(row.get("row_index", "")).strip()
        row_index = int(row_index_value) if row_index_value.isdigit() else ""
        run_rows.append(
            {
                "source_type": source_type,
                "source_file": row.get("source_file", row.get("source", "")),
                "source_path": row.get("source_path", row.get("source", "")),
                "row_index": row_index,
                "row_id": row.get("row_id", ""),
                "row_title": row.get("row_title", ""),
                "case_id": row.get("case_id", ""),
                "method": row.get("method", ""),
                "run": row.get("run", "1"),
                "runtime_seconds": runtime,
                "word_count": _safe_float(row.get("word_count", row.get("output_word_count", 0))) or 0.0,
                "paragraph_count": _safe_float(row.get("paragraph_count", 0)) or 0.0,
                "broken_sentence_count": int(_safe_float(row.get("broken_sentence_count", row.get("broken_sentences", 0))) or 0),
                "forbidden_formatting_count": int(_safe_float(row.get("forbidden_formatting_count", row.get("forbidden_formatting", 0))) or 0),
                "failed": str(row.get("failed", "")).lower() in {"true", "1", "yes"},
                "bertscore_f1": _safe_float(row.get("bertscore_f1")),
                "semantic_similarity": _safe_float(row.get("semantic_similarity")) or 0.0,
                "semantic_similarity_method": row.get("semantic_similarity_method", ""),
                "NRS_no_R": _safe_float(row.get("NRS_no_R")) or 0.0,
                "field_coverage": _safe_float(row.get("field_coverage")),
                "format_score": _safe_float(row.get("format_score")),
                "Q": _safe_float(row.get("Q")),
                "CSV_NRS": _safe_float(row.get("CSV_NRS")),
                "NRS": _safe_float(row.get("NRS")),
                "source_word_count": _safe_float(row.get("source_word_count", 0)) or 0.0,
                "output_word_count": _safe_float(row.get("output_word_count", row.get("word_count", 0))) or 0.0,
            }
        )
    return run_rows


def summarize_runs_csv(runs_csv: Path, outdir: Path) -> dict:
    run_rows = _load_run_rows_from_csv(runs_csv)
    case_rows = compute_case_method_summary(run_rows)
    model_rows = compute_model_overall_summary(case_rows)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "nrs_case_method_summary.csv", case_rows)
    write_csv(outdir / "nrs_model_overall_summary.csv", model_rows)
    return {"case_method": case_rows, "model": model_rows}


def evaluate_folder(*, sources_dir: Path | None = None, outputs_dir: Path | None = None, manifest: Path | None = None, outdir: Path = Path("benchmark_results"), semantic_method: str = "sentence-transformers", excel: bool = False) -> dict:
    if manifest is not None:
        pairs = load_manifest(manifest)
    else:
        if outputs_dir is not None:
            metadata_path = outputs_dir / "generation_metadata.csv"
            if metadata_path.exists():
                pairs = load_manifest(metadata_path)
            else:
                pairs = []
        else:
            pairs = []

        if not pairs and (sources_dir is None or outputs_dir is None):
            raise ValueError("Use either manifest or sources_dir + outputs_dir")
        if not pairs:
            pairs = enrich_pairs_with_runtime(match_outputs_to_sources(discover_sources(sources_dir), discover_outputs(outputs_dir)), outputs_dir)
        else:
            pairs = enrich_pairs_with_runtime(pairs, outputs_dir)

    if not pairs:
        raise RuntimeError("No source/output pairs could be matched.")

    run_rows = [evaluate_pair(pair, semantic_method=semantic_method) for pair in pairs if pair.source_path.exists() and pair.output_path.exists()]
    case_rows = compute_case_method_summary(run_rows)
    model_rows = compute_model_overall_summary(case_rows)

    robustness_map: dict[tuple[str, str, str], bool] = {}
    score_map: dict[tuple[str, str, str], dict] = {}
    for row in case_rows:
        source_type = str(row.get("source_type", "docx")).strip().lower() or "docx"
        if source_type == "csv":
            source_key = f"{row.get('source_file', '')}#{row.get('row_index', '') or row.get('row_id', '')}"
        else:
            source_key = str(row.get("case_id", ""))
        key = (source_type, source_key, str(row.get("method", "")))
        robustness_map[key] = bool(row.get("robustness_available"))
        score_map[key] = row

    for row in run_rows:
        source_type = str(row.get("source_type", "docx")).strip().lower() or "docx"
        if source_type == "csv":
            source_key = f"{row.get('source_file', '')}#{row.get('row_index', '') or row.get('row_id', '')}"
        else:
            source_key = str(row.get("case_id", ""))
        key = (source_type, source_key, str(row.get("method", "")))
        row["robustness_available"] = robustness_map.get(key, False)
        summary = score_map.get(key)
        if summary:
            row["R"] = summary.get("R")
            row["NRS"] = summary.get("NRS")
            row["CSV_NRS"] = summary.get("CSV_NRS")

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "nrs_runs.csv", run_rows)
    write_csv(outdir / "nrs_case_method_summary.csv", case_rows)
    write_csv(outdir / "nrs_model_overall_summary.csv", model_rows)

    if excel:
        try:
            import pandas as pd

            with pd.ExcelWriter(outdir / "nrs_report.xlsx") as writer:
                pd.DataFrame(run_rows).to_excel(writer, sheet_name="Runs", index=False)
                pd.DataFrame(case_rows).to_excel(writer, sheet_name="CaseMethodSummary", index=False)
                pd.DataFrame(model_rows).to_excel(writer, sheet_name="ModelOverallSummary", index=False)
        except Exception:
            warnings.warn("Excel report could not be written.", stacklevel=2)

    return {"runs": run_rows, "case_method": case_rows, "model": model_rows}
