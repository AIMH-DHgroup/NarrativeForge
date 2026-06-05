from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path

from .docx_reader import read_document
from .nrs import nrs, nrs_no_r, robustness_from_runs
from .text_metrics import bertscore_f1, broken_sentence_count, forbidden_formatting_count, is_failed_output, semantic_similarity_with_method, split_paragraphs, word_count
from .utils import case_id_from_path, read_csv, slugify, write_csv


@dataclass
class PairSpec:
    source_path: Path
    output_path: Path
    method: str
    run: str
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
        pairs.append(PairSpec(best[1], out, method, run))

    if unmatched:
        warnings.warn(f"Skipped {len(unmatched)} unmatched output file(s).", stacklevel=2)
    return pairs


def load_manifest(path: Path) -> list[PairSpec]:
    rows = read_csv(path)
    pairs: list[PairSpec] = []
    for row in rows:
        runtime = str(row.get("runtime_seconds", "")).strip()
        pairs.append(
            PairSpec(
                source_path=Path(row["source"]),
                output_path=Path(row["output"]),
                method=row.get("method", "unknown"),
                run=str(row.get("run", "1")),
                runtime_seconds=float(runtime) if runtime else None,
            )
        )
    return pairs


def _load_runtime_manifest(output_dir: Path) -> dict[str, float]:
    runtime_map: dict[str, float] = {}
    for candidate in (output_dir / "generation_runs.csv", output_dir / "runs.csv", output_dir / "manifest.csv"):
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
        enriched.append(PairSpec(pair.source_path, pair.output_path, pair.method, pair.run, runtime_seconds=runtime))
    return enriched


def write_manifest_template(path: Path) -> None:
    write_csv(
        path,
        [
            {"source": "case_studies/Form - Case Study_ValdeLoire.docx", "output": "outputs/ValdeLoire_qwen_run1.txt", "method": "qwen2.5:7b", "run": "1", "runtime_seconds": ""},
            {"source": "case_studies/Form - Case Study_ValdeLoire.docx", "output": "outputs/ValdeLoire_qwen_run2.txt", "method": "qwen2.5:7b", "run": "2", "runtime_seconds": ""},
            {"source": "case_studies/Form - Case Study_ValdeLoire.docx", "output": "outputs/ValdeLoire_qwen_run3.txt", "method": "qwen2.5:7b", "run": "3", "runtime_seconds": ""},
        ],
    )


def _read_source(path: Path) -> str:
    return read_document(path)


def _safe_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_pair(pair: PairSpec, semantic_method: str = "sentence-transformers") -> dict:
    source_text = _read_source(pair.source_path)
    output_text = _read_source(pair.output_path)

    bscore = bertscore_f1(source_text, output_text)
    semantic, semantic_method_label = semantic_similarity_with_method(source_text, output_text, semantic_method)
    paragraph_count = len(split_paragraphs(output_text))

    row = {
        "case_id": case_id_from_path(pair.source_path),
        "source_path": str(pair.source_path),
        "output_path": str(pair.output_path),
        "method": pair.method,
        "run": str(pair.run),
        "runtime_seconds": None if pair.runtime_seconds is None else round(pair.runtime_seconds, 3),
        "word_count": word_count(output_text),
        "paragraph_count": paragraph_count,
        "broken_sentence_count": broken_sentence_count(output_text),
        "forbidden_formatting_count": forbidden_formatting_count(output_text),
        "failed": is_failed_output(output_text),
        "bertscore_f1": None if bscore is None else round(max(0.0, min(1.0, bscore)), 6),
        "semantic_similarity": round(max(0.0, min(1.0, semantic)), 6),
        "semantic_similarity_method": semantic_method_label,
    }
    row["NRS_no_R"] = round(nrs_no_r(bscore, semantic), 3)
    row["robustness_available"] = False
    return row


def compute_case_method_summary(run_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in run_rows:
        groups.setdefault((row["case_id"], row["method"]), []).append(row)

    summaries: list[dict] = []
    for (case_id, method), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: str(r["run"]))
        robust = robustness_from_runs(rows)
        runtime_values = [value for r in rows if (value := _safe_float(r.get("runtime_seconds"))) is not None]
        mean_runtime_seconds = statistics.mean(runtime_values) if runtime_values else 0.0
        word_values = [value for r in rows if (value := _safe_float(r.get("word_count"))) is not None]
        paragraph_values = [value for r in rows if (value := _safe_float(r.get("paragraph_count"))) is not None]
        mean_word_count = statistics.mean(word_values) if word_values else 0.0
        mean_paragraph_count = statistics.mean(paragraph_values) if paragraph_values else 0.0
        total_broken_sentences = sum(int(r["broken_sentence_count"]) for r in rows)
        total_forbidden_formatting = sum(int(r["forbidden_formatting_count"]) for r in rows)
        failed_runs = sum(1 for r in rows if bool(r["failed"]))
        b_values = [value for r in rows if (value := _safe_float(r.get("bertscore_f1"))) is not None]
        mean_bertscore_f1 = statistics.mean(b_values) if b_values else None
        semantic_values = [value for r in rows if (value := _safe_float(r.get("semantic_similarity"))) is not None]
        mean_semantic_similarity = statistics.mean(semantic_values) if semantic_values else 0.0
        methods = sorted({str(r.get("semantic_similarity_method", "")).strip() for r in rows if str(r.get("semantic_similarity_method", "")).strip()})
        semantic_similarity_method = "; ".join(methods)
        nrs_no_r_values = [value for r in rows if (value := _safe_float(r.get("NRS_no_R"))) is not None]
        mean_nrs_no_r = statistics.mean(nrs_no_r_values)

        if robust["robustness_available"]:
            clean = nrs(mean_bertscore_f1, mean_semantic_similarity, robust["R"] or 0.0)
        else:
            clean = nrs_no_r(mean_bertscore_f1, mean_semantic_similarity)

        summary = {
            "case_id": case_id,
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
            "NRS_no_R": round(mean_nrs_no_r, 3),
            "NRS": round(clean, 3),
        }
        summaries.append(summary)
    return summaries


def compute_model_overall_summary(case_summaries: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in case_summaries:
        groups.setdefault(row["method"], []).append(row)

    summaries: list[dict] = []
    for method, rows in sorted(groups.items()):
        nrs_values = [value for r in rows if (value := _safe_float(r.get("NRS"))) is not None]
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

        summaries.append(
            {
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
                "std_NRS": None if len(nrs_values) < 2 else round(statistics.pstdev(nrs_values), 3),
                "rank": None,
            }
        )

    ranked = sorted([row for row in summaries if row["mean_NRS"] is not None], key=lambda row: row["mean_NRS"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    summaries.sort(key=lambda row: (row["rank"] is None, row["rank"] or 9999, -(row["mean_NRS"] or 0)))
    return summaries


def _load_run_rows_from_csv(runs_csv: Path) -> list[dict]:
    rows = read_csv(runs_csv)
    run_rows: list[dict] = []
    for row in rows:
        runtime_value = str(row.get("runtime_seconds", "")).strip()
        runtime = _safe_float(runtime_value)
        run_rows.append(
            {
                "case_id": row.get("case_id", ""),
                "method": row.get("method", ""),
                "run": row.get("run", "1"),
                "runtime_seconds": runtime,
                "word_count": _safe_float(row.get("word_count", 0)) or 0.0,
                "paragraph_count": _safe_float(row.get("paragraph_count", 0)) or 0.0,
                "broken_sentence_count": int(_safe_float(row.get("broken_sentence_count", row.get("broken_sentences", 0))) or 0),
                "forbidden_formatting_count": int(_safe_float(row.get("forbidden_formatting_count", row.get("forbidden_formatting", 0))) or 0),
                "failed": str(row.get("failed", "")).lower() in {"true", "1", "yes"},
                "bertscore_f1": _safe_float(row.get("bertscore_f1")),
                "semantic_similarity": _safe_float(row.get("semantic_similarity")) or 0.0,
                "semantic_similarity_method": row.get("semantic_similarity_method", ""),
                "NRS_no_R": _safe_float(row.get("NRS_no_R")) or 0.0,
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
        if sources_dir is None or outputs_dir is None:
            raise ValueError("Use either manifest or sources_dir + outputs_dir")
        pairs = enrich_pairs_with_runtime(match_outputs_to_sources(discover_sources(sources_dir), discover_outputs(outputs_dir)), outputs_dir)

    if not pairs:
        raise RuntimeError("No source/output pairs could be matched.")

    run_rows = [evaluate_pair(pair, semantic_method=semantic_method) for pair in pairs if pair.source_path.exists() and pair.output_path.exists()]
    case_rows = compute_case_method_summary(run_rows)
    model_rows = compute_model_overall_summary(case_rows)

    robustness_map = {(row["case_id"], row["method"]): bool(row.get("robustness_available")) for row in case_rows}
    for row in run_rows:
        row["robustness_available"] = robustness_map.get((row["case_id"], row["method"]), False)

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
