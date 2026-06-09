from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .csv_reader import load_csv_row, read_csv_rows
from .docx_reader import read_docx_text
from .utils import case_id_from_path, safe_model_name


@dataclass
class SourceRecord:
    source_type: str
    source_path: Path
    source_text: str
    prompt_kind: str
    row_index: int | None = None
    row_id: str = ""
    row_title: str = ""
    row: dict[str, str] | None = None


def _case_insensitive_get(row: dict[str, str], column: str) -> str:
    lookup = column.strip().lower()
    for key, value in row.items():
        if key.strip().lower() == lookup:
            return str(value).strip()
    return ""


def row_to_source_text(row: dict[str, str], include_columns: list[str] | None = None) -> str:
    ordered_columns = include_columns if include_columns is not None else list(row.keys())
    parts: list[str] = ["CSV VALUE-CHAIN RECORD", ""]
    for column in ordered_columns:
        value = _case_insensitive_get(row, column)
        if not value.strip():
            continue
        parts.append(f"{column}:")
        parts.append(value.strip())
        parts.append("")
    return "\n".join(parts).strip()


def resolve_prompt_kind(source_type: str, prompt_kind: str | None = None) -> str:
    kind = (prompt_kind or "auto").strip().lower()
    if kind != "auto":
        return kind
    return "value-chain" if source_type == "csv" else "cultural-heritage"


def build_csv_source_record(
    path: Path,
    row: dict[str, str],
    *,
    row_index: int,
    csv_id_column: str,
    csv_title_column: str,
    csv_text_columns: list[str] | None,
    csv_all_columns: bool,
    prompt_kind: str,
) -> SourceRecord:
    include_columns = csv_text_columns if csv_text_columns else None
    if include_columns is None and csv_all_columns:
        include_columns = None
    elif include_columns is None:
        include_columns = [c for c in (csv_id_column, csv_title_column) if _case_insensitive_get(row, c)]

    source_text = row_to_source_text(row, include_columns=include_columns)
    return SourceRecord(
        source_type="csv",
        source_path=path,
        source_text=source_text,
        prompt_kind=resolve_prompt_kind("csv", prompt_kind),
        row_index=row_index,
        row_id=_case_insensitive_get(row, csv_id_column),
        row_title=_case_insensitive_get(row, csv_title_column),
        row=row,
    )


def build_docx_source_record(path: Path, *, prompt_kind: str) -> SourceRecord:
    source_text = read_docx_text(path)
    return SourceRecord(
        source_type="docx",
        source_path=path,
        source_text=source_text,
        prompt_kind=resolve_prompt_kind("docx", prompt_kind),
    )


def iter_source_records(
    input_path: Path,
    *,
    csv_id_column: str = "Card ID",
    csv_title_column: str = "Descriptor of the value chain",
    csv_text_columns: list[str] | None = None,
    csv_all_columns: bool = True,
    csv_max_rows: int = 0,
    prompt_kind: str = "auto",
) -> list[SourceRecord]:
    records: list[SourceRecord] = []

    def _add_path(path: Path) -> None:
        if path.suffix.lower() == ".docx" and not path.name.startswith("~$"):
            records.append(build_docx_source_record(path, prompt_kind=prompt_kind))
            return
        if path.suffix.lower() == ".csv" and not path.name.startswith("~$"):
            rows = read_csv_rows(path)
            count = 0
            for index, row in enumerate(rows, start=1):
                if csv_max_rows and count >= csv_max_rows:
                    break
                if not any(value.strip() for value in row.values()):
                    continue
                count += 1
                records.append(
                    build_csv_source_record(
                        path,
                        row,
                        row_index=count,
                        csv_id_column=csv_id_column,
                        csv_title_column=csv_title_column,
                        csv_text_columns=csv_text_columns,
                        csv_all_columns=csv_all_columns if not csv_text_columns else False,
                        prompt_kind=prompt_kind,
                    )
                )

    if input_path.is_file():
        _add_path(input_path)
    elif input_path.is_dir():
        for path in sorted(p for p in input_path.rglob("*") if p.is_file()):
            _add_path(path)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not records:
        raise FileNotFoundError(f"No .docx or .csv source files found at {input_path}")
    return records


def source_text_word_count(record: SourceRecord) -> int:
    from .text_metrics import word_count

    return word_count(record.source_text)


def build_output_filename(record: SourceRecord, model: str, run: int, *, prompt_strategy: str, input_strategy: str | None = None) -> str:
    model_slug = safe_model_name(model)
    strategy_slug = safe_model_name(prompt_strategy)
    if record.source_type == "csv":
        parts = [safe_model_name(record.source_path.stem)]
        if record.row_index is not None:
            parts.append(f"row{record.row_index:03d}")
        if record.row_id:
            parts.append(safe_model_name(record.row_id))
        parts.extend([model_slug, strategy_slug, f"run{run}"])
        return "_".join(part for part in parts if part) + ".txt"

    case_id = case_id_from_path(record.source_path)
    input_slug = safe_model_name(input_strategy or "auto")
    return f"{case_id}_narrative_{model_slug}_{input_slug}_{strategy_slug}_run{run}.txt"
