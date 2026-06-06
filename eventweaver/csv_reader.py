from __future__ import annotations

import csv
from pathlib import Path


def _detect_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    except Exception:
        return csv.get_dialect("excel")


def _normalize_headers(headers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: dict[str, int] = {}
    for index, header in enumerate(headers, start=1):
        name = str(header).strip()
        if not name:
            name = f"unnamed_column_{index}"
        key = name.lower()
        if key in seen:
            seen[key] += 1
            name = f"{name}_{seen[key]}"
        else:
            seen[key] = 1
        normalized.append(name)
    return normalized


def iter_csv_rows(path: Path) -> list[dict[str, str]]:
    dialect = _detect_dialect(path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, dialect)
        try:
            headers = next(reader)
        except StopIteration:
            return rows

        headers = _normalize_headers(headers)
        for raw_row in reader:
            values = [str(cell).strip() for cell in raw_row]
            if not any(values):
                continue
            row: dict[str, str] = {}
            for index, header in enumerate(headers):
                row[header] = values[index].strip() if index < len(values) else ""
            if len(values) > len(headers):
                for extra_index in range(len(headers), len(values)):
                    row[f"unnamed_column_{extra_index + 1}"] = values[extra_index]
            if any(value for value in row.values()):
                rows.append(row)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    return iter_csv_rows(path)


def load_csv_row(path: Path, *, row_index: int | None = None, row_id: str | None = None, id_column: str = "Card ID") -> dict[str, str]:
    rows = read_csv_rows(path)
    if row_index is not None:
        if row_index < 1 or row_index > len(rows):
            raise IndexError(f"CSV row_index {row_index} is out of range for {path}")
        return rows[row_index - 1]

    if row_id is not None:
        target = row_id.strip().lower()
        for row in rows:
            candidate = row.get(id_column, "").strip().lower()
            if candidate == target:
                return row

    raise KeyError(f"Could not locate CSV row in {path} using row_index={row_index!r} row_id={row_id!r}")
