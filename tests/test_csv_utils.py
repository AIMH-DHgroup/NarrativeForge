from eventweaver.narrative.csv_utils import parse_csv_rows, split_csv_ids


def test_parse_csv_model_response_with_commas_and_fences():
    rows = parse_csv_rows(
        '```csv\nsummary,evidence_ids,date_hint\n"A fact, with a comma",DOC0001_C0001;DOC0001_C0002,2015\n```',
        {"summary", "evidence_ids", "date_hint"},
    )
    assert rows == [{
        "summary": "A fact, with a comma",
        "evidence_ids": "DOC0001_C0001;DOC0001_C0002",
        "date_hint": "2015",
    }]
    assert split_csv_ids(rows[0]["evidence_ids"]) == ["DOC0001_C0001", "DOC0001_C0002"]


def test_parse_semicolon_csv_and_unquoted_commas():
    semicolon_rows = parse_csv_rows(
        '"summary";"evidence_ids";"date_hint"\n"A fact";"DOC0001_C0001";"2015"',
        {"summary", "evidence_ids", "date_hint"},
    )
    assert semicolon_rows[0]["evidence_ids"] == "DOC0001_C0001"

    unquoted_rows = parse_csv_rows(
        "summary,evidence_ids,date_hint\nA fact, with a comma,DOC0001_C0001,2015",
        {"summary", "evidence_ids", "date_hint"},
    )
    assert unquoted_rows[0]["summary"] == "A fact, with a comma"
    assert unquoted_rows[0]["evidence_ids"] == "DOC0001_C0001"


def test_parse_csv_without_header():
    rows = parse_csv_rows(
        '"A fact, with a comma";"DOC0001_C0001";"2019"',
        {"summary", "evidence_ids", "date_hint"},
    )
    assert rows == [{
        "summary": "A fact, with a comma",
        "evidence_ids": "DOC0001_C0001",
        "date_hint": "2019",
    }]

    synthesis_rows = parse_csv_rows(
        '"A paragraph, with a comma";"EVT00025"',
        {"paragraph", "evidence_ids"},
    )
    assert synthesis_rows == [{"paragraph": "A paragraph, with a comma", "evidence_ids": "EVT00025"}]
