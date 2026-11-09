from eventweaver.narrative.marked_lines import parse_marked_lines


def test_parse_marked_event_with_punctuation():
    rows = parse_marked_lines(
        "EVENT|DOC0001_C0001;DOC0001_C0002|A fact, with commas and a | pipe.",
        "event",
    )
    assert rows == [{
        "evidence_ids": "DOC0001_C0001;DOC0001_C0002",
        "summary": "A fact, with commas and a | pipe.",
    }]


def test_parse_marked_check():
    rows = parse_marked_lines(
        "CHECK|2|supported|DOC0001_C0001|Supported, according to the evidence.",
        "check",
    )
    assert rows[0]["sentence_index"] == "2"
    assert rows[0]["status"] == "supported"
