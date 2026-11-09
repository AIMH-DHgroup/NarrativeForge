from __future__ import annotations

import unittest
from pathlib import Path

from eventweaver.generation import extract_docx_title, extract_music_nationality, prompt_format_values
from eventweaver.source_record import SourceRecord


class SimplePromptExtractionTests(unittest.TestCase):
    def test_extracts_labeled_title_and_nationality(self) -> None:
        source = SourceRecord(
            source_type="docx",
            source_path=Path("Form - Case Study_Hungarian Music.docx"),
            source_text='Title | "From the Music of Franz Liszt to the World of the "Táncház" (Dance House) Movement"\nCountry: Hungary',
            prompt_kind="cultural-heritage",
        )
        title = extract_docx_title(source)
        self.assertEqual(title, '"From the Music of Franz Liszt to the World of the "Táncház" (Dance House) Movement"')
        self.assertEqual(extract_music_nationality(source, title), "Hungarian")

    def test_prompt_format_values_for_simple_prompt(self) -> None:
        source = SourceRecord(
            source_type="docx",
            source_path=Path("Form - Case Study_02 - Georgian Poliphony.docx"),
            source_text="Some source text",
            prompt_kind="cultural-heritage",
        )
        values = prompt_format_values(source, "prepared text")
        self.assertEqual(values["title"], "A Journey Through Georgian Polyphonic Traditions")
        self.assertEqual(values["nation"], "Georgian")
        self.assertEqual(values["case_study_text"], "prepared text")

    def test_uses_known_case_study_title_when_label_is_missing(self) -> None:
        source = SourceRecord(
            source_type="docx",
            source_path=Path("Form - Case Study_ValdeLoire.docx"),
            source_text="Some source text",
            prompt_kind="cultural-heritage",
        )
        self.assertEqual(
            extract_docx_title(source),
            "Music in the Châteaux of the Loire Valley: From Tours and Saint‑Martin to the Cubiculum Musicae at Châteaudun",
        )


if __name__ == "__main__":
    unittest.main()
