from __future__ import annotations

import unittest

from eventweaver.cli import build_parser
from eventweaver.prompts import (
    GENERAL_PROMPT_TEMPLATE,
    DOCX_PROMPT_STRATEGIES,
    get_prompt_template,
    list_prompt_strategies,
    validate_prompt_strategy,
)


class PromptRegistryTests(unittest.TestCase):
    def test_standard_docx_prompt_is_restored(self) -> None:
        self.assertIs(get_prompt_template("cultural-heritage", "standard"), GENERAL_PROMPT_TEMPLATE)

    def test_docx_strategies_include_requested_variants(self) -> None:
        for strategy in ("short", "detailed", "simple"):
            self.assertIn(strategy, DOCX_PROMPT_STRATEGIES)
            validate_prompt_strategy("cultural-heritage", strategy)

    def test_simple_docx_prompt_template(self) -> None:
        template = get_prompt_template("cultural-heritage", "simple")
        rendered = template.format(nation="Hungarian", title='"From the Music of Franz Liszt"', source_text="", case_study_text="FORM TEXT")
        self.assertIn('I want to create a narrative about the Hungarian music titled "From the Music of Franz Liszt". The narrative has to be divided into events.', rendered)
        self.assertIn("CASE-STUDY FORM:\nFORM TEXT", rendered)

    def test_list_prompt_strategies_for_docx(self) -> None:
        strategies = list_prompt_strategies("cultural-heritage")
        self.assertIn("standard", strategies)
        self.assertIn("short", strategies)
        self.assertIn("detailed", strategies)

    def test_incompatible_csv_strategy_is_rejected_for_docx(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown prompt strategy: numeric_aware"):
            validate_prompt_strategy("cultural-heritage", "numeric_aware")

    def test_simplified_new_is_accepted_with_requested_length(self) -> None:
        args = build_parser().parse_args(["generate", "source.docx", "--prompt-strategy", "simplified_new"])
        self.assertEqual(args.prompt_strategy, "simplified_new")
        template = get_prompt_template("cultural-heritage", args.prompt_strategy)
        self.assertIn("at 500-600 words", template)


if __name__ == "__main__":
    unittest.main()
