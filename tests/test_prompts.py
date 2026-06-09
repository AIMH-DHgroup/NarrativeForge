from __future__ import annotations

import unittest

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
        for strategy in ("short", "detailed", "strict", "event_focused", "faithfulness_first", "digital_heritage_focused"):
            self.assertIn(strategy, DOCX_PROMPT_STRATEGIES)
            validate_prompt_strategy("cultural-heritage", strategy)

    def test_list_prompt_strategies_for_docx(self) -> None:
        strategies = list_prompt_strategies("cultural-heritage")
        self.assertIn("standard", strategies)
        self.assertIn("short", strategies)
        self.assertIn("detailed", strategies)

    def test_incompatible_csv_strategy_is_rejected_for_docx(self) -> None:
        with self.assertRaisesRegex(ValueError, "only valid for value-chain CSV prompts"):
            validate_prompt_strategy("cultural-heritage", "numeric_aware")


if __name__ == "__main__":
    unittest.main()
