from ..base import unavailable


class AlignScoreEvaluator:
    reference = "https://github.com/factiverse/alignscore/tree/247c7f8549694111155b6b13b55bc39103280ddc"
    method = "Reference AlignScore adapter not enabled in increment 1"

    def evaluate(self, inputs, config):
        return unavailable(
            "Not implemented: needs a verified checkpoint, isolated runtime and output contract. "
            "Factiverse nli_sp returns three class scores unlike upstream scalar AlignScore; "
            "no embedding or scalar substitute is computed."
        )
