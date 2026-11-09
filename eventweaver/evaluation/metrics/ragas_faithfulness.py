from ..base import unavailable


class RagasFaithfulnessEvaluator:
    reference = "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/"
    method = "Official RAGAS Faithfulness adapter not enabled in increment 1"

    def evaluate(self, inputs, config):
        return unavailable(
            "Not implemented: requires a pinned RAGAS runtime and independently validated judge. "
            "No FActScore claims are reused. No context is silently selected or truncated."
        )
