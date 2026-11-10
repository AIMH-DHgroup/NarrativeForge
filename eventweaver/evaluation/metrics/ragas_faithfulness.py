import asyncio
import importlib.metadata
import math
import os

from ..base import unavailable
from ..contexts import positive_integer, source_contexts

RAGAS_VERSION = "0.4.3"


def make_metric(llm):
    from ragas.metrics.collections import Faithfulness

    class TracedFaithfulness(Faithfulness):
        async def _create_statements(self, question, response):
            self.extracted_statements = await super()._create_statements(question, response)
            return self.extracted_statements

        async def _create_verdicts(self, statements, context):
            result = await super()._create_verdicts(statements, context)
            self.verdict_records = result.model_dump()["statements"]
            return result

    metric = TracedFaithfulness(llm=llm)
    metric.extracted_statements = []
    metric.verdict_records = []
    return metric


async def run_official(config, question, response, contexts):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    timeout = positive_integer(config, "timeout_seconds", 300)
    async with AsyncOpenAI(api_key=os.environ[config.get("api_key_env", "OPENAI_API_KEY")],
                           base_url=config.get("base_url"), timeout=timeout, max_retries=0) as client:
        llm = llm_factory(config["model"], client=client, temperature=0.0, top_p=1.0,
                          max_tokens=positive_integer(config, "max_tokens", 4096))
        metric = make_metric(llm)
        runtime = {"effective_model_parameters": llm._map_provider_params()}
        try:
            result = await asyncio.wait_for(metric.ascore(user_input=question, response=response,
                                                          retrieved_contexts=contexts), timeout=timeout)
            score = result.value
        except Exception as exc:
            runtime["error"] = f"{type(exc).__name__}: {exc}"
            score = None
        return score, metric.extracted_statements, metric.verdict_records, runtime


class RagasFaithfulnessEvaluator:
    reference = "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/"
    method = "Official RAGAS 0.4.3 collections.Faithfulness; independent extraction and NLI"

    def evaluate(self, inputs, config):
        if not config.get("model"):
            return unavailable("Configure ragas_faithfulness.model and an OpenAI-compatible structured-output judge")
        question = config.get("user_input") or inputs.original_prompt
        if not isinstance(question, str) or not question.strip():
            raise ValueError("RAGAS needs the original prompt or explicit user_input; no question is invented")
        contexts = source_contexts(inputs.corpus, positive_integer(config, "chunk_chars", 4000))
        if not inputs.narrative["body"].strip():
            raise ValueError("RAGAS response must not be empty")
        total = len(question) + len(inputs.narrative["body"]) + sum(len(item["text"]) for item in contexts)
        details = {"contexts_used": contexts, "context_strategy": "all pages, consecutive character chunks, zero overlap, no retrieval",
                   "runtime_configuration": {"ragas_version": RAGAS_VERSION, "judge_model": config["model"],
                                             "requested_temperature": 0.0, "requested_top_p": 1.0,
                                             "max_tokens": positive_integer(config, "max_tokens", 4096)},
                   "warnings": ["Judge-dependent metric; not scientifically validated for cultural heritage.",
                                "A judge model alias is not an immutable weight snapshot; provision/pin the model server separately for repeatability.",
                                "Input character budget is not a tokenizer bound; configure the judge context window accordingly.",
                                "Verdicts refer to the complete joined context; RAGAS does not supply exact evidence offsets."]}
        if total > positive_integer(config, "max_input_chars", 60000):
            return dict(unavailable("Full context exceeds max_input_chars; no evidence was silently truncated"), **details)
        try:
            installed = importlib.metadata.version("ragas")
        except importlib.metadata.PackageNotFoundError:
            return dict(unavailable("Install the optional evaluation-ragas dependencies (ragas==0.4.3)"), **details)
        if installed != RAGAS_VERSION:
            return dict(unavailable(f"Supported RAGAS version is {RAGAS_VERSION}; found {installed}"), **details)
        if not os.getenv(config.get("api_key_env", "OPENAI_API_KEY")):
            return dict(unavailable("Set the configured judge API-key environment variable; local compatible servers can use a placeholder key"), **details)
        try:
            score, statements, verdicts, runtime = asyncio.run(run_official(
                config, question, inputs.narrative["body"], [item["text"] for item in contexts]))
        except ImportError as exc:
            return dict(unavailable(f"Missing RAGAS runtime dependency: {exc}; install .[evaluation-ragas]"), **details)
        details.update(statements=statements, verdicts=verdicts, user_input=question)
        details["runtime_configuration"].update(runtime)
        effective = runtime.get("effective_model_parameters", {})
        if effective.get("temperature", 0.0) != 0.0 or effective.get("top_p", 1.0) != 1.0:
            details["warnings"].append("RAGAS remapped sampling parameters for this judge; effective_model_parameters records the actual provider settings")
        if "error" in runtime:
            return dict(details, status="failed", score=None, error=runtime["error"])
        if not statements and not verdicts:
            return dict(details, status="uncertain", score=None)
        if [item.get("statement") for item in verdicts] != statements:
            return dict(details, status="failed", score=None, error="Official verdict list does not match all extracted statements")
        if any(type(item.get("verdict")) is not int or item["verdict"] not in (0, 1) for item in verdicts):
            return dict(details, status="failed", score=None, error="Invalid RAGAS verdict value")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            return dict(details, status="uncertain", score=None)
        if not 0 <= score <= 1 or not math.isclose(score, sum(item["verdict"] for item in verdicts) / len(verdicts)):
            return dict(details, status="failed", score=None, error="Score disagrees with official verdict ratio")
        return dict(details, status="completed", score=score)
