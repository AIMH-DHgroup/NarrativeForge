from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import requests


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


@dataclass
class OllamaResult:
    text: str
    runtime_seconds: float
    error: str | None = None


def resolve_ollama_host(ollama_host: str | None = None) -> str:
    host = (ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).strip()
    if not host:
        host = DEFAULT_OLLAMA_HOST
    if not re.match(r"^https?://", host, flags=re.I):
        host = f"http://{host}"
    return host.rstrip("/")


def ollama_api_url(ollama_host: str | None, path: str) -> str:
    return urljoin(resolve_ollama_host(ollama_host) + "/", path.lstrip("/"))


def _connection_refused_message(host: str, exc: Exception) -> str:
    detail = str(exc)
    if "10061" in detail or "Connection refused" in detail or "Failed to establish a new connection" in detail:
        return (
            f"Windows refused the connection to Ollama at {host}. "
            "This usually means the Ollama server is not running or is not listening on port 11434. "
            "Start it with 'ollama serve', then verify with 'ollama list' and "
            f"'Invoke-WebRequest {ollama_api_url(host, '/api/tags')}'."
        )
    return f"Could not connect to Ollama at {host}: {detail}"


def list_ollama_models(ollama_host: str | None = None, timeout: int = 30) -> list[str]:
    host = resolve_ollama_host(ollama_host)
    try:
        response = requests.get(ollama_api_url(host, "/api/tags"), timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(_connection_refused_message(host, exc)) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Timed out while checking Ollama at {host}. Verify the server is responsive with 'ollama list'.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama health check failed at {host}: {exc}") from exc

    payload = response.json()
    return sorted(str(model.get("name", "")).strip() for model in payload.get("models", []) if str(model.get("name", "")).strip())


def preflight_ollama(models: list[str], ollama_host: str | None = None, timeout: int = 30) -> list[str]:
    host = resolve_ollama_host(ollama_host)
    available = list_ollama_models(host, timeout=timeout)
    missing = sorted(set(models) - set(available))
    if missing:
        raise RuntimeError(
            "The following requested Ollama model(s) are not installed locally: "
            + ", ".join(missing)
            + ". Install them with 'ollama pull <model>' or choose a different --models/--model-preset value."
        )
    return available


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def generate_ollama(
    prompt: str,
    model: str,
    temperature: float = 0.1,
    num_ctx: int = 8192,
    timeout: int = 900,
    ollama_host: str | None = None,
    retries: int = 1,
) -> OllamaResult:
    host = resolve_ollama_host(ollama_host)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    started = time.perf_counter()
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(ollama_api_url(host, "/api/generate"), json=payload, timeout=timeout)
            if response.status_code >= 500 and attempt < attempts:
                time.sleep(min(2.0, 0.5 * attempt))
                continue
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if "deepseek" in model.lower():
                text = remove_think_blocks(text)
            return OllamaResult(text=text, runtime_seconds=time.perf_counter() - started)
        except requests.exceptions.ConnectionError as exc:
            return OllamaResult(text="", runtime_seconds=time.perf_counter() - started, error=_connection_refused_message(host, exc))
        except requests.exceptions.Timeout:
            return OllamaResult(text="", runtime_seconds=time.perf_counter() - started, error=f"Timed out calling Ollama at {host} after {timeout} seconds.")
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            return OllamaResult(text="", runtime_seconds=time.perf_counter() - started, error=f"Ollama returned HTTP {status} for model {model} at {host}: {exc}")
        except requests.exceptions.RequestException as exc:
            return OllamaResult(text="", runtime_seconds=time.perf_counter() - started, error=f"Ollama request failed at {host}: {exc}")
        except Exception as exc:
            return OllamaResult(text="", runtime_seconds=time.perf_counter() - started, error=str(exc))
    return OllamaResult(text="", runtime_seconds=time.perf_counter() - started, error=f"Ollama generation failed at {host} after {attempts} attempts.")
