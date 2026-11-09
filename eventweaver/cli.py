from __future__ import annotations

import argparse
import sys
import sqlite3
from dataclasses import replace
from pathlib import Path

from .config import GenerateConfig
from .generation import generate_narratives
from .providers import build_provider
from .reorganize import reorganize_markdown
from . import prompts
from .resume import load_resume_config, resume_narratives


def build_parser():
    parser = argparse.ArgumentParser(prog="eventweaver", description="Thematic narratives based only on the supplied documents")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("generate", help="Analyze all documents, identify up to 5 themes, and generate the narratives")
    cmd.add_argument("inputs", nargs="*", type=Path, help="Files or directories (recursive): PDF, DOCX, TXT, MD, CSV")
    cmd.add_argument("--resume-run", type=Path, help="Resume reducing a saved draft in a new run")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--provider", choices=["ollama", "huggingface-api", "huggingface-local"], default="ollama")
    cmd.add_argument("--output-dir", type=Path, default=Path("outputs"))
    cmd.add_argument("--max-themes", type=int, default=5)
    cmd.add_argument("--titles", nargs="+", default=[], help="Narrative titles in quotes, from 1 to 5")
    cmd.add_argument("--max-words", type=int, default=None, help="Maximum words per narrative, excluding the title")
    cmd.add_argument("--language", default="English")
    cmd.add_argument("--write-prompt", default="WRITE", help="Named prompt constant from eventweaver/prompts.py")
    cmd.add_argument("--num-ctx", type=int, default=16384, help="Total context window also configured in the model/server")
    cmd.add_argument("--max-new-tokens", type=int, default=4096)
    cmd.add_argument("--chunk-chars", type=int, default=3000)
    cmd.add_argument("--overlap-chars", type=int, default=300)
    cmd.add_argument("--frequency-penalty", type=float, default=1.0,
                     help="Ollama penalty for token frequency in prose (0..2, default 1.0)")
    cmd.add_argument("--presence-penalty", type=float, default=0.8,
                     help="Ollama penalty for tokens already present in prose (0..2, default 0.8)")
    cmd.add_argument("--json-retries", type=int, default=2)
    cmd.add_argument("--ocr", choices=["auto", "always", "never"], default="auto")
    cmd.add_argument("--ocr-language", default="ita+eng")
    cmd.add_argument("--ocr-dpi", type=int, default=200)
    cmd.add_argument("--cache-dir", default=".eventweaver_cache",
                     help="Persistent cache for PDF text, relative to the current directory")
    cache_control = cmd.add_mutually_exclusive_group()
    cache_control.add_argument("--refresh-ocr", action="store_true",
                               help="Re-extract PDFs and update the cache, ignoring saved text")
    cache_control.add_argument("--no-cache", action="store_true",
                               help="Disable reading and writing the cache")
    cmd.add_argument("--ollama-host")
    cmd.add_argument("--ollama-timeout", type=int, default=900)
    cmd.add_argument("--ollama-retries", type=int, default=1)
    cmd.add_argument("--skip-ollama-preflight", action="store_true")
    cmd.add_argument("--hf-token-env", default="HF_TOKEN")
    cmd.add_argument("--hf-provider", default="auto")
    cmd.add_argument("--hf-timeout", type=int, default=120)
    reorganize = sub.add_parser("reorganize", help="Rewrite and regroup a Markdown story")
    reorganize.add_argument("input", type=Path, help="Markdown story to reorganize")
    prompt_source = reorganize.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt-file", type=Path, help="External prompt file")
    prompt_source.add_argument("--prompt-name", help="Named prompt constant from eventweaver/prompts.py")
    reorganize.add_argument("--output", type=Path, help="File Markdown di output")
    reorganize.add_argument("--model", required=True)
    reorganize.add_argument("--provider", choices=["ollama", "huggingface-api", "huggingface-local"], default="ollama")
    reorganize.add_argument("--num-ctx", type=int, default=32768)
    reorganize.add_argument("--max-new-tokens", type=int, default=12000)
    reorganize.add_argument("--ollama-host")
    reorganize.add_argument("--ollama-timeout", type=int, default=900)
    reorganize.add_argument("--ollama-retries", type=int, default=1)
    reorganize.add_argument("--skip-ollama-preflight", action="store_true")
    reorganize.add_argument("--frequency-penalty", type=float, default=1.0)
    reorganize.add_argument("--presence-penalty", type=float, default=0.8)
    return parser


def main(argv=None):
    parser = build_parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_args)
    if args.command == "reorganize":
        try:
            provider = build_provider(
                args.provider,
                args.model,
                num_ctx=args.num_ctx,
                ollama_host=args.ollama_host,
                ollama_timeout=args.ollama_timeout,
                ollama_retries=args.ollama_retries,
                skip_ollama_preflight=args.skip_ollama_preflight,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
            )
            prompt_path = args.prompt_file or Path(__file__).resolve().parent / "reorganize_prompt.txt"
            prompt_text = prompts.get_prompt(args.prompt_name) if args.prompt_name else None
            print(reorganize_markdown(
                args.input, prompt_path, provider, args.output, args.max_new_tokens, prompt_text
            ))
            return 0
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if not args.inputs and args.resume_run is None:
        parser.error("specify at least one input or --resume-run")
    if args.inputs and args.resume_run is not None:
        parser.error("do not specify input documents with --resume-run")
    provided = {value.split("=", 1)[0] for value in raw_args if value.startswith("--")}
    try:
        if args.provider != "ollama" and provided & {"--frequency-penalty", "--presence-penalty"}:
            raise ValueError("--frequency-penalty and --presence-penalty are available with --provider ollama")
        if args.resume_run is None:
            config = GenerateConfig(max_themes=args.max_themes, language=args.language,
                                     narrative_titles=tuple(args.titles), max_narrative_words=args.max_words,
                                     write_prompt_name=args.write_prompt,
                                    context_tokens=args.num_ctx, max_output_tokens=args.max_new_tokens,
                                    chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars,
                                    retries=args.json_retries, ocr_mode=args.ocr,
                                    frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty,
                                    ocr_language=args.ocr_language, ocr_dpi=args.ocr_dpi,
                                    cache_enabled=not args.no_cache, cache_dir=args.cache_dir,
                                    refresh_ocr=args.refresh_ocr)
        else:
            incompatible = provided & {"--titles", "--language", "--max-themes", "--ocr", "--ocr-language", "--ocr-dpi", "--overlap-chars", "--cache-dir", "--refresh-ocr", "--no-cache"}
            if incompatible:
                raise ValueError("Resume uses saved text and does not read or OCR documents; remove: " + ", ".join(sorted(incompatible)))
            config = load_resume_config(args.resume_run)
            mapping = {"--max-words": ("max_narrative_words", args.max_words),
                       "--num-ctx": ("context_tokens", args.num_ctx),
                       "--max-new-tokens": ("max_output_tokens", args.max_new_tokens),
                       "--json-retries": ("retries", args.json_retries),
                       "--chunk-chars": ("chunk_chars", args.chunk_chars),
                       "--write-prompt": ("write_prompt_name", args.write_prompt),
                       "--frequency-penalty": ("frequency_penalty", args.frequency_penalty),
                       "--presence-penalty": ("presence_penalty", args.presence_penalty)}
            config = replace(config, **{field: value for flag, (field, value) in mapping.items() if flag in provided})
        provider = build_provider(args.provider, args.model, num_ctx=config.context_tokens,
                                  ollama_host=args.ollama_host, ollama_timeout=args.ollama_timeout,
                                  ollama_retries=args.ollama_retries,
                                  skip_ollama_preflight=args.skip_ollama_preflight,
                                  json_mode=True, frequency_penalty=config.frequency_penalty,
                                  presence_penalty=config.presence_penalty, hf_token_env=args.hf_token_env,
                                  hf_provider=args.hf_provider, hf_timeout=args.hf_timeout)
        def progress(message):
            print(message, file=sys.stderr)
        if args.resume_run is not None:
            destination = args.output_dir if "--output-dir" in provided else args.resume_run.resolve().parent
            result = resume_narratives(args.resume_run, output_dir=destination,
                                       provider=provider, config=config, progress=progress)
        else:
            result = generate_narratives(args.inputs, output_dir=args.output_dir,
                                         provider=provider, config=config, progress=progress)
        print(result)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
