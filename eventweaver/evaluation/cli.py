from pathlib import Path

from .config import METRIC_IDS, load_config
from .runner import run_evaluation


def add_parser(subcommands):
    parser = subcommands.add_parser("evaluate", help="Independently evaluate an existing Markdown narrative")
    parser.add_argument("narrative", type=Path)
    parser.add_argument("--sources", required=True, nargs="+", type=Path)
    parser.add_argument("--metrics", default=",".join(METRIC_IDS),
                        help="Comma-separated independent metric IDs: " + ",".join(METRIC_IDS))
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; never overwrite an existing run")
    parser.add_argument("--config", type=Path, help="JSON object with separate configurations by metric ID")
    parser.add_argument("--narrative-id")
    parser.add_argument("--original-prompt", type=Path, help="UTF-8 file containing the original generation prompt")


def execute(args):
    config = load_config(args.config)
    prompt = args.original_prompt.read_text(encoding="utf-8") if args.original_prompt else None
    statuses = run_evaluation(
        args.narrative, args.sources, args.output_dir, args.metrics.split(","),
        config=config, narrative_id=args.narrative_id, original_prompt=prompt,
    )
    for name, status in statuses.items():
        print(f"{name}: {status}")
    print(args.output_dir)
    # Every requested metric must complete to return success; files still exist for other outcomes.
    return 0 if all(status == "completed" for status in statuses.values()) else 1
