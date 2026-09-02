#!/usr/bin/env python3
"""
AI Resume Screening — CLI entry point.

Usage:
    python main.py --input ./resumes --output ./output/results.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import settings
from src.pipeline import run_batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-assisted resume screening pipeline")
    parser.add_argument("--input", required=True, help="Directory containing PDF resumes")
    parser.add_argument("--output", required=True, help="Path to write results.json")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"Max concurrent workers (default: {settings.max_workers}, from MAX_WORKERS env var)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("main")

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not settings.cohere_enabled:
        logger.warning(
            "COHERE_API_KEY is not set — running with deterministic extraction only. "
            "Set COHERE_API_KEY (see .env) to enable semantic analysis."
        )

    try:
        batch_result = run_batch(input_dir, output_path, concurrency=args.concurrency)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    summary = batch_result.summary
    logger.info(
        "Done. discovered=%d parsed=%d eligible=%d rejected=%d failed=%d -> %s",
        summary.total_discovered,
        summary.successfully_parsed,
        summary.eligible_count,
        summary.rejected_count,
        summary.failed_count,
        output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
