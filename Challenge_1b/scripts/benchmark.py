#!/usr/bin/env python3
"""Measure Challenge 1B per-collection latency and process memory."""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import fitz
from sentence_transformers import SentenceTransformer


def max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def pdf_folder(collection: Path) -> Path:
    for name in ("PDFs", "pdf"):
        candidate = collection / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No PDF folder found in {collection}")


def count_pages(collection: Path) -> tuple[int, int]:
    folder = pdf_folder(collection)
    pdfs = sorted(folder.glob("*.pdf"))
    pages = 0
    for pdf in pdfs:
        with fitz.open(pdf) as doc:
            pages += doc.page_count
    return len(pdfs), pages


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--semantic-weight", type=float, default=0.7)
    parser.add_argument("--keyword-weight", type=float, default=0.3)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.root / "src"))
    from main import process_1b_collection

    model_start = time.perf_counter()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model_load_seconds = time.perf_counter() - model_start

    rows = []
    for collection in sorted(args.root.glob("Collection *")):
        input_file = collection / "challenge1b_input.json"
        if not input_file.exists():
            continue
        pdf_count, page_count = count_pages(collection)
        start = time.perf_counter()
        process_1b_collection(
            input_file,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            model=model,
        )
        seconds = time.perf_counter() - start
        rows.append([
            collection.name,
            str(pdf_count),
            str(page_count),
            f"{seconds:.2f}s",
            f"{max_rss_mb():.0f} MB",
        ])

    print(f"Model load time: {model_load_seconds:.2f}s")
    print_table(["Collection", "PDFs", "Pages", "Processing time", "Peak RSS"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
