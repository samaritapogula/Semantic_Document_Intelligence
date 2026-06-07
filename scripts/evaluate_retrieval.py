#!/usr/bin/env python3
"""Evaluate retrieval outputs and scoring-weight ablations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


GROUND_TRUTH = {
    "collection_1": {
        ("sump_guidelines_2019_interactive_document_1.pdf", "Urban Mobility Plans"),
        ("Final-version-Public-Transport-Declaration-Barcelona-FINAL.pdf", "We, the representatives of major European cities and public transport organisations, have"),
        ("Final-version-Public-Transport-Declaration-Barcelona-FINAL.pdf", "On the supply industry to further  develop innovation in public transport"),
        ("sump_guidelines_2019_interactive_document_1.pdf", "European learning programme for cities"),
        ("UITP-EU-Political-priorities.pdf", "FOR THE LEGISLATIVE TERM 2024-2029"),
    },
    "collection_2": {
        ("CourseBased_SLOmodules.pdf", "Benander created this set of modules so"),
        ("CourseBased_SLOmodules.pdf", "Student Learning Outcomes and Activities"),
        ("Course-Learning-Objectives-guide-2018.pdf", "Sample Verbs to Utilize*"),
        ("COE Writing Learning Objectives2023.pdf", "Compare, Correlate, Criticize, Discern, Deduce, Designate,"),
        ("COE Writing Learning Objectives2023.pdf", "Learning objectives (also called  \u201clearning outcomes\u201d  or  \u201clearning targets \u201d) are specific,"),
    },
    "collection_3": {
        ("IQBBA CFLBA Syllabus v. 3.0.pdf", "Introduction to this Syllabus"),
        ("IQBBA CFLBA Syllabus v. 3.0.pdf", "Table of Contents"),
    },
}


WEIGHTS = {
    "keyword 0/100": (0.0, 1.0),
    "balanced 50/50": (0.5, 0.5),
    "hybrid 70/30": (0.7, 0.3),
    "semantic-heavy 90/10": (0.9, 0.1),
    "semantic 100/0": (1.0, 0.0),
}


def normalize(item: tuple[str, str]) -> tuple[str, str]:
    return item[0].strip(), " ".join(item[1].split()).lower()


def section_key(section: dict) -> tuple[str, str]:
    return normalize((section["document"], section["section_title"]))


def metrics(sections: list[dict], relevant: set[tuple[str, str]], k: int = 5) -> tuple[float, float, int]:
    relevant_norm = {normalize(item) for item in relevant}
    top = sections[:k]
    hits = sum(1 for section in top if section_key(section) in relevant_norm)
    precision = hits / k
    recall = hits / len(relevant_norm) if relevant_norm else 0.0
    return precision, recall, hits


def evaluate_existing(root: Path, k: int) -> list[dict]:
    rows = []
    for collection, relevant in GROUND_TRUTH.items():
        output_path = root / "collections" /collection / "output.json"
        output = json.loads(output_path.read_text(encoding="utf-8"))
        precision, recall, hits = metrics(output["extracted_sections"], relevant, k)
        rows.append({
            "collection": collection,
            "variant": "stored output",
            "precision": precision,
            "recall": recall,
            "hits": hits,
        })
    return rows


def evaluate_ablation(root: Path, k: int) -> list[dict]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(root / "src"))
    from retrieval_pipeline import process_collection

    model = SentenceTransformer("all-MiniLM-L6-v2")
    rows = []
    for variant, (semantic_weight, keyword_weight) in WEIGHTS.items():
        precisions = []
        recalls = []
        hits_total = 0
        for collection, relevant in GROUND_TRUTH.items():
            result = process_collection(
                root /"collections" / collection / "input.json",
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                top_k=k,
                model=model,
            )
            precision, recall, hits = metrics(result["extracted_sections"], relevant, k)
            precisions.append(precision)
            recalls.append(recall)
            hits_total += hits
        rows.append({
            "variant": variant,
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "hits": hits_total,
        })
    return rows


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    existing = evaluate_existing(args.root, args.k)
    print("Existing output evaluation")
    print_table(
        ["Collection", f"Precision@{args.k}", f"Recall@{args.k}", "Hits"],
        [[row["collection"], f"{row['precision']:.2f}", f"{row['recall']:.2f}", str(row["hits"])] for row in existing],
    )

    if not args.skip_ablation:
        print()
        print("Weight ablation")
        ablation = evaluate_ablation(args.root, args.k)
        print_table(
            ["Scoring variant", f"Mean Precision@{args.k}", f"Mean Recall@{args.k}", "Relevant hits"],
            [[row["variant"], f"{row['precision']:.2f}", f"{row['recall']:.2f}", str(row["hits"])] for row in ablation],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
