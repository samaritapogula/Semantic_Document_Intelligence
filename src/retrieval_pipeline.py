# PDF Retrieval Pipeline

import json
import os
import re
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

HEADER_FOOTER_REPETITION_THRESHOLD = 0.5
MIN_HEADING_WORDS = 1
MAX_HEADING_WORDS = 12
MAX_SECTIONS_PER_DOCUMENT = 2
DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_TOP_K = 5
max_heading_levels=4
COLON_SPLIT_ENABLED = False
COLON_PREFIX_MAX_WORDS = 6
CHUNK_TOP_PADDING_RATIO = 0.2
MIN_SECTION_WORDS = 10
DEBUG_CHUNK_FILTERING = False


def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def keyword_score(text, keywords):
    if not keywords:
        return 0.0
    return sum(1 for word in keywords if word.lower() in text.lower()) / len(keywords)


def is_similar(t1, t2):
    t1 = re.sub(r'[^a-zA-Z0-9]', '', t1).lower()
    t2 = re.sub(r'[^a-zA-Z0-9]', '', t2).lower()
    return t1 in t2 or t2 in t1


def is_heading_like(text):
    return (
        1 <= len(text.split()) <= 12 and
        text[0].isupper() and
        text[-1] not in ".!"
        
    )


def generate_dynamic_queries(role: str, task: str):
    role = role.strip()
    task = task.strip()
    return [
        f"What parts of the document help a {role} to {task}?",
        f"Which sections are most useful for a {role} when trying to {task}?",
        f"As a {role}, which content supports the task to {task}?",
        f"Where are the objectives, outcomes, or results mentioned that would help {role}s in {task}?",
        f"Extract sections that describe how to {task} from a {role}'s perspective.",
        f"Which parts of the document explain what users should know or be able to do for {task}?",
        f"What instructional or learning goals are aligned with the task to {task}?",
        f"As a {role}, find passages that summarize student expectations or intended learning outcomes.",
        f"What content in this document would be relevant for someone designing learning materials to {task}?",
        f"What parts of this document clearly support the goal to {task}?"
    ]


def find_headings_and_title(doc):
    text_counts = {}
    blocks = []

    for page_num, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:

            if b["type"] != 0:
                continue

            for line in b["lines"]:

                if not line["spans"]:
                    continue

                text = " ".join(
                    [s["text"] for s in line["spans"]]
                ).strip()

                if COLON_SPLIT_ENABLED and ":" in text:

                    prefix = text.split(":", 1)[0].strip()

                    if len(prefix.split()) <= COLON_PREFIX_MAX_WORDS:

                        # Optional debug logging
                        # print(
                        #     f"Colon-truncated heading: "
                        #     f"'{text}' -> '{prefix}'"
                        # )

                        text = prefix

                if not text:
                    continue

                span = line["spans"][0]

                blocks.append({
                    "text": text,
                    "size": round(span["size"]),
                    "is_bold": (span["flags"] & 16) != 0,
                    "bbox": line["bbox"],
                    "page_num": page_num + 1
                })

                text_counts[text] = (
                    text_counts.get(text, 0) + 1
                )

    if not blocks:
        return {"title": "Error: No text", "outline": []}

    blocks.sort(key=lambda b: (b['page_num'], b['bbox'][1]))

    # Title extraction
    title, title_bbox = "", None
    common = {k: v for k, v in text_counts.items() if v > doc.page_count * HEADER_FOOTER_REPETITION_THRESHOLD}
    if common:
        title = max(common, key=common.get)
        header_footer_texts = set(common.keys())

    else:

        first_page_blocks = [
            b for b in blocks
            if b['page_num'] == 1
        ]

        if not first_page_blocks:
            return {"title": "Untitled", "outline": []}

        max_size = max(
            b['size'] for b in first_page_blocks
        )

        page_height = doc[0].rect.height

        best_score = -1

        for b in first_page_blocks:

            font_score = b['size'] / max_size

            # Higher score for text near top
            position_score = (
                1 - (b['bbox'][1] / page_height)
            )

            # Weighted combination
            score = (
                0.7 * font_score
                + 0.3 * position_score
            )

            if score > best_score:

                best_score = score
                title = b['text']
                title_bbox = b['bbox']

        header_footer_texts = set()

    # Heading levels
    usable = [b for b in blocks if b['text'] not in header_footer_texts and b['bbox'] != title_bbox]
    body_size = max(set([b['size'] for b in usable]), key=[b['size'] for b in usable].count)
    heading_sizes = sorted(set(b['size'] for b in usable if b['size'] > body_size or b['is_bold']), reverse=True)[:4]
    size_to_level = {sz: f"H{i+1}" for i, sz in enumerate(heading_sizes)}

    outline = []
    for b in usable:
        if b['size'] in size_to_level and is_heading_like(b['text']):
            outline.append({
                "level": size_to_level[b['size']],
                "text": b['text'],
                "page": b['page_num'],
                "bbox": b['bbox']
            })

    return {"title": title, "outline": outline}


def extract_chunks_from_doc(doc, doc_name):

    structure = find_headings_and_title(doc)
    headings = structure["outline"]

    chunks = []

    for i, h in enumerate(headings):

        page = doc[h['page'] - 1]

        heading_height = (
            h['bbox'][3] - h['bbox'][1]
        )

        padding = (
            heading_height * CHUNK_TOP_PADDING_RATIO
        )

        y0 = max(
            h['bbox'][1] - padding,
            0
        )

        y1 = page.rect.height

        for j in range(i + 1, len(headings)):

            if headings[j]['page'] == h['page']:

                y1 = headings[j]['bbox'][1]
                break

        clip = fitz.Rect(
            0,
            y0,
            page.rect.width,
            y1
        )

        txt = clean_text(
            page.get_text("text", clip=clip)
        )

        word_count = len(txt.split())

        if word_count >= MIN_SECTION_WORDS:

            chunks.append({
                "document": doc_name,
                "page_number": h['page'],
                "section_title": h['text'],
                "text": txt
            })

        elif DEBUG_CHUNK_FILTERING:

            print(
                f"Dropped short chunk "
                f"({word_count} words): "
                f"{h['text'][:80]}"
            )

    return chunks


def resolve_pdf_folder(input_json_path):
    base = Path(input_json_path).parent

    pdf_dirs = []

    for candidate in base.rglob("*"):
        if candidate.is_dir() and list(candidate.glob("*.pdf")):
            pdf_dirs.append(candidate)

    if len(pdf_dirs) == 1:
        return pdf_dirs[0]

    if len(pdf_dirs) > 1:
        raise ValueError(
            f"Multiple PDF directories found: {pdf_dirs}"
        )

    raise FileNotFoundError(
        f"No directory containing PDF files found beside {input_json_path}"
    )

def load_model():
    if os.getenv("MODEL_LOCAL_ONLY", "0") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return SentenceTransformer('all-MiniLM-L6-v2')


def process_collection(input_json_path, semantic_weight=0.7, keyword_weight=0.3, top_k=5, max_per_doc=MAX_SECTIONS_PER_DOCUMENT,model=None):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    role = input_data['persona']['role']
    task = input_data['job_to_be_done']['task']
    pdfs = [doc['filename'] for doc in input_data['documents']]
    keywords = set(role.lower().split() + task.lower().split())

    queries = generate_dynamic_queries(role, task)

    if model is None:
        model = load_model()
    q_embed = model.encode(queries, convert_to_tensor=True).mean(dim=0)

    all_chunks = []
    pdf_folder = resolve_pdf_folder(input_json_path)
    for fname in pdfs:
        doc = fitz.open(pdf_folder / fname)
        chunks = extract_chunks_from_doc(doc, fname)
        all_chunks.extend(chunks)

    if not all_chunks:
        return {
            "metadata": {
                "input_documents": pdfs,
                "persona": role,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

    c_embeds = model.encode([c['text'] for c in all_chunks], convert_to_tensor=True)
    scores = util.cos_sim(q_embed, c_embeds)[0]

    for i, c in enumerate(all_chunks):
        c['score'] = (
            semantic_weight * scores[i].item() +
            keyword_weight * keyword_score(c['text'], keywords)
        )

    top_chunks = sorted(all_chunks, key=lambda x: x['score'], reverse=True)
    selected, seen_docs, seen_titles = [], {}, []

    for chunk in top_chunks:
        if any(is_similar(chunk['section_title'], t) for t in seen_titles):
            continue
        if seen_docs.get(chunk['document'], 0) < max_per_doc:
            selected.append(chunk)
            seen_docs[chunk['document']] = seen_docs.get(chunk['document'], 0) + 1
            seen_titles.append(chunk['section_title'])
        if len(selected) >= top_k:
            break

    output = {
        "metadata": {
            "input_documents": pdfs,
            "persona": role,
            "job_to_be_done": task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": c['document'],
                "section_title": c['section_title'],
                "importance_rank": i + 1,
                "page_number": c['page_number'],
                "relevance_score": round(c['score'], 4)
            } for i, c in enumerate(selected)
        ],
        "subsection_analysis": [
            {
                "document": c['document'],
                "refined_text": c['text'],
                "page_number": c['page_number']
            } for c in selected
        ]
    }

    return output


def run_on_all_collections(base_path=None):
    base = Path(base_path) if base_path else Path(__file__).parent.parent
    semantic_weight = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    keyword_weight = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
    top_k = int(os.getenv("TOP_K", "5"))
    model = load_model()
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        input_file = folder / "input.json"
        if not input_file.exists():
            continue
        output_file = folder / "output.json"
        if input_file.exists():
            print(f"Processing: {input_file}")
            result = process_collection(
                input_file,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                top_k=top_k,
                model=model
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Root directory containing collection folders"
    )

    args = parser.parse_args()

    print("Starting processor")
    run_on_all_collections(args.base_path)
    print("All collections processed.")