# Challenge_1b/process_pdfs.py

import json
import os
import re
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import torch


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
        3 <= len(text.split()) <= 12 and
        text[0].isupper() and
        text[-1] not in ".!?" and
        not text.lower().startswith("here are") and
        not text.lower().startswith("for example")
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
        f"Which parts of the syllabus explain what students should know or be able to do, as needed by a {role}?",
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
                text = " ".join([s['text'] for s in line['spans']]).strip()
                if ":" in text and len(text.split(":")[0].split()) < 7:
                    text = text.split(":")[0]
                if not text:
                    continue
                span = line['spans'][0]
                blocks.append({
                    "text": text,
                    "size": round(span["size"]),
                    "is_bold": (span["flags"] & 16) != 0,
                    "bbox": line["bbox"],
                    "page_num": page_num + 1
                })
                text_counts[text] = text_counts.get(text, 0) + 1

    if not blocks:
        return {"title": "Error: No text", "outline": []}

    blocks.sort(key=lambda b: (b['page_num'], b['bbox'][1]))

    # Title extraction
    title, title_bbox = "", None
    common = {k: v for k, v in text_counts.items() if v > doc.page_count * 0.5}
    if common:
        title = max(common, key=common.get)
        header_footer_texts = set(common.keys())
    else:
        best_score = -1
        for b in [b for b in blocks if b['page_num'] == 1]:
            score = b['size'] + 1000 / (b['bbox'][1] + 1)
            if score > best_score:
                best_score, title, title_bbox = score, b['text'], b['bbox']
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
        y0 = max(h['bbox'][1] - 5, 0)
        y1 = page.rect.height
        for j in range(i+1, len(headings)):
            if headings[j]['page'] == h['page']:
                y1 = headings[j]['bbox'][1]
                break
        clip = fitz.Rect(0, y0, page.rect.width, y1)
        txt = clean_text(page.get_text("text", clip=clip))
        if len(txt.split()) >= 30:
            chunks.append({
                "document": doc_name,
                "page_number": h['page'],
                "section_title": h['text'],
                "text": txt
            })
    return chunks


def resolve_pdf_folder(input_json_path):
    base = Path(input_json_path).parent
    for name in ("PDFs", "pdf"):
        candidate = base / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No PDF folder found beside {input_json_path}; expected PDFs/ or pdf/")


def load_model():
    if os.getenv("MODEL_LOCAL_ONLY", "0") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return SentenceTransformer('all-MiniLM-L6-v2')


def process_1b_collection(input_json_path, semantic_weight=0.7, keyword_weight=0.3, top_k=5, model=None):
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
        if seen_docs.get(chunk['document'], 0) < 2:
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


def run_on_all_collections():
    base = Path(__file__).parent.parent  
    semantic_weight = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    keyword_weight = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
    top_k = int(os.getenv("TOP_K", "5"))
    model = load_model()
    for folder in base.glob("Collection*/"):
        input_file = folder / "challenge1b_input.json"
        output_file = folder / "challenge1b_output.json"
        if input_file.exists():
            print(f"Processing: {input_file}")
            result = process_1b_collection(
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
    print("Starting Challenge 1B processor")
    run_on_all_collections()
    print("All collections processed.")
