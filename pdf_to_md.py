"""
PDF → Markdown batch converter
Uses GPT-4o vision (page-by-page images) to handle text, tables, graphs, and diagrams.

Requirements:
    pip install openai pdf2image pillow
    Also needs poppler installed:
        macOS:   brew install poppler
        Ubuntu:  apt install poppler-utils
        Windows: download from https://github.com/oschwartz10612/poppler-windows/releases

Usage:
    python pdf_to_md.py /path/to/your/folder

Output:
    A mirrored folder structure at /path/to/your/folder_md/
    Every .pdf becomes a .md file. Other files are skipped.
"""

import os
import sys
import base64
import time
import threading
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set the OPENAI_API_KEY environment variable")
MODEL               = "gpt-4o"
MAX_TOKENS          = 4096         # per page — increase if pages get cut off
DPI                 = 100          # lowered from 150 for faster processing
OUTPUT_SUFFIX       = "_md"        # input folder + this suffix = output folder
SLEEP_BETWEEN_PAGES = 1          # seconds between pages within a PDF
MAX_WORKERS         = 2           # PDFs processed in parallel — reduce to 3 if you hit rate limits
# ───────────────────────────────────────────────────────────────────────────────

PROMPT = """Convert this PDF page to markdown for study notes.
Include: core concepts and explanations, definitions, formulas and their meaning,
examples, key arguments or reasoning, and detailed descriptions of any graphs,
charts or diagrams.
Exclude: administrative content, professor info, course logistics, deadlines,
and decorative elements.
Write clearly and preserve the logical structure of the content.
Output only the markdown — no preamble, no explanation."""

print_lock = threading.Lock()


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 string for the OpenAI API."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def convert_pdf_to_md(pdf_path: Path, client: OpenAI) -> str:
    """Convert a single PDF file to markdown using GPT-4o vision page by page."""
    log(f"  Converting: {pdf_path.name}")

    try:
        pages = convert_from_path(str(pdf_path), dpi=DPI)
    except Exception as e:
        return f"# Error converting {pdf_path.name}\n\n{e}"

    md_parts = []

    for i, page_img in enumerate(pages, start=1):
        log(f"    [{pdf_path.name}] Page {i}/{len(pages)}...")
        b64 = encode_image_to_base64(page_img)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            )
            page_md = response.choices[0].message.content.strip()
            md_parts.append(page_md)
            log(f"    [{pdf_path.name}] Page {i}/{len(pages)} done")
        except Exception as e:
            md_parts.append(f"<!-- Page {i} error: {e} -->")
            log(f"    [{pdf_path.name}] Page {i} error: {e}")

        if i < len(pages):
            time.sleep(SLEEP_BETWEEN_PAGES)

    return "\n\n---\n\n".join(md_parts)


def process_pdf(pdf_path: Path, input_root: Path, output_root: Path, client: OpenAI, idx: int, total: int):
    """Process a single PDF — called from thread pool."""
    relative = pdf_path.relative_to(input_root)
    out_dir  = output_root / relative.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (pdf_path.stem + ".md")

    if out_file.exists():
        log(f"[{idx}/{total}] Skipping (already done): {relative}")
        return

    log(f"[{idx}/{total}] {relative}")
    md_content = convert_pdf_to_md(pdf_path, client)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_path.stem}\n\n")
        f.write(md_content)
        f.write("\n")

    log(f"  Saved → {out_file.relative_to(output_root.parent)}\n")


def mirror_structure_and_convert(input_folder: str) -> None:
    input_root = Path(input_folder).resolve()
    if not input_root.exists():
        print(f"Error: folder not found: {input_root}")
        sys.exit(1)

    output_root = input_root.parent / (input_root.name + OUTPUT_SUFFIX)
    output_root.mkdir(parents=True, exist_ok=True)

    client    = OpenAI(api_key=OPENAI_API_KEY)
    pdf_files = list(input_root.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    total = len(pdf_files)
    print(f"Found {total} PDF(s). Output → {output_root}")
    print(f"Running with {MAX_WORKERS} parallel workers\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_pdf, pdf, input_root, output_root, client, idx, total): pdf
            for idx, pdf in enumerate(pdf_files, start=1)
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            pdf = futures[future]
            try:
                future.result()
            except Exception as e:
                log(f"  [FATAL] {pdf.name} — {e}")
            log(f"\nProgress: {done}/{total} PDFs completed\n")

    print("All done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_md.py /path/to/folder")
        sys.exit(1)

    mirror_structure_and_convert(sys.argv[1])
