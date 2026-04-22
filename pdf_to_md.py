"""
PDF → Markdown batch converter
Uses GPT-4o vision (page-by-page images) to handle text, tables, graphs, and diagrams.

Requirements:
    pip install openai pypdf pdf2image pillow
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
from pathlib import Path
from io import BytesIO

from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set the OPENAI_API_KEY environment variable")   # set via env var or paste here
MODEL = "gpt-4o"
MAX_TOKENS = 4096          # per page — increase if pages get cut off
DPI = 150                  # resolution for rasterising pages (150 is a good balance)
OUTPUT_SUFFIX = "_md"      # input folder + this suffix = output folder
SLEEP_BETWEEN_PAGES = 0.3  # seconds — avoids rate-limit bursts on big docs
# ───────────────────────────────────────────────────────────────────────────────

PROMPT = """Take this PDF page as input and convert it into markdown.
Do not leave any information out.
For pictures and graphs, describe them in detail using markdown (e.g. use a caption block or a descriptive paragraph).
Preserve tables using markdown table syntax.
Output only the markdown — no preamble, no explanation."""


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 string for the OpenAI API."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def convert_pdf_to_md(pdf_path: Path, client: OpenAI) -> str:
    """Convert a single PDF file to markdown using GPT-4o vision page by page."""
    print(f"  Converting: {pdf_path.name}")

    try:
        pages = convert_from_path(str(pdf_path), dpi=DPI)
    except Exception as e:
        return f"# Error converting {pdf_path.name}\n\n{e}"

    md_parts = []

    for i, page_img in enumerate(pages, start=1):
        print(f"    Page {i}/{len(pages)}...", end=" ", flush=True)
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
            print("done")
        except Exception as e:
            md_parts.append(f"<!-- Page {i} error: {e} -->")
            print(f"error: {e}")

        if i < len(pages):
            time.sleep(SLEEP_BETWEEN_PAGES)

    return "\n\n---\n\n".join(md_parts)


def mirror_structure_and_convert(input_folder: str) -> None:
    input_root = Path(input_folder).resolve()
    if not input_root.exists():
        print(f"Error: folder not found: {input_root}")
        sys.exit(1)

    output_root = input_root.parent / (input_root.name + OUTPUT_SUFFIX)
    output_root.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Collect all PDFs first so we can show progress
    pdf_files = list(input_root.rglob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Output → {output_root}\n")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        relative = pdf_path.relative_to(input_root)
        out_dir = output_root / relative.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (pdf_path.stem + ".md")

        if out_file.exists():
            print(f"[{idx}/{len(pdf_files)}] Skipping (already done): {relative}")
            continue

        print(f"[{idx}/{len(pdf_files)}] {relative}")
        md_content = convert_pdf_to_md(pdf_path, client)

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# {pdf_path.stem}\n\n")
            f.write(md_content)
            f.write("\n")

        print(f"  Saved → {out_file.relative_to(output_root.parent)}\n")

    print("All done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_md.py /path/to/folder")
        sys.exit(1)

    mirror_structure_and_convert(sys.argv[1])
