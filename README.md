# PDF to Markdown Converter

Converts an entire folder of PDFs to markdown using GPT-4o vision.
Handles text, tables, graphs, and diagrams — preserving visual content as detailed text descriptions, making it suitable for RAG pipelines.

## Requirements

pip install openai pdf2image pillow

brew install poppler  # macOS
sudo apt install poppler-utils  # Ubuntu

## Usage

export OPENAI_API_KEY=sk-yourkey

python pdf_to_md.py /path/to/your/folder

## Output

Creates a mirrored folder structure next to the original with `_md` suffix.
Every `.pdf` becomes a `.md` file. Subfolders are preserved.

## Cost

~$0.008 per page using GPT-4o (high detail vision).
A typical semester of lecture slides (~500 pages) runs ~$4.