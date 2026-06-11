"""
ingestion/parse_attachments.py

Extracts text from PDF / DOCX / TXT / HTML attachments.
Preserves page_no for PDFs (required for citations).
Returns list of chunk dicts ready for indexing.
"""

from __future__ import annotations
import re
from pathlib import Path

from utils.models import EmailRecord
from ingestion.chunker import chunk_attachment


def parse_attachments_for_thread(data_dir: Path, record: EmailRecord) -> list[dict]:
    """
    Scan data_dir for attachment files named by message_id.
    Real Enron data: attachments live in subdirs named after the mailbox + message.
    Adjust the path pattern to match your slice layout.
    """
    chunks: list[dict] = []
    safe_id = record.message_id.replace("/", "_").replace(":", "_").replace(" ", "_")
    att_dir = data_dir / "attachments" / safe_id
    if not att_dir.exists():
        return []

    for path in sorted(att_dir.iterdir()):
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                file_chunks = _parse_pdf(path)
            elif suffix in {".docx", ".doc"}:
                file_chunks = _parse_docx(path)
            elif suffix == ".txt":
                file_chunks = _parse_txt(path)
            elif suffix in {".html", ".htm"}:
                file_chunks = _parse_html(path)
            else:
                continue
        except Exception as e:
            print(f"[WARN] Could not parse {path}: {e}")
            continue

        for chunk_idx, tc in enumerate(file_chunks):
            chunks.append({
                "doc_id":     f"{record.message_id}__att__{path.name}__{tc.page_no or 0}_{chunk_idx}",
                "thread_id":  record.thread_id,
                "message_id": record.message_id,
                "source":     "attachment",
                "filename":   path.name,
                "text":       tc.text,
                "page_no":    tc.page_no,
                "embedding":  None,  # filled during ingest.py
            })

    return chunks


def _parse_pdf(path: Path) -> list:
    import pymupdf  # PyMuPDF
    doc = pymupdf.open(str(path))
    page_texts = []
    page_breaks = []

    full_text = ""
    for page in doc:
        text = page.get_text("text")
        page_breaks.append(len(full_text))
        full_text += text + "\n"

    doc.close()
    return chunk_attachment(full_text, page_breaks=page_breaks[1:])  # skip offset 0


def _parse_docx(path: Path) -> list:
    try:
        import docx
        doc = docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        # fallback: mammoth
        import mammoth
        with open(path, "rb") as f:
            result = mammoth.extract_raw_text(f)
        text = result.value
    return chunk_attachment(text)


def _parse_txt(path: Path) -> list:
    text = path.read_text(errors="replace")
    return chunk_attachment(text)


def _parse_html(path: Path) -> list:
    html = path.read_text(errors="replace")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return chunk_attachment(text)
