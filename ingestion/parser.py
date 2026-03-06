"""
ingestion/parser.py — Step 1: extract text, images, and tables from PDFs.

partition_pdf with hi_res strategy uses a layout detection model (detectron2)
to spatially distinguish text blocks, tables, and image regions on each page.

extract_image_block_to_payload=True stores each visual as base64 directly on
the element — no separate file I/O. chunking_strategy="by_title" groups content
under its section heading, so each chunk carries its document context.

Every chunk gets chunk_id, source_file, page_number, and section_title so that
provenance is traceable all the way to the final answer.
"""
import base64
import logging
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem, Text, Table, Image, CompositeElement
)

from config import PDF_STRATEGY, IMAGES_DIR, MAX_CHARS, COMBINE_CHARS
from utils.schemas import DocumentChunk, ChunkType
from utils.security import sanitize

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: Path) -> list[DocumentChunk]:
    """
    Extract all content from one PDF and return a flat list of DocumentChunks.

    Image and table chunks have empty text_content here — the captioner fills
    those in next. Text chunks are ready to embed immediately.
    """
    source = pdf_path.name
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=PDF_STRATEGY,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,        # base64 on the element
        extract_image_block_output_dir=str(IMAGES_DIR),   # also save to disk
        chunking_strategy="by_title",               # group content under headings
        max_characters=MAX_CHARS,
        combine_text_under_n_chars=COMBINE_CHARS,
        include_page_breaks=False,
    )

    chunks: list[DocumentChunk] = []
    current_section: str | None = None
    counters = {"text": 0, "image": 0, "table": 0}

    for el in elements:
        page = get_page(el)

        if isinstance(el, Title) and el.text:
            current_section = el.text.strip()

        if isinstance(el, (NarrativeText, ListItem, Text, Title, CompositeElement)):
            raw = sanitize(el.text or "").strip()
            if len(raw) < 30:   # skip noise: page numbers, watermarks, stray fragments
                continue
            counters["text"] += 1
            chunks.append(DocumentChunk(
                chunk_id      = f"{pdf_path.stem}_p{page}_text_{counters['text']}",
                source_file   = source,
                page_number   = page,
                chunk_type    = ChunkType.TEXT,
                text_content  = raw,
                section_title = current_section,
            ))

        elif isinstance(el, Table):
            counters["table"] += 1
            chunks.append(DocumentChunk(
                chunk_id      = f"{pdf_path.stem}_p{page}_table_{counters['table']}",
                source_file   = source,
                page_number   = page,
                chunk_type    = ChunkType.TABLE,
                text_content  = sanitize(el.text or ""),   # raw table text as fallback
                section_title = current_section,
                image_base64  = extract_b64(el),
            ))

        elif isinstance(el, Image):
            b64 = extract_b64(el)
            if not b64:
                continue   # skip images with no data
            counters["image"] += 1
            chunks.append(DocumentChunk(
                chunk_id      = f"{pdf_path.stem}_p{page}_image_{counters['image']}",
                source_file   = source,
                page_number   = page,
                chunk_type    = ChunkType.IMAGE,
                text_content  = "",   # filled by captioner
                section_title = current_section,
                image_base64  = b64,
            ))

    print(f"  {source}: {counters['text']} text, {counters['table']} tables, {counters['image']} images")
    return chunks


def parse_all(pdf_dir: Path) -> list[DocumentChunk]:
    """Parse every PDF in a directory. Returns all chunks combined."""
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    all_chunks = []
    for pdf in pdfs:
        all_chunks.extend(parse_pdf(pdf))
    print(f"Total: {len(all_chunks)} chunks from {len(pdfs)} PDF(s)")
    return all_chunks


def get_page(element) -> int:
    """Extract 1-indexed page number from an unstructured element."""
    try:
        return element.metadata.page_number or 1
    except AttributeError:
        return 1


def extract_b64(element) -> str | None:
    """Pull base64 image data from an element's metadata payload."""
    try:
        b64 = getattr(element.metadata, "image_base64", None)
        if b64:
            return b64
        payload = getattr(element.metadata, "image_data", None)
        if payload and isinstance(payload, bytes):
            return base64.b64encode(payload).decode("utf-8")
    except Exception:
        pass
    return None
