"""
ingestion/parser.py — Step 1: extract text, images, and tables from PDFs.

partition_pdf with hi_res strategy uses a layout detection model (detectron2)
to spatially distinguish text blocks, tables, and image regions on each page.

Images are extracted via PyMuPDF (reads directly from PDF binary) instead of
partition_pdf's image block detection, which is unreliable on colored/complex PDFs.

chunking_strategy="by_title" groups content under its section heading, so each
chunk carries its document context.

Every chunk gets chunk_id, source_file, page_number, and section_title so that
provenance is traceable all the way to the final answer.
"""
import base64
import hashlib
import logging
from pathlib import Path

import fitz  # pymupdf
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem, Text, Table, Image, CompositeElement
)

from config import PDF_STRATEGY, IMAGES_DIR, MAX_CHARS, COMBINE_CHARS
from utils.schemas import DocumentChunk, ChunkType
from utils.security import sanitize

logger = logging.getLogger(__name__)

MIN_IMAGE_PX = 50
MIN_IMAGE_BYTES = 2048


def parse_pdf(pdf_path: Path) -> list[DocumentChunk]:
    """
    Extract all content from one PDF and return a flat list of DocumentChunks.

    Text + tables come from partition_pdf (layout-aware chunking).
    Images come from PyMuPDF (reliable binary extraction).
    """
    source = pdf_path.name
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── partition_pdf for text + tables only ─────────────────────────────
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=PDF_STRATEGY,
        extract_images_in_pdf=False,              # PyMuPDF handles images
        chunking_strategy="by_title",
        max_characters=MAX_CHARS,
        combine_text_under_n_chars=COMBINE_CHARS,
        include_page_breaks=False,
        languages=["eng"],  
    )

    chunks: list[DocumentChunk] = []
    page_text_map: dict[int, list[str]] = {}      # for building image context
    current_section: str | None = None
    section_by_page: dict[int, str] = {}
    counters = {"text": 0, "image": 0, "table": 0}

    for el in elements:
        page = get_page(el)

        if isinstance(el, Title) and el.text:
            current_section = el.text.strip()
            section_by_page[page] = current_section

        if isinstance(el, (NarrativeText, ListItem, Text, Title, CompositeElement)):
            raw = sanitize(el.text or "").strip()
            if len(raw) < 30:
                continue
            counters["text"] += 1
            page_text_map.setdefault(page, []).append(raw)
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
                text_content  = sanitize(el.text or ""),
                section_title = current_section,
                image_base64  = extract_b64(el),
            ))

    # ── PyMuPDF for ALL images ───────────────────────────────────────────
    image_chunks = extract_images_pymupdf(pdf_path, source, page_text_map, section_by_page)
    chunks.extend(image_chunks)
    counters["image"] = len(image_chunks)

    print(f"  {source}: {counters['text']} text, {counters['table']} tables, {counters['image']} images")
    return chunks


def extract_images_pymupdf(
    pdf_path: Path,
    source: str,
    page_text_map: dict[int, list[str]],
    section_by_page: dict[int, str],
) -> list[DocumentChunk]:
    """
    Extract images from PDF using PyMuPDF.

    Reads directly from the PDF binary — no detectron2, no tesseract.
    Deduplicates identical images (logos, watermarks) via MD5 hash.
    Attaches surrounding page text as context for the captioner.
    """
    chunks: list[DocumentChunk] = []
    counter = 0
    seen: set[str] = set()

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1

            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < MIN_IMAGE_PX or pix.height < MIN_IMAGE_PX:
                        continue
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_bytes = pix.tobytes("png")
                    if len(img_bytes) < MIN_IMAGE_BYTES:
                        continue

                    # Deduplicate identical images across pages
                    digest = hashlib.md5(img_bytes).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)

                    b64 = base64.b64encode(img_bytes).decode("utf-8")

                    out_path = IMAGES_DIR / f"{pdf_path.stem}_p{page_num}_img_{counter}.png"
                    pix.save(str(out_path))

                    # Build surrounding text context for the captioner
                    context = build_surrounding_context(page_num, page_text_map, total_pages)

                    # Walk back to find nearest section title
                    section = None
                    for p in range(page_num, 0, -1):
                        if p in section_by_page:
                            section = section_by_page[p]
                            break

                    counter += 1
                    chunks.append(DocumentChunk(
                        chunk_id            = f"{pdf_path.stem}_p{page_num}_image_{counter}",
                        source_file         = source,
                        page_number         = page_num,
                        chunk_type          = ChunkType.IMAGE,
                        text_content        = "",
                        section_title       = section,
                        image_base64        = b64,
                        surrounding_context = context,
                    ))
                except Exception as e:
                    logger.warning(f"PyMuPDF could not extract image xref {xref} on page {page_num}: {e}")
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF fallback failed for {pdf_path.name}: {e}")

    return chunks


def build_surrounding_context(
    page_num: int,
    page_text_map: dict[int, list[str]],
    total_pages: int,
    max_chars: int = 1500,
) -> str:
    """
    Collect text from same page + adjacent pages so the captioner
    can reason about what the image represents in context.
    """
    parts: list[str] = []

    same = page_text_map.get(page_num, [])
    if same:
        parts.append(f"[Page {page_num}]: " + " ".join(same))

    if page_num > 1:
        prev = page_text_map.get(page_num - 1, [])
        if prev:
            parts.append(f"[Page {page_num - 1}]: " + " ".join(prev[-2:]))

    if page_num < total_pages:
        nxt = page_text_map.get(page_num + 1, [])
        if nxt:
            parts.append(f"[Page {page_num + 1}]: " + " ".join(nxt[:2]))

    context = "\n".join(parts)
    return context[:max_chars] if len(context) > max_chars else context


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