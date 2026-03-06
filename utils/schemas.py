"""
utils/schemas.py — Pydantic data contracts shared across the pipeline.

DocumentChunk flows through every stage: parser → captioner → embedder → retriever → generator.
RAGResponse is the validated final output returned to the caller.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    TABLE = "table"


class DocumentChunk(BaseModel):
    """
    One unit of extracted content from a PDF page.

    text_content holds raw text for TEXT chunks and the VLM caption for IMAGE/TABLE chunks —
    after captioning, all three types are embedded the same way.
    image_base64 is kept on the chunk so it can be passed directly to the LLM at generation time.
    """
    chunk_id:      str
    source_file:   str
    page_number:   int
    chunk_type:    ChunkType
    text_content:  str           = ""
    section_title: Optional[str] = None   # section heading from Unstructured Title elements
    image_base64:  Optional[str] = None   # set for IMAGE and TABLE chunks

    class Config:
        arbitrary_types_allowed = True


class RAGResponse(BaseModel):
    """Validated response from the full pipeline."""
    answer:                str
    confidence:            float        = Field(ge=0.0, le=1.0)
    sources:               list[str]    = []
    has_tables:            bool         = False
    follow_up_suggestions: list[str]    = []
    image_refs:            list[str]    = []   # chunk_ids referenced in the answer
