"""
generation/generator.py — Steps 6, 7, 9: generate a grounded multimodal answer.

Step 6: build a context string from retrieved chunks, then construct a multimodal
        user message that interleaves the text context with real images for the
        visual chunks. GPT-4o sees both captions and actual images simultaneously.

Step 7: after generation, the LLM returns chunk_ids of any visuals it referenced
        in its answer (via the JSON metadata block). resolve_image_refs() maps
        those IDs back to actual DocumentChunk objects so the caller can serve
        the real images to the user.

Step 9: parse_response() uses Pydantic (RAGResponse) to validate the output
        before it leaves this module.
"""
import json
import logging
import re
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, MAX_IMAGES, MAX_TOKENS, TEMPERATURE
from utils.schemas import DocumentChunk, ChunkType, RAGResponse
from utils.security import wrap_context, check_response
from prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate(question: str, chunks: list[DocumentChunk]) -> tuple[RAGResponse, list[DocumentChunk]]:
    """
    Generate a grounded answer from retrieved chunks.

    Returns (RAGResponse, resolved_image_chunks) where resolved_image_chunks
    are the actual DocumentChunks for any visuals the LLM cited — ready to serve.
    """
    text_chunks   = [c for c in chunks if c.chunk_type == ChunkType.TEXT]
    visual_chunks = [c for c in chunks if c.chunk_type in (ChunkType.IMAGE, ChunkType.TABLE)]
    visual_chunks = visual_chunks[:MAX_IMAGES]   # cap to avoid token overload

    context = format_context(text_chunks + visual_chunks)
    wrapped = wrap_context(context)

    # Build multimodal message: text context + real images interleaved
    content = [{"type": "text", "text": f"{wrapped}\n\nQuestion: {question}"}]
    for chunk in visual_chunks:
        if chunk.image_base64:
            label = (
                f"[{chunk.chunk_type.upper()} — {chunk.source_file}, "
                f"page {chunk.page_number}"
                + (f", section: {chunk.section_title}" if chunk.section_title else "")
                + f", chunk_id: {chunk.chunk_id}]"
            )
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{chunk.image_base64}",
                    "detail": "high",
                },
            })

    raw = call_llm(content)

    if not check_response(raw):
        raw = "Response blocked by safety filter. Please rephrase your question."

    response = parse_response(raw)

    # Step 7: map image_refs back to actual chunk objects for the caller to serve
    chunk_map       = {c.chunk_id: c for c in chunks}
    resolved_images = [chunk_map[ref] for ref in response.image_refs if ref in chunk_map]

    return response, resolved_images


def call_llm(content: list[dict]) -> str:
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
    )
    return resp.choices[0].message.content


def format_context(chunks: list[DocumentChunk]) -> str:
    """Format chunks into a structured context string with type/source/page headers."""
    lines = []
    for c in chunks:
        header = f"[{c.chunk_type.upper()} | {c.source_file} | page {c.page_number}"
        if c.section_title:
            header += f" | section: {c.section_title}"
        header += f" | id: {c.chunk_id}]"
        lines.append(header)
        lines.append(c.text_content)
        lines.append("")
    return "\n".join(lines)


def parse_response(raw: str) -> RAGResponse:
    """
    Split the prose answer from the trailing JSON metadata block.
    Pydantic validates the structure before it leaves this function.
    """
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    meta  = {}
    prose = raw.strip()

    if json_match:
        try:
            meta  = json.loads(json_match.group(1))
            prose = raw[: json_match.start()].strip()
        except json.JSONDecodeError:
            logger.warning("Could not parse metadata JSON from response")

    return RAGResponse(
        answer                = prose,
        sources               = meta.get("sources", []),
        has_tables            = bool(meta.get("has_tables", False)),
        follow_up_suggestions = meta.get("follow_up_suggestions", []),
        image_refs            = meta.get("image_refs", []),
    )
