"""
ingestion/captioner.py — Step 2: caption every image and table with GPT-4o Vision.

After this step all chunk types — text, image, table — have a populated
text_content field. That uniformity is what lets the embedder treat them
identically in the next step.

Separate prompts for images vs tables: images need trend/chart-type analysis,
tables need row/column and value extraction. Section title is appended to both
prompts when available, giving the VLM surrounding document context.

One failure never breaks the whole ingest — bad chunks get a placeholder caption.
"""
import logging
from openai import OpenAI

from config import OPENAI_API_KEY, VLM_MODEL
from utils.schemas import DocumentChunk, ChunkType

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

IMAGE_PROMPT = """\
Analyze this visual element from a financial document (annual report, earnings release, investor presentation).
Describe it precisely and searchably:
- Type (bar chart, line graph, pie chart, diagram, etc.)
- All visible numbers, percentages, and currency values
- Time periods or dates shown
- Trend direction (increasing / decreasing / stable)
- Key takeaway a financial analyst would care about
Under 200 words."""

TABLE_PROMPT = """\
Analyze this table from a financial document.
Describe it precisely:
- What it measures (revenue, expenses, ratios, headcount, etc.)
- Time period or comparison shown
- Most important values (highest, lowest, notable changes)
- Any totals or summary rows
Under 200 words."""


def caption_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    Fill text_content on every IMAGE and TABLE chunk via GPT-4o Vision.
    TEXT chunks pass through unchanged.
    """
    visual = [c for c in chunks if c.chunk_type in (ChunkType.IMAGE, ChunkType.TABLE)]
    print(f"Captioning {len(visual)} visual chunks via {VLM_MODEL}")

    for i, chunk in enumerate(visual):
        if not chunk.image_base64:
            chunk.text_content = (
                f"[{chunk.chunk_type.upper()} on page {chunk.page_number} "
                f"of {chunk.source_file} — no image data]"
            )
            continue

        prompt = TABLE_PROMPT if chunk.chunk_type == ChunkType.TABLE else IMAGE_PROMPT
        if chunk.section_title:
            prompt += f"\n\nThis appears under the section: '{chunk.section_title}'"

        try:
            resp = openai_client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{chunk.image_base64}",
                                "detail": "high",   # reads small text in tables
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            chunk.text_content = resp.choices[0].message.content.strip()
            print(f"  [{i+1}/{len(visual)}] {chunk.chunk_id}: {chunk.text_content[:60]}...")

        except Exception as e:
            chunk.text_content = (
                f"[Caption unavailable — {chunk.chunk_type} from "
                f"{chunk.source_file} page {chunk.page_number}]"
            )
            logger.error(f"  [{i+1}/{len(visual)}] caption failed for {chunk.chunk_id}: {e}")

    return chunks
