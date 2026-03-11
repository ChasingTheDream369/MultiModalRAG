"""
ingestion/captioner.py — Step 2: caption every image and table with GPT-4o Vision.

After this step all chunk types — text, image, table — have a populated
text_content field. That uniformity is what lets the embedder treat them
identically in the next step.

Separate prompts for images vs tables — both are general-purpose and work
across any document type (technical manuals, reports, presentations, etc.).
surrounding_context (nearby page text from the parser) is appended to both
prompts, giving the VLM real document context to reason with — not just the visual.

One failure never breaks the whole ingest — bad chunks get a placeholder caption.
"""
import logging
from openai import OpenAI

from config import OPENAI_API_KEY, VLM_MODEL
from utils.schemas import DocumentChunk, ChunkType

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

IMAGE_PROMPT = """\
You are a document analysis assistant. Your job is to produce a caption so \
detailed and complete that someone who has never seen this image can answer \
ANY specific question about it — including looking up a particular label, \
reading a specific data point at a given x/y value, or tracing a relationship \
between components.

Think step by step:

1. FIRST LOOK — reason about what you see before extracting details:
   - What type of visual is this? (bar chart, line graph, pie chart, scatter \
plot, heatmap, flowchart, diagram, schematic, photograph, screenshot, map, \
illustration, infographic, etc.)
   - What is the intent behind this visual — what question is it trying to \
answer or what story is it telling?
   - How is the information organised — what is being compared, grouped, or \
sequenced?

2. STRUCTURE & LAYOUT:
   - For charts/graphs: state the exact axis labels, units, scale/range, and \
legend entries. Note if axes are logarithmic, inverted, or have breaks.
   - For diagrams/flowcharts: list every node/box and every connection/arrow \
between them, with labels.
   - For photographs/illustrations: describe the scene, all visible objects, \
spatial arrangement, and any overlaid text or annotations.

3. EXHAUSTIVE DATA EXTRACTION — this is the most important part:
   - Extract EVERY visible data point, value, label, percentage, date, and \
number exactly as shown. Do not summarise or skip any.
   - For charts: read off each bar height, line point, slice percentage, etc. \
Format as "Label: Value" pairs or a mini-table so any specific x→y lookup is \
possible from your caption alone.
   - For diagrams: capture every label on every component and connector.
   - For images with text: transcribe ALL visible text verbatim.

4. REASONING & INTERPRETATION — now that you have all the data, think about it:
   - What patterns emerge? (increasing, decreasing, stable, cyclical)
   - Which values are highest/lowest and by how much?
   - Are there outliers, anomalies, or sudden changes? What might explain them?
   - If the surrounding document context is provided below, connect the visual \
to that context — does it support, illustrate, or contradict the nearby text?

5. METADATA: any visible title, subtitle, footnotes, source attribution, \
date stamps, or watermarks.

Be exhaustive and factual. Only describe what is visible — never hallucinate \
values. Use as many words as needed to capture everything."""

TABLE_PROMPT = """\
You are a document analysis assistant. Your job is to produce a caption so \
detailed and complete that someone who has never seen this table can answer \
ANY specific question about it — including looking up a value at a particular \
row/column intersection.

Think step by step:

1. FIRST LOOK — reason about the table before extracting data:
   - What does this table represent?
   - What is being measured, compared, or tracked?
   - How is it organised — what do rows represent vs columns?

2. STRUCTURE: List every column header and every row label exactly as shown, \
preserving hierarchy if there are merged cells or sub-headers.

3. EXHAUSTIVE DATA EXTRACTION — this is the most important part:
   - Reproduce EVERY cell value in the table. Format them as structured \
"Row | Column: Value" pairs or as a markdown table so that any specific \
row×column lookup is possible from your caption alone.
   - Do not skip, summarise, or round any values. Capture units, currencies, \
percentages, and decimal places exactly.

4. REASONING & INTERPRETATION — now that you have all the data, think about it:
   - What stands out? Largest, smallest, and any notable changes between \
rows or columns.
   - Are there any row-over-row or column-over-column trends?
   - Highlighted, bold, or colour-coded cells — what do they signify?
   - Totals, subtotals, averages, or summary rows — do they add up correctly?
   - If the surrounding document context is provided below, connect the table \
to that context — does it support or elaborate on the nearby text?

5. METADATA: table title, caption, footnotes, source line, or date if visible.

Be exhaustive and factual. Only describe what is visible — never hallucinate \
values. Use as many words as needed to capture everything."""


def build_caption_prompt(chunk: DocumentChunk) -> str:
    """
    Build the full prompt for captioning a visual chunk.

    Starts with the base prompt (image vs table), then appends
    section title and surrounding page text so the VLM can reason
    about what the visual represents in the document's context.
    """
    prompt = TABLE_PROMPT if chunk.chunk_type == ChunkType.TABLE else IMAGE_PROMPT

    if chunk.section_title:
        prompt += f"\n\nDocument section: '{chunk.section_title}'"

    context = getattr(chunk, "surrounding_context", "") or ""
    if context.strip():
        prompt += (
            f"\n\nSurrounding text from the document for context:\n"
            f"---\n{context}\n---\n"
            f"Use this context to better interpret what the visual is showing."
        )

    return prompt


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

        prompt = build_caption_prompt(chunk)

        try:
            resp = openai_client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{chunk.image_base64}",
                                "detail": "high",
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