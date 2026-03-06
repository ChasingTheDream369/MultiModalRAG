import os
from pathlib import Path

BASE_DIR         = Path(__file__).parent
PDF_DIR          = BASE_DIR / "data" / "pdfs"
IMAGES_DIR       = BASE_DIR / "data" / "extracted_images"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VLM_MODEL      = "gpt-4o"               # used for captioning images/tables
LLM_MODEL      = "gpt-4o"               # used for final answer synthesis
EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536

# Ingestion
PDF_STRATEGY  = "hi_res"               # "fast" skips table/image detection
MAX_CHARS     = 2000                   # max chars per text chunk
COMBINE_CHARS = 200                    # merge fragments smaller than this

# Retrieval
TOP_K = 6                              # chunks retrieved per query

# Generation
MAX_IMAGES  = 3                        # max real images sent to LLM per query
MAX_TOKENS  = 1500
TEMPERATURE = 0.2

# Security
MAX_QUERY_LEN = 1000
