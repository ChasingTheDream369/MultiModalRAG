"""
main.py — CLI entrypoint.

Commands:
  python main.py ingest
  python main.py ingest --pdf_dir path/to/pdfs
  python main.py query "What was Q3 2024 revenue growth?"
  python main.py query "Show me the cost breakdown" --show_captions
"""
import argparse
import sys


def cmd_ingest(args):
    from pathlib import Path
    from config import PDF_DIR
    from ingestion.parser import parse_all
    from ingestion.captioner import caption_chunks
    from ingestion.embedder import embed_and_store

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else PDF_DIR
    if not pdf_dir.exists():
        print(f"Directory not found: {pdf_dir}")
        sys.exit(1)

    chunks = parse_all(pdf_dir)
    chunks = caption_chunks(chunks)
    embed_and_store(chunks)

    text  = sum(1 for c in chunks if c.chunk_type.value == "text")
    imgs  = sum(1 for c in chunks if c.chunk_type.value == "image")
    tabs  = sum(1 for c in chunks if c.chunk_type.value == "table")
    print(f"\nDone — {len(chunks)} chunks ({text} text, {imgs} images, {tabs} tables)")
    print('Ready: python main.py query "your question"')


def cmd_query(args):
    from utils.security import check_query
    from ingestion.embedder import load
    from retrieval.retriever import HybridRetriever
    from generation.generator import generate

    chk = check_query(args.question)
    if chk.blocked:
        print("\nQuery blocked — potential prompt injection detected.")
        print("Please rephrase your question.\n")
        sys.exit(1)

    index, chunks = load()
    retriever     = HybridRetriever(index, chunks)
    retrieved     = retriever.retrieve(chk.text)
    response, image_chunks = generate(chk.text, retrieved)

    sep = "─" * 60
    print(f"\n{sep}\nQ: {chk.text}\n{sep}\n")
    print(response.answer)
    print(f"\nConfidence:  {response.confidence:.2f}")
    if response.sources:
        print(f"Sources:     {', '.join(response.sources)}")

    if image_chunks:
        print(f"\nImages referenced:")
        for c in image_chunks:
            print(f"  {c.chunk_id}  ({c.source_file} p.{c.page_number})")
            if args.show_captions:
                print(f"  Caption: {c.text_content[:120]}...")

    if response.follow_up_suggestions:
        print(f"\nSuggested follow-ups:")
        for q in response.follow_up_suggestions:
            print(f"  - {q}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Multimodal Document RAG")
    sub    = parser.add_subparsers(dest="command", required=True)

    p_i = sub.add_parser("ingest", help="Parse PDFs and build the vector index")
    p_i.add_argument("--pdf_dir", default=None, help="Override default PDF directory")

    p_q = sub.add_parser("query", help="Ask a question against the indexed documents")
    p_q.add_argument("question")
    p_q.add_argument("--show_captions", action="store_true",
                     help="Print VLM captions for images referenced in the answer")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "query": cmd_query}[args.command](args)


if __name__ == "__main__":
    main()
