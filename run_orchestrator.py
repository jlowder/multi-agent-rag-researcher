from __future__ import annotations

import argparse
from uuid import uuid4
from pathlib import Path

from memory import init_memory, set_debug_mode
from memory.save_report import save_report
from orchestrator_agent import orchestrator_agent
from qdrant_vector_database import close_qdrant_client, ingest_documents

def initialize_app(pdf_dir: Path) -> dict:
    init_memory()
    print("Ingesting documents...")
    info = ingest_documents(pdf_dir)
    print(
        f"{info['num_pdfs']} PDFs ingested to Qdrant collection {info['collection_name']}"
    )
    return info

def chat_with_supervisor(session_id: str | None = None, debug: bool = False) -> None:
    set_debug_mode(debug)

    if session_id is None:
        session_id = str(uuid4())

    print("Analyze your pdfs!! \n")
    print("Use 'q', 'exit', or 'exist' to end chat. \n")
    while True:
        user_query = input("User: ").strip()

        if not user_query:
            continue

        if user_query.lower() in {"q", "exit", "exist"}:
            print("Exiting chat loop.")
            break

        result = orchestrator_agent(user_query, session_id=session_id, verbose=True, debug_enabled=debug)
        answer = result.get("final_answer", "")
        if answer:
            print(f"\nAssistant: {answer}\n")

        if answer:
            saved_path = save_report(answer, query=user_query, session_id=session_id)
            print(f"Report saved to: {saved_path}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Researcher Orchestrator")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    pdf_dir = Path("docs")
    try:
        initialize_app(pdf_dir)
        chat_with_supervisor(debug=args.debug)
    finally:
        close_qdrant_client()
