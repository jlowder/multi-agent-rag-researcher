from __future__ import annotations

import argparse
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any

from memory import init_memory, set_debug_mode
from memory.save_report import save_report, ReportConfig, build_enriched_markdown
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
    
    # Research configuration: uncapped body/verification, capped snippets
    report_config = ReportConfig.research()
    
    while True:
        user_query = input("User: ").strip()

        if not user_query:
            continue

        if user_query.lower() in {"q", "exit", "exist"}:
            print("Exiting chat loop.")
            break

        # Call orchestrator and get comprehensive state with evidence
        result = orchestrator_agent(
            user_query, 
            session_id=session_id, 
            verbose=True, 
            debug_enabled=debug
        )
        
        # Extract the final answer from the orchestration result
        answer = result.get("final_answer", "")
        
        if answer:
            print(f"\nAssistant: {answer}\n")

        # Build and save enriched markdown report with evidence from state
        if answer:
            # Pass the full orchestration state to save_report for enriched reporting
            saved_path = save_report(
                content=answer,
                query=user_query,
                session_id=session_id,
                state=result,  # Pass orchestration state with evidence, verification, etc.
                config=report_config  # Use research configuration (uncapped body)
            )
            print(f"Report saved to: {saved_path}")
            
            # Optional: Log additional metadata for debugging
            if debug:
                evidence_json = result.get("evidence_json", "")
                verification = result.get("verification", "")
                status = result.get("verification_status", {})
                print(f"[DEBUG] Evidence length: {len(evidence_json)} chars")
                print(f"[DEBUG] Verification: {'present' if verification else 'missing'}")
                print(f"[DEBUG] Status: {status}")
        
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
