from __future__ import annotations

import argparse
import dataclasses
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any

from memory import init_memory, set_debug_mode
from memory.save_report import save_report, ReportConfig, build_enriched_markdown
from orchestrator_agent import orchestrator_agent
from deep_research_orchestrator import deep_research
from qdrant_vector_database import close_qdrant_client, ingest_documents

def initialize_app(pdf_dir: Path) -> dict:
    init_memory()
    print("Ingesting documents...")
    info = ingest_documents(pdf_dir)
    print(
        f"{info['num_pdfs']} PDFs ingested to Qdrant collection {info['collection_name']}"
    )
    return info

def _deep_state(result: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a deep_research envelope for the save flow.

    deep_research() returns {"final_answer", "state", "stats"}; save_report
    consumes the NESTED state (evidence_json, verification,
    verification_status live at its top level, not the envelope's). A result
    without a usable "state" dict yields {} so save_report degrades to a
    plain markdown report instead of crashing.
    """
    state = result.get("state") if isinstance(result, dict) else None
    return state if isinstance(state, dict) else {}

def chat_with_supervisor(
    session_id: str | None = None, debug: bool = False, mode: str = "standard"
) -> None:
    set_debug_mode(debug)

    if session_id is None:
        session_id = str(uuid4())

    print("Analyze your pdfs!! \n")
    print("Use 'q', 'exit', or 'exist' to end chat. \n")
    if mode == "deep":
        print("Mode: deep research (5-stage pipeline: decompose -> investigate -> "
              "per-section draft -> critic -> assembly)\n")

    # Research configuration: uncapped body/verification, capped snippets
    report_config = ReportConfig.research()

    while True:
        user_query = input("User: ").strip()

        if not user_query:
            continue

        if user_query.lower() in {"q", "exit", "exist"}:
            print("Exiting chat loop.")
            break

        if mode == "deep":
            # Deep pipeline (P1-3/P1-4): 5 stages, per-sub-question
            # investigation, per-section drafting, critic, assembly.
            result = deep_research(user_query, verbose=True)
        else:
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

        # Standard mode's result IS the orchestration state; deep mode's
        # result is an envelope whose nested "state" is what save_report
        # consumes (evidence_json, verification, verification_status).
        save_state = _deep_state(result) if mode == "deep" else result
        # Deep parity with the UI save: the verification summary renders in
        # the evidence side file, so deep saves enable the dump (standard
        # keeps it off — P0 parity).
        save_config = report_config
        if mode == "deep":
            save_config = dataclasses.replace(
                report_config, include_evidence_dump=True
            )

        # Build and save the report. Deep mode saves the canonical
        # structured document (JSON + sources + Markdown side export) when
        # the pipeline produced one (state['report_json']); everything else
        # keeps the legacy markdown path.
        if mode == "deep" and save_state.get("report_json"):
            from memory.save_report import save_structured_report
            from models.report_schema import ResearchReport

            saved_path = save_structured_report(
                ResearchReport.model_validate_json(save_state["report_json"]),
                state=save_state,
                config=save_config,
            )
            print(f"Report saved to: {saved_path}")
        elif answer:
            # Pass the full orchestration state to save_report for enriched reporting
            saved_path = save_report(
                content=answer,
                query=user_query,
                session_id=session_id,
                state=save_state,  # state with evidence, verification, etc.
                config=save_config  # research config; deep adds the evidence dump
            )
            print(f"Report saved to: {saved_path}")
            
            # Optional: Log additional metadata for debugging
            if debug:
                evidence_json = save_state.get("evidence_json", "")
                verification = save_state.get("verification", "")
                status = save_state.get("verification_status", {})
                print(f"[DEBUG] Evidence length: {len(evidence_json)} chars")
                print(f"[DEBUG] Verification: {'present' if verification else 'missing'}")
                print(f"[DEBUG] Status: {status}")
        
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Researcher Orchestrator")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--mode",
        choices=["standard", "deep"],
        default="standard",
        help=(
            "Research mode: 'standard' runs the original orchestrator loop "
            "(default); 'deep' runs the 5-stage deep-research pipeline."
        ),
    )
    args = parser.parse_args()

    pdf_dir = Path("docs")
    try:
        initialize_app(pdf_dir)
        chat_with_supervisor(debug=args.debug, mode=args.mode)
    finally:
        close_qdrant_client()
