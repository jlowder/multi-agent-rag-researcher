from .model_runner import run_model
from typing import Optional
from utils.config import get_config

"""
Writer Agent 
=====================================================================================
It used by the Orchestrator to write a report on the retrieved information
from the retriever agent.
Main role is to draft a clear, grounded response from the evidence it receives.
"""
def writer_agent(
    user_query: str,
    evidence_text: str,
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Execute the writer agent.
    
    Args:
        user_query: The user's original query
        evidence_text: The evidence to base the report on
        verbose: Whether to print debug information
        endpoint: Optional custom endpoint URL
        api_key: Optional custom API key
        
    Returns:
        The written report as a string
    """
    if verbose:
        print("[Writer Agent] Writing report...")

    instructions = (
        """
        Answer the user using only the evidence.
        Start with the answer.
        Provide a thorough, detailed, and comprehensive report.
        Include the key supporting details needed to fully answer the question.
        When the evidence supports it, include the most important comparisons, caveats, or specific facts rather than giving a minimal summary.
        For judgments, comparisons, recommendations, or conclusions state the best-supported conclusion clearly.
        For short follow-ups, answer comprehensively and with detail.
        Do not add unsupported facts.
        Use the exact citation field from Document evidence for PDF citations.
        If web evidence is present, synthesize it into a comprehensive, self-contained answer and include explicit web citations with the exact title and exact URL from Web evidence.
        When citing web sources, use Markdown links in the form [Exact Source Title](Exact URL).
        Do not omit web source URLs when web evidence is used.
        If the evidence is weak or incomplete, say so.
        Do not end with a question, a suggestion for the user to ask a follow-up, or an offer for more help.

        REPORT DEPTH REQUIREMENTS:
        - Every concept should be fully explained, not just named. For example, don't just define "multi-head attention" — explain how it works, why it's better than single-head, the mathematical formulation, and practical implications.
        - Include mathematical formulas, architectural details, and quantitative comparisons where available in the evidence.
        - Include specific examples, analogies, and visual descriptions from the evidence.
        - Organize the report with clear section headings and sub-headings for each major topic.
        - Aim for a comprehensive, graduate-level technical report that could serve as a standalone reference on the topic.

        REQUIRED OUTLINE (follow this two-step process):
        - Step 1: Begin the report with a "## Report Outline" section listing the 7-8 "##" sections you will write. Use this default set, adapting section names to the topic when a better fit exists:
          1. Definition & Background
          2. Core Components & Mechanics
          3. Major Variants & Alternative Approaches
          4. Dynamics / Evaluation / Analysis
          5. Applications
          6. Tools & Ecosystem
          7. Limitations & Open Problems
          8. Synthesis
        - Step 2: After the outline, write each section under its own "##" heading with at least 300 words of substance — concrete facts, mechanisms, comparisons, and examples grounded in the evidence, not filler.
        """
    )

    # Pass the user query together with the retrieval context for drafting.
    input_text = (
        f"User query: {user_query}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    config = get_config()
    response = run_model(
        instructions=instructions,
        input_data=input_text,
        reasoning_effort=config.get_reasoning_effort("writer"),
        max_output_tokens=config.get_max_output_tokens("writer"),
        tools=None,
        agent_name="writer",
        endpoint=endpoint,
        api_key=api_key,
    )
    return response.output_text