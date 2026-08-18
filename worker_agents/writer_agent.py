from .model_runner import run_model
from typing import Optional

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
        Be clear, complete, and concise.
        Include the key supporting details needed to fully answer the question.
        When the evidence supports it, include the most important comparisons, caveats, or specific facts rather than giving a minimal summary.
        For judgments, comparisons, recommendations, or conclusions state the best-supported conclusion clearly.
        For short follow-ups, answer briefly and directly.
        Do not add unsupported facts.
        Use the exact citation field from Document evidence for PDF citations.
        If web evidence is present, synthesize it into a concise, self-contained answer and include explicit web citations with the exact title and exact URL from Web evidence.
        When citing web sources, use Markdown links in the form [Exact Source Title](Exact URL).
        Do not omit web source URLs when web evidence is used.
        If the evidence is weak or incomplete, say so.
        Do not end with a question, a suggestion for the user to ask a follow-up, or an offer for more help.
        """
    )

    # Pass the user query together with the retrieval context for drafting.
    input_text = (
        f"User query: {user_query}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        reasoning_effort="low",
        tools=None,
        agent_name="writer",
        endpoint=endpoint,
        api_key=api_key,
    )
    return response.output_text