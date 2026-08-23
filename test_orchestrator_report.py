"""
Test script to verify the orchestrator report integration works correctly.
"""

import json
from pathlib import Path

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from memory.save_report import save_report, ReportConfig

def test_report_generation():
    """Test report generation with orchestrator-like state."""
    
    # Simulate orchestration state
    state = {
        "user_query": "What are transformer models?",
        "current_date": "2024-01-15",
        "final_answer": "Transformer models use attention mechanisms to process sequences in parallel, enabling more efficient training on large datasets.",
        "evidence_json": json.dumps({
            "document_evidence": {
                "chunks": [
                    {
                        "chunk_id": "chunk_1",
                        "source": "attention_is_all_you_need.pdf",
                        "content": "The Transformer architecture uses a self-attention mechanism to process input sequences in parallel, rather than sequentially like RNNs. This allows for much faster training on large datasets.",
                        "score": 0.95
                    },
                    {
                        "chunk_id": "chunk_2",
                        "source": "transformer_tutorial.md",
                        "content": "Self-attention computes the attention scores between all pairs of positions in the sequence, allowing the model to weigh the importance of different tokens.",
                        "score": 0.88
                    }
                ]
            },
            "web_evidence": {
                "results": [
                    {
                        "url": "https://example.com/transformers",
                        "snippet": "Transformers have revolutionized NLP by enabling parallel processing and capturing long-range dependencies.",
                        "relevance_score": 0.82
                    }
                ]
            }
        }),
        "written_draft": "Transformer models use attention mechanisms...",
        "verification": "EVIDENCE_STATUS:\n  confidence: 0.92\n  coverage: comprehensive\n  gaps: []",
        "verification_status": {
            "confidence": 0.92,
            "coverage": "comprehensive",
            "gaps": [],
            "re_retrieve": False
        },
        "retrieval_attempted": True,
        "re_retrieve_rounds": 0,
        "needs_more_evidence": False,
        "gap_queries": []
    }
    
    # Test 1: Basic report generation
    print("Test 1: Basic report generation")
    config = ReportConfig.default()
    filepath = save_report(
        content=state["final_answer"],
        query=state["user_query"],
        session_id="test_session_1",
        state=state,
        config=config
    )
    print(f"  ✓ Report saved to: {filepath}")
    
    # Test 2: Strict security configuration
    print("\nTest 2: Strict security configuration")
    strict_config = ReportConfig.strict()
    filepath = save_report(
        content=state["final_answer"],
        query=state["user_query"],
        session_id="test_session_2",
        state=state,
        config=strict_config
    )
    print(f"  ✓ Strict config report saved to: {filepath}")
    
    # Test 3: Report with minimal state
    print("\nTest 3: Report with minimal state (no evidence)")
    minimal_state = {
        "final_answer": "Simple answer",
        "user_query": "What is 2+2?"
    }
    filepath = save_report(
        content=minimal_state["final_answer"],
        query=minimal_state["user_query"],
        session_id="test_session_3",
        state=minimal_state
    )
    print(f"  ✓ Minimal report saved to: {filepath}")
    
    # Test 4: Empty state
    print("\nTest 4: Empty state (should still generate report)")
    filepath = save_report(
        content="Answer with no evidence",
        query="Test query",
        session_id="test_session_4",
        state={}
    )
    print(f"  ✓ Empty state report saved to: {filepath}")
    
    # Test 5: Very long content (should be truncated)
    print("\nTest 5: Very long content (should be truncated)")
    long_content = "Long content " * 1000  # 11000 characters
    filepath = save_report(
        content=long_content,
        query="Long content test",
        session_id="test_session_5",
        state=state,
        config=strict_config
    )
    print(f"  ✓ Long content report saved to: {filepath}")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_report_generation()