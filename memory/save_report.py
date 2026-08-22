"""Save research reports to markdown files."""
import os
import json
from datetime import datetime


def save_report(content: str, query: str = "", session_id: str = "default") -> str:
    """Save a research report to a markdown file.
    
    Args:
        content: The report text to save.
        query: The original user query.
        session_id: The session identifier.
        
    Returns:
        The file path where the report was saved.
    """
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Clean filename from query
    safe_query = "".join(c for c in query if c.isalnum() or c in " -_").lower()[:60]
    if not safe_query:
        safe_query = "untitled"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_query}_{timestamp}.md"
    filepath = os.path.join(reports_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {query}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Session:** {session_id}\n\n")
        f.write("---\n\n")
        f.write(content)
    
    return filepath
