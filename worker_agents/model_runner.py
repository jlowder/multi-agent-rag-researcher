import os
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Compute UTILS_DIR before any imports that depend on it
UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
ENV_FILE_PATH = UTILS_DIR / "var.env"

load_dotenv(ENV_FILE_PATH)

from utils.config import (
    Config,
    LLMConfig,
    create_openai_client,
    get_client_for_endpoint,
    get_config,
    reset_config,
)
from utils.json_extract import extract_json_payload

# The SDK's own text-format builder (private but stable across openai 2.x):
# converts a pydantic model into the strict json_schema text.format the
# Responses API expects.
try:
    from openai.lib._parsing._responses import type_to_text_format_param
except ImportError:  # pragma: no cover - SDK layout drift
    type_to_text_format_param = None


def _text_format_param(text_format: Any) -> Dict[str, Any]:
    """Strict json_schema text.format entry for a pydantic text_format model."""
    if type_to_text_format_param is not None:
        try:
            return type_to_text_format_param(text_format)
        except Exception:
            pass
    schema = text_format.model_json_schema()
    return {
        "type": "json_schema",
        "strict": True,
        "name": schema.get("title") or "structured_output",
        "schema": schema,
    }


# Credential-shaped strings that must never reach a log line verbatim.
_SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9_\-]{8,}"
    r"|omlx-[A-Za-z0-9_\-]{8,}"
    r"|\bBearer\s+[A-Za-z0-9._\-]{8,}"
    r"|\b(api[_\-]?key|token|secret|authorization|password)[\"']?\s*[:=]\s*[\"']?[\w.\-]{6,}"
)


def _sanitize_span(span: str) -> str:
    """Escape control chars and redact credential-shaped strings; never raises."""
    one_line = span.replace("\r", "").replace("\t", "\\t").replace("\n", "\\n")
    one_line = re.sub(r"[\x00-\x1f\x7f]+", " ", one_line)
    one_line = _SECRET_RE.sub("[REDACTED]", one_line)
    return re.sub(r"\s{2,}", " ", one_line).strip()


def _sanitize_output_snippet(
    text: str, head_limit: int = 120, tail_limit: int = 80, full_limit: int = 200
) -> str:
    """One-line snippet of raw LLM output for parse-failure warnings.

    Shows BOTH ends of the output — first ~120 chars and last ~80 chars,
    joined by " …[N chars elided]… " (N = raw middle chars omitted) — so a
    failed parse exposes start-malformations (e.g. unquoted keys at char 1)
    AND end-truncation (unclosed JSON); the elided count makes
    suspiciously-short totals (token-cap cuts) visible. Output of
    <= full_limit chars is logged whole, without elision. Newlines/tabs are
    shown literally (one log line) and credential-shaped strings become
    [REDACTED] (re-checked across the assembled snippet, so secrets
    straddling the head/tail seam are caught too). Purely cosmetic; never
    raises.
    """
    if not text:
        return "(empty)"
    if len(text) <= full_limit:
        return _sanitize_span(text) or "(whitespace only)"
    head_n = min(head_limit, len(text))
    tail_n = min(tail_limit, len(text) - head_n)
    elided = len(text) - head_n - tail_n
    head = _sanitize_span(text[:head_n]).rstrip()
    tail = _sanitize_span(text[len(text) - tail_n:]).lstrip()
    return _SECRET_RE.sub("[REDACTED]", f"{head} …[{elided} chars elided]… {tail}")


def _normalize_endpoint_url(endpoint: str) -> str:
    """
    Normalize an endpoint URL to ensure consistent formatting.
    
    Args:
        endpoint: The endpoint URL to normalize
        
    Returns:
        Normalized endpoint URL with trailing slash removed
    """
    # Strip trailing slashes for consistency
    normalized = endpoint.rstrip('/')
    
    # Check for existing scheme properly
    if not (normalized.startswith('http://') or normalized.startswith('https://')):
        if ':' in normalized or '/' in normalized:
            normalized = f"http://{normalized}"
        else:
            normalized = f"https://{normalized}"
    
    return normalized


def get_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_name: Optional[str] = None
) -> OpenAI:
    """
    Get an OpenAI client for the specified endpoint.
    
    Args:
        endpoint: The base URL for the LLM API. If None, uses default endpoint.
        api_key: The API key for authentication. If None, uses default API key.
        agent_name: If provided, uses the agent-specific configuration.
        
    Returns:
        An OpenAI client instance
    """
    if agent_name:
        config = get_config()
        agent_config = config.get_agent_config(agent_name)
        endpoint = endpoint or agent_config.endpoint
        api_key = api_key or agent_config.api_key
    else:
        config = get_config()
        endpoint = endpoint or config.default_endpoint
        api_key = api_key or config.default_api_key
    
    # Normalize endpoint for consistent caching
    if endpoint:
        endpoint = _normalize_endpoint_url(endpoint)
    
    return get_client_for_endpoint(endpoint, api_key)


def get_openai_client() -> OpenAI:
    """Get the default OpenAI client (backward compatibility)."""
    config = get_config()
    return get_client_for_endpoint(config.default_endpoint, config.default_api_key)


""" Helper function to be used by the agents to run the LLM """
def run_model(
    *,
    instructions: str,
    input_data: Any,
    tools: Optional[List[Dict[str, Any]]] = None,
    previous_response_id: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    text_format: Any = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_name: Optional[str] = None,
):
    """
    Run an LLM model with the given parameters.
    
    Args:
        instructions: System instructions for the model
        input_data: Input data to process
        tools: Optional list of tool schemas
        previous_response_id: ID of previous response for multi-turn conversations
        model: Model name (uses agent-specific or default if not provided)
        reasoning_effort: Reasoning effort level (e.g., "low", "medium", "high")
        max_output_tokens: Optional cap on model output length (Responses API
            max_output_tokens); omitted from the request when None
        text_format: Optional Pydantic model for structured output
        endpoint: Custom endpoint URL (overrides default)
        api_key: Custom API key (overrides default)
        agent_name: Agent name to use configuration from (e.g., "retriever", "writer")
        
    Returns:
        The model response
    """
    logger = logging.getLogger(__name__)
    
    # ============================================
    # STEP 1: Log incoming parameters
    # ============================================
    logger.info("=" * 80)
    logger.info("RUN_MODEL CALLED WITH PARAMETERS:")
    logger.info(f"  instructions length: {len(instructions)} chars")
    logger.info(f"  input_data type: {type(input_data).__name__}")
    if isinstance(input_data, str):
        logger.info(f"  input_data preview: {input_data[:100]}...")
    logger.info(f"  tools: {len(tools) if tools else 0} tools")
    logger.info(f"  previous_response_id: {previous_response_id}")
    logger.info(f"  model (incoming): {model} (type: {type(model).__name__ if model is not None else 'NoneType'})")
    logger.info(f"  reasoning_effort: {reasoning_effort}")
    logger.info(f"  max_output_tokens: {max_output_tokens}")
    logger.info(f"  text_format: {text_format}")
    logger.info(f"  endpoint: {endpoint}")
    logger.info(f"  api_key: {'***' if api_key else '(none)'}")
    logger.info(f"  agent_name: {agent_name}")
    logger.info("=" * 80)
    
    # Get client based on parameters
    client = get_client(endpoint=endpoint, api_key=api_key, agent_name=agent_name)
    
    # ============================================
    # STEP 2: Get model name from agent_name if model is None
    # ============================================
    if model is None and agent_name:
        logger.info(f"Model is None, looking up config for agent_name: '{agent_name}'")
        config = get_config()
        agent_config = config.get_agent_config(agent_name)
        logger.info(f"Agent config retrieved:")
        logger.info(f"  endpoint: {agent_config.endpoint}")
        logger.info(f"  model: {agent_config.model} (type: {type(agent_config.model).__name__})")
        logger.info(f"  api_key: {'***' if agent_config.api_key else '(none)'}")
        model = agent_config.model
    
    # ============================================
    # STEP 3: Fallback to default model if still None
    # ============================================
    if model is None:
        logger.warning("Model is still None after agent_name lookup, using default_model")
        config = get_config()
        model = config.default_model
        logger.info(f"Using default_model: {model} (type: {type(model).__name__})")
    
    # ============================================
    # STEP 4: Validate model is a string
    # ============================================
    if not isinstance(model, str):
        logger.error(f"CRITICAL: Model is not a string! Type: {type(model)}, Value: {model}")
        raise ValueError(f"Model must be a non-empty string, got: {type(model)} - {model}")
    
    if not model:
        logger.error("CRITICAL: Model is an empty string!")
        raise ValueError("Model must be a non-empty string, got empty string")
    
    logger.info(f"VALIDATED: model is a valid string: '{model}'")
    
    # ============================================
    # STEP 5: Build request dictionary
    # ============================================
    request = dict(
        model=model,
        instructions=instructions,
        input=input_data,
        tools=tools or [],
        # Force tool usage if tools are provided (prevents LLM from answering
        # from training data instead of using retrieval tools)
        tool_choice = "required" if tools else "auto",
        previous_response_id=previous_response_id,
        parallel_tool_calls=False,
    )
    
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}
        logger.info(f"Added reasoning_effort: {reasoning_effort}")
    
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens
        logger.info(f"Added max_output_tokens: {max_output_tokens}")
    
    logger.info("=" * 80)
    logger.info("FINAL REQUEST DICT (about to send to API):")
    logger.info(f"  model: '{request['model']}' (type: {type(request['model']).__name__})")
    logger.info(f"  instructions length: {len(request['instructions'])} chars")
    logger.info(f"  input type: {type(request['input']).__name__}")
    logger.info(f"  tools count: {len(request['tools'])}")
    logger.info(f"  tool_choice: {request['tool_choice']}")
    logger.info(f"  previous_response_id: {request['previous_response_id']}")
    logger.info(f"  parallel_tool_calls: {request['parallel_tool_calls']}")
    if "reasoning" in request:
        logger.info(f"  reasoning: {request['reasoning']}")
    if "max_output_tokens" in request:
        logger.info(f"  max_output_tokens: {request['max_output_tokens']}")
    logger.info("=" * 80)
    
    if text_format is not None:
        # Send the strict schema for servers that honor text.format (the
        # local MLX server ignores it), but call .create — NOT .parse: the
        # SDK's parse runs model_validate_json on the FULL raw text
        # client-side and raises on any conversational preamble, before any
        # agent fallback can see the raw output.
        request["text"] = {"format": _text_format_param(text_format)}
        logger.info("Calling client.responses.create() with structured output")
    else:
        logger.info("Calling client.responses.create()")
    response = client.responses.create(**request)
    # The local MLX server occasionally returns a response whose `output`
    # is None (model glitch); reading .output_text on it raises TypeError.
    # Normalize to an empty list so every caller's fallback path engages
    # (decomposer fallback plan, neutral critic, empty prose) instead of
    # crashing the whole run.
    if getattr(response, "output", None) is None:
        logger.warning(
            f"Model returned a response with output=None "
            f"(model={request.get('model')}); normalizing to empty output"
        )
        response.output = []

    if text_format is not None:
        # Tolerant client-side parse: recover the JSON payload from whatever
        # the server actually returned (preamble/postamble/fences) and
        # validate it. Invariant: never raises on malformed or preambled
        # content — output_parsed is the validated model on success, None
        # otherwise, so callers run their existing fallbacks. The raw
        # output_text is preserved untouched for those fallbacks.
        try:
            response.output_parsed = text_format.model_validate(
                extract_json_payload(getattr(response, "output_text", "") or "")
            )
        except Exception:
            response.output_parsed = None
            # Diagnostic only: every text_format parse failure shows what
            # the model actually emitted (sanitized to one line, secrets
            # redacted) so "could not be parsed" warnings are debuggable.
            raw_text = getattr(response, "output_text", "") or ""
            logger.warning(
                "no structured output on response (text_format=%s, model=%s, "
                "raw output_text %d chars, starts: %s)",
                getattr(text_format, "__name__", str(text_format)),
                request.get("model"),
                len(raw_text),
                _sanitize_output_snippet(raw_text),
            )

    return response


# Legacy function name for backward compatibility
run_llm = run_model