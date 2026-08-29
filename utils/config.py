"""
Configuration module for multi-agent RAG researcher.

Supports OpenAI and local LLMs with OpenAI-compatible endpoints.
"""

import math
import os
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv


# Load environment variables
UTILS_DIR = Path(__file__).resolve().parent
ENV_FILE_PATH = UTILS_DIR / "var.env"


def _normalize_endpoint_url(endpoint: str) -> str:
    """
    Normalize an endpoint URL to ensure consistent formatting.
    
    Args:
        endpoint: The endpoint URL to normalize
        
    Returns:
        Normalized endpoint URL with scheme and trailing slash removed
    """
    # Strip trailing slashes for consistency
    normalized = endpoint.rstrip('/')
    
    # Ensure URL has a scheme (check for :// or : followed by path)
    if '://' not in normalized:
        # If it contains a port number or path, assume http
        if ':' in normalized or '/' in normalized:
            normalized = f"http://{normalized}"
        else:
            normalized = f"https://{normalized}"
    
    return normalized


def _ensure_env_file_exists() -> None:
    """
    Ensure var.env file exists, creating it from .env.example if missing.
    Warns if neither file exists.
    """
    var_env_exists = ENV_FILE_PATH.exists()
    env_example_path = UTILS_DIR.parent / ".env.example"
    env_example_exists = env_example_path.exists()
    
    if not var_env_exists and not env_example_exists:
        print("⚠️  WARNING: No configuration files found.")
        print("   Create a .env.example file with your API keys, or run:")
        print("   cp .env.example .env")
        return
    
    if not var_env_exists and env_example_exists:
        try:
            shutil.copy2(env_example_path, ENV_FILE_PATH)
            print(f"✅ Created {ENV_FILE_PATH} from .env.example")
            print("   Please update the values in .env with your actual API keys.")
        except Exception as e:
            print(f"⚠️  WARNING: Could not create {ENV_FILE_PATH}: {e}")
            print("   Please create it manually from .env.example")
    
    # Log environment file being used
    logger = logging.getLogger(__name__)
    logger.info(f"Using configuration file: {ENV_FILE_PATH}")


# Load environment variables after ensuring file exists
_ensure_env_file_exists()
load_dotenv(ENV_FILE_PATH)


@dataclass
class LLMConfig:
    """Configuration for a single LLM endpoint."""
    endpoint: str
    api_key: str
    model: str
    name: str = "default"
    
    def __post_init__(self):
        # Validate endpoint URL
        if not self.endpoint:
            raise ValueError("Endpoint cannot be empty")
        
        # Normalize and validate endpoint
        self.endpoint = _normalize_endpoint_url(self.endpoint)
        
        # Validate it looks like a valid URL
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, self.endpoint, re.IGNORECASE):
            raise ValueError(f"Invalid endpoint URL: {self.endpoint}")


@dataclass
class Config:
    """Main configuration object."""
    # Global defaults
    default_endpoint: str
    default_api_key: str
    default_model: str
    
    # Per-agent overrides
    retriever_endpoint: Optional[str] = None
    retriever_api_key: Optional[str] = None
    retriever_model: Optional[str] = None
    
    writer_endpoint: Optional[str] = None
    writer_api_key: Optional[str] = None
    writer_model: Optional[str] = None
    
    verifier_endpoint: Optional[str] = None
    verifier_api_key: Optional[str] = None
    verifier_model: Optional[str] = None
    
    orchestrator_endpoint: Optional[str] = None
    orchestrator_api_key: Optional[str] = None
    orchestrator_model: Optional[str] = None

    # Decomposer model override only (endpoint/api_key stay global): the
    # structured research plan needs a schema-compliant model, which may
    # differ from the global default (e.g. DECOMPOSER_MODEL=Qwen... while
    # LLM_MODEL is a weaker local model).
    decomposer_model: Optional[str] = None
    
    # Agent-specific defaults (reasoning effort, etc.)
    retriever_reasoning_effort: str = "low"
    writer_reasoning_effort: str = "medium"
    verifier_reasoning_effort: str = "medium"
    orchestrator_reasoning_effort: str = "low"
    decomposer_reasoning_effort: str = "low"
    sufficiency_reasoning_effort: str = "low"
    
    # Per-agent max output tokens (Responses API `max_output_tokens`)
    retriever_max_output_tokens: int = 2000
    writer_max_output_tokens: int = 16000
    verifier_max_output_tokens: int = 16000
    orchestrator_max_output_tokens: int = 2000
    decomposer_max_output_tokens: int = 2000
    sufficiency_max_output_tokens: int = 1000

    # Deep-mode evidence cache (P2-2): reuse previously retrieved
    # per-sub-question evidence within the TTL instead of re-retrieving.
    # Disable with EVIDENCE_CACHE_ENABLED=false.
    evidence_cache_enabled: bool = True
    evidence_cache_ttl_days: int = 30

    # Minimum cosine similarity for document retrieval results (the only
    # relevance floor on qdrant hits; see retriever_agent.retrieve_document).
    # Weak embedders that score weakly-relevant chunks near 1.0 need a
    # higher value: DOC_SCORE_THRESHOLD=0.6. Cosine is a raw score, not a
    # calibrated probability, so there is no "correct" value per model.
    doc_score_threshold: float = 0.2
    
    # Cached clients
    _clients: Dict[str, Any] = field(default_factory=dict)
    
    def get_agent_config(self, agent_name: str) -> LLMConfig:
        """Get configuration for a specific agent."""
        # Get overrides for this agent. The override fields are optional:
        # agents without dedicated fields (e.g. sufficiency, and the
        # decomposer's endpoint/api_key) inherit the global defaults via
        # the None fallback. The decomposer additionally supports a model
        # override (DECOMPOSER_MODEL) for schema-compliance routing.
        endpoint = getattr(self, f"{agent_name}_endpoint", None) or self.default_endpoint
        api_key = getattr(self, f"{agent_name}_api_key", None) or self.default_api_key
        model = getattr(self, f"{agent_name}_model", None) or self.default_model
        
        return LLMConfig(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            name=agent_name
        )
    
    def get_reasoning_effort(self, agent_name: str) -> str:
        """Get reasoning effort for a specific agent."""
        return getattr(self, f"{agent_name}_reasoning_effort")
    
    def get_max_output_tokens(self, agent_name: str) -> int:
        """Get the max output tokens for a specific agent."""
        return getattr(self, f"{agent_name}_max_output_tokens")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration."""
    global _config
    
    if _config is None:
        # Safe-int: a bad EVIDENCE_CACHE_TTL_DAYS must not crash config
        # loading app-wide (falls back to the 30-day default).
        try:
            evidence_cache_ttl_days = int(os.getenv("EVIDENCE_CACHE_TTL_DAYS", "30"))
        except (TypeError, ValueError):
            logger = logging.getLogger(__name__)
            logger.warning(
                "Invalid EVIDENCE_CACHE_TTL_DAYS value; falling back to 30 days."
            )
            evidence_cache_ttl_days = 30

        # Safe-float: a bad DOC_SCORE_THRESHOLD must not crash config
        # loading app-wide (falls back to the 0.2 default). Non-finite
        # values ("nan", "inf") parse fine as floats but are unusable as a
        # relevance floor, so they are treated as invalid too.
        try:
            doc_score_threshold = float(os.getenv("DOC_SCORE_THRESHOLD", "0.2"))
            if not math.isfinite(doc_score_threshold):
                raise ValueError(f"non-finite: {doc_score_threshold!r}")
        except (TypeError, ValueError):
            logger = logging.getLogger(__name__)
            logger.warning(
                "Invalid DOC_SCORE_THRESHOLD value; falling back to 0.2."
            )
            doc_score_threshold = 0.2

        _config = Config(
            # Global defaults from environment
            default_endpoint=os.getenv("LLM_ENDPOINT", os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")),
            default_api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            default_model=os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4")),
            
            # Retriever agent overrides
            retriever_endpoint=os.getenv("RETRIEVER_ENDPOINT"),
            retriever_api_key=os.getenv("RETRIEVER_API_KEY"),
            retriever_model=os.getenv("RETRIEVER_MODEL"),
            
            # Writer agent overrides
            writer_endpoint=os.getenv("WRITER_ENDPOINT"),
            writer_api_key=os.getenv("WRITER_API_KEY"),
            writer_model=os.getenv("WRITER_MODEL"),
            
            # Verifier agent overrides
            verifier_endpoint=os.getenv("VERIFIER_ENDPOINT"),
            verifier_api_key=os.getenv("VERIFIER_API_KEY"),
            verifier_model=os.getenv("VERIFIER_MODEL"),
            
            # Orchestrator agent overrides
            orchestrator_endpoint=os.getenv("ORCHESTRATOR_ENDPOINT"),
            orchestrator_api_key=os.getenv("ORCHESTRATOR_API_KEY"),
            orchestrator_model=os.getenv("ORCHESTRATOR_MODEL"),
            
            # Decomposer model override (endpoint/api_key stay global)
            decomposer_model=os.getenv("DECOMPOSER_MODEL"),
            
            # Per-agent reasoning effort (documented in get_env_example)
            retriever_reasoning_effort=os.getenv("RETRIEVER_REASONING_EFFORT", "low"),
            writer_reasoning_effort=os.getenv("WRITER_REASONING_EFFORT", "medium"),
            verifier_reasoning_effort=os.getenv("VERIFIER_REASONING_EFFORT", "medium"),
            orchestrator_reasoning_effort=os.getenv("ORCHESTRATOR_REASONING_EFFORT", "low"),
            decomposer_reasoning_effort=os.getenv("DECOMPOSER_REASONING_EFFORT", "low"),
            sufficiency_reasoning_effort=os.getenv("SUFFICIENCY_REASONING_EFFORT", "low"),
            
            # Per-agent max output tokens (Responses API max_output_tokens)
            retriever_max_output_tokens=int(os.getenv("RETRIEVER_MAX_OUTPUT_TOKENS", "2000")),
            writer_max_output_tokens=int(os.getenv("WRITER_MAX_OUTPUT_TOKENS", "16000")),
            verifier_max_output_tokens=int(os.getenv("VERIFIER_MAX_OUTPUT_TOKENS", "16000")),
            orchestrator_max_output_tokens=int(os.getenv("ORCHESTRATOR_MAX_OUTPUT_TOKENS", "2000")),
            decomposer_max_output_tokens=int(os.getenv("DECOMPOSER_MAX_OUTPUT_TOKENS", "2000")),
            sufficiency_max_output_tokens=int(os.getenv("SUFFICIENCY_MAX_OUTPUT_TOKENS", "1000")),
            evidence_cache_enabled=os.getenv(
                "EVIDENCE_CACHE_ENABLED", "true"
            ).strip().lower() in ("1", "true", "yes", "on"),
            evidence_cache_ttl_days=evidence_cache_ttl_days,
            doc_score_threshold=doc_score_threshold,
        )
        
        # Validate configurations and issue warnings
        _validate_config(_config)
    
    return _config


def _validate_config(config: Config) -> None:
    """Validate configuration and print warnings."""
    logger = logging.getLogger(__name__)
    
    # Check if API keys are missing
    if not config.default_api_key:
        print("⚠️  WARNING: No API key found. Set LLM_API_KEY or OPENAI_API_KEY in environment.")
    
    # Check if using OpenAI without API key
    if "openai.com" in config.default_endpoint and not config.default_api_key:
        print("⚠️  WARNING: Using OpenAI endpoint but no API key is set. LLM calls may fail.")
    
    # Print agent-specific configuration
    agents = ["retriever", "writer", "verifier", "orchestrator", "decomposer"]
    for agent in agents:
        agent_config = config.get_agent_config(agent)
        if agent_config.endpoint != config.default_endpoint:
            logger.info(f"{agent.capitalize()} Agent using custom endpoint: {agent_config.endpoint}")
        if agent_config.model != config.default_model:
            logger.info(f"{agent.capitalize()} Agent using custom model: {agent_config.model}")


def create_openai_client(endpoint: str, api_key: str) -> Any:
    """
    Create an OpenAI-compatible client for a given endpoint.
    
    Args:
        endpoint: The base URL for the LLM API (e.g., https://api.openai.com/v1)
        api_key: The API key for authentication (can be empty for local LLMs)
        
    Returns:
        An OpenAI client instance configured for the endpoint
    """
    try:
        from openai import OpenAI
        # Only set api_key if non-empty (some local LLMs don't require auth)
        if api_key and api_key.lower() not in ('n/a', 'none', 'null', ''):
            return OpenAI(base_url=endpoint.rstrip('/'), api_key=api_key)
        else:
            # For local LLMs without auth (Ollama, etc.)
            return OpenAI(base_url=endpoint.rstrip('/'), api_key='dummy')
    except ImportError:
        raise ImportError(
            "OpenAI library not installed. Install with: pip install openai"
        )


def get_client_for_endpoint(endpoint: str, api_key: str) -> Any:
    """
    Get or create a cached client for a specific endpoint.
    
    Args:
        endpoint: The base URL for the LLM API
        api_key: The API key for authentication
        
    Returns:
        A cached OpenAI client instance
    """
    config = get_config()
    
    # Normalize endpoint for consistent caching
    normalized_endpoint = _normalize_endpoint_url(endpoint)
    cache_key = f"{normalized_endpoint}:{api_key}"
    
    if cache_key not in config._clients:
        config._clients[cache_key] = create_openai_client(endpoint, api_key)
        print(f"✅ Created new client for endpoint: {normalized_endpoint}")
    
    return config._clients[cache_key]


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


def get_env_example() -> str:
    """Generate an example .env file content."""
    return '''# ========================================
# Multi-Agent RAG Researcher Configuration
# ========================================

# ----------------------------------------
# Global Settings (default for all agents)
# ----------------------------------------
# LLM Endpoint (OpenAI-compatible)
# For OpenAI: https://api.openai.com/v1
# For Ollama: http://localhost:11434/v1
# For LM Studio: http://localhost:1234/v1
# For Azure OpenAI: https://your-resource-name.openai.azure.com/
LLM_ENDPOINT=https://api.openai.com/v1

# API Key for the LLM endpoint
LLM_API_KEY=your_api_key_here

# Default model to use
LLM_MODEL=gpt-5.4

# ----------------------------------------
# Per-Agent Configuration (optional overrides)
# ----------------------------------------
# Each agent can use a different endpoint, API key, or model

# Retriever Agent
RETRIEVER_ENDPOINT=https://api.openai.com/v1
RETRIEVER_API_KEY=your_retriever_api_key_here
RETRIEVER_MODEL=gpt-5.4-mini

# Writer Agent
WRITER_ENDPOINT=https://api.openai.com/v1
WRITER_API_KEY=your_writer_api_key_here
WRITER_MODEL=gpt-5.4

# Verifier Agent
VERIFIER_ENDPOINT=https://api.openai.com/v1
VERIFIER_API_KEY=your_verifier_api_key_here
VERIFIER_MODEL=gpt-5.4

# Orchestrator Agent
ORCHESTRATOR_ENDPOINT=https://api.openai.com/v1
ORCHESTRATOR_API_KEY=your_orchestrator_api_key_here
ORCHESTRATOR_MODEL=gpt-5.4-mini

# ----------------------------------------
# Agent Reasoning Effort (optional)
# ----------------------------------------
# Control the reasoning effort for each agent (low, medium, high)
# Lower effort = faster, less accurate; Higher effort = slower, more accurate

RETRIEVER_REASONING_EFFORT=low
WRITER_REASONING_EFFORT=medium
VERIFIER_REASONING_EFFORT=medium
ORCHESTRATOR_REASONING_EFFORT=low
DECOMPOSER_REASONING_EFFORT=low
SUFFICIENCY_REASONING_EFFORT=low

# ----------------------------------------
# Agent Max Output Tokens (optional)
# ----------------------------------------
# Cap each agent's model output length (Responses API max_output_tokens)
RETRIEVER_MAX_OUTPUT_TOKENS=2000
WRITER_MAX_OUTPUT_TOKENS=16000
VERIFIER_MAX_OUTPUT_TOKENS=16000
ORCHESTRATOR_MAX_OUTPUT_TOKENS=2000
DECOMPOSER_MAX_OUTPUT_TOKENS=2000
SUFFICIENCY_MAX_OUTPUT_TOKENS=1000

# ----------------------------------------
# Deep-Mode Evidence Cache (optional)
# ----------------------------------------
# Reuses previously retrieved per-sub-question evidence within the TTL
# instead of re-retrieving; disabled by setting EVIDENCE_CACHE_ENABLED=false
EVIDENCE_CACHE_ENABLED=true
EVIDENCE_CACHE_TTL_DAYS=30

# ----------------------------------------
# Document Retrieval Relevance Floor (optional)
# ----------------------------------------
# Minimum cosine similarity for qdrant document hits (default 0.2).
# Raise it (e.g. 0.6) if a weak embedder ranks off-topic chunks highly.
DOC_SCORE_THRESHOLD=0.2

# ----------------------------------------
# Local LLM Examples
# ----------------------------------------
# Ollama (local)
# LLM_ENDPOINT=http://localhost:11434/v1
# LLM_API_KEY=n/a  # Ollama doesn't require API keys by default

# LM Studio (local)
# LLM_ENDPOINT=http://localhost:1234/v1
# LLM_API_KEY=lm-studio

# ----------------------------------------
# Other API Keys
# ----------------------------------------
# OpenAI API Key (backward compatibility)
OPENAI_ENDPOINT=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.4

# Tavily Web Search API Key
TAVILY_API_KEY=your_tavily_api_key_here
'''


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("=== Configuration ===")
    print(f"Default Endpoint: {config.default_endpoint}")
    print(f"Default Model: {config.default_model}")
    print()
    
    # Test agent configs
    agents = ["retriever", "writer", "verifier", "orchestrator", "decomposer"]
    for agent in agents:
        agent_config = config.get_agent_config(agent)
        print(f"{agent.capitalize()} Agent:")
        print(f"  Endpoint: {agent_config.endpoint}")
        print(f"  Model: {agent_config.model}")
        print()