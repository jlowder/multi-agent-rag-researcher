# Local LLM Support and API Token Authentication

This document describes the new configuration system for supporting local LLMs with OpenAI-compatible endpoints and API token authentication in the multi-agent RAG researcher.

## Features

- ✅ **OpenAI-compatible endpoints**: Works with any OpenAI-compatible API (Ollama, LM Studio, etc.)
- ✅ **API token authentication**: Supports custom API keys for local LLMs
- ✅ **Per-agent configuration**: Each agent can use a different endpoint/model
- ✅ **Backward compatible**: Existing OpenAI configuration continues to work
- ✅ **Validation**: URL validation and helpful warnings
- ✅ **Client caching**: Efficient client reuse across multiple calls

## Configuration Overview

Configuration is managed through environment variables, with support for:

1. **Global defaults** - Applied to all agents unless overridden
2. **Per-agent overrides** - Specific endpoints/models for each agent

## Environment Variables

### Global Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENDPOINT` | `https://api.openai.com/v1` | Base URL for the LLM API |
| `LLM_API_KEY` | (required) | API key for authentication |
| `LLM_MODEL` | `gpt-5.4` | Default model name |

### Per-Agent Settings

Each agent (retriever, writer, verifier, orchestrator) can have its own endpoint, API key, and model:

| Agent | Endpoint Var | API Key Var | Model Var |
|-------|--------------|-------------|-----------|
| Retriever | `RETRIEVER_ENDPOINT` | `RETRIEVER_API_KEY` | `RETRIEVER_MODEL` |
| Writer | `WRITER_ENDPOINT` | `WRITER_API_KEY` | `WRITER_MODEL` |
| Verifier | `VERIFIER_ENDPOINT` | `VERIFIER_API_KEY` | `VERIFIER_MODEL` |
| Orchestrator | `ORCHESTRATOR_ENDPOINT` | `ORCHESTRATOR_API_KEY` | `ORCHESTRATOR_MODEL` |

### Backward Compatibility

The following variables are still supported for backward compatibility:

| Variable | Description |
|----------|-------------|
| `OPENAI_ENDPOINT` | OpenAI endpoint (alias for `LLM_ENDPOINT`) |
| `OPENAI_API_KEY` | OpenAI API key (alias for `LLM_API_KEY`) |
| `OPENAI_MODEL` | OpenAI model (alias for `LLM_MODEL`) |

## Usage Examples

### Example 1: Using OpenAI (Default)

```bash
# No configuration needed - uses defaults
python3 run_orchestrator.py
```

Or explicitly set:

```bash
export LLM_ENDPOINT=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-5.4
python3 run_orchestrator.py
```

### Example 2: Using Ollama (Local LLM)

```bash
# Start Ollama first: https://ollama.ai/download
# Then configure the agents:

# Global config - all agents use Ollama
export LLM_ENDPOINT=http://localhost:11434/v1
export LLM_API_KEY=n/a  # Ollama doesn't require API keys by default
export LLM_MODEL=llama3.1:8b

# Or per-agent config
export RETRIEVER_ENDPOINT=http://localhost:11434/v1
export RETRIEVER_MODEL=llama3.1:8b
export WRITER_ENDPOINT=http://localhost:11434/v1
export WRITER_MODEL=gemma2:9b
export VERIFIER_ENDPOINT=http://localhost:11434/v1
export VERIFIER_MODEL=mistral:7b
```

### Example 3: Using LM Studio (Local LLM)

```bash
# Start LM Studio and load a model
# Then configure:

export LLM_ENDPOINT=http://localhost:1234/v1
export LLM_API_KEY=lm-studio  # LM Studio requires this value
export LLM_MODEL=your-loaded-model-name
```

### Example 4: Using Azure OpenAI

```bash
export LLM_ENDPOINT=https://your-resource-name.openai.azure.com/
export LLM_API_KEY=your_azure_api_key
export LLM_MODEL=gpt-4
```

### Example 5: Mixed Configuration

Use different models for different agents:

```bash
# Global default
export LLM_ENDPOINT=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-5.4

# Retriever uses a cheaper/faster model
export RETRIEVER_MODEL=gpt-5.4-mini

# Writer uses the full model
export WRITER_MODEL=gpt-5.4

# Verifier uses a different model for fact-checking
export VERIFIER_MODEL=gpt-4.1
```

## Agent Reasoning Effort

Agents can be configured with different reasoning effort levels:

| Variable | Default | Options |
|----------|---------|---------|
| `RETRIEVER_REASONING_EFFORT` | `low` | `low`, `medium`, `high` |
| `WRITER_REASONING_EFFORT` | `low` | `low`, `medium`, `high` |
| `VERIFIER_REASONING_EFFORT` | `low` | `low`, `medium`, `high` |
| `ORCHESTRATOR_REASONING_EFFORT` | `low` | `low`, `medium`, `high` |

## Programmatic Configuration

You can also configure agents programmatically:

```python
from orchestrator_agent import orchestrator_agent

# Use a custom endpoint for a single call
result = orchestrator_agent(
    user_query="What is this document about?",
    session_id="my-session",
    endpoint="http://localhost:11434/v1",
    api_key="n/a",
    verbose=True
)
```

Each agent function also supports these parameters:

```python
from worker_agents.retriever_agent import retriever_agent

result = retriever_agent(
    user_query="Find information about X",
    endpoint="http://localhost:11434/v1",
    api_key="n/a"
)
```

## Validation and Warnings

The configuration system includes validation and helpful warnings:

1. **Missing API Key**: Warns if no API key is set when using OpenAI
2. **Invalid URL**: Validates endpoint URLs and adds default scheme if missing
3. **Agent-Specific Config**: Logs when agents use non-default configurations
4. **Client Creation**: Logs when new clients are created for endpoints

Example output:
```
✅ Created new client for endpoint: http://localhost:11434/v1
ℹ️  Retriever Agent using custom model: llama3.1:8b
⚠️  WARNING: No API key found. Set LLM_API_KEY or OPENAI_API_KEY in environment.
```

## File Structure

- `utils/config.py`: Configuration module with validation
- `worker_agents/model_runner.py`: Updated to support multiple clients
- `worker_agents/retriever_agent.py`: Updated with endpoint parameters
- `worker_agents/writer_agent.py`: Updated with endpoint parameters
- `worker_agents/verifier_agent.py`: Updated with endpoint parameters
- `orchestrator_agent.py`: Updated with endpoint parameters
- `.env.example`: Example configuration file

## Migration Guide

### From Old Configuration

If you were using `utils/var.env` with OpenAI settings:

```env
# Old format (still works)
OPENAI_API_KEY=sk-...
```

The system automatically falls back to OpenAI with the old environment variable.

### New Configuration

For local LLMs or custom endpoints:

```env
# New format
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b
```

## Troubleshooting

### Connection Refused

If you see connection refused errors:
- Ensure the LLM server is running (Ollama/LM Studio)
- Verify the endpoint URL is correct
- Check firewall settings

### Authentication Errors

If you get authentication errors:
- Verify your API key is correct
- Check if the endpoint requires authentication
- For Ollama, use `n/a` as the API key

### Model Not Found

If the model is not found:
- Verify the model name is correct
- For Ollama, ensure the model is pulled: `ollama pull llama3.1:8b`
- For LM Studio, ensure the model is loaded in the UI

### Performance Issues

If using local LLMs is slow:
- Use smaller models (e.g., `llama3.1:8b` vs `llama3.1:70b`)
- Reduce `reasoning_effort` settings
- Use GPU acceleration if available