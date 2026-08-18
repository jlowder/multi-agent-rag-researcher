# Local LLM Setup Guide

This guide explains how to configure the multi-agent RAG researcher to work with local LLMs and any OpenAI-compatible API endpoint.

## Overview

The system supports:
- ✅ **OpenAI API** (backward compatible)
- ✅ **Local LLMs** (Ollama, LocalAI, LM Studio)
- ✅ **Any OpenAI-compatible endpoint** with API token authentication
- ✅ **Per-agent configuration** - each agent can use a different endpoint/model
- ✅ **Mixed configurations** - use different models for different agents

## Quick Start

### 1. For OpenAI (Default)
No configuration needed. The system uses default settings:
- Endpoint: `https://api.openai.com/v1`
- Model: `gpt-5.4`

### 2. For Local LLMs (Ollama Example)
```bash
# Start Ollama first
ollama serve

# Then in your terminal:
export LLM_ENDPOINT=http://localhost:11434/v1
export LLM_MODEL=llama3.1:8b
export LLM_API_KEY=n/a  # Ollama doesn't require API keys by default

# Run the orchestrator
python3 run_orchestrator.py
```

### 3. For LM Studio
```bash
# Start LM Studio and load a model
# Then:

export LLM_ENDPOINT=http://localhost:1234/v1
export LLM_API_KEY=lm-studio
export LLM_MODEL=your-loaded-model-name

python3 run_orchestrator.py
```

## Configuration Methods

### Method 1: Using Environment Variables

Create or edit `utils/var.env` in your project directory:

```env
# Global settings for all agents
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-5.4
```

### Method 2: Using .env.example

Copy the example configuration:
```bash
cp .env.example utils/var.env
# Then edit utils/var.env with your values
```

### Method 3: Programmatic Configuration

You can also pass endpoint and API key directly to agent functions:

```python
from orchestrator_agent import orchestrator_agent

result = orchestrator_agent(
    user_query="What is this document about?",
    session_id="my-session",
    endpoint="http://localhost:11434/v1",
    api_key="n/a",
    verbose=True
)
```

## Environment Variables Reference

### Global Settings

These settings apply to all agents unless overridden:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENDPOINT` | `https://api.openai.com/v1` | Base URL for the LLM API |
| `LLM_API_KEY` | (required) | API key for authentication (use `n/a` for Ollama) |
| `LLM_MODEL` | `gpt-5.4` | Default model name |

### Per-Agent Settings

Each agent can have its own endpoint, API key, and model:

| Agent | Endpoint Variable | API Key Variable | Model Variable | Default Model |
|-------|-------------------|------------------|----------------|---------------|
| Retriever | `RETRIEVER_ENDPOINT` | `RETRIEVER_API_KEY` | `RETRIEVER_MODEL` | `gpt-5.4-mini` |
| Writer | `WRITER_ENDPOINT` | `WRITER_API_KEY` | `WRITER_MODEL` | `gpt-5.4` |
| Verifier | `VERIFIER_ENDPOINT` | `VERIFIER_API_KEY` | `VERIFIER_MODEL` | `gpt-5.4` |
| Orchestrator | `ORCHESTRATOR_ENDPOINT` | `ORCHESTRATOR_API_KEY` | `ORCHESTRATOR_MODEL` | `gpt-5.4-mini` |

### Agent Reasoning Effort

Control the reasoning effort for each agent:

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `RETRIEVER_REASONING_EFFORT` | `low` | `low`, `medium`, `high` | Lower effort = faster, less accurate |
| `WRITER_REASONING_EFFORT` | `low` | `low`, `medium`, `high` | Higher effort = slower, more accurate |
| `VERIFIER_REASONING_EFFORT` | `low` | `low`, `medium`, `high` | Higher effort = slower, more accurate |
| `ORCHESTRATOR_REASONING_EFFORT` | `low` | `low`, `medium`, `high` | Higher effort = slower, more accurate |

### Backward Compatibility Variables

The following variables are still supported:

| Variable | Alias For | Description |
|----------|-----------|-------------|
| `OPENAI_ENDPOINT` | `LLM_ENDPOINT` | OpenAI endpoint |
| `OPENAI_API_KEY` | `LLM_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | `LLM_MODEL` | OpenAI model |

## Configuration Examples

### Example 1: Using Ollama (Local LLM)

Ollama runs locally on port 11434 by default and doesn't require API keys.

```env
# Global config - all agents use Ollama
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b

# Optional: Different models for different agents
# RETRIEVER_MODEL=llama3.1:8b
# WRITER_MODEL=gemma2:9b
# VERIFIER_MODEL=mistral:7b
```

**Prerequisites:**
1. Install Ollama: https://ollama.ai/download
2. Pull the model: `ollama pull llama3.1:8b`
3. Ensure Ollama is running: `ollama serve`

### Example 2: Using LocalAI (Local LLM)

LocalAI is a self-hosted OpenAI-compatible API server.

```env
# Global config
LLM_ENDPOINT=http://localhost:8080/v1
LLM_API_KEY=n/a  # Or your LocalAI API key
LLM_MODEL=llama3-8b

# Per-agent configuration (optional)
RETRIEVER_ENDPOINT=http://localhost:8080/v1
RETRIEVER_MODEL=llama3-8b
WRITER_ENDPOINT=http://localhost:8080/v1
WRITER_MODEL=llama3-70b
```

**Prerequisites:**
1. Deploy LocalAI: https://localai.io/basics/getting_started/
2. Load your model: `curl http://localhost:8080/models/ensure -X POST -H "Content-Type: application/json" -d '{"model": "llama3-8b"}'`

### Example 3: Using LM Studio (Local LLM)

LM Studio provides a GUI for loading local models.

```env
# Global config
LLM_ENDPOINT=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=your-loaded-model-name

# Get the exact model name from LM Studio's model dropdown
```

**Prerequisites:**
1. Install LM Studio: https://lmstudio.ai/
2. Load a model through the LM Studio interface
3. Start the local server (default port: 1234)

### Example 4: Using Azure OpenAI

```env
LLM_ENDPOINT=https://your-resource-name.openai.azure.com/
LLM_API_KEY=your_azure_api_key
LLM_MODEL=gpt-4
```

### Example 5: Using OpenAI (Default)

No configuration needed, but you can explicitly set it:

```env
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=sk-your_openai_api_key
LLM_MODEL=gpt-5.4
```

### Example 6: Mixed Configuration

Use different models for different agents to optimize cost and performance:

```env
# Global default
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-5.4

# Retriever uses a cheaper/faster model
RETRIEVER_MODEL=gpt-5.4-mini

# Writer uses the full model for better quality
WRITER_MODEL=gpt-5.4

# Verifier uses a different model for fact-checking
VERIFIER_MODEL=gpt-4.1
```

### Example 7: Using Any OpenAI-Compatible Service

For services like Groq, DeepSeek, or other providers:

```env
# Groq example
LLM_ENDPOINT=https://api.groq.com/openai/v1/
LLM_API_KEY=gsk_...
LLM_MODEL=llama3-70b-8192

# DeepSeek example
LLM_ENDPOINT=https://api.deepseek.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=deepseek-chat
```

## API Token Authentication

### Services Requiring API Keys

Most cloud-based services and some local LLM setups require API tokens:

```env
# Ollama with authentication
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=your_ollama_api_token

# LocalAI with authentication
LLM_ENDPOINT=http://localhost:8080/v1
LLM_API_KEY=your_localai_api_key

# Any service
LLM_ENDPOINT=https://api.your-service.com/v1
LLM_API_KEY=your_api_key_here
```

### No Authentication Required

Some local LLM servers run without authentication:

```env
# Ollama (default installation)
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a  # or any non-empty string, auth is disabled
```

## Migration Guide

### From OpenAI to Local LLMs

#### Step 1: Verify Local LLM Server is Running

**For Ollama:**
```bash
# Check if Ollama is running
curl http://localhost:11434

# Should return: {" models ":"..."}
```

**For LocalAI:**
```bash
# Check if LocalAI is running
curl http://localhost:8080/models

# Should return a list of available models
```

**For LM Studio:**
1. Open LM Studio
2. Click the server icon in the bottom left
3. Verify the server is running on port 1234

#### Step 2: Update Configuration

Create or edit `utils/var.env`:

```env
# Old configuration (OpenAI)
# OPENAI_API_KEY=sk-...

# New configuration (Ollama)
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b
```

#### Step 3: Test the Configuration

Run the orchestrator:
```bash
python3 run_orchestrator.py
```

You should see output like:
```
✅ Created new client for endpoint: http://localhost:11434/v1
```

#### Step 4: Troubleshoot if Needed

If you see connection errors:
- Ensure the LLM server is running
- Verify the endpoint URL is correct
- Check firewall settings

### From Old Configuration Format

If you were using `utils/var.env` with the old format:

```env
# Old format (still works for backward compatibility)
OPENAI_API_KEY=sk-...
```

Update to the new format:

```env
# New format (recommended)
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-5.4
```

## Troubleshooting

### Connection Refused

**Error:**
```
ConnectionRefusedError: [Errno 61] Connection refused
```

**Solutions:**
1. Ensure the LLM server is running:
   ```bash
   # Ollama
   ollama serve
   
   # LocalAI
   ./local-ai
   
   # LM Studio
   # Check server is running in LM Studio UI
   ```

2. Verify the endpoint URL:
   ```bash
   curl http://localhost:11434
   curl http://localhost:8080/models
   curl http://localhost:1234/models
   ```

3. Check firewall settings

### Authentication Errors

**Error:**
```
AuthenticationError: Incorrect API key provided
```

**Solutions:**
1. Verify your API key is correct
2. Check if the endpoint requires authentication
3. For Ollama without auth, use `n/a` as the API key
4. For services with auth, ensure you're using a valid API key

### Model Not Found

**Error:**
```
NotFoundError: model 'llama3.1:8b' not found
```

**Solutions:**
1. For Ollama, pull the model:
   ```bash
   ollama pull llama3.1:8b
   ```

2. Verify the model name:
   ```bash
   # Ollama
   ollama list
   
   # LocalAI
   curl http://localhost:8080/models
   ```

3. For LM Studio, ensure the model is loaded in the UI

### Performance Issues with Local LLMs

**Solutions:**
1. Use smaller models:
   - `llama3.1:8b` instead of `llama3.1:70b`
   - `gemma2:2b` instead of `gemma2:9b`

2. Reduce reasoning effort:
   ```env
   RETRIEVER_REASONING_EFFORT=low
   WRITER_REASONING_EFFORT=low
   ```

3. Use GPU acceleration if available:
   - Ollama: Set `OLLAMA_GPU_OVERHEAD` environment variable
   - LocalAI: Use GPU-enabled builds

4. Increase context window if needed:
   - For Ollama: `OLLAMA_MAX_LOADED_MODELS=1`
   - For LocalAI: Configure context window in model settings

## Advanced Configuration

### Custom Model Parameters

Some local LLM servers support additional parameters:

**Ollama with custom parameters:**
```bash
# Start Ollama with custom parameters
OLLAMA_FLASH_ATTN=1 ollama serve
```

**LocalAI with model configuration:**
```yaml
# .env file
LLM_ENDPOINT=http://localhost:8080/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3-8b

# Model-specific settings in LocalAI
# See: https://localai.io/models/
```

### Multiple Endpoint Support

You can configure different endpoints for different use cases:

```env
# Default endpoint
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-5.4

# Local endpoint for specific agents
RETRIEVER_ENDPOINT=http://localhost:11434/v1
RETRIEVER_API_KEY=n/a
RETRIEVER_MODEL=llama3.1:8b
```

### Programmatic Configuration Examples

```python
from orchestrator_agent import orchestrator_agent
from worker_agents.retriever_agent import retriever_agent
from worker_agents.writer_agent import writer_agent
from worker_agents.verifier_agent import verifier_agent

# Use local LLM for a single call
result = orchestrator_agent(
    user_query="What is this document about?",
    session_id="my-session",
    endpoint="http://localhost:11434/v1",
    api_key="n/a",
    verbose=True
)

# Use different endpoints for different agents
retriever_result = retriever_agent(
    user_query="Find information about X",
    endpoint="http://localhost:8080/v1",
    api_key="local-key"
)

writer_result = writer_agent(
    user_query="What is this document about?",
    evidence_text=retriever_result.summary,
    endpoint="https://api.openai.com/v1",
    api_key="sk-..."
)
```

## Performance Benchmarks

Expected performance on typical hardware:

| Configuration | Response Time | Memory Usage | Use Case |
|---------------|---------------|--------------|----------|
| OpenAI GPT-5.4 | 2-5 seconds | N/A | Fast, reliable |
| Ollama llama3.1:8b | 5-10 seconds | 4-8 GB | Balanced |
| Ollama llama3.1:4b | 3-6 seconds | 2-4 GB | Fast, low RAM |
| LocalAI llama3-8b | 5-10 seconds | 4-8 GB | Self-hosted |
| LM Studio llama3-8b | 5-10 seconds | 4-8 GB | GUI-based |

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your LLM server is running correctly
3. Ensure all environment variables are set correctly
4. Check the system logs for detailed error messages

## Related Documentation

- [README.md](README.md) - Main project documentation
- [utils/config.py](utils/config.py) - Configuration implementation
- [utils/model_runner.py](utils/model_runner.py) - Model execution logic
- [worker_agents/](worker_agents/) - Worker agent implementations