# Model Configuration for Local LLMs

This guide explains how to configure LLM models for the Multi-Agent RAG Researcher, with a focus on using local LLMs with OpenAI-compatible endpoints.

## Table of Contents

- [Overview](#overview)
- [How the Configuration System Works](#how-the-configuration-system-works)
- [Setting Up Local Models](#setting-up-local-models)
- [Global Defaults vs. Per-Agent Configuration](#global-defaults-vs-per-agent-configuration)
- [Switching Between Local and OpenAI Models](#switching-between-local-and-openai-models)
- [Important Considerations](#important-considerations)
- [Examples](#examples)

---

## Overview

The Multi-Agent RAG Researcher uses four specialized agents:

- **Orchestrator**: Coordinates all other agents and manages the overall research workflow
- **Retriever**: Searches documents and the web for relevant information
- **Writer**: Creates reports based on retrieved evidence
- **Verifier**: Fact-checks and validates the generated content

Each agent can be configured to use different LLM endpoints and models, enabling flexible deployment strategies.

### Configuration Sources

Configuration is managed through:

1. **Environment variables** in `utils/var.env` (primary method)
2. **Programmatic parameters** passed to agent functions (override env vars)
3. **Default values** from the configuration system

---

## How the Configuration System Works

### Environment Variables

The configuration system uses the following environment variables:

| Variable | Scope | Default | Description |
|----------|-------|---------|-------------|
| `LLM_ENDPOINT` | Global | `https://api.openai.com/v1` | Base URL for the LLM API |
| `LLM_API_KEY` | Global | Empty | API key for authentication |
| `LLM_MODEL` | Global | `gpt-5.4` | Default model name |
| `LLM_REASONING_EFFORT` | Global | `low` | Reasoning effort level |

### Per-Agent Configuration

Each agent can override the global defaults:

| Agent | Endpoint Var | API Key Var | Model Var | Effort Var |
|-------|--------------|-------------|-----------|------------|
| Retriever | `RETRIEVER_ENDPOINT` | `RETRIEVER_API_KEY` | `RETRIEVER_MODEL` | `RETRIEVER_REASONING_EFFORT` |
| Writer | `WRITER_ENDPOINT` | `WRITER_API_KEY` | `WRITER_MODEL` | `WRITER_REASONING_EFFORT` |
| Verifier | `VERIFIER_ENDPOINT` | `VERIFIER_API_KEY` | `VERIFIER_MODEL` | `VERIFIER_REASONING_EFFORT` |
| Orchestrator | `ORCHESTRATOR_ENDPOINT` | `ORCHESTRATOR_API_KEY` | `ORCHESTRATOR_MODEL` | `ORCHESTRATOR_REASONING_EFFORT` |

### Configuration Priority

The system uses this priority order (highest to lowest):

1. **Programmatic parameters** (passed directly to agent functions)
2. **Per-agent environment variables** (e.g., `RETRIEVER_MODEL`)
3. **Global environment variables** (e.g., `LLM_MODEL`)
4. **Default values** (hardcoded fallbacks)

### Configuration Flow

```
Agent Call → run_model() → get_config() → Environment Variables
                                   ↓
                            get_agent_config()
                                   ↓
                            LLMConfig(endpoint, api_key, model)
                                   ↓
                            get_client_for_endpoint()
                                   ↓
                            OpenAI Client (cached)
```

### Code Implementation

The configuration is implemented in `utils/config.py`:

```python
@dataclass
class LLMConfig:
    """Configuration for a single LLM endpoint."""
    endpoint: str
    api_key: str
    model: str
    name: str = "default"

@dataclass
class Config:
    """Main configuration object."""
    # Global defaults
    default_endpoint: str
    default_api_key: str
    default_model: str
    
    # Per-agent overrides
    retriever_endpoint: Optional[str] = None
    retriever_model: Optional[str] = None
    # ... (similar for other agents)
    
    def get_agent_config(self, agent_name: str) -> LLMConfig:
        """Get configuration for a specific agent."""
        endpoint = getattr(self, f"{agent_name}_endpoint") or self.default_endpoint
        api_key = getattr(self, f"{agent_name}_api_key") or self.default_api_key
        model = getattr(self, f"{agent_name}_model") or self.default_model
        return LLMConfig(endpoint=endpoint, api_key=api_key, model=model)
```

---

## Setting Up Local Models

### Prerequisites

1. **Local LLM Server**: Install and start one of:
   - [Ollama](https://ollama.ai/download)
   - [LocalAI](https://localai.io/basics/getting_started/)
   - [LM Studio](https://lmstudio.ai/)

2. **Embedding Model**: Configure local embeddings (see [local-embeddings.md](./local-embeddings.md))

### Step-by-Step Setup

#### Step 1: Start Your Local LLM Server

**Ollama:**
```bash
ollama serve
ollama pull llama3.1:8b  # Pull a model if needed
```

**LocalAI:**
```bash
# Download LocalAI
./localai-ollama.sh
```

**LM Studio:**
```bash
# Launch LM Studio from GUI
# Load a model through the interface
```

#### Step 2: Identify Available Models

**Ollama:**
```bash
ollama list
```

**LocalAI:**
```bash
curl http://localhost:8080/v1/models
```

**LM Studio:**
```bash
curl http://localhost:1234/v1/models
```

#### Step 3: Update Configuration

Edit `utils/var.env`:

```env
# Global settings for local LLM
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a  # or dummy, or actual key if required
LLM_MODEL=llama3.1:8b

# Optional: Override per-agent
WRITER_MODEL=llama3.1:8b
RETRIEVER_MODEL=gemma2:9b
```

#### Step 4: Test Configuration

```bash
# Test that the configuration loads correctly
python3 -c "from utils.config import get_config; c = get_config(); print(f'Model: {c.default_model}, Endpoint: {c.default_endpoint}')"

# Run the orchestrator
python3 run_orchestrator.py
```

---

## Global Defaults vs. Per-Agent Configuration

### Global Defaults (Simple Approach)

Use global defaults when:
- All agents use the same model
- You want simpler configuration
- You're just getting started

**Example `utils/var.env`:**
```env
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b
```

All four agents (orchestrator, retriever, writer, verifier) will use `llama3.1:8b`.

### Per-Agent Configuration (Advanced Approach)

Use per-agent configuration when:
- Different agents use different models
- You want to optimize cost and performance
- You need specific models for specific tasks

**Example `utils/var.env`:**
```env
# Global default (used by orchestrator)
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b

# Retriever uses a smaller, faster model
RETRIEVER_MODEL=gemma2:9b

# Writer uses a more capable model
WRITER_MODEL=llama3.1:8b

# Verifier uses a different model for fact-checking
VERIFIER_MODEL=mistral:7b

# Orchestrator uses the default
ORCHESTRATOR_MODEL=llama3.1:8b
```

### When to Use Each Approach

#### Use Global Defaults For:
- **Initial setup and testing**: Get everything working quickly
- **Resource-constrained environments**: Use one smaller model
- **Consistency requirements**: Same model across all agents

#### Use Per-Agent Configuration For:
- **Cost optimization**: Use cheaper models for simpler tasks
- **Performance tuning**: Use specialized models for specific tasks
- **Model capabilities**: Leverage model strengths (e.g., reasoning vs. coding)

### Cost and Performance Considerations

| Agent | Recommended Model Size | Reason |
|-------|----------------------|--------|
| Retriever | Small (1-4B params) | Fast, low resource usage |
| Writer | Medium (7-13B params) | Needs good writing capability |
| Verifier | Medium (7-13B params) | Needs good reasoning |
| Orchestrator | Medium (7-13B params) | Coordinates complex workflow |

### Example: Mixed Configuration

```env
# Global (Orchestrator)
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b

# Retriever: Small, fast model
RETRIEVER_MODEL=gemma2:2b

# Writer: Capable writing model
WRITER_MODEL=llama3.1:8b

# Verifier: Different model for cross-checking
VERIFIER_MODEL=mistral:7b

# Reasoning effort (controls how much computation each agent uses)
RETRIEVER_REASONING_EFFORT=low
WRITER_REASONING_EFFORT=medium
VERIFIER_REASONING_EFFORT=high
ORCHESTRATOR_REASONING_EFFORT=medium
```

---

## Switching Between Local and OpenAI Models

### From OpenAI to Local LLMs

1. **Identify your current OpenAI configuration**:
   ```env
   LLM_ENDPOINT=https://api.openai.com/v1
   LLM_API_KEY=sk-...
   LLM_MODEL=gpt-5.4
   ```

2. **Set up local LLM server** (see [Setting Up Local Models](#setting-up-local-models))

3. **Update configuration**:
   ```env
   # Change endpoint
   LLM_ENDPOINT=http://localhost:11434/v1
   
   # Change API key (often empty or dummy for local)
   LLM_API_KEY=n/a
   
   # Change model name
   LLM_MODEL=llama3.1:8b
   ```

4. **Test the change**:
   ```bash
   python3 run_orchestrator.py
   ```

### From Local LLMs to OpenAI

1. **Update configuration**:
   ```env
   LLM_ENDPOINT=https://api.openai.com/v1
   LLM_API_KEY=sk-your-openai-key-here
   LLM_MODEL=gpt-5.4
   ```

2. **Ensure API key is valid**:
   ```bash
   # Test OpenAI connectivity
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer sk-..."
   ```

### Migrating Between Different Local Models

When switching between local models:

1. **Check embedding compatibility** (see [local-embeddings.md](./local-embeddings.md))
2. **Clear Qdrant storage** if vector sizes differ:
   ```bash
   rm -rf utils/qdrant_storage/
   ```
3. **Re-ingest documents** if needed

---

## Important Considerations

### 1. Vector Size Compatibility

When switching embedding models, ensure the Qdrant vector size matches:

| Embedding Model | Vector Size |
|----------------|-------------|
| `text-embedding-3-small` (OpenAI) | 1536 |
| `nomic-embed-text` (Ollama) | 768 |
| `all-MiniLM-L6-v2` (LocalAI) | 384 |
| `modernbert-embed` | 768 |

If vector sizes don't match, clear Qdrant storage and re-ingest documents.

### 2. API Key Requirements

Some local LLM servers require API keys:

| Server | API Key Requirement | Example |
|--------|-------------------|---------|
| Ollama | Not required | `LLM_API_KEY=n/a` |
| LocalAI | Optional | `LLM_API_KEY=dummy` or empty |
| LM Studio | Required | `LLM_API_KEY=lm-studio` |
| Self-hosted | Configurable | Your choice |

### 3. Model Naming Conventions

Model names vary between servers:

**Ollama**: `model:tag` (e.g., `llama3.1:8b`, `mistral:7b`)
**LocalAI**: Model filename without extension (e.g., `llama-3-8b`)
**LM Studio**: Exact model name as shown in UI

### 4. Reasoning Effort

Controls computational resources:

- **low**: Fast, less accurate, minimal resources
- **medium**: Balanced speed and quality
- **high**: Slow, most accurate, high resources

```env
WRITER_REASONING_EFFORT=medium
VERIFIER_REASONING_EFFORT=high
```

### 5. Resource Requirements

Estimate RAM requirements for local models:

| Model Size | RAM Required | GPU Required |
|------------|-------------|--------------|
| 1-4B params | 2-4 GB | CPU only |
| 7-13B params | 8-16 GB | CPU or GPU |
| 34B+ params | 32+ GB | GPU recommended |

### 6. Performance Optimization

**For slow local LLMs:**
```env
# Use smaller models
RETRIEVER_MODEL=gemma2:2b
WRITER_MODEL=llama3.1:8b

# Reduce reasoning effort
RETRIEVER_REASONING_EFFORT=low
WRITER_REASONING_EFFORT=low

# Use batching for embeddings
# (handled automatically in qdrant_vector_database)
```

**For better quality:**
```env
# Use larger models
WRITER_MODEL=llama3.1:70b

# Increase reasoning effort
WRITER_REASONING_EFFORT=high
VERIFIER_REASONING_EFFORT=high
```

---

## Examples

### Example 1: Single Local Model (Simple)

Use one local model for all agents.

**`utils/var.env`:**
```env
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b
```

### Example 2: Hybrid Local (Ollama + OpenAI)

Use local models for most agents, OpenAI for writer.

**`utils/var.env`:**
```env
# Global (Orchestrator, Retriever, Verifier)
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b

# Writer uses OpenAI for better quality
WRITER_ENDPOINT=https://api.openai.com/v1
WRITER_API_KEY=sk-your-key-here
WRITER_MODEL=gpt-5.4
```

### Example 3: Multi-Model Local

Different local models for different agents.

**`utils/var.env`:**
```env
# Global
LLM_ENDPOINT=http://localhost:11434/v1
LLM_API_KEY=n/a
LLM_MODEL=llama3.1:8b

# Different models per agent
RETRIEVER_MODEL=gemma2:2b
WRITER_MODEL=llama3.1:8b
VERIFIER_MODEL=mistral:7b
ORCHESTRATOR_MODEL=llama3.1:8b

# Reasoning effort
RETRIEVER_REASONING_EFFORT=low
WRITER_REASONING_EFFORT=medium
VERIFIER_REASONING_EFFORT=high
ORCHESTRATOR_REASONING_EFFORT=medium
```

### Example 4: LocalAI Setup

Using LocalAI with embedding support.

**Start LocalAI:**
```bash
./localai-ollama.sh --models-path ./models
```

**`utils/var.env`:**
```env
LLM_ENDPOINT=http://localhost:8080/v1
LLM_API_KEY=dummy
LLM_MODEL=llama-3-8b

EMBEDDING_ENDPOINT=http://localhost:8080/v1
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_API_KEY=dummy
```

### Example 5: Full Hybrid (All Different)

Each agent uses a different provider.

**`utils/var.env`:**
```env
# Orchestrator: OpenAI
ORCHESTRATOR_ENDPOINT=https://api.openai.com/v1
ORCHESTRATOR_API_KEY=sk-key1
ORCHESTRATOR_MODEL=gpt-5.4

# Retriever: Local (fast)
RETRIEVER_ENDPOINT=http://localhost:11434/v1
RETRIEVER_API_KEY=n/a
RETRIEVER_MODEL=gemma2:2b

# Writer: OpenAI (quality)
WRITER_ENDPOINT=https://api.openai.com/v1
WRITER_API_KEY=sk-key2
WRITER_MODEL=gpt-5.4

# Verifier: Local (different model)
VERIFIER_ENDPOINT=http://localhost:11434/v1
VERIFIER_API_KEY=n/a
VERIFIER_MODEL=mistral:7b
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Model not found"

**Cause**: Model name doesn't match server's model list.

**Solution**:
```bash
# Check available models
curl http://localhost:11434/v1/models  # Ollama
curl http://localhost:8080/v1/models   # LocalAI
```

#### Issue 2: "Connection refused"

**Cause**: LLM server not running or wrong port.

**Solution**:
```bash
# Start server
ollama serve

# Check endpoint URL
echo $LLM_ENDPOINT
```

#### Issue 3: "Authentication failed"

**Cause**: Missing or incorrect API key.

**Solution**:
```env
# For Ollama (no auth required)
LLM_API_KEY=n/a

# For other servers
LLM_API_KEY=your-key
```

#### Issue 4: "Vector size mismatch"

**Cause**: Embedding model changed without re-ingesting.

**Solution**:
```bash
rm -rf utils/qdrant_storage/
# Re-run ingestion
```

### Debugging Configuration

```python
# Test your configuration
python3 -c "
from utils.config import get_config
c = get_config()

print('=== Configuration ===')
print(f'Default Endpoint: {c.default_endpoint}')
print(f'Default Model: {c.default_model}')
print()

agents = ['retriever', 'writer', 'verifier', 'orchestrator']
for agent in agents:
    cfg = c.get_agent_config(agent)
    print(f'{agent.capitalize()} Agent:')
    print(f'  Endpoint: {cfg.endpoint}')
    print(f'  Model: {cfg.model}')
    print()
"
```

---

## See Also

- [local-embeddings.md](./local-embeddings.md) - Embedding model configuration
- [LOCAL_LLM_SUPPORT.md](../LOCAL_LLM_SUPPORT.md) - General local LLM documentation
- [utils/config.py](../utils/config.py) - Configuration source code