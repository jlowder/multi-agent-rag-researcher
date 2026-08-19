# Local LLM Embeddings Documentation

This guide provides comprehensive documentation for using local embedding models with the Multi-Agent RAG Researcher. Local embeddings allow you to run embedding generation entirely on your machine without relying on external API services like OpenAI.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [OpenAIEmbeddings Settings](#openaiembeddings-settings)
- [Switching Between Local and OpenAI Embeddings](#switching-between-local-and-openai-embeddings)
- [Common Embedding Models](#common-embedding-models)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Overview

The Multi-Agent RAG Researcher uses embeddings to convert text chunks into vector representations for similarity search in Qdrant. By default, the system supports:

- **Local embeddings**: Run embedding models on your own machine using OpenAI-compatible endpoints (Ollama, LocalAI, LM Studio, etc.)
- **OpenAI embeddings**: Use OpenAI's `text-embedding-3-small` or `text-embedding-3-large` models

### Benefits of Local Embeddings

- **Privacy**: Documents never leave your machine
- **Cost**: No per-request charges after model download
- **Speed**: Faster for local documents (no network latency)
- **Offline**: Works without internet connection
- **Control**: Choose your embedding model

---

## Configuration

Embedding configuration is managed through environment variables in `utils/var.env` or `.env.example`. This is separate from LLM model configuration. For configuring LLM models (LLM_ENDPOINT, LLM_MODEL, per-agent settings), see [model-configuration.md](./model-configuration.md).

### Environment Variables

Embedding configuration is managed through environment variables in `utils/var.env` or `.env.example`.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_ENDPOINT` | `https://api.openai.com/v1` | Base URL for the embedding API endpoint |
| `EMBEDDING_MODEL` | `nomicai-modernbert-embed-base-bf16` | Name of the embedding model to use |
| `EMBEDDING_API_KEY` | Empty string | API key for authentication (dummy for local LLMs) |

**Note**: Embedding configuration is separate from LLM model configuration. For configuring LLM endpoints, models, and per-agent settings, see [model-configuration.md](./model-configuration.md).

### Basic Configuration Files

#### `utils/var.env` (Active Configuration)

This file contains your current embedding configuration:

```env
# ----------------------------------------
# Embedding Configuration (Local LLM)
# ----------------------------------------
# Embedding endpoint (same server as LLM_ENDPOINT but different path)
# OpenAIEmbeddings appends /embeddings automatically, so just provide base URL
# For Ollama with nomic-embed: http://localhost:11434/v1
EMBEDDING_ENDPOINT=http://localhost:8080/v1

# Embedding model name (nomicai-modernbert-embed-base-bf16)
EMBEDDING_MODEL=nomicai-modernbert-embed-base-bf16

# API Key for the embedding endpoint (same as LLM_API_KEY if using same server)
# For local LLMs like Ollama, this can be empty or a dummy value
EMBEDDING_API_KEY=your_embedding_api_key_here
```

#### `.env.example` (Template)

This file serves as a template for new deployments:

```env
# ----------------------------------------
# Embedding Configuration (Local LLM)
# ----------------------------------------
# Embedding endpoint (same server as LLM_ENDPOINT but different path)
# OpenAIEmbeddings appends /embeddings automatically, so just provide base URL
# For Ollama with nomic-embed: http://localhost:11434/v1
EMBEDDING_ENDPOINT=https://api.openai.com/v1

# Embedding model name (nomicai-modernbert-embed-base-bf16 for local LLMs)
EMBEDDING_MODEL=text-embedding-3-small

# API Key for the embedding endpoint (empty for local LLMs like Ollama)
EMBEDDING_API_KEY=your_embedding_api_key_here
```

### Complete Configuration Reference

For comprehensive LLM configuration (including per-agent settings, reasoning effort, and endpoint management), see [model-configuration.md](./model-configuration.md).

### Complete Configuration Reference

For comprehensive LLM configuration (including per-agent settings, reasoning effort, and endpoint management), see [model-configuration.md](./model-configuration.md).

### Configuration Examples

#### Example 1: Ollama with nomic-embed

```bash
# Start Ollama
ollama serve

# In another terminal, pull a model
ollama pull nomic-embed-text

# Configure var.env
EMBEDDING_ENDPOINT=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_API_KEY=
```

#### Example 2: LocalAI with all-MiniLM-L6-v2

```bash
# Start LocalAI with a model that supports embeddings
./localai-ollama.sh --models-path ./models

# Configure var.env
EMBEDDING_ENDPOINT=http://localhost:8080/v1
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_API_KEY=dummy
```

#### Example 3: LM Studio with built-in embeddings

```bash
# Start LM Studio and load a model with embedding support
# Configure var.env
EMBEDDING_ENDPOINT=http://localhost:1234/v1
EMBEDDING_MODEL=embedding-model-name
EMBEDDING_API_KEY=lm-studio
```

#### Example 4: OpenAI (Remote)

```bash
# Configure var.env
EMBEDDING_ENDPOINT=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-your-openai-api-key-here
```

---

## OpenAIEmbeddings Settings

The system uses LangChain's `OpenAIEmbeddings` class with specific settings for LocalAI compatibility. Here's the configuration from `qdrant_vector_database/vector_store.py`:

```python
embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=EMBEDDING_API_KEY if EMBEDDING_API_KEY and EMBEDDING_API_KEY != "your_embedding_api_key_here" else "dummy",
    openai_api_base=EMBEDDING_ENDPOINT + "/" if not EMBEDDING_ENDPOINT.endswith("/") else EMBEDDING_ENDPOINT,
    check_embedding_ctx_length=False,
)
```

### Key Settings Explained

#### `check_embedding_ctx_length=False` (Critical)

**Why this is important:**

LocalAI-compatible servers (Ollama, LocalAI, LM Studio) often don't implement the same tokenization endpoints as OpenAI's API. When `check_embedding_ctx_length=True` (the default), LangChain attempts to call the model's tokenization endpoint to determine the maximum context length. This call fails with local LLMs, causing errors like:

```
AttributeError: 'NoneType' object has no attribute 'tokenize'
```

Setting `check_embedding_ctx_length=False` skips this check and allows the embeddings to work with any LocalAI-compatible server.

**When to keep it False:**
- Using Ollama, LocalAI, LM Studio, or any OpenAI-compatible server
- Using custom embedding models

**When you might want to set it True:**
- Using OpenAI's embedding API (though even then, it's often unnecessary)

#### `openai_api_base`

This parameter specifies the base URL for the embedding API. The `OpenAIEmbeddings` class automatically appends `/embeddings` to this URL when making API calls.

Example:
- If `EMBEDDING_ENDPOINT=http://localhost:11434/v1`
- The actual API call goes to `http://localhost:11434/v1/embeddings`

#### `openai_api_key`

For local LLMs, this can be:
- An empty string `""`
- A dummy value like `"dummy"`
- Or the API key if your local server requires authentication

For OpenAI embeddings, this should be your actual OpenAI API key.

---

## Switching Between Local and OpenAI Embeddings

### Quick Switch Guide

#### To Use Local Embeddings

1. **Start your local LLM server** (Ollama, LocalAI, etc.)

   ```bash
   # Ollama
   ollama serve
   
   # LocalAI
   ./localai
   ```

2. **Pull or load an embedding model**

   ```bash
   # Ollama
   ollama pull nomic-embed-text
   
   # LocalAI - load model via API or config
   ```

3. **Update `utils/var.env`**

   ```env
   EMBEDDING_ENDPOINT=http://localhost:11434/v1  # or your local server URL
   EMBEDDING_MODEL=nomic-embed-text
   EMBEDDING_API_KEY=
   ```

4. **Re-ingest documents**

   ```bash
   # Clear existing embeddings (optional)
   rm -rf utils/qdrant_storage/
   
   # Re-run ingestion
   python3 run_orchestrator.py
   ```

#### To Use OpenAI Embeddings

1. **Update `utils/var.env`**

   ```env
   EMBEDDING_ENDPOINT=https://api.openai.com/v1
   EMBEDDING_MODEL=text-embedding-3-small
   EMBEDDING_API_KEY=sk-your-openai-api-key-here
   ```

2. **Re-ingest documents**

   ```bash
   # Clear existing embeddings (optional)
   rm -rf utils/qdrant_storage/
   
   # Re-run ingestion
   python3 run_orchestrator.py
   ```

### Verifying Your Configuration

After configuration, check the logs when running the orchestrator:

```bash
python3 run_orchestrator.py
```

You should see log messages like:

```
INFO:root:Using embedding model: nomic-embed-text
INFO:root:Embedding endpoint: http://localhost:11434/v1
```

### Switching Without Re-ingesting

**Important**: Embeddings from different models are not interoperable. If you switch embedding models:

1. You must re-ingest all documents
2. The existing Qdrant collection will need to be rebuilt
3. Vector dimensions differ between models (e.g., 768 for ModernBert vs 1536 for OpenAI)

---

## Common Embedding Models

### Local Embedding Models

#### Ollama Models

| Model | Vector Size | Command | Notes |
|-------|-------------|---------|-------|
| `nomic-embed-text` | 768 | `ollama pull nomic-embed-text` | Small, fast, good for most use cases |
| `mxbai-embed-large` | 1024 | `ollama pull mxbai-embed-large` | Larger, higher quality |

#### LocalAI Models

| Model | Vector Size | Installation | Notes |
|-------|-------------|--------------|-------|
| `all-MiniLM-L6-v2` | 384 | Load via LocalAI UI | Lightweight, very fast |
| `all-MiniLM-L12-v2` | 384 | Load via LocalAI UI | Better quality than L6 |
| `all-MiniLM-L24-v2` | 384 | Load via LocalAI UI | Best quality MiniLM |
| `modernbert-embed` | 768 | Load via LocalAI UI | Modern, high quality |

### OpenAI Embeddings

| Model | Vector Size | Cost | Notes |
|-------|-------------|------|-------|
| `text-embedding-3-small` | 1536 | $0.02/1M tokens | Default choice, good quality |
| `text-embedding-3-large` | 3072 | $0.13/1M tokens | Highest quality, expensive |

### Embedding Model Comparison

| Model | Size | Speed | Quality | Privacy | Cost |
|-------|------|-------|---------|---------|------|
| `nomic-embed-text` | 768 | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ Local | Free |
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡⚡ | ⭐⭐⭐ | ✅ Local | Free |
| `modernbert-embed` | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ Local | Free |
| `text-embedding-3-small` | 1536 | ⚡ | ⭐⭐⭐⭐⭐ | ❌ Remote | $0.02/1M tokens |

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Connection Refused

**Error:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:**
- Ensure your local LLM server is running (Ollama, LocalAI, etc.)
- Verify the `EMBEDDING_ENDPOINT` URL is correct
- Check that the server is listening on the expected port

```bash
# Test Ollama
curl http://localhost:11434/v1/models

# Test LocalAI
curl http://localhost:8080/v1/models
```

#### Issue 2: Model Not Found

**Error:**
```
ValueError: Error 404: Model not found
```

**Solution:**
- Ensure the embedding model is pulled/loaded
- Verify the model name matches exactly (case-sensitive)

```bash
# For Ollama
ollama list  # Check available models
ollama pull nomic-embed-text  # Pull if missing
```

#### Issue 3: Tokenization Error

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'tokenize'
```

**Solution:**
This is caused by `check_embedding_ctx_length=True` trying to call a non-existent tokenization endpoint. Ensure your configuration has:

```python
# In qdrant_vector_database/vector_store.py (line 207)
check_embedding_ctx_length=False,
```

If you've modified this, revert to `False`.

#### Issue 4: Wrong Vector Size

**Error:**
```
QdrantException: Wrong vector: wrong size: got 768, expected 1536
```

**Solution:**
This happens when you switch embedding models without re-ingesting documents. The Qdrant collection has vectors of one size, but the new model produces different-sized vectors.

**Fix:**
```bash
# Clear the Qdrant storage
rm -rf utils/qdrant_storage/

# Re-run ingestion
python3 run_orchestrator.py
```

#### Issue 5: API Key Required

**Error:**
```
AuthenticationError: Incorrect API key provided
```

**Solution:**
For local LLMs, set an empty or dummy API key:

```env
EMBEDDING_API_KEY=
# or
EMBEDDING_API_KEY=dummy
```

For OpenAI, ensure you have a valid API key:

```env
EMBEDDING_API_KEY=sk-your-valid-openai-key-here
```

#### Issue 6: Slow Embedding Generation

**Symptoms:**
- Embedding generation takes a very long time
- High CPU usage

**Solutions:**
- Use a smaller model (`nomic-embed-text` instead of `mxbai-embed-large`)
- Use GPU acceleration if available
- For Ollama, ensure CUDA is enabled:
  ```bash
  CUDA_VISIBLE_DEVICES=0 ollama serve
  ```

#### Issue 7: Inconsistent Results

**Symptoms:**
- Similar queries return different results across runs
- Vector search quality varies

**Solutions:**
- Ensure you're using the same embedding model for ingestion and query
- Check that `EMBEDDING_ENDPOINT` and `EMBEDDING_MODEL` are consistent
- Re-ingest documents to ensure consistency

### Debug Checklist

If you're having issues with embeddings, go through this checklist:

- [ ] Local LLM server is running
- [ ] Embedding model is pulled/loaded
- [ ] `EMBEDDING_ENDPOINT` URL is correct and accessible
- [ ] `EMBEDDING_MODEL` name matches exactly
- [ ] `EMBEDDING_API_KEY` is set correctly (empty/dummy for local, valid for OpenAI)
- [ ] `check_embedding_ctx_length=False` in `vector_store.py`
- [ ] Qdrant storage was cleared if switching models
- [ ] No network/firewall issues blocking the connection

### Getting Help

If you're still having issues:

1. Check the logs for detailed error messages
2. Test your embedding endpoint directly:
   ```bash
   curl -X POST http://localhost:11434/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{
       "model": "nomic-embed-text",
       "input": ["test"]
     }'
   ```
3. Verify your configuration matches one of the working examples in this document
4. Check that all environment variables are loaded (restart your terminal if needed)

---

## Advanced Configuration

### Custom Embedding Code

If you need more control over the embedding process, you can modify the `EmbeddingConfig` in `qdrant_vector_database/vector_store.py`:

```python
# Embedding model configuration from environment variables with type hints
EmbeddingConfig = dict[str, str]
embedding_config: EmbeddingConfig = {
    "endpoint": os.getenv("EMBEDDING_ENDPOINT", ""),
    "model": os.getenv("EMBEDDING_MODEL", "nomicai-modernbert-embed-base-bf16"),
    "api_key": os.getenv("EMBEDDING_API_KEY", ""),
}

EMBEDDING_MODEL_NAME = embedding_config["model"]
EMBEDDING_ENDPOINT = embedding_config["endpoint"].rstrip("/")
EMBEDDING_API_KEY = embedding_config["api_key"]

# Determine vector size based on model name
if "modernbert" in EMBEDDING_MODEL_NAME.lower():
    EMBEDDING_VECTOR_SIZE = 768
elif "minilm" in EMBEDDING_MODEL_NAME.lower():
    EMBEDDING_VECTOR_SIZE = 384
elif "nomic" in EMBEDDING_MODEL_NAME.lower():
    EMBEDDING_VECTOR_SIZE = 768
else:
    EMBEDDING_VECTOR_SIZE = 1536  # OpenAI default
```

### Custom Embedding Class

For advanced use cases, you can implement a custom embedding class:

```python
from langchain_core.embeddings import Embeddings
import requests

class CustomEmbeddings(Embeddings):
    def __init__(self, endpoint: str, model: str, api_key: str = ""):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Your custom embedding logic here
        pass
    
    def embed_query(self, text: str) -> list[float]:
        # Your custom query embedding logic here
        pass
```

---

## See Also

- [model-configuration.md](./model-configuration.md) - LLM model configuration (global and per-agent settings)
- [LOCAL_LLM_SUPPORT.md](../LOCAL_LLM_SUPPORT.md) - General local LLM documentation
- [LOCAL_LLM_SETUP.md](../LOCAL_LLM_SETUP.md) - Local LLM setup instructions
- [README.md](../README.md) - Project overview

---

## Summary

Local LLM embeddings provide a privacy-preserving, cost-effective alternative to cloud-based embedding services. By configuring the right combination of endpoint, model, and API key, you can run the entire RAG pipeline entirely on your machine.

### Key Takeaways

1. **Configuration**: Use `EMBEDDING_ENDPOINT`, `EMBEDDING_MODEL`, and `EMBEDDING_API_KEY` in `utils/var.env`
2. **Critical Setting**: Always set `check_embedding_ctx_length=False` for LocalAI compatibility
3. **Model Choice**: `nomic-embed-text` is a good starting point for Ollama
4. **Switching**: Re-ingest documents when changing embedding models
5. **Troubleshooting**: Use the checklist to diagnose common issues

For additional help, refer to:
- [LOCAL_LLM_SUPPORT.md](../LOCAL_LLM_SUPPORT.md) - General local LLM configuration
- [LOCAL_LLM_SETUP.md](../LOCAL_LLM_SETUP.md) - Local LLM setup instructions
- [README.md](../README.md) - Project overview