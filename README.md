# Thunk - Deep Research Agent

A sophisticated AI-powered research assistant that autonomously performs multi-step web research and synthesizes findings into comprehensive reports using Google's Vertex AI and Gemini models.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/wbeard/thunk)

## 🎯 Overview

Thunk replicates advanced research capabilities similar to Google's Gemini Deep Research, providing:

- **Autonomous Research Planning** - Breaks down complex queries into structured research steps
- **Intelligent Web Search** - Performs targeted searches with AI-powered quality filtering  
- **Content Analysis** - Fetches and processes web pages, PDFs, and documents
- **Vertex AI RAG Integration** - Stores and manages research findings using Google's RAG engine
- **Interactive CLI** - User-friendly command-line interface with clarification support
- **Report Synthesis** - Generates comprehensive reports with proper citations

## 🏗️ Architecture & Research Flow

```mermaid
graph TD
    A[User Query] --> B[Clarification Questions]
    B --> C[Research Planning]
    C --> D[Generate Search Queries]
    D --> E[Parallel Web Search]
    E --> F[Source Evaluation & Filtering]
    F --> G[Content Fetching]
    G --> H[Content Summarization]
    H --> I[Document Storage]
    I --> J{Research Complete?}
    J -->|No| K[Focused Research]
    K --> D
    J -->|Yes| L[Vertex AI RAG Synthesis]
    L --> M[Final Report]
    
    N[Existing Corpus] --> L
    
    subgraph "Storage Layer"
        O[Vertex AI RAG Corpus]
        P[Local Backup]
    end
    
    I --> O
    I --> P
    N --> O
```

The research flow follows these phases:

1. **Query Analysis** - Analyzes complexity and generates clarifying questions
2. **Research Planning** - Creates structured multi-step research plan
3. **Parallel Execution** - Concurrent search queries with rate limiting
4. **Quality Filtering** - AI-powered source relevance evaluation
5. **Content Processing** - Fetch, summarize, and store documents
6. **Completeness Assessment** - Determines if additional focused research is needed
7. **Synthesis** - Uses Vertex AI RAG for enhanced report generation

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **uv** - Fast Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
3. **Google Cloud Project** with Vertex AI API enabled
4. **API Keys** for SerpAPI and Google Cloud authentication

### Installation

```bash
# Clone the repository
git clone git@github.com:wbeard/thunk.git
cd thunk

# Install dependencies with uv
uv sync

# Set up Google Cloud authentication
gcloud auth application-default login
```

### Configuration

Create a `.env` file with your API keys:

```bash
# Required
SERPAPI_KEY=your_serpapi_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id

# Optional (with defaults)
VERTEX_AI_LOCATION=us-central1
RAG_CORPUS_NAME=research_corpus_6
RAG_MODEL_NAME=gemini-2.5-flash-preview-05-20
MODEL_NAME=gemini-2.5-flash-preview-05-20
```

### API Keys Setup

**SerpAPI Key:**
1. Sign up at [SerpAPI](https://serpapi.com)
2. Get your API key from the dashboard
3. Add to `.env` file as `SERPAPI_KEY`

**Google Cloud Setup:**
1. Create a Google Cloud Project
2. Enable Vertex AI API in the console
3. Set up authentication: `gcloud auth application-default login`
4. Set project ID in `.env` as `GOOGLE_CLOUD_PROJECT`

### Verify Configuration

```bash
# Check configuration
uv run thunk --check-config
```

## 🖥️ CLI Usage

The CLI provides multiple ways to interact with the research agent:

### Basic Research Query

```bash
# Simple research query
uv run thunk "Latest developments in quantum computing 2024"

# With specific corpus and output options
uv run thunk "AI safety research trends" --corpus my_research --output report.md --full

# Quiet mode (minimal output)
uv run thunk "research query" --quiet
```

### Interactive Mode

```bash
# Start interactive session
uv run thunk --interactive

# Available commands in interactive mode:
# <query>                     - Run research with clarification
# regenerate <query>          - Regenerate from existing corpus
# regenerate-no-rag <query>   - Regenerate without Vertex AI RAG
# corpus-info                 - Show corpus information
# quit                        - Exit interactive mode
```

### Regeneration Mode

```bash
# Regenerate report from existing corpus (uses Vertex AI RAG)
uv run thunk --regenerate "quantum computing trends"

# Regenerate without using Vertex AI RAG
uv run thunk --regenerate "AI developments" --no-rag

# Regenerate with specific corpus
uv run thunk --regenerate "research topic" --corpus specific_corpus
```

### Debug and Configuration

```bash
# Check configuration and API keys
uv run thunk --check-config

# Enable debug mode with verbose output
uv run thunk "research query" --debug

# Use specific RAG corpus
uv run thunk "query" --corpus my_research_corpus
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--interactive` | `-i` | Run in interactive mode |
| `--query` | `-q` | Research query (alternative to positional) |
| `--regenerate` | `-r` | Regenerate from existing corpus |
| `--output` | `-o` | Output file for report |
| `--corpus` | `-c` | Name of Vertex AI RAG corpus to use |
| `--no-save` | | Don't save report to file |
| `--full` | | Display full report (don't truncate) |
| `--quiet` | `-k` | Minimal output mode |
| `--debug` | `-d` | Enable debug mode with verbose output |
| `--no-rag` | | Don't use Vertex AI RAG for regeneration |
| `--check-config` | | Check configuration and exit |

### Example Workflows

```bash
# First-time setup
uv run thunk --check-config

# Interactive research session
uv run thunk --interactive

# Quick research with custom corpus
uv run thunk "AI trends 2024" --corpus ai_research --debug

# Regenerate previous research
uv run thunk --regenerate "previous query" --corpus ai_research
```

## ⚙️ Configuration Options

### Environment Variables

The following environment variables can be configured:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SERPAPI_KEY` | ✅ Yes | - | SerpAPI key for web search |
| `GOOGLE_CLOUD_PROJECT` | ✅ Yes | - | Google Cloud project ID |
| `VERTEX_AI_LOCATION` | No | `us-central1` | Vertex AI region |
| `RAG_CORPUS_NAME` | No | `research_corpus_6` | Name for RAG corpus |
| `RAG_MODEL_NAME` | No | `gemini-2.5-flash-preview-05-20` | Model for RAG synthesis |
| `MODEL_NAME` | No | `gemini-2.5-flash-preview-05-20` | Primary Gemini model |

### ResearchConfig Class

The `ResearchConfig` class automatically loads and validates configuration:

```python
from src.thunk.types import ResearchConfig

# Automatically loads from environment variables
config = ResearchConfig(corpus_display_name="my_corpus")

# Validates required configuration
try:
    config._validate_config()
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Troubleshooting Configuration

**Common configuration issues:**

1. **Missing API Keys**: Ensure `SERPAPI_KEY` and `GOOGLE_CLOUD_PROJECT` are set
2. **Authentication**: Run `gcloud auth application-default login`
3. **Corpus Name**: Provide via `--corpus` argument or `RAG_CORPUS_NAME` environment variable

```bash
# Debug configuration issues
uv run thunk --check-config

# Example error messages:
# "Missing required configuration: SERPAPI_KEY, GOOGLE_CLOUD_PROJECT"
# "Missing required configuration: --corpus argument or RAG_CORPUS_NAME environment variable"
```

## 🤝 Contributing

The system is designed to be extensible:
- Add new content fetchers for different file types
- Implement custom RAG engines beyond Vertex AI
- Create specialized event subscribers for different use cases
- Extend the CLI with additional features

## 📄 License

[Add your license information here]

---

*Thunk provides a powerful foundation for automated research tasks while maintaining flexibility for customization and extension.*