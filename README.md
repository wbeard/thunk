# Deep Research Agent

A sophisticated AI-powered research assistant that autonomously performs multi-step web research and synthesizes findings into comprehensive reports using Google's Vertex AI and Gemini models.

## 🎯 Overview

The Deep Research Agent replicates advanced research capabilities similar to Google's Gemini Deep Research, providing:

- **Autonomous Research Planning** - Breaks down complex queries into structured research steps
- **Intelligent Web Search** - Performs targeted searches with AI-powered quality filtering  
- **Content Analysis** - Fetches and processes web pages, PDFs, and documents
- **Vertex AI RAG Integration** - Stores and manages research findings using Google's RAG engine
- **Interactive CLI** - User-friendly command-line interface with clarification support
- **Report Synthesis** - Generates comprehensive reports with proper citations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Gemini LLM     │───▶│ Research Plan   │
└─────────────────┘    │ (Planner)       │    └─────────────────┘
                       └─────────────────┘              │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Final Report   │◀───│   Synthesizer   │◀───│ Search Queries  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        ▲                      │
         │                        │                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Vertex AI RAG   │    │   Summarizer    │    │ Web Search API  │
│   Engine        │    │   & Analyzer    │    │ (SerpAPI)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **uv** - Fast Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
3. **Google Cloud Project** with Vertex AI enabled
4. **API Keys** for SerpAPI and Google Cloud

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-research-agent

# Install dependencies with uv
uv sync

# Alternatively, if you don't have a pyproject.toml:
uv pip install -r requirements.txt

# Set up Google Cloud authentication
gcloud auth application-default login
```

### Configuration

Create a `.env` file with your API keys:

```bash
# Required
SERPAPI_KEY=your_serpapi_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id

# Optional
VERTEX_AI_LOCATION=us-central1
RAG_CORPUS_NAME=research_corpus_6
RAG_MODEL_NAME=gemini-2.5-flash-preview-05-20
MODEL_NAME=gemini-2.5-flash-preview-05-20
```

### Get API Keys

**SerpAPI Key:**
1. Sign up at [SerpAPI](https://serpapi.com)
2. Get your API key from the dashboard

**Google Cloud Setup:**
1. Create a Google Cloud Project
2. Enable Vertex AI API
3. Set up authentication: `gcloud auth application-default login`

## 🖥️ CLI Usage

The CLI provides multiple ways to interact with the research agent:

### Basic Research Query

```bash
# Simple research query
uv run python cli.py "Latest developments in quantum computing 2024"

# With output options
uv run python cli.py "AI safety research trends" --full --output report.md

# Or if you have activated the environment:
python cli.py "Latest developments in quantum computing 2024"
```

### Interactive Mode

```bash
# Start interactive session
uv run python cli.py --interactive

# Available commands in interactive mode:
# <query>                    - Run research with clarification
# skip-clarification <query> - Run research without clarification  
# regenerate <query>         - Regenerate from existing corpus
# regenerate-no-rag <query>  - Regenerate without Vertex AI RAG
# corpus-info               - Show corpus information
# quit                      - Exit
```

### Regeneration Mode

```bash
# Regenerate report from existing corpus
uv run python cli.py --regenerate "quantum computing trends"

# Regenerate without using Vertex AI RAG
uv run python cli.py --regenerate "AI developments" --no-rag
```

### Debug and Configuration

```bash
# Check configuration
uv run python cli.py --check-config

# Enable debug mode
uv run python cli.py "research query" --debug

# Quiet mode (minimal output)
uv run python cli.py "research query" --quiet
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--interactive`, `-i` | Run in interactive mode |
| `--query`, `-q` | Research query (alternative to positional) |
| `--regenerate`, `-r` | Regenerate from existing corpus |
| `--output`, `-o` | Output file for report |
| `--no-save` | Don't save report to file |
| `--full` | Display full report (don't truncate) |
| `--quiet`, `-k` | Minimal output mode |
| `--debug`, `-d` | Enable debug mode |
| `--no-rag` | Don't use Vertex AI RAG for regeneration |
| `--check-config` | Check configuration and exit |

## 🔬 DeepResearchAgent API

### Basic Usage

```python
import asyncio
from deep_research_agent import DeepResearchAgent

async def main():
    agent = DeepResearchAgent(
        serpapi_key="your_serpapi_key",
        project_id="your_gcp_project",
        location="us-central1",
        corpus_display_name="research_corpus"
    )
    
    # Conduct research
    report = await agent.research("Latest AI breakthroughs 2024")
    print(report)

asyncio.run(main())
```

### Advanced Features

#### Research with Context

```python
# Research with clarification context
context = """
Focus on commercial applications rather than academic research.
Time frame: Last 6 months.
Geographic focus: North America and Europe.
"""

report = await agent.research_with_context(
    "AI in healthcare developments", 
    context=context
)
```

#### Regenerate from Existing Corpus

```python
# Regenerate summary from stored documents
summary = await agent.regenerate_summary(
    "quantum computing trends",
    use_rag=True  # Use Vertex AI RAG for enhanced synthesis
)
```

#### Corpus Management

```python
# Get corpus statistics
summary = agent.get_corpus_summary()
print(f"Vertex AI documents: {summary['vertex_ai_rag']['file_count']}")
print(f"Local documents: {summary['local_backup']['local_documents']}")

# Performance statistics
stats = agent.get_performance_stats()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average docs per query: {stats['avg_docs_per_query']}")

# Export local corpus
agent.export_corpus("research_backup.json")
```

### Callback System

The agent supports a comprehensive callback system for monitoring research progress:

```python
from callbacks import ConsoleResearchCallbacks

# Create callbacks for progress monitoring
callbacks = ConsoleResearchCallbacks(debug_mode=True, quiet_mode=False)

agent = DeepResearchAgent(
    serpapi_key="key",
    project_id="project",
    callbacks=callbacks
)
```

#### Available Callbacks

- **Research Lifecycle**: `on_research_start`, `on_research_complete`
- **Search Events**: `on_search_start`, `on_search_results_found`
- **Content Processing**: `on_content_fetch_start`, `on_summary_generated`
- **Document Storage**: `on_document_stored`
- **Error Handling**: `on_error`, `on_debug_message`

## 🔄 Research Workflow

### 1. Query Analysis & Planning
- Analyzes user query complexity
- Generates clarifying questions if needed
- Creates structured research plan with specific steps

### 2. Iterative Research Execution
```
For each research step:
├── Generate targeted search queries
├── Perform web search via SerpAPI
├── AI-powered source evaluation and filtering
├── Fetch high-quality content from selected sources
├── Generate focused summaries using Gemini
└── Store documents in Vertex AI RAG corpus
```

### 3. Completeness Assessment
- Evaluates research coverage and depth
- Identifies information gaps
- Decides whether additional research is needed

### 4. Report Synthesis
- Synthesizes findings using Vertex AI RAG (when available)
- Generates structured reports with citations
- Formats output in markdown with proper sections

## 📊 Example Output

```markdown
# Research Report: AI Safety Trends 2024

**Query:** Latest developments in AI safety research 2024
**Generated:** 2024-05-27 14:30:15
**Duration:** 180.5 seconds
**RAG Engine:** Vertex AI

## Executive Summary

Recent developments in AI safety research have focused on constitutional AI, 
alignment techniques, and regulatory frameworks...

## Key Findings

### Constitutional AI Development
According to Anthropic's latest research [1], constitutional AI methods have 
shown significant progress in creating more reliable and controllable AI systems...

### Alignment Breakthroughs
Stanford researchers reported [2] breakthrough results in AI alignment using 
novel training methodologies that improve model interpretability...

### Regulatory Landscape
The EU AI Act implementation [3] has established comprehensive guidelines 
for AI safety compliance across member states...

## References

[1] Constitutional AI Research - https://anthropic.com/research
[2] Stanford AI Alignment Study - https://hai.stanford.edu/research
[3] EU AI Act Documentation - https://digital-strategy.ec.europa.eu
```

## 🛠️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SERPAPI_KEY` | Yes | - | SerpAPI key for web search |
| `GOOGLE_CLOUD_PROJECT` | Yes | - | Google Cloud project ID |
| `VERTEX_AI_LOCATION` | No | `us-central1` | Vertex AI region |
| `RAG_CORPUS_NAME` | No | `research_corpus_6` | Name for RAG corpus |
| `RAG_MODEL_NAME` | No | `gemini-2.5-flash-preview-05-20` | Model for RAG synthesis |
| `MODEL_NAME` | No | `gemini-2.5-flash-preview-05-20` | Primary Gemini model |

### Customization Options

```python
# Adjust research parameters
agent.max_retries = 5
agent.set_debug_mode(True)

# Customize LLM behavior
agent.llm.generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "max_output_tokens": 4096
}
```

## 🔍 Features in Detail

### Intelligent Source Evaluation
- AI-powered relevance scoring for search results
- Authority assessment based on domain and content quality
- Duplicate detection and filtering
- Configurable acceptance thresholds

### Content Processing
- Support for HTML, PDF, and text documents
- Clean text extraction from web pages
- Content length optimization for LLM processing
- Automatic content summarization with query focus

### Vertex AI RAG Integration
- Automatic document storage in Vertex AI RAG corpus
- Semantic search capabilities for document retrieval
- Enhanced report synthesis using RAG context
- Persistent storage across research sessions

### Interactive Clarification
- Automatic generation of clarifying questions
- Context-aware research planning
- User-guided research focus refinement
- Skip option for clear queries

## 🛟 Troubleshooting

### Common Issues

**Authentication Errors:**
```bash
# Ensure Google Cloud authentication is set up
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**API Rate Limits:**
- The system includes built-in rate limiting and retry logic
- Increase delays between requests if needed
- Check your SerpAPI usage quota

**Memory Usage:**
- Large research sessions may consume significant memory
- Use `agent.clear_corpus()` to free local memory
- Vertex AI RAG corpus persists independently

**Content Fetch Failures:**
- Some websites may block automated requests
- The system automatically retries failed requests
- Check network connectivity and firewall settings

### Debug Mode

Enable debug mode for detailed logging:

```bash
uv run python cli.py "your query" --debug
```

Or programmatically:

```python
agent.set_debug_mode(True)
```

## 📈 Performance

Typical performance characteristics:
- **Research Time**: 2-5 minutes per complex query
- **Source Coverage**: 10-30 documents per research session
- **Accuracy**: High relevance scoring (>0.7 average)
- **Storage**: Documents persist in Vertex AI RAG corpus

## 🔐 Security & Privacy

- No personal data storage in local components
- All data stored in your Google Cloud project
- Respects robots.txt and rate limits
- Transparent source attribution
- Content is temporarily cached for processing only

## 📚 Requirements

Install dependencies using uv:

```bash
uv add google-genai>=0.5.4
uv add vertexai>=1.60.0
uv add requests>=2.31.0
uv add beautifulsoup4>=4.12.2
uv add pypdf>=4.2.0
uv add python-dotenv>=1.0.0
```

Or if using requirements.txt:
```txt
google-genai>=0.5.4
vertexai>=1.60.0
requests>=2.31.0
beautifulsoup4>=4.12.2
pypdf>=4.2.0
python-dotenv>=1.0.0
```

## 🤝 Contributing

The system is designed to be extensible:
- Add new content fetchers for different file types
- Implement custom RAG engines beyond Vertex AI
- Create specialized callbacks for different use cases
- Extend the CLI with additional features

## 📄 License

[Add your license information here]

---

*This Deep Research Agent provides a powerful foundation for automated research tasks while maintaining flexibility for customization and extension.*