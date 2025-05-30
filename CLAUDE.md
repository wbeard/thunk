# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thunk is a sophisticated AI-powered research assistant that autonomously performs multi-step web research using Google's Vertex AI and Gemini models. It replicates advanced research capabilities similar to Google's Gemini Deep Research, providing autonomous research planning, intelligent web search, content analysis, and comprehensive report synthesis.

## Key Architecture Components

### Core Classes and Flow
- **DeepResearchAgent** (`src/thunk/deep_research_agent.py`): Main orchestrator that coordinates the entire research process
- **CLI** (`src/thunk/cli.py`): Command-line interface with event-driven progress reporting
- **Vertex AI RAG Integration** (`src/thunk/vertex_rag_engine.py`): Handles document storage and enhanced report synthesis
- **Multi-step Research Flow**: Query analysis → Research planning → Parallel search execution → Quality filtering → Content processing → Completeness assessment → Focused research (if needed) → Synthesis

### Event-Driven Architecture
The system uses a publish-subscribe event system where components emit events (`research_start`, `search_complete`, `synthesis_start`, etc.) that subscribers can listen to for progress tracking, logging, and UI updates.

### Asynchronous Processing
- Parallel search execution with semaphore-based rate limiting
- Concurrent content fetching with configurable concurrency limits
- Recursive focused research with iteration limits to prevent infinite loops

## Development Commands

### Running the Application
```bash
# Basic research query
uv run thunk "research topic"

# Interactive mode
uv run thunk --interactive

# Regenerate from existing corpus
uv run thunk --regenerate "query" --corpus corpus_name

# Check configuration
uv run thunk --check-config
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_deep_research_agent.py

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src/thunk
```

### Development Setup
```bash
# Install dependencies
uv sync

# Set up Google Cloud authentication
gcloud auth application-default login
```

## Configuration Requirements

The application requires specific environment variables in a `.env` file:

**Required:**
- `SERPAPI_KEY`: SerpAPI key for web search
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID

**Optional with defaults:**
- `VERTEX_AI_LOCATION`: Vertex AI region (default: us-central1)
- `RAG_CORPUS_NAME`: Name for RAG corpus (default: research_corpus_6)
- `RAG_MODEL_NAME`: Model for RAG synthesis (default: gemini-2.5-flash-preview-05-20)
- `MODEL_NAME`: Primary Gemini model (default: gemini-2.5-flash-preview-05-20)

The `ResearchConfig` class in `src/thunk/types.py` handles configuration validation.

## Key Patterns

### Event Subscription
When adding new functionality, subscribe to relevant events:
```python
agent.subscribe('research_start', callback_function)
agent.subscribe('document_stored', callback_function)
```

### Async Pattern for Research Operations
All research operations are async and use semaphores for rate limiting:
```python
async with self._search_semaphore:
    await asyncio.sleep(self.search_delay)
    return await self._search_and_analyze(query, focus_area)
```

### Error Handling with Events
Errors are emitted as events for consistent handling:
```python
self._emit('error', exception, context_string)
```

### Focused Research Recursion
The system uses recursive focused research with proper iteration limits to handle complex queries requiring multiple research phases.

## Testing Patterns

- Use comprehensive mocks for external services (Vertex AI, SerpAPI, Gemini)
- Test both unit functionality and integration flows
- Mock async operations with `AsyncMock`
- Use fixtures for common test objects (`mock_agent`, `sample_documents`)

## CLI Architecture Notes

The CLI uses a dual-subscriber pattern:
- `CLIEventSubscriber`: Handles user-facing progress updates and session statistics
- `ResearchEventLogger`: Provides structured logging for debugging

When modifying the CLI, ensure both subscribers are properly configured to maintain visibility into the research process.