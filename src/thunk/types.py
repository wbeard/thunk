import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str
    rank: int
    relevance_score: float = 0.0


@dataclass
class Document:
    id: str
    title: str
    url: str
    content: str
    summary: str
    source_type: str
    date_collected: str
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class ResearchPlan:
    query: str
    steps: List[str]
    confidence_threshold: float = 0.8
    max_iterations: int = 3


@dataclass
class ResearchConfig:
    """Configuration for the research agent"""

    def __init__(self, corpus_display_name: str = None, search_provider: str = "web"):
        # Load from environment variables
        # self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.search_provider = search_provider

        # Vertex AI configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.corpus_display_name = corpus_display_name or os.getenv("RAG_CORPUS_NAME")
        self.rag_model_name = os.getenv(
            "RAG_MODEL_NAME", "gemini-2.5-flash-preview-05-20"
        )
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-preview-05-20")

        # Validate required environment variables
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration is present"""
        missing = []

        # Only require SERPAPI_KEY for web search provider
        if self.search_provider == "web" and not self.serpapi_key:
            missing.append("SERPAPI_KEY")
        if not self.project_id:
            missing.append("GOOGLE_CLOUD_PROJECT")
        if not self.corpus_display_name:
            missing.append("--corpus argument or RAG_CORPUS_NAME environment variable")

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
