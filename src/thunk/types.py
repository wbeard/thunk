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

    def __init__(self):
        # Load from environment variables
        # self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.serpapi_key = os.getenv("SERPAPI_KEY")

        # Vertex AI configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.corpus_display_name = os.getenv("RAG_CORPUS_NAME", "research_corpus_6")
        self.rag_model_name = os.getenv(
            "RAG_MODEL_NAME", "gemini-2.5-flash-preview-05-20"
        )
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-preview-05-20")

        # Validate required environment variables
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration is present"""
        missing = []

        if not self.serpapi_key:
            missing.append("SERPAPI_KEY")
        if not self.project_id:
            missing.append("GOOGLE_CLOUD_PROJECT")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
