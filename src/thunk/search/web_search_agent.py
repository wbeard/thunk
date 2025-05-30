import logging
import requests
from typing import List

from ..types import SearchResult
from .search_provider import SearchProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchAgent(SearchProvider):
    """Handles web search operations"""

    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key

    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform web search and return results"""
        try:
            # Using SerpAPI for search
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results,
            }

            response = requests.get(
                "https://serpapi.com/search", params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for i, result in enumerate(data.get("organic_results", []), 1):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    domain=self._extract_domain(result.get("link", "")),
                    rank=i,
                )
                results.append(search_result)

            return results
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def provider_name(self) -> str:
        """Return the name of this search provider"""
        return "web"

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc
        except Exception:
            return ""
