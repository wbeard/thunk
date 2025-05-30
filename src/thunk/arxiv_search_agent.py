import logging
from typing import List
from urllib.parse import urlparse

from .types import SearchResult
from .search_provider import SearchProvider
from arxiv import Client, Search, SortCriterion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivSearchAgent(SearchProvider):
    """Handles arXiv search operations"""

    def __init__(self):
        self.client = Client()

    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform arXiv search and return results"""
        try:
            # Create arXiv search object
            arxiv_search = Search(
                query=query,
                max_results=num_results,
                sort_by=SortCriterion.Relevance
            )

            # Execute search
            results = []
            for i, result in enumerate(self.client.results(arxiv_search), 1):
                search_result = SearchResult(
                    title=result.title,
                    url=result.entry_id,
                    snippet=result.summary[:300] + "..." if len(result.summary) > 300 else result.summary,
                    domain=self._extract_domain(result.entry_id),
                    rank=i,
                )
                results.append(search_result)

            return results
        except Exception as e:
            logger.error(f"ArXiv search failed for query '{query}': {e}")
            return []

    def provider_name(self) -> str:
        """Return the name of this search provider"""
        return "arxiv"

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return "arxiv.org"