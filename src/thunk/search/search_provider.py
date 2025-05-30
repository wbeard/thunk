from abc import ABC, abstractmethod
from typing import List
from ..types import SearchResult


class SearchProvider(ABC):
    """Abstract base class for search providers"""

    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Perform search and return results

        Args:
            query: Search query string
            num_results: Maximum number of results to return

        Returns:
            List[SearchResult]: List of search results
        """
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the name of this search provider

        Returns:
            str: Provider name (e.g., "web", "arxiv")
        """
        pass
