import pytest
from unittest.mock import Mock, patch
from thunk.search.arxiv_search_agent import ArxivSearchAgent
from thunk.types import SearchResult


class MockArxivResult:
    """Mock ArXiv result object"""

    def __init__(self, title, entry_id, summary):
        self.title = title
        self.entry_id = entry_id
        self.summary = summary


class TestArxivSearchAgent:
    """Test suite for ArxivSearchAgent"""

    @pytest.fixture
    @patch("arxiv.Client")
    def agent(self, mock_client_class):
        """Create an ArxivSearchAgent instance for testing"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        agent = ArxivSearchAgent()
        agent.client = mock_client  # Ensure we have access to the mock
        return agent

    def test_provider_name(self, agent):
        """Test that provider_name returns 'arxiv'"""
        assert agent.provider_name() == "arxiv"

    def test_extract_domain(self, agent):
        """Test domain extraction from URLs"""
        test_url = "https://arxiv.org/abs/2107.05580v1"
        expected_domain = "arxiv.org"

        result = agent._extract_domain(test_url)
        assert result == expected_domain

    def test_extract_domain_fallback(self, agent):
        """Test domain extraction fallback for invalid URLs"""
        invalid_url = "not-a-valid-url"
        expected_domain = "arxiv.org"

        result = agent._extract_domain(invalid_url)
        assert result == expected_domain

    def test_search_success(self, agent):
        """Test successful search with mocked results"""
        # Create mock ArXiv results
        mock_results = [
            MockArxivResult(
                title="Quantum Computing with Superconducting Qubits",
                entry_id="https://arxiv.org/abs/2107.05580v1",
                summary="This paper presents a comprehensive study of quantum computing using superconducting qubits. We explore the latest developments in quantum error correction and demonstrate improved fidelity rates across multiple quantum operations.",
            ),
            MockArxivResult(
                title="Machine Learning for Quantum State Preparation",
                entry_id="https://arxiv.org/abs/2108.12345v1",
                summary="We introduce a novel machine learning approach for quantum state preparation that reduces preparation time by 40% compared to traditional methods. Our approach leverages reinforcement learning to optimize quantum gate sequences.",
            ),
        ]

        # Mock the client.results method to return our mock results
        agent.client.results.return_value = mock_results

        # Perform search
        results = agent.search("quantum computing", num_results=2)

        # Verify results
        assert len(results) == 2
        assert all(isinstance(result, SearchResult) for result in results)

        # Check first result
        first_result = results[0]
        assert first_result.title == "Quantum Computing with Superconducting Qubits"
        assert first_result.url == "https://arxiv.org/abs/2107.05580v1"
        assert first_result.domain == "arxiv.org"
        assert first_result.rank == 1
        assert "comprehensive study of quantum computing" in first_result.snippet

        # Check second result
        second_result = results[1]
        assert second_result.title == "Machine Learning for Quantum State Preparation"
        assert second_result.url == "https://arxiv.org/abs/2108.12345v1"
        assert second_result.rank == 2

    def test_search_empty_results(self, agent):
        """Test search with no results"""
        # Mock client to return empty results
        agent.client.results.return_value = []

        results = agent.search("nonexistent topic", num_results=5)

        assert results == []

    def test_search_exception_handling(self, agent):
        """Test search exception handling"""
        # Mock client to raise an exception
        agent.client.results.side_effect = Exception("ArXiv API error")

        results = agent.search("quantum computing", num_results=5)

        assert results == []

    def test_snippet_truncation(self, agent):
        """Test that long summaries are properly truncated"""
        long_summary = (
            "This is a very long summary that exceeds 300 characters. " * 10
        )  # Much longer than 300 chars
        mock_results = [
            MockArxivResult(
                title="Test Paper",
                entry_id="https://arxiv.org/abs/test",
                summary=long_summary,
            )
        ]
        agent.client.results.return_value = mock_results

        results = agent.search("test", num_results=1)

        assert len(results) == 1
        snippet = results[0].snippet
        assert len(snippet) <= 304  # 300 chars + "..."
        assert snippet.endswith("...")

    def test_snippet_no_truncation_needed(self, agent):
        """Test that short summaries are not truncated"""
        short_summary = "This is a short summary."
        mock_results = [
            MockArxivResult(
                title="Test Paper",
                entry_id="https://arxiv.org/abs/test",
                summary=short_summary,
            )
        ]
        agent.client.results.return_value = mock_results

        results = agent.search("test", num_results=1)

        assert len(results) == 1
        snippet = results[0].snippet
        assert snippet == short_summary
        assert not snippet.endswith("...")
