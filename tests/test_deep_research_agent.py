"""
Comprehensive test suite for DeepResearchAgent
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from src.thunk.deep_research_agent import DeepResearchAgent


class MockResearchPlan:
    """Mock research plan for testing"""

    def __init__(self, steps: List[str], max_iterations: int = 3):
        self.steps = steps
        self.max_iterations = max_iterations


class MockGeminiLLM:
    """Mock GeminiLLM for testing"""

    def __init__(self):
        self.plan_research = Mock(
            return_value=MockResearchPlan(
                ["Background research", "Current developments", "Future prospects"]
            )
        )
        self.generate_search_queries = Mock(
            return_value=["test query 1", "test query 2"]
        )
        self.assess_research_completeness = Mock(return_value=(True, None))
        self.evaluate_source_relevance = Mock(
            return_value=("HIGHLY_RELEVANT", 0.8, "Good source")
        )
        self.summarize_content = Mock(return_value="Test summary")
        self.synthesize_final_report = Mock(return_value="Test report")


class MockSearchResult:
    """Mock search result for testing"""

    def __init__(
        self, title: str = "Test Article", url: str = "https://test.com", rank: int = 1
    ):
        self.title = title
        self.url = url
        self.rank = rank
        self.domain = "test.com"
        self.relevance_score = 0.8


class MockDocument:
    """Mock document for testing"""

    def __init__(
        self, id: str = "doc_1", title: str = "Test Doc", url: str = "https://test.com"
    ):
        self.id = id
        self.title = title
        self.url = url
        self.content = "Test content"
        self.summary = "Test summary"
        self.source_type = "web"
        self.date_collected = "2024-01-01T00:00:00"
        self.relevance_score = 0.8
        self.metadata = {"domain": "test.com", "search_rank": 1}


@pytest.fixture
def mock_agent():
    """Create a mock DeepResearchAgent for testing"""
    with (
        patch("src.thunk.deep_research_agent.GeminiLLM") as mock_llm_class,
        patch("src.thunk.deep_research_agent.ContentFetcher") as mock_fetcher_class,
        patch("src.thunk.deep_research_agent.VertexRagEngineAPI") as mock_rag_class,
    ):
        # Setup mocks
        mock_llm_class.return_value = MockGeminiLLM()
        mock_fetcher_class.return_value = Mock()
        mock_rag_class.return_value = Mock()

        # Create mock search provider
        mock_search_provider = Mock()
        mock_search_provider.search.return_value = []
        mock_search_provider.provider_name.return_value = "web"

        agent = DeepResearchAgent(
            search_provider=mock_search_provider,
            project_id="test_project",
            location="us-central1",
        )

        return agent


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        MockDocument("doc_1", "Article 1", "https://test1.com"),
        MockDocument("doc_2", "Article 2", "https://test2.com"),
        MockDocument("doc_3", "Article 3", "https://test3.com"),
    ]


class TestDeepResearchAgent:
    """Test suite for DeepResearchAgent"""

    def test_initialization(self, mock_agent):
        """Test agent initialization"""
        assert mock_agent is not None
        assert mock_agent.iteration_count == 0
        assert mock_agent.max_retries == 3
        assert mock_agent.corpus == []
        assert mock_agent.max_concurrent_searches == 10
        assert mock_agent.max_concurrent_fetches == 20

    def test_event_subscription(self, mock_agent):
        """Test event subscription system"""
        handler_called = False

        def test_handler(*args, **kwargs):
            nonlocal handler_called
            handler_called = True

        # Subscribe to event
        mock_agent.subscribe("test_event", test_handler)

        # Emit event
        mock_agent._emit("test_event")

        assert handler_called

        # Test unsubscribe
        mock_agent.unsubscribe("test_event", test_handler)
        handler_called = False
        mock_agent._emit("test_event")
        assert not handler_called

    def test_synthesize_report_from_documents(self, mock_agent, sample_documents):
        """Test report synthesis from documents"""
        with (
            patch.object(
                mock_agent.rag_engine,
                "get_corpus_summary",
                return_value={"file_count": 5},
            ),
            patch.object(
                mock_agent.rag_engine, "generate_with_rag", return_value="RAG report"
            ),
            patch.object(
                mock_agent,
                "_generate_references",
                return_value="## References\n[1] Test ref",
            ),
        ):
            result = mock_agent._synthesize_report_from_documents(
                "test query", sample_documents, use_rag=True
            )

            assert "RAG report" in result
            assert "## References" in result

    @pytest.mark.asyncio
    async def test_search_and_analyze(self, mock_agent):
        """Test search and analyze functionality"""
        # Mock search results
        mock_search_results = [
            MockSearchResult("Article 1", "https://test1.com", 1),
            MockSearchResult("Article 2", "https://test2.com", 2),
        ]

        mock_agent.search_provider.search.return_value = mock_search_results
        mock_agent.content_fetcher.fetch_content.return_value = "Test content"

        # Mock the fetch and process method to return documents
        with patch.object(
            mock_agent, "_fetch_and_process_content", return_value=MockDocument()
        ) as _mock_fetch:
            results = await mock_agent._search_and_analyze("test query", "test focus")

            assert len(results) <= 2  # Should process up to top 3 results
            mock_agent.search_provider.search.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_execute_focused_research_base_case(self, mock_agent):
        """Test focused research stops when sufficient"""
        sample_docs = [MockDocument()]

        # Mock completeness check to return sufficient
        mock_agent.llm.assess_research_completeness.return_value = (True, None)

        result = await mock_agent._execute_focused_research(
            "test query", "focus area", sample_docs, 0, 3
        )

        assert result == sample_docs
        mock_agent.llm.generate_search_queries.assert_called_once_with("focus area")

    @pytest.mark.asyncio
    async def test_execute_focused_research_max_iterations(self, mock_agent):
        """Test focused research stops at max iterations"""
        sample_docs = [MockDocument()]

        result = await mock_agent._execute_focused_research(
            "test query",
            "focus area",
            sample_docs,
            3,
            3,  # Already at max
        )

        assert result == sample_docs
        # Should not call generate_search_queries when at max iterations
        mock_agent.llm.generate_search_queries.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_focused_research_recursion(self, mock_agent):
        """Test focused research recursion with multiple iterations"""
        sample_docs = [MockDocument()]

        # First call: not sufficient, has next focus
        # Second call: sufficient
        mock_agent.llm.assess_research_completeness.side_effect = [
            (False, "deeper focus"),  # First iteration needs more
            (True, None),  # Second iteration sufficient
        ]

        with patch.object(
            mock_agent,
            "_search_and_analyze_with_semaphore",
            return_value=[MockDocument()],
        ):
            result = await mock_agent._execute_focused_research(
                "test query", "focus area", sample_docs, 0, 3
            )

            # Should have made recursive call
            assert len(result) >= len(sample_docs)
            assert mock_agent.llm.generate_search_queries.call_count >= 1

    @pytest.mark.asyncio
    async def test_research_with_context_integration(self, mock_agent):
        """Test full research with context integration"""
        # Mock existing corpus search
        mock_agent.rag_engine.search_corpus.return_value = [MockDocument("existing_1")]

        # Mock search and analyze
        with (
            patch.object(
                mock_agent,
                "_search_and_analyze_with_semaphore",
                return_value=[MockDocument()],
            ) as _mock_search,
            patch.object(
                mock_agent,
                "_synthesize_report_from_documents",
                return_value="Final report",
            ) as mock_synthesize,
        ):
            result = await mock_agent.research_with_context(
                "test query", "test context"
            )

            assert result == "Final report"
            assert mock_agent.stats["queries_processed"] == 1
            mock_synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_with_context_focused_research_integration(self, mock_agent):
        """Test research with context when focused research is needed"""
        # Mock existing corpus search
        mock_agent.rag_engine.search_corpus.return_value = []

        # Mock initial research step as insufficient, requiring focused research
        mock_agent.llm.assess_research_completeness.side_effect = [
            (False, "need more specific data"),  # Initial step needs more
            (True, None),  # Focused research sufficient
        ]

        with (
            patch.object(
                mock_agent,
                "_search_and_analyze_with_semaphore",
                return_value=[MockDocument()],
            ) as _mock_search,
            patch.object(
                mock_agent,
                "_execute_focused_research",
                return_value=[MockDocument(), MockDocument()],
            ) as mock_focused,
            patch.object(
                mock_agent,
                "_synthesize_report_from_documents",
                return_value="Final report",
            ) as _mock_synthesize,
        ):
            result = await mock_agent.research_with_context("test query")

            assert result == "Final report"
            mock_focused.assert_called_once()
            # Verify focused research was called with correct parameters
            args, kwargs = mock_focused.call_args
            assert args[1] == "need more specific data"  # next_focus

    @pytest.mark.asyncio
    async def test_research_with_retry(self, mock_agent):
        """Test research with retry functionality"""
        # Mock research_with_context to fail twice then succeed
        mock_agent.research_with_context = AsyncMock(
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
                "Success",
            ]
        )

        result = await mock_agent.research_with_retry("test query", max_retries=3)
        assert result == "Success"
        assert mock_agent.research_with_context.call_count == 3

    @pytest.mark.asyncio
    async def test_research_with_retry_exhausted(self, mock_agent):
        """Test research with retry when all attempts fail"""
        mock_agent.research_with_context = AsyncMock(
            side_effect=Exception("Always fails")
        )

        with pytest.raises(Exception, match="Always fails"):
            await mock_agent.research_with_retry("test query", max_retries=2)

        assert mock_agent.research_with_context.call_count == 2

    @pytest.mark.asyncio
    async def test_regenerate_summary(self, mock_agent, sample_documents):
        """Test summary regeneration from existing corpus"""
        # Mock existing documents in RAG corpus
        mock_agent.rag_engine.search_corpus.return_value = sample_documents
        mock_agent.corpus = [MockDocument("local_1")]

        with patch.object(
            mock_agent,
            "_synthesize_report_from_documents",
            return_value="Regenerated report",
        ) as mock_synthesize:
            result = await mock_agent.regenerate_summary("test query")

            assert result == "Regenerated report"
            mock_synthesize.assert_called_once()
            # Should combine documents from both sources
            call_args = mock_synthesize.call_args[0]
            assert len(call_args[1]) == 4  # 3 from RAG + 1 from local

    @pytest.mark.asyncio
    async def test_regenerate_summary_no_documents(self, mock_agent):
        """Test regenerate summary when no documents exist"""
        mock_agent.rag_engine.search_corpus.return_value = []
        mock_agent.corpus = []

        result = await mock_agent.regenerate_summary("test query")

        assert "No documents found in corpus" in result

    def test_get_corpus_summary(self, mock_agent, sample_documents):
        """Test corpus summary generation"""
        mock_agent.corpus = sample_documents
        mock_agent.rag_engine.get_corpus_summary.return_value = {"file_count": 5}

        summary = mock_agent.get_corpus_summary()

        assert "vertex_ai_rag" in summary
        assert "local_backup" in summary
        assert summary["local_backup"]["local_documents"] == 3

    def test_get_performance_stats(self, mock_agent):
        """Test performance statistics"""
        # Set some test stats
        mock_agent.stats["queries_processed"] = 5
        mock_agent.stats["documents_collected"] = 15
        mock_agent.stats["api_errors"] = 1
        mock_agent.stats["total_search_time"] = 25.0

        stats = mock_agent.get_performance_stats()

        assert stats["success_rate"] == 5 / 6  # 5 successful out of 6 total
        assert stats["avg_docs_per_query"] == 3  # 15/5
        assert stats["avg_search_time"] == 5.0  # 25/5
        assert "local_corpus_size" in stats

    def test_clear_corpus(self, mock_agent, sample_documents):
        """Test corpus clearing"""
        mock_agent.corpus = sample_documents
        assert len(mock_agent.corpus) == 3

        mock_agent.clear_corpus()
        assert len(mock_agent.corpus) == 0

    def test_export_corpus(self, mock_agent, sample_documents, tmp_path):
        """Test corpus export to JSON"""
        mock_agent.corpus = sample_documents
        export_path = tmp_path / "test_corpus.json"

        mock_agent.export_corpus(str(export_path))

        assert export_path.exists()

        import json

        with open(export_path) as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["id"] == "doc_1"
        assert data[0]["title"] == "Article 1"

    def test_debug_mode(self, mock_agent):
        """Test debug mode toggle"""

        # Test enabling debug mode
        mock_agent.set_debug_mode(True)
        assert mock_agent.debug_mode is True

        # Test disabling debug mode
        mock_agent.set_debug_mode(False)
        assert mock_agent.debug_mode is False


@pytest.mark.asyncio
class TestDeepResearchAgentIntegration:
    """Integration tests for DeepResearchAgent"""

    async def test_full_research_flow_mock(self, mock_agent):
        """Test complete research flow with all mocks"""
        # Setup comprehensive mocks for full flow
        mock_agent.rag_engine.search_corpus.return_value = []

        # Mock research plan
        research_plan = MockResearchPlan(["Step 1", "Step 2"], max_iterations=2)
        mock_agent.llm.plan_research.return_value = research_plan

        # Mock search queries
        mock_agent.llm.generate_search_queries.return_value = ["query1", "query2"]

        # Mock search results
        mock_search_results = [MockSearchResult()]
        mock_agent.search_provider.search.return_value = mock_search_results

        # Mock content fetching
        mock_agent.content_fetcher.fetch_content.return_value = "Test content"

        # Mock completeness assessment (sufficient after first step)
        mock_agent.llm.assess_research_completeness.return_value = (True, None)

        with (
            patch.object(
                mock_agent, "_fetch_and_process_content", return_value=MockDocument()
            ) as _mock_fetch,
            patch.object(
                mock_agent,
                "_synthesize_report_from_documents",
                return_value="Complete report",
            ) as mock_synthesize,
        ):
            result = await mock_agent.research_with_context("test research query")

            assert result == "Complete report"
            assert mock_agent.stats["queries_processed"] == 1
            mock_synthesize.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
