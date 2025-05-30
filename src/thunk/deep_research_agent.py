import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime
import time
from collections import defaultdict

from .vertex_rag_engine import VertexRagEngineAPI
from .types import SearchResult, Document
from .gemini_llm import GeminiLLM
from .content_fetcher import ContentFetcher
from .search.search_provider import SearchProvider

logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Main orchestrator for the deep research agent with Vertex AI RAG"""

    def __init__(
        self,
        search_provider: SearchProvider,
        project_id: str,
        location: str = "us-central1",
        corpus_display_name: str = "research_corpus",
        max_concurrent_searches: int = 10,
        max_concurrent_fetches: int = 20,
        search_delay: float = 0.1,
        fetch_delay: float = 0.05,
    ):
        """
        Initialize the Deep Research Agent with Vertex AI RAG

        Args:
            search_provider: Search provider instance (WebSearchAgent or ArxivSearchAgent)
            project_id: Google Cloud project ID
            location: Vertex AI location (default: us-central1)
            corpus_display_name: Name for the RAG corpus
            max_concurrent_searches: Maximum concurrent search operations (default: 10)
            max_concurrent_fetches: Maximum concurrent content fetch operations (default: 20)
            search_delay: Delay between search operations in seconds (default: 0.1)
            fetch_delay: Delay between fetch operations in seconds (default: 0.05)
        """
        self.llm = GeminiLLM(project_name=project_id, location=location)
        self.search_provider = search_provider
        self.content_fetcher = ContentFetcher()

        # Initialize Vertex AI RAG Engine
        self.rag_engine = VertexRagEngineAPI(
            project_id=project_id,
            location=location,
            corpus_display_name=corpus_display_name,
        )

        # Event system using Python primitives
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        self.corpus = []  # Local backup corpus
        self.iteration_count = 0
        self.max_retries = 3
        self.debug_mode = False

        # Rate limiting configuration
        self.max_concurrent_searches = max_concurrent_searches
        self.max_concurrent_fetches = max_concurrent_fetches
        self.search_delay = search_delay
        self.fetch_delay = fetch_delay

        # Create semaphores for concurrency control
        self._search_semaphore = asyncio.Semaphore(max_concurrent_searches)
        self._fetch_semaphore = asyncio.Semaphore(max_concurrent_fetches)

        # Performance tracking
        self.stats = {
            "queries_processed": 0,
            "documents_collected": 0,
            "failed_fetches": 0,
            "api_errors": 0,
            "total_search_time": 0.0,
            "parallel_search_time": 0.0,
            "parallel_fetch_time": 0.0,
            "concurrent_searches_executed": 0,
            "concurrent_fetches_executed": 0,
            "max_concurrent_searches_used": 0,
            "max_concurrent_fetches_used": 0,
        }

    def subscribe(self, event_name: str, handler: Callable):
        """Subscribe to an event with a handler function"""
        self._event_handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: Callable):
        """Unsubscribe from an event"""
        if handler in self._event_handlers[event_name]:
            self._event_handlers[event_name].remove(handler)

    def _emit(self, event_name: str, *args, **kwargs):
        """Emit an event to all subscribers"""
        for handler in self._event_handlers[event_name]:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")

    def _synthesize_report_from_documents(
        self,
        query: str,
        documents: List[Document],
        use_rag: bool = True,
        context: str = None,
    ) -> str:
        """Helper method to synthesize a report from given documents"""
        # Try to use Vertex AI RAG for synthesis if enabled and corpus has content
        self._emit("synthesis_before", len(documents), use_rag)

        if use_rag:
            try:
                corpus_summary = self.rag_engine.get_corpus_summary()
                if corpus_summary.get("file_count", 0) > 0:
                    self._emit(
                        "debug_message",
                        "Using Vertex AI RAG for enhanced report synthesis",
                    )
                    self._emit("rag_synthesis_before", query)
                    # Create enhanced query that includes clarification context
                    context_section = ""

                    if context:
                        context_section = f"""
                    CLARIFICATION CONTEXT PROVIDED:
                    {context}
                    
                    IMPORTANT: Use this clarification context to focus your analysis on the specific aspects mentioned. The user has provided additional details about what they're looking for, so tailor your report accordingly."""

                    rag_prompt = f"""
                    You are a senior research analyst tasked with creating an extensive, in-depth research report. Based on the comprehensive research corpus available and the following query, produce a detailed analytical report of at least 3,000-5,000 words:

                    RESEARCH QUERY: {query}
                    {context_section}

                    REPORT STRUCTURE (Required sections with substantial content):

                    # Research Report: [Title]

                    **Query:** {query}
                    **Generated:** [Current timestamp]
                    **Methodology:** Comprehensive analysis using Vertex AI RAG corpus
                    **Document Sources:** [Number] research documents analyzed

                    ## 1. Executive Summary (400-600 words)
                    - Synthesize the most critical findings and insights
                    - Highlight key trends, developments, and implications
                    - Present main conclusions and recommendations
                    - Include quantitative data and statistics where available

                    ## 2. Background & Context (600-800 words)
                    - Provide comprehensive background on the topic
                    - Explain the current landscape and key players
                    - Discuss historical context and evolution
                    - Identify driving forces and market dynamics

                    ## 3. Current State Analysis (800-1000 words)
                    - Detailed examination of current developments
                    - Analysis of recent trends and patterns
                    - Key statistics, metrics, and performance indicators
                    - Comparison across different regions/sectors/approaches
                    - Current challenges and opportunities

                    ## 4. Key Findings & Insights (1000-1200 words)
                    Break this into subsections covering:
                    - Major discoveries or developments
                    - Technological advances or innovations
                    - Market trends and business implications
                    - Regulatory and policy developments
                    - Expert opinions and industry perspectives
                    - Data analysis and statistical insights

                    ## 5. Comparative Analysis (400-600 words)
                    - Compare different approaches, solutions, or perspectives
                    - Analyze advantages and disadvantages
                    - Benchmark against industry standards or competitors
                    - Regional or demographic comparisons

                    ## 6. Future Outlook & Implications (600-800 words)
                    - Predictions and forecasts based on current data
                    - Potential future developments and scenarios
                    - Long-term implications and consequences
                    - Emerging opportunities and threats

                    ## 7. Recommendations & Conclusions (400-600 words)
                    - Strategic recommendations based on findings
                    - Key takeaways for different stakeholders
                    - Action items and next steps
                    - Final synthesis of insights

                    CRITICAL REQUIREMENTS:
                    - TARGET LENGTH: 3,000-5,000 words minimum
                    - Include detailed citations [1], [2], etc. throughout (at least 2-3 per major paragraph)
                    - Provide specific examples, case studies, and data points
                    - Use quantitative data wherever possible (percentages, numbers, dates)
                    - Include direct quotes from sources when relevant
                    - Maintain analytical depth - don't just summarize, but analyze and synthesize
                    - Cross-reference multiple sources to validate claims
                    - Identify patterns, contradictions, and knowledge gaps
                    - Write with authority and confidence based on the research
                    - Use professional, academic tone with clear, engaging prose
                    - Include specific company names, research institutions, dates, and locations
                    - Provide context for technical terms and industry jargon

                    ANALYSIS DEPTH GUIDELINES:
                    - For each major point, provide supporting evidence from multiple sources
                    - Explain the significance and implications of findings
                    - Connect findings to broader trends and contexts
                    - Identify cause-and-effect relationships
                    - Highlight areas of consensus and disagreement in the literature
                    - Discuss limitations and uncertainties in the data

                    Write in markdown format with clear headings and subheadings. Ensure every section meets the minimum word count guidelines to produce a comprehensive, publication-quality research report."""

                    final_report = self.rag_engine.generate_with_rag(rag_prompt)
                    self._emit("rag_synthesis_after", query, True)
                else:
                    # Fallback to standard synthesis
                    final_report = self.llm.synthesize_final_report(query, documents)
            except Exception:
                final_report = self.llm.synthesize_final_report(query, documents)
        else:
            # Use standard synthesis method
            final_report = self.llm.synthesize_final_report(query, documents)

        # Add references
        references = self._generate_references(documents)
        complete_report = final_report + "\n\n" + references

        self._emit("synthesis_after", len(documents), use_rag, True)
        return complete_report

    async def regenerate_summary(self, query: str, use_rag: bool = True) -> str:
        """
        Regenerate a summary from existing corpus without conducting new research

        Args:
            query: The research query to regenerate summary for
            use_rag: Whether to use Vertex AI RAG for synthesis (default: True)

        Returns:
            Generated report string
        """
        self._emit("regeneration_before", query, use_rag)
        logger.debug(f"Regenerating summary for: {query}")

        try:
            # Get documents from existing corpus
            all_documents = []
            vertex_count = 0
            local_count = 0

            # First, try to get documents from Vertex AI RAG corpus
            try:
                existing_docs = self.rag_engine.search_corpus(query, limit=20)
                if existing_docs:
                    all_documents.extend(existing_docs)
                    vertex_count = len(existing_docs)
                    logger.debug(
                        f"Retrieved {vertex_count} documents from Vertex AI RAG corpus"
                    )
            except Exception as e:
                logger.warning(f"Failed to search Vertex AI corpus: {e}")
                self._emit("error", e, "vertex_corpus_search")

            # Add local corpus documents
            if self.corpus:
                all_documents.extend(self.corpus)
                local_count = len(self.corpus)
                logger.debug(f"Added {local_count} documents from local corpus")

            # Check if we have any documents
            if not all_documents:
                error_msg = "No documents found in corpus. Cannot regenerate summary."
                logger.warning(error_msg)
                self._emit("regeneration_after", False, Exception(error_msg))
                return "No documents found in corpus. Please conduct research first before regenerating summary."

            # Remove duplicates based on URL
            seen_urls = set()
            unique_documents = []
            for doc in all_documents:
                if doc.url not in seen_urls:
                    unique_documents.append(doc)
                    seen_urls.add(doc.url)

            unique_count = len(unique_documents)
            logger.debug(f"Using {unique_count} unique documents for regeneration")
            self._emit("regeneration_documents_before", vertex_count, local_count)
            self._emit(
                "regeneration_documents_after", vertex_count, local_count, unique_count
            )

            # Synthesize report from existing documents
            complete_report = self._synthesize_report_from_documents(
                query, unique_documents, use_rag, context=None
            )

            logger.debug("Summary regeneration completed successfully")
            self._emit("regeneration_after", True)
            return complete_report

        except Exception as e:
            logger.error(f"Summary regeneration failed: {e}")
            self._emit("regeneration_after", False, e)
            raise

    async def _execute_focused_research(
        self,
        query: str,
        next_focus: str,
        all_documents: List[Document],
        current_iteration: int,
        max_iterations: int,
    ) -> List[Document]:
        """Recursively execute focused research based on next_focus areas"""
        if current_iteration >= max_iterations:
            logger.debug(
                f"Reached maximum iterations ({max_iterations}), stopping focused research"
            )
            return all_documents

        try:
            # Generate search queries from the next_focus area (similar to research steps)
            focused_search_queries = self.llm.generate_search_queries(next_focus)
            logger.debug(
                f"Generated {len(focused_search_queries)} focused search queries from: {next_focus}"
            )

            # Execute focused search queries in parallel
            search_tasks = [
                self._search_and_analyze_with_semaphore(query_text, next_focus)
                for query_text in focused_search_queries
            ]

            parallel_search_start = time.time()
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            parallel_search_time = time.time() - parallel_search_start

            # Update performance metrics
            self.stats["parallel_search_time"] += parallel_search_time
            self.stats["concurrent_searches_executed"] += len(search_tasks)
            self.stats["max_concurrent_searches_used"] = max(
                self.stats["max_concurrent_searches_used"], len(search_tasks)
            )

            # Process results and handle exceptions
            for query_text, result in zip(focused_search_queries, search_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Failed to process focused query '{query_text}': {result}"
                    )
                    self._emit(
                        "error", result, f"focused_query_processing: {query_text}"
                    )
                    self.stats["api_errors"] += 1
                else:
                    all_documents.extend(result)

            # Check if we need even more focused research
            summaries = [doc.summary for doc in all_documents]
            is_sufficient, next_next_focus = self.llm.assess_research_completeness(
                query, summaries
            )

            self._emit(
                "focused_research_completeness_before",
                is_sufficient,
                next_next_focus,
                current_iteration,
            )
            self._emit(
                "focused_research_completeness_after",
                is_sufficient,
                next_next_focus,
                current_iteration,
            )

            if is_sufficient:
                logger.debug(
                    f"Focused research complete after {current_iteration + 1} iterations"
                )
                return all_documents
            elif next_next_focus:
                # Recursive call for deeper focused research
                logger.debug(
                    f"Continuing focused research (iteration {current_iteration + 1}): {next_next_focus}"
                )
                return await self._execute_focused_research(
                    query,
                    next_next_focus,
                    all_documents,
                    current_iteration + 1,
                    max_iterations,
                )
            else:
                # No more focus areas identified, stop here
                logger.debug(
                    "No additional focus areas identified, stopping focused research"
                )
                return all_documents

        except Exception as e:
            logger.error(f"Failed focused research iteration {current_iteration}: {e}")
            self._emit("error", e, f"focused_research_iteration_{current_iteration}")
            return all_documents

    async def research_with_context(self, query: str, context: str = None) -> str:
        """Main research method with optional clarification context"""
        start_time = time.time()
        self._emit("research_before", query, context)
        logger.debug(f"Starting research for: {query}")
        if context:
            logger.debug("Using clarification context for targeted research")

        try:
            # Step 1: Plan research with context
            research_plan = self.llm.plan_research(query, context)
            logger.debug(f"Research plan created with {len(research_plan.steps)} steps")

            self._emit("research_plan_before", research_plan.steps)
            self._emit("research_plan_after", research_plan.steps)

            # Step 2: Check existing corpus first
            existing_docs = []
            try:
                existing_docs = self.rag_engine.search_corpus(query, limit=20)
                if existing_docs:
                    self._emit("existing_documents_before", len(existing_docs))
                    self._emit("existing_documents_after", len(existing_docs))
                    logger.debug(
                        f"Found {len(existing_docs)} relevant documents in existing corpus"
                    )
            except Exception as e:
                logger.warning(f"Failed to search existing corpus: {e}")
                self._emit("error", e, "existing_corpus_search")

            # Step 3: Execute research plan
            all_documents = existing_docs.copy()

            for step_num, research_step in enumerate(research_plan.steps):
                self._emit("research_step_before", step_num, research_step)
                logger.debug(f"Executing step {step_num}: {research_step}")

                # Generate search queries for this step
                search_queries = self.llm.generate_search_queries(research_step)

                # Process all search queries in parallel with rate limiting
                search_tasks = [
                    self._search_and_analyze_with_semaphore(query_text, research_step)
                    for query_text in search_queries
                ]

                # Track parallel search performance
                parallel_search_start = time.time()
                search_results = await asyncio.gather(
                    *search_tasks, return_exceptions=True
                )
                parallel_search_time = time.time() - parallel_search_start

                # Update performance metrics
                self.stats["parallel_search_time"] += parallel_search_time
                self.stats["concurrent_searches_executed"] += len(search_tasks)
                self.stats["max_concurrent_searches_used"] = max(
                    self.stats["max_concurrent_searches_used"], len(search_tasks)
                )

                # Process results and handle exceptions
                for query_text, result in zip(search_queries, search_results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Failed to process query '{query_text}': {result}"
                        )
                        self._emit("error", result, f"query_processing: {query_text}")
                        self.stats["api_errors"] += 1
                    else:
                        all_documents.extend(result)

                # Check if we need more information
                summaries = [doc.summary for doc in all_documents]
                is_sufficient, next_focus = self.llm.assess_research_completeness(
                    query, summaries
                )

                self._emit("research_completeness_before", is_sufficient, next_focus)
                self._emit("research_completeness_after", is_sufficient, next_focus)

                if is_sufficient:
                    break
                elif next_focus:
                    # Use recursive focused research instead of loop
                    logger.debug(f"Starting focused research: {next_focus}")
                    self._emit(
                        "focused_research_before",
                        next_focus,
                        0,
                        research_plan.max_iterations,
                    )
                    all_documents = await self._execute_focused_research(
                        query,
                        next_focus,
                        all_documents,
                        0,
                        research_plan.max_iterations,
                    )
                    self._emit("focused_research_after", next_focus, len(all_documents))

                self._emit(
                    "research_step_after", step_num, research_step, len(all_documents)
                )

            # Step 4: Synthesize final report using helper method
            complete_report = self._synthesize_report_from_documents(
                query, all_documents, use_rag=True, context=context
            )

            # Update statistics
            self.stats["queries_processed"] += 1
            self.stats["total_search_time"] += time.time() - start_time

            logger.debug("Research completed successfully")
            self._emit("research_after", True)
            return complete_report

        except Exception as e:
            logger.error(f"Research failed: {e}")
            self._emit("research_after", False, e)
            self.stats["api_errors"] += 1
            raise

    async def research(self, query: str) -> str:
        """Main research method - orchestrates the entire process"""
        return await self.research_with_context(query, None)

    async def research_with_retry(self, query: str, max_retries: int = None) -> str:
        """Research with automatic retry on failure"""
        max_retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                result = await self.research_with_context(query)
                return result
            except Exception as e:
                last_error = e
                logger.error(f"Research attempt {attempt + 1} failed: {e}")
                self._emit("error", e, f"research_attempt_{attempt + 1}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = 2**attempt
                    logger.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts exhausted")

        raise last_error

    async def _search_and_analyze_with_semaphore(
        self, query: str, focus_area: str
    ) -> List[Document]:
        """Search and analyze with semaphore-based rate limiting"""
        async with self._search_semaphore:
            # Add delay for rate limiting
            if self.search_delay > 0:
                await asyncio.sleep(self.search_delay)

            return await self._search_and_analyze(query, focus_area)

    async def _fetch_and_process_content(
        self, result, focus_area: str, corpus_offset: int
    ) -> Optional[Document]:
        """Fetch and process content for a single search result with rate limiting"""
        async with self._fetch_semaphore:
            # Add delay for rate limiting
            if self.fetch_delay > 0:
                await asyncio.sleep(self.fetch_delay)

            try:
                self._emit("content_fetch_before", result.title, result.url)

                # Note: content_fetcher.fetch_content is not async, so we'll run it in executor
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(
                    None, self.content_fetcher.fetch_content, result.url
                )

                if not content or len(content) <= 100:  # Minimum content threshold
                    self._emit("content_fetch_after", result.title, False)
                    return None

                self._emit("content_fetch_after", result.title, True, len(content))

                # Generate summary (also not async, run in executor)
                summary = await loop.run_in_executor(
                    None, self.llm.summarize_content, content, focus_area, result.title
                )

                if not summary:
                    self._emit("summary_generation_after", result.title, False)
                    return None

                self._emit("summary_generation_before", result.title)
                self._emit("summary_generation_after", result.title, True, summary)

                # Create document
                doc = Document(
                    id=f"doc_{corpus_offset + 1}",
                    title=result.title,
                    url=result.url,
                    content=content,
                    summary=summary,
                    source_type=self.search_provider.provider_name(),
                    date_collected=datetime.now().isoformat(),
                    relevance_score=result.relevance_score,
                    metadata={
                        "domain": result.domain,
                        "search_rank": result.rank,
                        "focus_area": focus_area,
                        "content_length": len(content),
                    },
                )

                # Store in Vertex AI RAG engine
                try:
                    self._emit("document_storage_before", doc.id, "vertex_rag")
                    stored_id = await loop.run_in_executor(
                        None, self.rag_engine.store_document, doc
                    )
                    self._emit("document_storage_after", stored_id, "vertex_rag", True)
                    logger.debug(f"Stored document in Vertex AI RAG: {stored_id}")
                except Exception as e:
                    logger.warning(f"Failed to store document in Vertex AI RAG: {e}")
                    self._emit("document_storage_after", doc.id, "vertex_rag", False, e)
                    self._emit("error", e, f"vertex_rag_storage: {doc.id}")

                logger.debug(f"Processed document: {result.title[:50]}...")
                return doc

            except Exception as e:
                self._emit("content_fetch_after", result.title, False, 0, e)
                logger.error(f"Failed to process {result.url}: {e}")
                return None

    async def _search_and_analyze(self, query: str, focus_area: str) -> List[Document]:
        """Search, filter, fetch and analyze content for a specific query"""
        try:
            # Step 1: Web search
            self._emit("search_before", query)

            search_results = self.search_provider.search(query)

            if not search_results:
                self._emit("search_no_results")
                self._emit("search_after", query, 0)
                return []
            else:
                self._emit("search_results_found", len(search_results), query)
                self._emit("search_after", query, len(search_results))
                logger.debug(f"Found {len(search_results)} search results for: {query}")

            # Step 2: Filter and rank results
            filtered_results: list[SearchResult] = []
            for result in search_results:
                try:
                    rating, score, reason = self.llm.evaluate_source_relevance(
                        query, result
                    )
                    result.relevance_score = score

                    accepted = (
                        rating in ["HIGHLY_RELEVANT", "SOMEWHAT_RELEVANT"]
                        and score > 0.3
                    )
                    self._emit("source_evaluation_before", result.title)
                    self._emit(
                        "source_evaluation_after", result.title, accepted, score, reason
                    )

                    if accepted:
                        filtered_results.append(result)
                        logger.debug(
                            f"Accepted source: {result.title} (score: {score:.2f}) (reason: {reason})"
                        )
                    else:
                        logger.debug(
                            f"Rejected source: {result.title} (score: {score:.2f}) (reason: {reason})"
                        )
                except Exception as e:
                    logger.warning(f"Failed to evaluate source {result.url}: {e}")
                    self._emit("error", e, f"source_evaluation: {result.url}")
                    continue

            # Sort by relevance score and take top 3
            filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
            top_results = filtered_results[:3]

            logger.debug(f"Selected {len(top_results)} high-quality sources")

            # Step 3: Fetch and analyze content in parallel
            fetch_tasks = [
                self._fetch_and_process_content(result, focus_area, len(self.corpus))
                for result in top_results
            ]

            # Track parallel fetch performance
            parallel_fetch_start = time.time()
            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            parallel_fetch_time = time.time() - parallel_fetch_start

            # Update performance metrics
            self.stats["parallel_fetch_time"] += parallel_fetch_time
            self.stats["concurrent_fetches_executed"] += len(fetch_tasks)
            self.stats["max_concurrent_fetches_used"] = max(
                self.stats["max_concurrent_fetches_used"], len(fetch_tasks)
            )

            # Collect successful documents and update stats
            documents = []
            for result, fetch_result in zip(top_results, fetch_results):
                if isinstance(fetch_result, Exception):
                    logger.error(f"Failed to process {result.url}: {fetch_result}")
                    self._emit(
                        "error", fetch_result, f"content_processing: {result.url}"
                    )
                    self.stats["failed_fetches"] += 1
                elif fetch_result is not None:
                    documents.append(fetch_result)

                    # Store in local corpus as backup
                    self.corpus.append(fetch_result)
                    self._emit("document_storage_after", fetch_result.id, "local", True)
                    self.stats["documents_collected"] += 1

            return documents

        except Exception as e:
            logger.error(f"Search and analysis failed for query '{query}': {e}")
            self._emit("error", e, f"search_and_analyze: {query}")
            return []

    def _generate_references(self, documents: List[Document]) -> str:
        """Generate a references section for the report"""
        references = ["## References\n"]

        for i, doc in enumerate(documents, 1):
            ref = f"[{i}] {doc.title} - {doc.url}"
            if doc.metadata.get("domain"):
                ref += f" ({doc.metadata['domain']})"
            references.append(ref)

        return "\n".join(references)

    def get_corpus_summary(self) -> Dict[str, Any]:
        """Get summary of collected documents including Vertex AI RAG corpus"""
        # Get Vertex AI corpus summary
        vertex_summary = {}
        try:
            vertex_summary = self.rag_engine.get_corpus_summary()
        except Exception as e:
            logger.warning(f"Failed to get Vertex AI corpus summary: {e}")
            self._emit("error", e, "get_vertex_corpus_summary")

        # Local corpus summary
        local_summary = {
            "local_documents": len(self.corpus),
            "sources_by_domain": {},
            "average_relevance": 0.0,
            "source_types": {},
        }

        if self.corpus:
            domains = [doc.metadata.get("domain", "unknown") for doc in self.corpus]
            for domain in domains:
                local_summary["sources_by_domain"][domain] = (
                    local_summary["sources_by_domain"].get(domain, 0) + 1
                )

            source_types = [doc.source_type for doc in self.corpus]
            for source_type in source_types:
                local_summary["source_types"][source_type] = (
                    local_summary["source_types"].get(source_type, 0) + 1
                )

            local_summary["average_relevance"] = sum(
                doc.relevance_score for doc in self.corpus
            ) / len(self.corpus)

        # Combine summaries
        combined_summary = {
            "vertex_ai_rag": vertex_summary,
            "local_backup": local_summary,
        }

        return combined_summary

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        parallel_efficiency = {
            "avg_parallel_search_time": self.stats["parallel_search_time"]
            / max(1, self.stats["queries_processed"]),
            "avg_parallel_fetch_time": self.stats["parallel_fetch_time"]
            / max(1, self.stats["queries_processed"]),
            "avg_concurrent_searches_per_operation": self.stats[
                "concurrent_searches_executed"
            ]
            / max(1, self.stats["queries_processed"]),
            "avg_concurrent_fetches_per_operation": self.stats[
                "concurrent_fetches_executed"
            ]
            / max(1, self.stats["queries_processed"]),
            "search_concurrency_utilization": self.stats["max_concurrent_searches_used"]
            / max(1, self.max_concurrent_searches),
            "fetch_concurrency_utilization": self.stats["max_concurrent_fetches_used"]
            / max(1, self.max_concurrent_fetches),
        }

        return {
            **self.stats,
            "success_rate": self.stats["queries_processed"]
            / max(1, self.stats["queries_processed"] + self.stats["api_errors"]),
            "avg_docs_per_query": self.stats["documents_collected"]
            / max(1, self.stats["queries_processed"]),
            "avg_search_time": self.stats["total_search_time"]
            / max(1, self.stats["queries_processed"]),
            "local_corpus_size": len(self.corpus),
            **parallel_efficiency,
        }

    def clear_corpus(self):
        """Clear the local corpus (Vertex AI corpus remains)"""
        self.corpus.clear()
        logger.debug("Local corpus cleared")
        self._emit("debug_message", "Local corpus cleared")

    def delete_vertex_corpus(self) -> bool:
        """Delete the Vertex AI RAG corpus (use with caution)

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            success = self.rag_engine.delete_corpus()
            if success:
                logger.debug("Vertex AI RAG corpus deleted")
                self._emit("debug_message", "Vertex AI RAG corpus deleted")
            return success
        except Exception as e:
            logger.error(f"Failed to delete Vertex AI corpus: {e}")
            self._emit("error", e, "delete_vertex_corpus")
            return False

    def export_corpus(self, filepath: str):
        """Export local corpus to JSON file"""
        try:
            corpus_data = []
            for doc in self.corpus:
                doc_data = {
                    "id": doc.id,
                    "title": doc.title,
                    "url": doc.url,
                    "summary": doc.summary,
                    "source_type": doc.source_type,
                    "date_collected": doc.date_collected,
                    "relevance_score": doc.relevance_score,
                    "metadata": doc.metadata,
                }
                corpus_data.append(doc_data)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(corpus_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Local corpus exported to {filepath}")
            self._emit("debug_message", f"Local corpus exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export corpus: {e}")
            self._emit("error", e, f"export_corpus: {filepath}")

    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
        if enabled:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
            self._emit("debug_message", "Debug mode enabled")
        else:
            logging.getLogger().setLevel(logging.INFO)
            logger.debug("Debug mode disabled")
            self._emit("debug_message", "Debug mode disabled")
