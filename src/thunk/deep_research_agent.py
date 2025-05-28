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
from .web_search_agent import WebSearchAgent

logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Main orchestrator for the deep research agent with Vertex AI RAG"""

    def __init__(
        self,
        serpapi_key: str,
        project_id: str,
        location: str = "us-central1",
        corpus_display_name: str = "research_corpus",
    ):
        """
        Initialize the Deep Research Agent with Vertex AI RAG

        Args:
            serpapi_key: API key for SerpAPI web search
            project_id: Google Cloud project ID
            location: Vertex AI location (default: us-central1)
            corpus_display_name: Name for the RAG corpus
        """
        self.llm = GeminiLLM(project_name=project_id, location=location)
        self.search_agent = WebSearchAgent(serpapi_key)
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

        # Performance tracking
        self.stats = {
            "queries_processed": 0,
            "documents_collected": 0,
            "failed_fetches": 0,
            "api_errors": 0,
            "total_search_time": 0.0,
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
        self, query: str, documents: List[Document], use_rag: bool = True, context: str = None
    ) -> str:
        """Helper method to synthesize a report from given documents"""
        # Try to use Vertex AI RAG for synthesis if enabled and corpus has content
        self._emit('synthesis_start', len(documents), use_rag)

        if use_rag:
            try:
                corpus_summary = self.rag_engine.get_corpus_summary()
                if corpus_summary.get("file_count", 0) > 0:
                    self._emit('debug_message', "Using Vertex AI RAG for enhanced report synthesis")
                    # Create enhanced query that includes clarification context
                    enhanced_query = query
                    context_section = ""
                    
                    if context:
                        enhanced_query = f"{query}\n\nCLARIFICATION CONTEXT:\n{context}"
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
                else:
                    # Fallback to standard synthesis
                    final_report = self.llm.synthesize_final_report(query, documents)
            except Exception as e:
                logger.warning(f"RAG synthesis failed, using standard method: {e}")
                final_report = self.llm.synthesize_final_report(query, documents)
        else:
            # Use standard synthesis method
            final_report = self.llm.synthesize_final_report(query, documents)

        # Add references
        references = self._generate_references(documents)
        complete_report = final_report + "\n\n" + references

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
        self._emit('regeneration_start', query, use_rag)
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
                self._emit('error', e, "vertex_corpus_search")

            # Add local corpus documents
            if self.corpus:
                all_documents.extend(self.corpus)
                local_count = len(self.corpus)
                logger.debug(f"Added {local_count} documents from local corpus")

            # Check if we have any documents
            if not all_documents:
                error_msg = "No documents found in corpus. Cannot regenerate summary."
                logger.warning(error_msg)
                self._emit('regeneration_complete', False, Exception(error_msg))
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
            self._emit('regeneration_documents_found', vertex_count, local_count, unique_count)

            # Synthesize report from existing documents
            complete_report = self._synthesize_report_from_documents(
                query, unique_documents, use_rag, context=None
            )

            logger.debug("Summary regeneration completed successfully")
            self._emit('regeneration_complete', True)
            return complete_report

        except Exception as e:
            logger.error(f"Summary regeneration failed: {e}")
            self._emit('regeneration_complete', False, e)
            raise

    async def research_with_context(self, query: str, context: str = None) -> str:
        """Main research method with optional clarification context"""
        start_time = time.time()
        self._emit('research_start', query, context)
        logger.debug(f"Starting research for: {query}")
        if context:
            logger.debug("Using clarification context for targeted research")

        try:
            # Step 1: Plan research with context
            research_plan = self.llm.plan_research(query, context)
            logger.debug(f"Research plan created with {len(research_plan.steps)} steps")

            self._emit('research_plan_created', research_plan, research_plan.steps)

            # Step 2: Check existing corpus first
            existing_docs = []
            try:
                existing_docs = self.rag_engine.search_corpus(query, limit=20)
                if existing_docs:
                    self._emit('existing_documents_found', len(existing_docs))
                    logger.debug(
                        f"Found {len(existing_docs)} relevant documents in existing corpus"
                    )
            except Exception as e:
                logger.warning(f"Failed to search existing corpus: {e}")
                self._emit('error', e, "existing_corpus_search")

            # Step 3: Execute research plan
            all_documents = existing_docs.copy()

            for step_num, research_step in enumerate(research_plan.steps, 1):
                self._emit('research_step_start', step_num, research_step)
                logger.debug(f"Executing step {step_num}: {research_step}")

                # Generate search queries for this step
                search_queries = self.llm.generate_search_queries(research_step)

                for query_text in search_queries:
                    try:
                        documents = await self._search_and_analyze(
                            query_text, research_step
                        )
                        all_documents.extend(documents)

                        # Brief pause between searches to respect rate limits
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Failed to process query '{query_text}': {e}")
                        self._emit('error', e, f"query_processing: {query_text}")
                        self.stats["api_errors"] += 1
                        continue

                # Check if we need more information
                summaries = [doc.summary for doc in all_documents]
                is_sufficient, next_focus = self.llm.assess_research_completeness(
                    query, summaries
                )

                self._emit('research_completeness_check', is_sufficient, next_focus)

                if is_sufficient:
                    break
                elif next_focus and self.iteration_count < research_plan.max_iterations:
                    try:
                        focused_docs = await self._search_and_analyze(
                            next_focus, next_focus
                        )
                        all_documents.extend(focused_docs)
                        self.iteration_count += 1
                    except Exception as e:
                        logger.error(f"Failed focused search on '{next_focus}': {e}")
                        self._emit('error', e, f"focused_search: {next_focus}")

            # Step 4: Synthesize final report using helper method
            complete_report = self._synthesize_report_from_documents(
                query, all_documents, use_rag=True, context=context
            )

            # Update statistics
            self.stats["queries_processed"] += 1
            self.stats["total_search_time"] += time.time() - start_time

            logger.debug("Research completed successfully")
            self._emit('research_complete', True)
            return complete_report

        except Exception as e:
            logger.error(f"Research failed: {e}")
            self._emit('research_complete', False, e)
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
                self._emit('error', e, f"research_attempt_{attempt + 1}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = 2**attempt
                    logger.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts exhausted")

        raise last_error

    async def _search_and_analyze(self, query: str, focus_area: str) -> List[Document]:
        """Search, filter, fetch and analyze content for a specific query"""
        try:
            # Step 1: Web search
            self._emit('search_start', query)

            search_results = self.search_agent.search(query)

            if not search_results:
                self._emit('search_no_results')
                logger.warning(f"No search results found for query: {query}")
                return []
            else:
                self._emit('search_results_found', len(search_results))
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
                    self._emit('source_evaluation', result.title, accepted, score, reason)

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
                    self._emit('error', e, f"source_evaluation: {result.url}")
                    continue

            # Sort by relevance score and take top 3
            filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
            top_results = filtered_results[:3]

            logger.debug(f"Selected {len(top_results)} high-quality sources")

            # Step 3: Fetch and analyze content
            documents = []
            for result in top_results:
                try:
                    self._emit('content_fetch_start', result.title, result.url)
                    content = self.content_fetcher.fetch_content(result.url)

                    if content and len(content) > 100:  # Minimum content threshold
                        self._emit('content_fetch_complete', result.title, True, len(content))

                        # Generate summary
                        summary = self.llm.summarize_content(
                            content, focus_area, result.title
                        )

                        if not summary:
                            logger.warning(
                                f"Failed to generate summary for {result.url}"
                            )
                            continue

                        self._emit('summary_generated', result.title, summary)

                        # Create document
                        doc = Document(
                            id=f"doc_{len(self.corpus) + len(documents) + 1}",
                            title=result.title,
                            url=result.url,
                            content=content,
                            summary=summary,
                            source_type="web",
                            date_collected=datetime.now().isoformat(),
                            relevance_score=result.relevance_score,
                            metadata={
                                "domain": result.domain,
                                "search_rank": result.rank,
                                "focus_area": focus_area,
                                "content_length": len(content),
                            },
                        )

                        documents.append(doc)

                        # Store in Vertex AI RAG engine
                        try:
                            stored_id = self.rag_engine.store_document(doc)
                            self._emit('document_stored', stored_id, "vertex_rag")
                            logger.debug(
                                f"Stored document in Vertex AI RAG: {stored_id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to store document in Vertex AI RAG: {e}"
                            )
                            self._emit('error', e, f"vertex_rag_storage: {doc.id}")

                        # Store in local corpus as backup
                        self.corpus.append(doc)
                        self._emit('document_stored', doc.id, "local")
                        self.stats["documents_collected"] += 1

                        logger.debug(f"Processed document: {result.title[:50]}...")
                    else:
                        self._emit('content_fetch_complete', result.title, False)
                        logger.warning(f"Insufficient content from {result.url}")
                        self.stats["failed_fetches"] += 1

                except Exception as e:
                    self._emit('content_fetch_complete', result.title, False)
                    logger.error(f"Failed to process {result.url}: {e}")
                    self._emit('error', e, f"content_processing: {result.url}")
                    self.stats["failed_fetches"] += 1
                    continue

            return documents

        except Exception as e:
            logger.error(f"Search and analysis failed for query '{query}': {e}")
            self._emit('error', e, f"search_and_analyze: {query}")
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
            self._emit('error', e, "get_vertex_corpus_summary")

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
        return {
            **self.stats,
            "success_rate": self.stats["queries_processed"]
            / max(1, self.stats["queries_processed"] + self.stats["api_errors"]),
            "avg_docs_per_query": self.stats["documents_collected"]
            / max(1, self.stats["queries_processed"]),
            "avg_search_time": self.stats["total_search_time"]
            / max(1, self.stats["queries_processed"]),
            "local_corpus_size": len(self.corpus),
        }

    def clear_corpus(self):
        """Clear the local corpus (Vertex AI corpus remains)"""
        self.corpus.clear()
        logger.debug("Local corpus cleared")
        self._emit('debug_message', "Local corpus cleared")

    def delete_vertex_corpus(self):
        """Delete the Vertex AI RAG corpus (use with caution)"""
        try:
            self.rag_engine.delete_corpus()
            logger.debug("Vertex AI RAG corpus deleted")
            self._emit('debug_message', "Vertex AI RAG corpus deleted")
        except Exception as e:
            logger.error(f"Failed to delete Vertex AI corpus: {e}")
            self._emit('error', e, "delete_vertex_corpus")

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
            self._emit('debug_message', f"Local corpus exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export corpus: {e}")
            self._emit('error', e, f"export_corpus: {filepath}")

    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
        if enabled:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
            self._emit('debug_message', "Debug mode enabled")
        else:
            logging.getLogger().setLevel(logging.INFO)
            logger.debug("Debug mode disabled")
            self._emit('debug_message', "Debug mode disabled")
