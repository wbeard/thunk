from abc import ABC
from typing import Dict, Any, Optional, List


class ResearchCallbacks(ABC):
    """Abstract base class for research lifecycle callbacks"""

    # Research Lifecycle
    def on_research_start(self, query: str, context: Optional[str] = None):
        """Called when research begins"""
        pass

    def on_research_plan_created(self, plan: Any, steps: List[str]):
        """Called when research plan is created"""
        pass

    def on_existing_documents_found(self, count: int):
        """Called when existing documents are found in corpus"""
        pass

    def on_research_step_start(self, step_num: int, step_description: str):
        """Called when starting a research step"""
        pass

    def on_research_completeness_check(
        self, is_sufficient: bool, next_focus: Optional[str] = None
    ):
        """Called after checking if research is complete"""
        pass

    def on_synthesis_start(self, document_count: int, use_rag: bool = True):
        """Called when starting report synthesis"""
        pass

    def on_research_complete(self, success: bool, error: Optional[Exception] = None):
        """Called when research completes (success or failure)"""
        pass

    # Search and Analysis Lifecycle
    def on_search_start(self, query: str):
        """Called when starting web search"""
        pass

    def on_search_results_found(self, count: int):
        """Called when search results are found"""
        pass

    def on_search_no_results(self):
        """Called when no search results found"""
        pass

    def on_source_evaluation(
        self, source_title: str, accepted: bool, score: float, reason: str
    ):
        """Called when evaluating source relevance"""
        pass

    def on_content_fetch_start(self, title: str, url: str):
        """Called when starting to fetch content"""
        pass

    def on_content_fetch_complete(
        self, title: str, success: bool, content_length: int = 0
    ):
        """Called when content fetch completes"""
        pass

    def on_summary_generated(self, title: str, summary_preview: str):
        """Called when content summary is generated"""
        pass

    def on_document_stored(self, doc_id: str, storage_type: str):
        """Called when document is stored"""
        pass

    # Regeneration Lifecycle
    def on_regeneration_start(self, query: str, use_rag: bool):
        """Called when starting summary regeneration"""
        pass

    def on_regeneration_documents_found(
        self, vertex_count: int, local_count: int, unique_count: int
    ):
        """Called when documents found for regeneration"""
        pass

    def on_regeneration_complete(
        self, success: bool, error: Optional[Exception] = None
    ):
        """Called when regeneration completes"""
        pass

    # General Lifecycle
    def on_error(self, error: Exception, context: str):
        """Called when an error occurs"""
        pass

    def on_debug_message(self, message: str):
        """Called for debug-level messages"""
        pass

    def on_progress_update(self, message: str, context: Dict[str, Any] = None):
        """Called for general progress updates"""
        pass


class ConsoleResearchCallbacks(ResearchCallbacks):
    """Console-based implementation of research callbacks"""

    def __init__(self, debug_mode: bool = False, quiet_mode: bool = False):
        self.debug_mode = debug_mode
        self.quiet_mode = quiet_mode

    def _print(self, message: str, force: bool = False):
        """Print message unless in quiet mode"""
        if force or not self.quiet_mode:
            print(message)

    def _debug_print(self, message: str):
        """Print debug message if debug mode enabled"""
        if self.debug_mode:
            print(message)

    # Research Lifecycle
    def on_research_start(self, query: str, context: Optional[str] = None):
        self._print(f"🔍 Starting research for: {query}")
        if context:
            self._debug_print("Using clarification context for targeted research")

    def on_research_plan_created(self, plan: Any, steps: List[str]):
        self._print("🧑‍🔬  Research plan created")
        for index, step in enumerate(steps, 1):
            self._print(f"{index}. {step}")

    def on_existing_documents_found(self, count: int):
        if count > 0:
            self._print("📄  Found relevant documents in existing corpus")

    def on_research_step_start(self, step_num: int, step_description: str):
        self._print(f"🧑‍🔬  Executing research step {step_num}: {step_description}")

    def on_research_completeness_check(
        self, is_sufficient: bool, next_focus: Optional[str] = None
    ):
        if is_sufficient:
            self._print("✅  Research deemed sufficient, proceeding to synthesis")
        elif next_focus:
            self._print(f"🔍  Additional research needed on: {next_focus}")

    def on_synthesis_start(self, document_count: int, use_rag: bool = True):
        self._print(f"🔮 Synthesizing final report from {document_count} documents")
        if self.debug_mode:
            self._debug_print(f"**************** USE RAG: {use_rag} ****************")

    def on_research_complete(self, success: bool, error: Optional[Exception] = None):
        if success:
            self._print("✅  Research completed successfully")
        else:
            self._print(f"❌  Research failed: {error}")

    # Search and Analysis Lifecycle
    def on_search_start(self, query: str):
        self._print(f"🔍  Searching for: {query}")

    def on_search_results_found(self, count: int):
        self._print(f"🔍  Found {count} search results")

    def on_search_no_results(self):
        self._print("❌  No search results found")

    def on_source_evaluation(
        self, source_title: str, accepted: bool, score: float, reason: str
    ):
        if self.debug_mode:
            if accepted:
                self._debug_print(
                    f"✅ Accepted source: {source_title} (score: {score:.2f}) (reason: {reason})"
                )
            else:
                self._debug_print(
                    f"❌ Rejected source: {source_title} (score: {score:.2f}) (reason: {reason})"
                )

    def on_content_fetch_start(self, title: str, url: str):
        title_preview = title[:30] + "..." if len(title) > 30 else title
        self._print(f"🌐  Fetching content from {title_preview}")

    def on_content_fetch_complete(
        self, title: str, success: bool, content_length: int = 0
    ):
        if not success:
            self._debug_print(f"⚠️  Failed to fetch content from {title}")

    def on_summary_generated(self, title: str, summary_preview: str):
        self._print(f"🧑‍🔬  Generated summary: {summary_preview}")

    def on_document_stored(self, doc_id: str, storage_type: str):
        if storage_type == "vertex_rag":
            self._print(f"📄  Stored document in Vertex AI RAG: {doc_id}")
        else:
            self._debug_print(f"📄  Stored document locally: {doc_id}")

    # Regeneration Lifecycle
    def on_regeneration_start(self, query: str, use_rag: bool):
        self._print(f"🔄 Regenerating summary for: {query}")
        if self.debug_mode:
            self._debug_print(f"Using RAG: {use_rag}")

    def on_regeneration_documents_found(
        self, vertex_count: int, local_count: int, unique_count: int
    ):
        if vertex_count > 0:
            self._debug_print(
                f"Retrieved {vertex_count} documents from Vertex AI RAG corpus"
            )
        if local_count > 0:
            self._debug_print(f"Added {local_count} documents from local corpus")
        self._print(f"🔮 Regenerating summary from {unique_count} existing documents")

    def on_regeneration_complete(
        self, success: bool, error: Optional[Exception] = None
    ):
        if success:
            self._debug_print("Summary regeneration completed successfully")
        else:
            self._print(f"❌ Summary regeneration failed: {error}")

    # General Lifecycle
    def on_error(self, error: Exception, context: str):
        self._print(f"❌ Error in {context}: {error}")

    def on_debug_message(self, message: str):
        self._debug_print(message)

    def on_progress_update(self, message: str, context: Dict[str, Any] = None):
        self._print(message)


class SilentResearchCallbacks(ResearchCallbacks):
    """Silent implementation that does nothing - useful for batch processing"""

    pass


class LoggingResearchCallbacks(ResearchCallbacks):
    """Logging-based implementation of research callbacks"""

    def __init__(self, logger):
        self.logger = logger

    def on_research_start(self, query: str, context: Optional[str] = None):
        self.logger.info(f"Research started for query: {query}")
        if context:
            self.logger.debug("Using clarification context")

    def on_research_plan_created(self, plan: Any, steps: List[str]):
        self.logger.info(f"Research plan created with {len(steps)} steps")
        for i, step in enumerate(steps, 1):
            self.logger.debug(f"Step {i}: {step}")

    def on_existing_documents_found(self, count: int):
        if count > 0:
            self.logger.info(f"Found {count} existing documents in corpus")

    def on_research_step_start(self, step_num: int, step_description: str):
        self.logger.info(f"Executing research step {step_num}: {step_description}")

    def on_research_completeness_check(
        self, is_sufficient: bool, next_focus: Optional[str] = None
    ):
        if is_sufficient:
            self.logger.info("Research deemed sufficient")
        elif next_focus:
            self.logger.info(f"Additional research needed on: {next_focus}")

    def on_synthesis_start(self, document_count: int, use_rag: bool = True):
        self.logger.info(
            f"Starting synthesis from {document_count} documents (RAG: {use_rag})"
        )

    def on_research_complete(self, success: bool, error: Optional[Exception] = None):
        if success:
            self.logger.info("Research completed successfully")
        else:
            self.logger.error(f"Research failed: {error}")

    def on_search_start(self, query: str):
        self.logger.debug(f"Starting search for: {query}")

    def on_search_results_found(self, count: int):
        self.logger.debug(f"Found {count} search results")

    def on_search_no_results(self):
        self.logger.warning("No search results found")

    def on_source_evaluation(
        self, source_title: str, accepted: bool, score: float, reason: str
    ):
        status = "accepted" if accepted else "rejected"
        self.logger.debug(f"Source {status}: {source_title} (score: {score:.2f})")

    def on_content_fetch_start(self, title: str, url: str):
        self.logger.debug(f"Fetching content from: {title}")

    def on_content_fetch_complete(
        self, title: str, success: bool, content_length: int = 0
    ):
        if success:
            self.logger.debug(f"Content fetched: {title} ({content_length} chars)")
        else:
            self.logger.warning(f"Failed to fetch content: {title}")

    def on_summary_generated(self, title: str, summary_preview: str):
        self.logger.debug(f"Summary generated for: {title}")

    def on_document_stored(self, doc_id: str, storage_type: str):
        self.logger.debug(f"Document stored ({storage_type}): {doc_id}")

    def on_regeneration_start(self, query: str, use_rag: bool):
        self.logger.info(f"Starting regeneration for: {query} (RAG: {use_rag})")

    def on_regeneration_documents_found(
        self, vertex_count: int, local_count: int, unique_count: int
    ):
        self.logger.info(
            f"Found documents - Vertex: {vertex_count}, Local: {local_count}, Unique: {unique_count}"
        )

    def on_regeneration_complete(
        self, success: bool, error: Optional[Exception] = None
    ):
        if success:
            self.logger.info("Regeneration completed successfully")
        else:
            self.logger.error(f"Regeneration failed: {error}")

    def on_error(self, error: Exception, context: str):
        self.logger.error(f"Error in {context}: {error}")

    def on_debug_message(self, message: str):
        self.logger.debug(message)

    def on_progress_update(self, message: str, context: Dict[str, Any] = None):
        self.logger.info(message)
