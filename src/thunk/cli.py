#!/usr/bin/env python3
"""
Basic CLI for Deep Research Agent
Simple command-line interface without TUI complexity
"""

import asyncio
import sys
import argparse
import time
import random
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from .deep_research_agent import (
    DeepResearchAgent,
)
from .types import ResearchConfig
from .state_logger import ResearchEventLogger
from .vertex_rag_engine import VertexRagEngineAPI

load_dotenv()


def generate_random_corpus_name() -> str:
    """Generate a random corpus name using adjective-noun combinations"""
    adjectives = [
        "agile", "bold", "calm", "deep", "eager", "fast", "gentle", "happy", "intense", "jolly",
        "keen", "lively", "merry", "noble", "optimistic", "peaceful", "quick", "radiant", "swift", "tranquil",
        "unique", "vibrant", "wise", "xenial", "youthful", "zealous", "ancient", "brilliant", "clever", "dynamic",
        "elegant", "fierce", "graceful", "humble", "innovative", "joyful", "kind", "luminous", "majestic", "nimble",
        "organic", "precise", "quiet", "robust", "serene", "thoughtful", "uplifting", "vivid", "warm", "xylem",
        "young", "zestful", "amber", "blue", "crimson", "dark", "emerald", "fiery", "golden", "icy",
        "jade", "lavender", "mystic", "neon", "orange", "purple", "royal", "silver", "teal", "ultra"
    ]
    
    nouns = [
        "panda", "tiger", "eagle", "dolphin", "fox", "wolf", "bear", "lion", "owl", "deer",
        "hawk", "shark", "whale", "penguin", "falcon", "jaguar", "lynx", "otter", "raven", "swan",
        "bicycle", "rocket", "compass", "telescope", "camera", "piano", "guitar", "violin", "drums", "flute",
        "mountain", "river", "forest", "ocean", "valley", "canyon", "meadow", "glacier", "volcano", "desert",
        "crystal", "diamond", "ruby", "sapphire", "emerald", "pearl", "opal", "quartz", "amber", "jade",
        "thunder", "lightning", "storm", "rainbow", "sunrise", "sunset", "aurora", "comet", "meteor", "nebula",
        "galaxy", "planet", "star", "moon", "sun", "cosmos", "orbit", "void", "dimension", "universe"
    ]
    
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    return f"{adjective}-{noun}"


class CLIEventSubscriber:
    """Event subscriber for CLI that handles research agent events"""

    def __init__(
        self,
        debug_mode: bool = False,
        quiet_mode: bool = False,
        show_full_report: bool = False,
    ):
        self.debug_mode = debug_mode
        self.quiet_mode = quiet_mode
        self.show_full_report = show_full_report
        self.session_stats = {
            "start_time": None,
            "research_queries": 0,
            "regeneration_queries": 0,
            "documents_processed": 0,
            "errors": 0,
        }
    
    def _print(self, message: str):
        """Print message if not in quiet mode"""
        if not self.quiet_mode:
            print(message)
    
    def _debug_print(self, message: str):
        """Print debug message if debug mode is enabled"""
        if self.debug_mode and not self.quiet_mode:
            print(f"[DEBUG] {message}")
    
    def subscribe_to_agent(self, agent: DeepResearchAgent):
        """Subscribe to all relevant events from the research agent"""
        # Research lifecycle events
        agent.subscribe('research_start', self.on_research_start)
        agent.subscribe('research_complete', self.on_research_complete)
        agent.subscribe('research_plan_created', self.on_research_plan_created)
        agent.subscribe('research_step_start', self.on_research_step_start)
        agent.subscribe('research_completeness_check', self.on_research_completeness_check)
        
        # Regeneration events
        agent.subscribe('regeneration_start', self.on_regeneration_start)
        agent.subscribe('regeneration_complete', self.on_regeneration_complete)
        agent.subscribe('regeneration_documents_found', self.on_regeneration_documents_found)
        
        # Search and content events
        agent.subscribe('search_start', self.on_search_start)
        agent.subscribe('search_results_found', self.on_search_results_found)
        agent.subscribe('search_no_results', self.on_search_no_results)
        agent.subscribe('source_evaluation', self.on_source_evaluation)
        agent.subscribe('content_fetch_start', self.on_content_fetch_start)
        agent.subscribe('content_fetch_complete', self.on_content_fetch_complete)
        agent.subscribe('summary_generated', self.on_summary_generated)
        
        # Document storage events
        agent.subscribe('document_stored', self.on_document_stored)
        agent.subscribe('existing_documents_found', self.on_existing_documents_found)
        
        # Synthesis events
        agent.subscribe('synthesis_start', self.on_synthesis_start)
        
        # Error and debug events
        agent.subscribe('error', self.on_error)
        agent.subscribe('debug_message', self.on_debug_message)
    
    # Event handlers - Before events show current status
    def on_research_before(self, query: str, context: Optional[str] = None):
        if self.session_stats["start_time"] is None:
            self.session_stats["start_time"] = time.time()
        self.session_stats["research_queries"] += 1

        self._print(f"\nStarting research on: '{query}'")
        self._print("=" * 60)

        if context:
            self._debug_print("Using clarification context for targeted research")
        self._print("Status: Research starting...")

    def on_regeneration_before(self, query: str, use_rag: bool):
        if self.session_stats["start_time"] is None:
            self.session_stats["start_time"] = time.time()
        self.session_stats["regeneration_queries"] += 1

        self._print(f"\nRegenerating summary for: '{query}'")
        self._print("=" * 60)
        self._print("Status: Regeneration starting...")
    
    def on_research_plan_before(self, steps):
        self._print("Status: Creating research plan...")
    
    def on_research_plan_after(self, steps):
        self._debug_print(f"Research plan created with {len(steps)} steps")
        if self.debug_mode:
            for i, step in enumerate(steps, 1):
                self._debug_print(f"  Step {i}: {step}")
        self._print("Status: Research plan ready")
    
    def on_research_step_before(self, step_num: int, research_step: str):
        self._debug_print(f"Executing step {step_num}: {research_step}")
        self._print(f"Status: Executing research step {step_num}...")
        
    def on_research_step_after(self, step_num: int, research_step: str, doc_count: int):
        self._print(f"Status: Step {step_num} completed ({doc_count} documents collected)")
    
    def on_research_completeness_before(self, is_sufficient: bool, next_focus: str):
        self._print("Status: Checking research completeness...")
        
    def on_research_completeness_after(self, is_sufficient: bool, next_focus: str):
        if is_sufficient:
            self._debug_print("Research deemed sufficient")
            self._print("Status: Research complete, proceeding to synthesis")
        elif next_focus:
            self._debug_print(f"Additional focus needed: {next_focus}")
            self._print(f"Status: Additional research needed on: {next_focus[:50]}...")
    
    def on_regeneration_documents_before(self, vertex_count: int, local_count: int):
        self._print("Status: Searching for existing documents...")
        
    def on_regeneration_documents_after(self, vertex_count: int, local_count: int, unique_count: int):
        self._debug_print(f"Found {vertex_count} Vertex AI docs, {local_count} local docs, {unique_count} unique")
        self._print(f"Status: Found {unique_count} documents for regeneration")
    
    def on_search_before(self, query: str):
        self._debug_print(f"Starting web search for: {query}")
        self._print(f"Status: Searching for '{query[:50]}...'")
        
    def on_search_after(self, query: str, result_count: int):
        self._print(f"Status: Search completed ({result_count} results)")
    
    def on_search_results_found(self, count: int, query: str):
        self._print(f"Found {count} search results for query {query}")
    
    def on_search_no_results(self):
        self._print("No search results found")
    
    def on_source_evaluation_before(self, title: str):
        self._print(f"Status: Evaluating '{title[:30]}...'")
        
    def on_source_evaluation_after(self, title: str, accepted: bool, score: float, reason: str):
        if accepted:
            self._debug_print(f"Accepted: {title[:50]}... (score: {score:.2f})")
            self._print(f"Status: Accepted source (score: {score:.2f})")
        else:
            self._debug_print(f"Rejected: {title[:50]}... (score: {score:.2f}) - {reason}")
    
    def on_content_fetch_before(self, title: str, url: str):
        self._debug_print(f"Fetching content from: {title[:50]}...")
        self._print(f"Status: Fetching '{title[:30]}...'")
    
    def on_content_fetch_after(self, title: str, success: bool, content_length: int = 0, error: Exception = None):
        if success:
            self._print(f"Processed: {title[:50]}... ({content_length} chars)")
            self._print("Status: Content fetched successfully")
        else:
            self._print(f"Failed to fetch: {title[:50]}...")
            self._print(f"Status: Content fetch failed")
    
    def on_summary_generation_before(self, title: str):
        self._print(f"Status: Generating summary for '{title[:30]}...'")
        
    def on_summary_generation_after(self, title: str, success: bool, summary: str = ""):
        if success:
            self._debug_print(f"Generated summary for: {title[:50]}...")
            self._print("Status: Summary generated")
        else:
            self._print("Status: Summary generation failed")
    
    def on_document_storage_before(self, doc_id: str, storage_type: str):
        self._print(f"Status: Storing document ({storage_type})...")
        
    def on_document_storage_after(self, doc_id: str, storage_type: str, success: bool, error: Exception = None):
        if success:
            self.session_stats["documents_processed"] += 1
            if storage_type == "vertex_rag":
                self._debug_print(f"Stored in Vertex AI RAG: {doc_id}")
            else:
                self._debug_print(f"Stored locally: {doc_id}")
            self._print(f"Status: Document stored ({storage_type})")
        else:
            self._print(f"Status: Document storage failed ({storage_type})")
    
    def on_existing_documents_before(self, count: int):
        self._print("Status: Searching existing corpus...")
        
    def on_existing_documents_after(self, count: int):
        self._print(f"Found {count} existing documents in corpus")
        self._print(f"Status: Corpus search completed ({count} docs)")
    
    def on_synthesis_before(self, doc_count: int, use_rag: bool):
        rag_status = "with Vertex AI RAG" if use_rag else "without RAG"
        self._print(f"Synthesizing report from {doc_count} documents {rag_status}")
        self._print("Status: Starting report synthesis...")
        
    def on_synthesis_after(self, doc_count: int, use_rag: bool, success: bool):
        if success:
            self._print("Status: Report synthesis completed")
        else:
            self._print("Status: Report synthesis failed")
    
    def on_focused_research_before(self, focus_area: str, iteration: int, max_iterations: int):
        self._print(f"Status: Starting focused research on '{focus_area[:40]}...' (iteration {iteration + 1}/{max_iterations})")
        
    def on_focused_research_after(self, focus_area: str, doc_count: int):
        self._print(f"Status: Focused research completed ({doc_count} total documents)")
    
    def on_error(self, error: Exception, context: str):
        self.session_stats["errors"] += 1
        self._print(f"Error in {context}: {error}")
    
    def on_debug_message(self, message: str):
        self._debug_print(message)
    
    # Missing event handler methods
    def on_research_start(self, query: str, context: str = None):
        pass
    
    def on_research_complete(self, success: bool, result: str = None):
        pass
    
    def on_research_plan_created(self, steps: list):
        pass
    
    def on_research_step_start(self, step_num: int, research_step: str):
        pass
    
    def on_research_completeness_check(self, is_sufficient: bool, next_focus: str = None):
        pass
    
    def on_regeneration_start(self, query: str, use_rag: bool):
        pass
    
    def on_regeneration_complete(self, success: bool, result: str = None):
        pass
    
    def on_regeneration_documents_found(self, vertex_count: int, local_count: int, unique_count: int):
        pass
    
    def on_search_start(self, query: str):
        pass
    
    def on_source_evaluation(self, title: str, accepted: bool, score: float, reason: str):
        pass
    
    def on_content_fetch_start(self, title: str, url: str):
        pass
    
    def on_content_fetch_complete(self, title: str, success: bool, content_length: int = 0, error: Exception = None):
        pass
    
    def on_summary_generated(self, title: str, success: bool, summary: str = ""):
        pass
    
    def on_document_stored(self, doc_id: str, storage_type: str, success: bool, error: Exception = None):
        pass
    
    def on_existing_documents_found(self, count: int):
        pass
    
    def on_synthesis_start(self, doc_count: int, use_rag: bool):
        pass
    
    def on_research_after(self, success: bool, error: Optional[Exception] = None):
        if success:
            duration = time.time() - (self.session_stats["start_time"] or time.time())
            self._print(f"\nResearch completed in {duration:.1f} seconds")
        else:
            self._print(f"\nResearch failed: {error}")

    def on_regeneration_after(self, success: bool, error: Optional[Exception] = None):
        if success:
            duration = time.time() - (self.session_stats["start_time"] or time.time())
            self._print(f"\nSummary regenerated in {duration:.1f} seconds")
        else:
            self._print(f"\nSummary regeneration failed: {error}")

    def display_session_summary(self):
        """Display summary of the current session"""
        if self.session_stats["start_time"]:
            total_time = time.time() - self.session_stats["start_time"]
            self._print("\nSession Summary:")
            self._print(f"   Duration: {total_time:.1f} seconds")
            self._print(
                f"   Research queries: {self.session_stats['research_queries']}"
            )
            self._print(
                f"   Regeneration queries: {self.session_stats['regeneration_queries']}"
            )
            self._print(
                f"   Documents processed: {self.session_stats['documents_processed']}"
            )
            self._print(f"   Errors: {self.session_stats['errors']}")


class BasicResearchCLI:
    """Simple command-line research interface with Vertex AI RAG support"""

    def __init__(self, corpus_display_name: str = None):
        self.agent = None
        
        # Generate random corpus name if none provided
        if not corpus_display_name:
            corpus_display_name = self._generate_unique_corpus_name()
        
        self.config = ResearchConfig(corpus_display_name=corpus_display_name)
        self.event_subscriber = None
        self.logger_subscriber = None

    def _generate_unique_corpus_name(self) -> str:
        """Generate a unique corpus name that doesn't already exist"""
        try:
            # Create a temporary config to get project details for checking
            temp_config = ResearchConfig()
            
            # Create temporary VertexRagEngineAPI instance for checking existence
            temp_rag_engine = VertexRagEngineAPI(
                project_id=temp_config.project_id,
                location=temp_config.location,
                lazy_corpus_init=True
            )
            
            # Generate names until we find one that doesn't exist
            max_attempts = 100  # Prevent infinite loop
            for attempt in range(max_attempts):
                candidate_name = generate_random_corpus_name()
                if not temp_rag_engine.corpus_exists(candidate_name):
                    return candidate_name
            
            # If we somehow can't find a unique name after 100 attempts,
            # fall back to a timestamped name
            import time
            return f"research-corpus-{int(time.time())}"
            
        except Exception as e:
            # If there's any error with uniqueness checking, fall back to random name
            print(f"⚠️ Could not check corpus uniqueness ({e}), using random name")
            return generate_random_corpus_name()

    def setup_agent(self, debug_mode: bool = False, quiet_mode: bool = False) -> bool:
        """Initialize the research agent with Vertex AI RAG"""
        try:
            if not quiet_mode:
                print("🔧 Initializing research agent with Vertex AI RAG...")

            # Create event subscribers
            # Use structured logger for all events
            self.logger_subscriber = ResearchEventLogger(
                debug_mode=debug_mode
            )
            # Keep CLI subscriber for session stats and status updates
            self.event_subscriber = CLIEventSubscriber(
                debug_mode=debug_mode, quiet_mode=quiet_mode
            )

            # Create agent with Vertex AI configuration
            self.agent = DeepResearchAgent(
                serpapi_key=self.config.serpapi_key,
                project_id=self.config.project_id,
                location=self.config.location,
                corpus_display_name=self.config.corpus_display_name,
            )
            
            # Subscribe to agent events
            self.event_subscriber.subscribe_to_agent(self.agent)
            if self.logger_subscriber:
                self.logger_subscriber.subscribe_to_agent(self.agent)

            if not quiet_mode:
                print("Research agent initialized successfully")
                print(
                    f"📁 Using Vertex AI RAG corpus: {self.config.corpus_display_name}"
                )
                print(
                    f"🌍 Project: {self.config.project_id}, Location: {self.config.location}"
                )
            return True

        except Exception as e:
            print(f"Failed to initialize agent: {e}")
            if not quiet_mode:
                print("Check your environment variables:")
                print("  - GEMINI_API_KEY")
                print("  - SERPAPI_KEY")
                print("  - GOOGLE_CLOUD_PROJECT")
                print("  - VERTEX_AI_LOCATION (optional)")
                print("And provide the RAG corpus name via:")
                print("  - --corpus argument (recommended)")
                print("  - RAG_CORPUS_NAME environment variable")
                print("\nAlso ensure Google Cloud authentication is set up:")
                print("  gcloud auth application-default login")
            return False

    def gather_clarifications(self, query: str) -> Optional[str]:
        """Gather clarifying information from the user"""
        print("\nAnalyzing query for clarification needs...")

        try:
            # Generate clarifying questions
            questions = self.agent.llm.generate_clarifying_questions(query)

            if not questions:
                print("Query is clear - proceeding with research")
                return None

            # Use state-based logger if available
            if self.logger_subscriber and hasattr(self.logger_subscriber, 'logger'):
                self.logger_subscriber.logger.add_clarification_questions(questions)
            else:
                print(
                    f"\nFound {len(questions)} aspects that could benefit from clarification:"
                )
                print("=" * 50)

            responses = []
            for i, question in enumerate(questions, 1):
                if not (self.logger_subscriber and hasattr(self.logger_subscriber, 'logger')):
                    print(f"\n{i}. {question}")

                while True:
                    response = input("   Your answer: ").strip()
                    if response:
                        if self.logger_subscriber and hasattr(self.logger_subscriber, 'logger'):
                            self.logger_subscriber.logger.add_clarification_response(question, response)
                        responses.append(f"Q{i}: {question}\nA{i}: {response}")
                        break
                    else:
                        print("   Please provide an answer")

            # Format context
            context = f"""Original Query: {query}
            
                        Clarification Dialogue:
                        {chr(10).join(responses)}

                        Based on this clarification, please create a research plan that addresses the specific aspects mentioned above."""

            return context

        except Exception as e:
            print(f"⚠️ Error during clarification: {e}")
            print("Proceeding with original query...")
            return None

    async def conduct_research(
        self, query: str, save_report: bool = True
    ) -> Optional[str]:
        """Conduct research on the given query with clarification"""

        context = self.gather_clarifications(query)

        # Structured logging will handle all output automatically
        # The callbacks will handle all the progress reporting
        start_time = time.time()

        try:
            # Run the research with context
            result = await self.agent.research_with_context(query, context)

            duration = time.time() - start_time

            # Get statistics for reporting
            stats = self.agent.get_performance_stats()
            corpus_summary = self.agent.get_corpus_summary()

            # Display enhanced statistics
            vertex_info = corpus_summary.get("vertex_ai_rag", {})
            local_info = corpus_summary.get("local_backup", {})

            print(f"Vertex AI RAG documents: {vertex_info.get('file_count', 0)}")
            print(
                f"Local documents collected: {local_info.get('local_documents', 0)}"
            )
            print(f"Queries processed: {stats.get('queries_processed', 0)}")
            print(f"Success rate: {stats.get('success_rate', 0):.1%}")

            # Save report if requested
            if save_report:
                filename = self.save_report(query, result, duration, context)
                if filename:
                    print(f"Report saved to: {filename}")

            return result

        except Exception as e:
            print(f"Research failed: {e}")
            return None
        
        finally:
            # Logging is automatic, no cleanup needed
            pass

    async def regenerate_summary(
        self, query: str, save_report: bool = True, use_rag: bool = True
    ) -> Optional[str]:
        """Regenerate summary from existing corpus without new research"""

        # Structured logging will handle all output automatically

        start_time = time.time()

        try:
            # Run the regeneration - callbacks will handle progress reporting
            result = await self.agent.regenerate_summary(query, use_rag)

            duration = time.time() - start_time

            # Get corpus information for reporting
            corpus_summary = self.agent.get_corpus_summary()
            vertex_info = corpus_summary.get("vertex_ai_rag", {})
            local_info = corpus_summary.get("local_backup", {})

            print(f"Used Vertex AI RAG: {use_rag}")
            print(f"Vertex AI RAG documents: {vertex_info.get('file_count', 0)}")
            print(
                f"Local documents available: {local_info.get('local_documents', 0)}"
            )

            # Save report if requested
            if save_report:
                filename = self.save_report(
                    f"REGENERATED: {query}", result, duration, regenerated=True
                )
                if filename:
                    print(f"Regenerated report saved to: {filename}")

            return result

        except Exception as e:
            print(f"Summary regeneration failed: {e}")
            return None
        
        finally:
            # Logging is automatic, no cleanup needed
            pass

    def save_report(
        self,
        query: str,
        report: str,
        duration: float,
        context: str = None,
        regenerated: bool = False,
    ) -> Optional[str]:
        """Save research report to file with clarification context"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create safe filename from query
            safe_query = "".join(
                c for c in query if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()[:50]
            safe_query = safe_query.replace(" ", "_")

            if regenerated:
                filename = f"regenerated_{safe_query}_{timestamp}.md"
            else:
                filename = f"research_{safe_query}_{timestamp}.md"

            with open(filename, "w", encoding="utf-8") as f:
                f.write("# Research Report\n\n")
                f.write(f"**Query:** {query}\n")
                f.write(
                    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"**Duration:** {duration:.1f} seconds\n")
                f.write("**RAG Engine:** Vertex AI\n")
                if regenerated:
                    f.write("**Type:** Regenerated Summary\n")
                f.write("\n")

                if context:
                    f.write("## Research Clarifications\n\n")
                    # Extract just the Q&A pairs for the report
                    lines = context.split("\n")
                    in_dialogue = False
                    for line in lines:
                        if "Clarification Dialogue:" in line:
                            in_dialogue = True
                            continue
                        elif in_dialogue and line.strip():
                            if line.startswith("Q") or line.startswith("A"):
                                f.write(f"{line}\n")
                    f.write("\n")

                f.write("---\n\n")
                f.write(report)

            return filename

        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
            return None

    def display_report(self, report: str, max_lines: int = 50):
        """Display report with optional truncation"""
        print("\n📄 Research Report:")
        print("=" * 60)

        lines = report.split("\n")

        for i, line in enumerate(lines):
            if i >= max_lines:
                remaining = len(lines) - i
                print(f"\n... ({remaining} more lines)")
                print("💡 Use --full to see complete report or check saved file")
                break
            print(line)

    def show_corpus_info(self):
        """Display information about the current corpus"""
        if not self.agent:
            print("Agent not initialized")
            return

        try:
            summary = self.agent.get_corpus_summary()
            vertex_info = summary.get("vertex_ai_rag", {})
            local_info = summary.get("local_backup", {})

            print("\n📚 Corpus Information:")
            print("=" * 40)

            print("\n🔹 Vertex AI RAG Corpus:")
            print(f"  Name: {vertex_info.get('display_name', 'Unknown')}")
            print(f"  Files: {vertex_info.get('file_count', 0)}")
            print(f"  Created: {vertex_info.get('create_time', 'Unknown')}")

            print("\n🔹 Local Backup:")
            print(f"  Documents: {local_info.get('local_documents', 0)}")
            print(f"  Avg Relevance: {local_info.get('average_relevance', 0):.2f}")

            if local_info.get("sources_by_domain"):
                print("  Top Domains:")
                for domain, count in list(local_info["sources_by_domain"].items())[:5]:
                    print(f"    {domain}: {count}")

        except Exception as e:
            print(f"Failed to get corpus info: {e}")

    def delete_corpus(self) -> bool:
        """Delete the current corpus with confirmation"""
        if not self.agent:
            print("Agent not initialized")
            return False

        try:
            corpus_name = self.config.corpus_display_name
            print(f"⚠️  About to delete corpus: '{corpus_name}'")
            print("This action cannot be undone and will permanently remove all documents from the corpus.")
            
            confirmation = input(f"Type '{corpus_name}' to confirm deletion: ").strip()
            
            if confirmation == corpus_name:
                print("🗑️  Deleting corpus...")
                success = self.agent.delete_vertex_corpus()
                if success:
                    print(f"✅ Successfully deleted corpus: '{corpus_name}'")
                    return True
                else:
                    print(f"❌ Failed to delete corpus: '{corpus_name}'")
                    return False
            else:
                print("❌ Deletion cancelled - corpus name did not match")
                return False
                
        except Exception as e:
            print(f"Failed to delete corpus: {e}")
            return False

    async def interactive_mode(self, debug_mode: bool = False):
        """Run in interactive mode with clarification support"""
        print(f"🔬 Deep Research Agent - Interactive Mode (Vertex AI RAG)")
        print("=" * 60)
        print("Commands:")
        print("  <query>                    - Run research with clarification")
        print("  regenerate <query>         - Regenerate summary from existing corpus")
        print("  regenerate-no-rag <query>  - Regenerate without Vertex AI RAG")
        print("  corpus-info                - Show corpus information")
        print("  delete-corpus              - Delete current corpus (closes session)")
        print("  quit                       - Exit")
        print()

        session_count = 0

        while True:
            try:
                # Get query from user
                user_input = input("> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if user_input.lower() == "corpus-info":
                    self.show_corpus_info()
                    continue
                    
                if user_input.lower() == "delete-corpus":
                    success = self.delete_corpus()
                    if success:
                        print("\n🔒 Session ending due to corpus deletion.")
                        print("Commands like 'regenerate' will no longer work.")
                        break
                    continue

                if not user_input:
                    print("⚠️ Please enter a query or command")
                    continue

                # Check for different command types
                regenerate_mode = False
                use_rag = True
                query = user_input

                if user_input.lower().startswith("regenerate-no-rag "):
                    regenerate_mode = True
                    use_rag = False
                    query = user_input[18:].strip()  # Remove the prefix
                elif user_input.lower().startswith("regenerate "):
                    regenerate_mode = True
                    use_rag = True
                    query = user_input[11:].strip()  # Remove the prefix

                if not query:
                    print("⚠️ Please provide a query after the command")
                    continue

                session_count += 1
                print(f"\n📋 Session #{session_count}")

                # Run research or regeneration
                if regenerate_mode:
                    result = await self.regenerate_summary(query, use_rag=use_rag)
                else:
                    result = await self.conduct_research(query)

                if result:
                    # Ask if user wants to see the report
                    show = input("\nDisplay report? (y/n): ").lower()
                    if show in ["y", "yes"]:
                        self.display_report(result)

                print("\n" + "=" * 60)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        # Display session summary using event subscriber - wrap in try-except for graceful exit
        try:
            if self.event_subscriber:
                self.event_subscriber.display_session_summary()

            print(
                f"\n👋 Interactive session ended. Completed {session_count} research queries."
            )
            print("🗄️ Documents remain stored in Vertex AI RAG corpus for future sessions.")
        except KeyboardInterrupt:
            # User pressed Ctrl+C during cleanup - exit silently
            pass

    async def interactive_mode_async(self, debug_mode: bool = False):
        """Async wrapper for interactive mode to handle async research calls"""
        # Need to handle this differently since we can't use await in the regular interactive_mode
        # This is a simplified version focusing on the CLI aspects
        await self.interactive_mode(debug_mode)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - AI-powered research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                  %(prog)s "Latest developments in quantum computing"
                  %(prog)s --interactive
                  %(prog)s "AI safety research 2024" --no-save --full
                  %(prog)s --query "Climate change solutions" --output report.md
                  %(prog)s --regenerate "quantum computing trends"
        """,
    )

    # Query input options
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("query", nargs="?", help="Research query to process")
    query_group.add_argument(
        "--query", "-q", help="Research query (alternative to positional argument)"
    )
    query_group.add_argument(
        "--regenerate",
        "-r",
        help="Regenerate summary from existing corpus for given query",
    )
    query_group.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )

    # Output options
    parser.add_argument(
        "--output", "-o", help="Output file for the report (default: auto-generated)"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save report to file"
    )
    parser.add_argument(
        "--full", action="store_true", help="Display full report (don't truncate)"
    )
    parser.add_argument(
        "--quiet", "-k", action="store_true", help="Minimal output (just show results)"
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode with verbose output",
    )

    # Regeneration options
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Don't use Vertex AI RAG for regeneration (use local synthesis only)",
    )

    # RAG corpus configuration
    parser.add_argument(
        "--corpus", "-c", help="Name of the Vertex AI RAG corpus to use"
    )
    parser.add_argument(
        "--delete-corpus", "-D", help="Delete the specified corpus (requires corpus name)"
    )


    # Debug options
    parser.add_argument(
        "--check-config", action="store_true", help="Check configuration and exit"
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = BasicResearchCLI(corpus_display_name=args.corpus)

    # Check configuration only
    if args.check_config:
        print("🔧 Checking configuration...")
        try:
            cli.config._validate_config()
            print("Configuration is valid")
            return 0
        except Exception as e:
            print(f"Configuration has issues: {e}")
            return 1

    # Setup agent
    if not cli.setup_agent(debug_mode=args.debug, quiet_mode=args.quiet):
        return 1

    # Delete corpus mode
    if args.delete_corpus:
        if not args.corpus:
            print("Error: --delete-corpus requires --corpus to specify which corpus to delete")
            return 1
        
        if args.delete_corpus != args.corpus:
            print(f"Error: --delete-corpus value '{args.delete_corpus}' must match --corpus value '{args.corpus}'")
            return 1
            
        try:
            print("🗑️  Deep Research Agent - Delete Corpus")
            print("=" * 40)
            
            success = cli.delete_corpus()
            if success:
                print("\\n✅ Corpus deletion completed successfully")
                return 0
            else:
                print("\\n❌ Corpus deletion failed")
                return 1
                
        except KeyboardInterrupt:
            print("\\n👋 Deletion cancelled by user")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    # Interactive mode
    if args.interactive:
        try:
            asyncio.run(cli.interactive_mode(debug_mode=args.debug))
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully - don't show stacktrace
            print("\n👋 Goodbye!")
        return 0

    # Regenerate mode
    if args.regenerate:
        try:
            if not args.quiet:
                print("Deep Research Agent - Regenerate Summary")
                print("=" * 40)

            # Regenerate summary
            use_rag = not args.no_rag
            result = asyncio.run(
                cli.regenerate_summary(
                    args.regenerate, save_report=not args.no_save, use_rag=use_rag
                )
            )

            if not result:
                return 1

            # Display results
            if not args.quiet:
                max_lines = None if args.full else 50
                cli.display_report(result, max_lines)
            else:
                print(result)

            return 0

        except KeyboardInterrupt:
            print("\n👋 Regeneration cancelled by user")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

    # Get query for regular research
    query = args.query or args.query
    if not query:
        print("No query provided")
        print("Use --help for usage information")
        return 1

    # Run research
    try:
        if not args.quiet:
            print("🔬 Deep Research Agent")
            print("=" * 30)

        # Conduct research
        result = asyncio.run(cli.conduct_research(query, save_report=not args.no_save))

        if not result:
            return 1

        # Display results
        if not args.quiet:
            max_lines = None if args.full else 50
            cli.display_report(result, max_lines)
        else:
            print(result)

        return 0

    except KeyboardInterrupt:
        print("\n👋 Research cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
