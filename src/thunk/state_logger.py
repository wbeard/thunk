"""
State-based Structured Logger for Deep Research Agent
Builds complete display state and re-renders instead of streaming events
"""

import os
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    """Log levels with corresponding colors"""
    QUESTION = "question"      # Cyan - main research question
    PLAN = "plan"             # Blue - research planning
    STEP = "step"             # Magenta - research steps
    SEARCH = "search"         # Yellow - search operations
    RESULT = "result"         # Green/Red - results and evaluations
    CONTENT = "content"       # White - content operations
    SYNTHESIS = "synthesis"   # Cyan - synthesis phase
    ERROR = "error"           # Red - errors
    SUCCESS = "success"       # Green - success messages
    INFO = "info"             # Default - general info


class Colors:
    """ANSI color codes"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


@dataclass
class SourceState:
    """State of a source during evaluation and fetching"""
    title: str
    accepted: Optional[bool] = None
    score: Optional[float] = None
    reason: str = ""
    fetching: bool = False
    fetch_complete: bool = False
    fetch_success: bool = False
    content_length: int = 0
    summary_generated: bool = False
    document_stored: bool = False
    storage_type: str = ""


@dataclass
class SearchState:
    """State of a search operation"""
    query: str
    results_count: Optional[int] = None
    no_results: bool = False
    sources: List[SourceState] = field(default_factory=list)
    evaluation_started: bool = False


@dataclass
class ResearchStepState:
    """State of a research step"""
    step_num: int
    name: str
    searches: List[SearchState] = field(default_factory=list)
    completed: bool = False


@dataclass
class ResearchState:
    """Complete state of the research process"""
    # Basic info
    query: str = ""
    start_time: Optional[float] = None
    
    # Clarification
    clarification_questions: List[str] = field(default_factory=list)
    clarification_responses: Dict[str, str] = field(default_factory=dict)
    
    # Research plan
    research_steps: List[ResearchStepState] = field(default_factory=list)
    
    # Synthesis
    synthesis_started: bool = False
    synthesis_doc_count: int = 0
    synthesis_use_rag: bool = True
    synthesis_progress: List[str] = field(default_factory=list)
    
    # Completion
    completed: bool = False
    success: bool = False
    duration: Optional[float] = None
    error_message: str = ""


class StateBasedLogger:
    """State-based logger that renders complete display"""
    
    def __init__(self, use_colors: bool = True, auto_render: bool = True):
        self.use_colors = use_colors
        self.auto_render = auto_render
        self.state = ResearchState()
        
        # Color mapping
        self.color_map = {
            LogLevel.QUESTION: Colors.BRIGHT_CYAN,
            LogLevel.PLAN: Colors.BRIGHT_BLUE,
            LogLevel.STEP: Colors.BRIGHT_MAGENTA,
            LogLevel.SEARCH: Colors.BRIGHT_YELLOW,
            LogLevel.RESULT: Colors.WHITE,
            LogLevel.CONTENT: Colors.WHITE,
            LogLevel.SYNTHESIS: Colors.BRIGHT_CYAN,
            LogLevel.ERROR: Colors.BRIGHT_RED,
            LogLevel.SUCCESS: Colors.BRIGHT_GREEN,
            LogLevel.INFO: Colors.WHITE,
        }
    
    def _colorize(self, text: str, level: LogLevel) -> str:
        """Apply color to text based on log level"""
        if not self.use_colors:
            return text
        
        color = self.color_map.get(level, Colors.WHITE)
        return f"{color}{text}{Colors.RESET}"
    
    def _get_indent_chars(self, level: int) -> str:
        """Get proper indentation characters for nesting level"""
        if level == 0:
            return ""
        
        chars = []
        for i in range(level):
            if i == level - 1:
                chars.append("⎿ ")
            else:
                chars.append("  ")
        return "".join(chars)
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _build_display_lines(self) -> List[str]:
        """Build the complete display from current state"""
        lines = []
        
        # Research question
        if self.state.query:
            lines.append(self._colorize(f"🔍 Research Question: {self.state.query}", LogLevel.QUESTION))
            lines.append("")
        
        # Clarification questions
        if self.state.clarification_questions:
            lines.append(self._colorize("💭 Clarification Questions:", LogLevel.QUESTION))
            for i, question in enumerate(self.state.clarification_questions, 1):
                lines.append(self._get_indent_chars(1) + self._colorize(f"{i}. {question}", LogLevel.INFO))
                if question in self.state.clarification_responses:
                    response = self.state.clarification_responses[question]
                    lines.append(self._get_indent_chars(1) + self._colorize(f"A: {response}", LogLevel.SUCCESS))
            lines.append("")
        
        # Research plan
        if self.state.research_steps:
            step_names = [step.name for step in self.state.research_steps]
            lines.append(self._colorize(f"📋 Research Plan Created ({len(step_names)} steps):", LogLevel.PLAN))
            for i, name in enumerate(step_names, 1):
                lines.append(self._get_indent_chars(1) + self._colorize(f"Step {i}: {name}", LogLevel.STEP))
            lines.append("")
        
        # Research steps execution
        for step in self.state.research_steps:
            lines.append(self._colorize(f"🔬 Executing Step {step.step_num}: {step.name}", LogLevel.STEP))
            
            for search in step.searches:
                lines.append(self._get_indent_chars(1) + self._colorize(f"🔍 Search: \"{search.query}\"", LogLevel.SEARCH))
                
                if search.no_results:
                    lines.append(self._get_indent_chars(2) + self._colorize("No results found", LogLevel.ERROR))
                elif search.results_count is not None:
                    lines.append(self._get_indent_chars(2) + self._colorize(f"Found {search.results_count} results", LogLevel.RESULT))
                    
                    if search.evaluation_started and search.sources:
                        lines.append(self._get_indent_chars(3) + self._colorize("📊 Evaluating sources:", LogLevel.RESULT))
                        
                        for source in search.sources:
                            # Source evaluation result
                            if source.accepted is not None:
                                if source.accepted:
                                    icon = "✅"
                                    level = LogLevel.SUCCESS
                                    score_text = f"(score: {source.score:.2f})"
                                else:
                                    icon = "❌"
                                    level = LogLevel.ERROR
                                    score_text = f"(score: {source.score:.2f}) - {source.reason}"
                                
                                truncated_title = source.title[:60] + "..." if len(source.title) > 60 else source.title
                                lines.append(self._get_indent_chars(4) + self._colorize(f"{icon} \"{truncated_title}\" {score_text}", level))
                                
                                # Content fetching for accepted sources
                                if source.accepted:
                                    if source.fetching or source.fetch_complete:
                                        truncated_fetch_title = source.title[:50] + "..." if len(source.title) > 50 else source.title
                                        
                                        if source.fetch_complete:
                                            if source.fetch_success:
                                                lines.append(self._get_indent_chars(5) + self._colorize(f"📥 Fetched: \"{truncated_fetch_title}\"", LogLevel.CONTENT))
                                                lines.append(self._get_indent_chars(6) + self._colorize(f"✅ Stored ({source.content_length:,} chars)", LogLevel.SUCCESS))
                                            else:
                                                lines.append(self._get_indent_chars(5) + self._colorize(f"📥 Fetched: \"{truncated_fetch_title}\"", LogLevel.CONTENT))
                                                lines.append(self._get_indent_chars(6) + self._colorize("❌ Failed to fetch", LogLevel.ERROR))
                                        else:
                                            lines.append(self._get_indent_chars(5) + self._colorize(f"📥 Fetching: \"{truncated_fetch_title}\"", LogLevel.CONTENT))
            
            lines.append("")
        
        # Synthesis
        if self.state.synthesis_started:
            rag_status = "with Vertex AI RAG" if self.state.synthesis_use_rag else "without RAG"
            lines.append(self._colorize(f"📝 Synthesizing report from {self.state.synthesis_doc_count} documents {rag_status}", LogLevel.SYNTHESIS))
            
            for progress in self.state.synthesis_progress:
                lines.append(self._get_indent_chars(1) + self._colorize(progress, LogLevel.INFO))
            
            lines.append("")
        
        # Completion
        if self.state.completed:
            if self.state.success:
                lines.append(self._colorize(f"✅ Research completed successfully in {self.state.duration:.1f} seconds", LogLevel.SUCCESS))
            else:
                lines.append(self._colorize(f"❌ Research failed: {self.state.error_message}", LogLevel.ERROR))
        
        return lines
    
    def render(self):
        """Render the complete display"""
        self._clear_screen()
        lines = self._build_display_lines()
        for line in lines:
            print(line)
    
    def _update_and_render(self):
        """Update display if auto-render is enabled"""
        if self.auto_render:
            self.render()
    
    # Research state update methods
    
    def research_start(self, query: str):
        """Start research"""
        self.state.query = query
        self.state.start_time = time.time()
        self._update_and_render()
    
    def add_clarification_questions(self, questions: List[str]):
        """Add clarification questions"""
        self.state.clarification_questions = questions
        self._update_and_render()
    
    def add_clarification_response(self, question: str, response: str):
        """Add clarification response"""
        self.state.clarification_responses[question] = response
        self._update_and_render()
    
    def set_research_plan(self, steps: List[str]):
        """Set research plan"""
        self.state.research_steps = [
            ResearchStepState(step_num=i+1, name=step) 
            for i, step in enumerate(steps)
        ]
        self._update_and_render()
    
    def start_research_step(self, step_num: int, step_name: str):
        """Start a research step"""
        # Find or create the step
        step = None
        for s in self.state.research_steps:
            if s.step_num == step_num:
                step = s
                break
        
        if not step:
            step = ResearchStepState(step_num=step_num, name=step_name)
            self.state.research_steps.append(step)
        
        self._update_and_render()
    
    def start_search(self, query: str, step_num: int):
        """Start a search within a research step"""
        # Find the step
        step = None
        for s in self.state.research_steps:
            if s.step_num == step_num:
                step = s
                break
        
        if step:
            search = SearchState(query=query)
            step.searches.append(search)
            self._update_and_render()
            return search
        return None
    
    def set_search_results(self, query: str, step_num: int, count: int):
        """Set search results count"""
        step = self._find_step(step_num)
        if step:
            search = self._find_search(step, query)
            if search:
                search.results_count = count
                search.evaluation_started = True
                self._update_and_render()
    
    def set_search_no_results(self, query: str, step_num: int):
        """Set search as having no results"""
        step = self._find_step(step_num)
        if step:
            search = self._find_search(step, query)
            if search:
                search.no_results = True
                self._update_and_render()
    
    def add_source_evaluation(self, query: str, step_num: int, title: str, accepted: bool, score: float, reason: str = ""):
        """Add source evaluation result"""
        step = self._find_step(step_num)
        if step:
            search = self._find_search(step, query)
            if search:
                source = SourceState(title=title, accepted=accepted, score=score, reason=reason)
                search.sources.append(source)
                self._update_and_render()
                return source
        return None
    
    def start_content_fetch(self, title: str, step_num: int):
        """Start content fetching for a source"""
        source = self._find_source_by_title(step_num, title)
        if source:
            source.fetching = True
            self._update_and_render()
    
    def complete_content_fetch(self, title: str, step_num: int, success: bool, content_length: int = 0):
        """Complete content fetching for a source"""
        source = self._find_source_by_title(step_num, title)
        if source:
            source.fetching = False
            source.fetch_complete = True
            source.fetch_success = success
            source.content_length = content_length
            self._update_and_render()
    
    def start_synthesis(self, doc_count: int, use_rag: bool):
        """Start synthesis phase"""
        self.state.synthesis_started = True
        self.state.synthesis_doc_count = doc_count
        self.state.synthesis_use_rag = use_rag
        self._update_and_render()
    
    def add_synthesis_progress(self, activity: str):
        """Add synthesis progress"""
        self.state.synthesis_progress.append(activity)
        self._update_and_render()
    
    def complete_research(self, success: bool, duration: float, error: str = ""):
        """Complete research"""
        self.state.completed = True
        self.state.success = success
        self.state.duration = duration
        self.state.error_message = error
        self._update_and_render()
    
    # Helper methods
    
    def _find_step(self, step_num: int) -> Optional[ResearchStepState]:
        """Find research step by number"""
        for step in self.state.research_steps:
            if step.step_num == step_num:
                return step
        return None
    
    def _find_search(self, step: ResearchStepState, query: str) -> Optional[SearchState]:
        """Find search by query within a step"""
        for search in step.searches:
            if search.query == query:
                return search
        return None
    
    def _find_source_by_title(self, step_num: int, title: str) -> Optional[SourceState]:
        """Find source by title across all searches in a step"""
        step = self._find_step(step_num)
        if step:
            for search in step.searches:
                for source in search.sources:
                    if source.title == title:
                        return source
        return None


class ResearchEventLogger:
    """Event subscriber that updates state-based logger"""
    
    def __init__(self, debug_mode: bool = False, use_colors: bool = True):
        self.logger = StateBasedLogger(use_colors=use_colors)
        self.debug_mode = debug_mode
        self.research_query = ""
        self.start_time = None
        
        # Track current context
        self.current_step_num = 0
        self.current_search_query = ""
    
    def subscribe_to_agent(self, agent):
        """Subscribe to all relevant events from the research agent"""
        # Research lifecycle events
        agent.subscribe('research_start', self.on_research_start)
        agent.subscribe('research_complete', self.on_research_complete)
        agent.subscribe('research_plan_created', self.on_research_plan_created)
        agent.subscribe('research_step_start', self.on_research_step_start)
        
        # Search and content events
        agent.subscribe('search_start', self.on_search_start)
        agent.subscribe('search_results_found', self.on_search_results_found)
        agent.subscribe('search_no_results', self.on_search_no_results)
        agent.subscribe('source_evaluation', self.on_source_evaluation)
        agent.subscribe('content_fetch_start', self.on_content_fetch_start)
        agent.subscribe('content_fetch_complete', self.on_content_fetch_complete)
        
        # Synthesis events
        agent.subscribe('synthesis_start', self.on_synthesis_start)
        
        # Regeneration events
        agent.subscribe('regeneration_start', self.on_regeneration_start)
        agent.subscribe('regeneration_complete', self.on_regeneration_complete)
        
        # Error events
        agent.subscribe('error', self.on_error)
    
    def on_research_start(self, query: str, context: Optional[str] = None):
        """Research started"""
        self.research_query = query
        self.start_time = time.time()
        self.logger.research_start(query)
    
    def on_regeneration_start(self, query: str, use_rag: bool):
        """Regeneration started"""
        self.research_query = query
        self.start_time = time.time()
        self.logger.research_start(f"🔄 Regenerating: {query}")
    
    def on_research_plan_created(self, research_plan, steps: List[str]):
        """Research plan created"""
        self.logger.set_research_plan(steps)
    
    def on_research_step_start(self, step_num: int, research_step: str):
        """Research step started"""
        self.current_step_num = step_num
        self.logger.start_research_step(step_num, research_step)
    
    def on_search_start(self, query: str):
        """Search started"""
        self.current_search_query = query
        self.logger.start_search(query, self.current_step_num)
    
    def on_search_results_found(self, count: int, query: str):
        """Search results found"""
        self.logger.set_search_results(query, self.current_step_num, count)
    
    def on_search_no_results(self):
        """No search results found"""
        self.logger.set_search_no_results(self.current_search_query, self.current_step_num)
    
    def on_source_evaluation(self, title: str, accepted: bool, score: float, reason: str):
        """Source evaluated"""
        self.logger.add_source_evaluation(self.current_search_query, self.current_step_num, title, accepted, score, reason)
    
    def on_content_fetch_start(self, title: str, url: str):
        """Content fetch started"""
        self.logger.start_content_fetch(title, self.current_step_num)
    
    def on_content_fetch_complete(self, title: str, success: bool, content_length: int = 0):
        """Content fetch completed"""
        self.logger.complete_content_fetch(title, self.current_step_num, success, content_length)
    
    def on_synthesis_start(self, doc_count: int, use_rag: bool):
        """Synthesis started"""
        self.logger.start_synthesis(doc_count, use_rag)
    
    def on_research_complete(self, success: bool, error: Optional[Exception] = None):
        """Research completed"""
        duration = time.time() - (self.start_time or time.time())
        error_msg = str(error) if error else ""
        self.logger.complete_research(success, duration, error_msg)
    
    def on_regeneration_complete(self, success: bool, error: Optional[Exception] = None):
        """Regeneration completed"""
        duration = time.time() - (self.start_time or time.time())
        error_msg = str(error) if error else ""
        self.logger.complete_research(success, duration, error_msg)
    
    def on_error(self, error: Exception, context: str):
        """Error occurred"""
        # For now, just complete with error
        duration = time.time() - (self.start_time or time.time()) if self.start_time else 0
        self.logger.complete_research(False, duration, f"{context}: {error}")