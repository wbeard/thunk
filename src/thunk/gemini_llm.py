import re
from typing import List, Optional, Tuple
import google.genai as genai
import logging

from .types import (
    ResearchPlan,
    SearchResult,
    Document,
)

logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class GeminiLLM:
    """Wrapper for Gemini API with research-specific methods"""

    def __init__(
        self,
        project_name: str,
        location: str,
        model_name: str = "gemini-2.5-flash-preview-05-20",
    ):
        self.client = genai.Client(
            vertexai=True, project=project_name, location=location
        )
        self.model = model_name
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

    def generate(self, prompt: str, temperature: float = None) -> str:
        """Generate text with optional temperature override"""
        config = self.generation_config.copy()
        if temperature is not None:
            config["temperature"] = temperature

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception:
            return ""

    def plan_research(self, query: str, context: str = None) -> ResearchPlan:
        """Generate a research plan for the given query with optional context"""
        base_prompt = f"""You are an expert research assistant. Break down the following research query into a structured plan.

                      Query: "{query}"
                      """

        if context:
            base_prompt += f"""
                      
                      Additional Context from clarification:
                      {context}
                      
                      Use this context to make your research plan more targeted and specific.
                      """

        base_prompt += """
                      Provide a numbered list of 3-5 specific research steps that would comprehensively address this query. Each step should be actionable and focused on gathering specific types of information.

                      Format your response as:
                      1. [Research step 1]
                      2. [Research step 2]
                      3. [Research step 3]
                      ...

                      Focus on covering different aspects like: current trends, key players, technical details, market analysis, recent developments, challenges, and future outlook as applicable.
                  """

        response = self.generate(base_prompt, temperature=0.5)

        # Parse the response to extract steps
        steps = []
        for line in response.split("\n"):
            line = line.strip()
            if re.match(r"^\d+\.", line):
                step = re.sub(r"^\d+\.\s*", "", line)
                if step:
                    steps.append(step)

        return ResearchPlan(query=query, steps=steps if steps else [query])

    def generate_search_queries(self, research_step: str) -> List[str]:
        """Generate specific search queries for a research step"""
        prompt = f"""Generate 2-3 specific, focused search queries that would help find information about: "{research_step}"

                  Make the queries:
                  - Specific and targeted
                  - Likely to return relevant, authoritative sources
                  - Varied in approach (e.g., technical, market, recent news)

                  Return only the search queries, one per line."""

        response = self.generate(prompt, temperature=0.4)
        queries = [q.strip() for q in response.split("\n") if q.strip()]
        return queries[:3]  # Limit to 3 queries

    def evaluate_source_relevance(
        self, query: str, result: SearchResult
    ) -> Tuple[str, float]:
        """Evaluate if a search result is relevant and high-quality"""
        prompt = f"""Evaluate this search result for the research query.

                      Query: "{query}"
                      Result Title: "{result.title}"
                      Result Snippet: "{result.snippet}"
                      Domain: "{result.domain}"

                      Rate this source on relevance and quality. Consider:
                      - How well does it match the query?
                      - Is the source likely to be authoritative?
                      - Does the snippet suggest substantive content?

                      Respond with ONLY:
                      HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, or IRRELEVANT
                      Then a score from 0.0 to 1.0
                      Then a brief reason

                      Format: RATING|SCORE|REASON"""

        response = self.generate(prompt, temperature=0.1)

        try:
            parts = response.strip().split("|", 2)
            rating = parts[0].strip()
            score = float(parts[1].strip())
            reason = parts[2].strip() if len(parts) > 2 else ""

            return rating, score, reason
        except Exception:
            return "SOMEWHAT_RELEVANT", 0.5

    def summarize_content(
        self, content: str, focus_query: str, source_title: str
    ) -> str:
        """Summarize content with focus on the research query"""
        prompt = f"""Summarize the following content, focusing specifically on information relevant to: "{focus_query}"

                    Source: {source_title}

                    Extract and summarize:
                    - Key facts and findings
                    - Important statistics or data
                    - Main conclusions or insights
                    - Any relevant quotes or specific statements

                    Keep the summary concise but comprehensive. Include attribution where possible (e.g., "According to [Source]...").

                    Content:
                    {content[:8000]}  # Limit content length

                    Provide a focused summary in 3-5 bullet points."""

        return self.generate(prompt, temperature=0.2)

    def assess_research_completeness(
        self, query: str, collected_info: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Assess if enough information has been gathered"""
        info_summary = "\n".join(
            [f"- {info}" for info in collected_info[-10:]]
        )  # Last 10 pieces

        prompt = f"""Assess whether we have sufficient information to comprehensively answer this research query.

                      Original Query: "{query}"

                      Information Gathered:
                      {info_summary}

                      Evaluate:
                      1. Have we covered the main aspects of the query?
                      2. Is there sufficient depth and breadth of information?
                      3. Are there any critical gaps that need more research?

                      Respond with:
                      SUFFICIENT if we can provide a comprehensive answer
                      NEEDS_MORE followed by what specific aspect needs more research

                      Format: STATUS|NEXT_FOCUS (if NEEDS_MORE)"""

        response = self.generate(prompt, temperature=0.2)

        try:
            parts = response.strip().split("|", 1)
            status = parts[0].strip()

            if status == "SUFFICIENT":
                return True, None
            else:
                next_focus = parts[1].strip() if len(parts) > 1 else None
                return False, next_focus
        except Exception:
            return True, None  # Default to sufficient

    def synthesize_final_report(self, query: str, documents: List[Document]) -> str:
        """Generate the final research report"""
        # Prepare document summaries
        doc_summaries = []
        for i, doc in enumerate(documents, 1):
            doc_summaries.append(
                f"[{i}] {doc.title}\nSource: {doc.url}\nSummary: {doc.summary}\n"
            )

        summaries_text = "\n".join(doc_summaries)

        prompt = f"""Write a comprehensive research report addressing the query: "{query}"

                      Use the following research findings to create a well-structured, insightful report:

                      {summaries_text}

                      Structure your report with:
                      1. Executive Summary (key findings)
                      2. Main sections covering different aspects
                      3. Supporting details with citations [1], [2], etc.
                      4. Conclusion with key insights

                      Requirements:
                      - Include citations in brackets [1], [2], etc. throughout the text
                      - Provide balanced, objective analysis
                      - Highlight key trends, statistics, and insights
                      - Use clear section headings
                      - Be comprehensive but well-organized

                      Write in markdown format with proper headings."""

        return self.generate(prompt, temperature=0.3)

    def generate_clarifying_questions(
        self, query: str, max_questions: int = 4
    ) -> List[str]:
        """Generate clarifying questions to better understand the research scope"""
        prompt = f"""Analyze this research query and identify aspects that could be ambiguous or would benefit from clarification: "{query}"

        Generate up to {max_questions} specific clarifying questions that would help you:
        - Better understand the scope and focus
        - Identify the target audience or use case
        - Clarify time frames, geographic focus, or specific domains
        - Understand the depth and type of analysis needed

        Make questions:
        - Specific and actionable
        - Focused on different aspects of the query
        - Helpful for scoping the research properly
        - Clear and easy to answer

        Return ONLY the questions, one per line, without numbering or bullet points.
        If the query is already very specific and clear, return fewer questions or indicate "CLEAR" if no clarification needed."""

        response = self.generate(prompt, temperature=0.4)

        # Parse the response
        if "CLEAR" in response.upper():
            return []

        questions = []
        for line in response.split("\n"):
            line = line.strip()
            # Remove numbering, bullets, or question marks if they exist
            line = re.sub(r"^\d+\.?\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            if line and line.endswith("?"):
                questions.append(line)
            elif line and not line.endswith("?"):
                questions.append(line + "?")

        return questions[:max_questions]
