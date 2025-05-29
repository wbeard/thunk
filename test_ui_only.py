#!/usr/bin/env python3
"""
Quick State-Based Logger Test Script
Tests the state-based logging display without full simulation
"""

import asyncio
import time
from src.thunk.state_logger import StateBasedLogger


async def test_state_based_logger():
    """Test the state-based logger with sample research flow"""
    print("🪵 Testing State-Based Logger...")
    
    logger = StateBasedLogger(use_colors=True, auto_render=True)
    
    # Research start
    logger.research_start("Latest developments in quantum computing and AI applications")
    await asyncio.sleep(1)
    
    # Clarification questions
    questions = [
        "What specific aspects of quantum computing interest you most?",
        "Are you looking for recent breakthroughs or general overview?",
        "Should we focus on any particular applications?"
    ]
    logger.add_clarification_questions(questions)
    await asyncio.sleep(1)
    
    # Clarification responses
    for i, question in enumerate(questions):
        response = f"Sample response {i+1} to demonstrate the logging system"
        logger.add_clarification_response(question, response)
        await asyncio.sleep(0.5)
    
    # Research plan
    steps = [
        "Background research on quantum computing",
        "Current developments and breakthroughs", 
        "AI applications in quantum systems",
        "Market trends and industry adoption",
        "Future prospects and challenges"
    ]
    logger.set_research_plan(steps)
    await asyncio.sleep(1)
    
    # Execute research steps
    for i, step in enumerate(steps, 1):
        logger.start_research_step(i, step)
        await asyncio.sleep(0.5)
        
        # Multiple searches per step to test parallel handling
        for search_num in range(1, 3):  # 2 searches per step
            search_query = f"quantum computing {step.split()[0].lower()} research {search_num}"
            
            # Start search
            logger.start_search(search_query, i)
            await asyncio.sleep(0.3)
            
            # Add search results
            logger.set_search_results(search_query, i, 6)
            await asyncio.sleep(0.2)
            
            # Source evaluations
            sources = [
                (f"Research paper {search_num}A on quantum {step.split()[0].lower()}", True, 0.92, ""),
                (f"Commercial article {search_num}B", False, 0.25, "Low relevance"),
                (f"Academic study {search_num}C on quantum systems", True, 0.88, ""),
                (f"Basic tutorial {search_num}D", False, 0.15, "Too basic")
            ]
            
            # Add all source evaluations
            accepted_sources = []
            for title, accepted, score, reason in sources:
                logger.add_source_evaluation(search_query, i, title, accepted, score, reason)
                if accepted:
                    accepted_sources.append(title)
                await asyncio.sleep(0.1)
            
            # Start content fetching for all accepted sources (this is where parallel fetching happens)
            for title in accepted_sources:
                logger.start_content_fetch(title, i)
                await asyncio.sleep(0.1)  # Small delay to show fetching state
            
            # Complete content fetching with varying timing (simulates parallel completion)
            for j, title in enumerate(accepted_sources):
                await asyncio.sleep(0.2 + j * 0.1)  # Staggered completion
                success = True  # Most succeed
                content_length = 3245 + j * 500
                logger.complete_content_fetch(title, i, success, content_length)
        
        await asyncio.sleep(0.5)  # Pause between steps
    
    # Synthesis phase
    logger.start_synthesis(12, True)
    await asyncio.sleep(0.5)
    
    synthesis_steps = [
        "Analyzing collected documents",
        "Identifying key themes and patterns",
        "Cross-referencing information", 
        "Generating report structure",
        "Writing executive summary",
        "Creating detailed analysis",
        "Adding citations and references",
        "Final formatting and review"
    ]
    
    for step in synthesis_steps:
        logger.add_synthesis_progress(step)
        await asyncio.sleep(0.4)
    
    # Completion
    logger.complete_research(True, 45.2)
    await asyncio.sleep(2)
    
    print("\n✅ State-based logger test complete!")


async def main():
    """Run state-based logger test"""
    try:
        await test_state_based_logger()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())