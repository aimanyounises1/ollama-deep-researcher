"""
Core functionality for the research package.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langsmith import traceable
from src.utils.config import Configuration
from .types import ResearchState
from .utils import (
    extract_confluence_content,
    report_deduplication,
    clean_llm_output,
    deduplicate_lines
)
from .search import search_all_sources
from .tools import tool_manager

logger = logging.getLogger(__name__)

@traceable
async def generate_query(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Generate a search query from the research topic."""
    try:
        # Extract research topic
        topic = state.get("research_topic", "")
        if not topic:
            logger.warning("No research topic provided")
            state["search_query"] = ""
            return state
            
        # If we're in a subsequent iteration and have alternative queries, use one
        if state.get("research_loop_count", 0) > 0 and state.get("alternative_terms", []):
            alternative = state["alternative_terms"][0]
            state["alternative_terms"] = state["alternative_terms"][1:]  # Remove the used term
            state["search_query"] = alternative
            logger.info(f"Using alternative query: {alternative}")
            return state
            
        # For first iteration, use the topic directly with some enhancements
        # Extract issue identifiers (like MTV-1234 or JIRA-5678)
        issue_pattern = r'([A-Z]+-\d+)'
        issues = re.findall(issue_pattern, topic)
        
        if issues:
            # If we have issue IDs, prioritize them
            state["search_query"] = " ".join(issues)
            logger.info(f"Generated search query from issue IDs: {state['search_query']}")
        else:
            # Otherwise, use the full topic
            state["search_query"] = topic
            logger.info(f"Using topic as search query: {state['search_query']}")
            
        return state
    except Exception as e:
        logger.error(f"Error generating query: {e}")
        # Fallback to using the topic directly
        state["search_query"] = state.get("research_topic", "")
        state.setdefault("_errors", []).append({
            "stage": "generate_query",
            "message": str(e)
        })
        return state

@traceable
async def enterprise_research(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Search enterprise sources for information."""
    try:
        # Get the search query
        query = state.get("search_query", "")
        if not query:
            logger.warning("No search query available")
            state["enterprise_research_results"] = []
            state["sources_gathered"] = []
            return state
            
        # Search different enterprise sources in parallel
        from asyncio import gather
        
        # Import search functions
        from .search import search_jira, search_perforce, search_confluence, search_vectorstore
        
        # Run searches in parallel
        jira_task = search_jira(query)
        perforce_task = search_perforce(query)
        confluence_task = search_confluence(query)
        vectorstore_task = search_vectorstore(query)
        
        # Gather results
        results = await gather(jira_task, perforce_task, confluence_task, vectorstore_task)
        jira_results, perforce_results, confluence_results, vectorstore_results = results
        
        # Format results with headers
        formatted_results = [
            f"=== Jira Results ===\n{jira_results}",
            f"=== Perforce Results ===\n{perforce_results}",
            f"=== Confluence Results ===\n{confluence_results}",
            f"=== Vectorstore Results ===\n{vectorstore_results}"
        ]
        
        # Store results in state
        state["enterprise_research_results"] = formatted_results
        state["sources_gathered"] = formatted_results
        
        # Increment research loop count
        state["research_loop_count"] = state.get("research_loop_count", 0) + 1
        
        return state
    except Exception as e:
        logger.error(f"Error in enterprise research: {e}")
        state.setdefault("_errors", []).append({
            "stage": "enterprise_research",
            "message": str(e)
        })
        return state

@traceable
async def check_retrieval_relevance(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Check if retrieved data is relevant to the research topic."""
    try:
        # Get results to check
        results = state.get("enterprise_research_results", [])
        if not results:
            logger.info("No results to check relevance")
            state["_relevance_check"] = "irrelevant"
            return state
            
        # Check if we have specific keywords in the results
        topic = state.get("research_topic", "")
        topic_terms = set(term.lower() for term in topic.split() if len(term) > 3)
        
        # Count how many topic terms appear in the results
        term_matches = 0
        for result in results:
            result_lower = result.lower()
            for term in topic_terms:
                if term in result_lower:
                    term_matches += 1
                    
        # Determine relevance based on matching terms
        if term_matches > len(topic_terms) / 2:  # More than half of the terms match
            state["_relevance_check"] = "relevant"
            logger.info("Results determined to be relevant")
        else:
            state["_relevance_check"] = "irrelevant"
            logger.info("Results determined to be irrelevant")
            
        return state
    except Exception as e:
        logger.error(f"Error checking relevance: {e}")
        state.setdefault("_errors", []).append({
            "stage": "check_retrieval_relevance",
            "message": str(e)
        })
        # Default to assuming relevant in case of error
        state["_relevance_check"] = "relevant"
        return state

@traceable
async def filter_irrelevant_data(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Filter out irrelevant data from search results."""
    try:
        # Get results to filter
        results = state.get("enterprise_research_results", [])
        if not results:
            logger.info("No results to filter")
            return state
            
        # Filter out empty or error results
        filtered_results = []
        for result in results:
            # Skip empty results
            if "No results found" in result or "Error" in result:
                continue
                
            # Skip very short results
            if len(result.split("\n")) < 3:
                continue
                
            filtered_results.append(result)
            
        # Update state with filtered results
        state["enterprise_research_results"] = filtered_results
        state["sources_gathered"] = filtered_results
        
        # Extract JIRA issues for special handling
        jira_results = next((r for r in results if "=== Jira Results ===" in r), None)
        if jira_results:
            # Try to parse JSON content
            import json
            try:
                # Extract the JSON part
                json_content = jira_results.split("=== Jira Results ===\n")[1]
                issues = json.loads(json_content)
                
                # Filter out cancelled issues
                active_issues = [issue for issue in issues if issue.get("status", "").lower() != "cancelled"]
                
                # Store filtered issues
                state["filtered_jira_results"] = active_issues
                logger.info(f"Filtered JIRA issues: {len(active_issues)} active out of {len(issues)} total")
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Error parsing JIRA results: {e}")
                state["filtered_jira_results"] = []
                
        return state
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        state.setdefault("_errors", []).append({
            "stage": "filter_irrelevant_data",
            "message": str(e)
        })
        return state

def should_process_next_confluence_chunk(state: ResearchState) -> bool:
    """Check if there are more Confluence chunks to process."""
    return bool(state.get("full_confluence_text", ""))

@traceable
async def process_confluence_chunk(state: ResearchState) -> ResearchState:
    """Process the next chunk of Confluence content."""
    try:
        # Get the full text and current chunk index
        full_text = state.get("full_confluence_text", "")
        chunk_idx = state.get("confluence_chunk_idx", 1)
        
        if not full_text:
            logger.info("No Confluence chunks to process")
            return state
            
        # Define chunk size
        chunk_size = 20000
        
        # Extract the next chunk
        current_chunk = full_text[:chunk_size]
        remaining_text = full_text[chunk_size:]
        
        # Format the chunk
        formatted_chunk = f"=== Confluence Results (Chunk {chunk_idx}) ===\n{current_chunk}"
        
        # Add to research results
        state.setdefault("enterprise_research_results", []).append(formatted_chunk)
        state.setdefault("sources_gathered", []).append(formatted_chunk)
        
        # Update state for next iteration
        state["full_confluence_text"] = remaining_text
        state["confluence_chunk_idx"] = chunk_idx + 1
        
        # If we just consumed the last chunk, clean up the bookkeeping keys
        if not remaining_text:
            state.pop("full_confluence_text", None)
            logger.info(f"Finished processing Confluence chunks ({chunk_idx})")
        else:
            logger.info(f"Processed Confluence chunk {chunk_idx}, more remaining")
            
        return state
    except Exception as e:
        logger.error(f"Error processing Confluence chunk: {e}")
        state.setdefault("_errors", []).append({
            "stage": "process_confluence_chunk",
            "message": str(e)
        })
        # Clean up in case of error to avoid infinite loops
        state.pop("full_confluence_text", None)
        return state 
