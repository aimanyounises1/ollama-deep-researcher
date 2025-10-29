"""
Graph-based research assistant implementation.
"""

import asyncio
import json
import logging
import operator
import os
import re
import ssl
from typing import List, Dict, Any, Optional, Union, Annotated
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langsmith import traceable

from .configuration import Configuration, SearchAPI
from .research.types import ResearchState
from .research.tools import tool_manager
from .research.utils import (
    deduplicate_and_format_sources,
    tavily_search,
    format_sources,
    perplexity_search,
    duckduckgo_search,
    initialize_ssl_fixes
)
from .research.prompts import (
    reflection_instructions,
    query_writer_instructions,
    summarizer_instructions,
    security_analysis_instructions,
    technical_validation_instructions,
    knowledge_graph_instructions,
)

# Initialize SSL fixes globally
initialize_ssl_fixes()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG', 'false').lower() == 'true' else logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

@traceable
async def generate_query(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Generate a search query from the research topic."""
    try:
        # Format the prompt
        query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.get("research_topic", ""))

        # Generate a query
        configurable = Configuration.from_runnable_config(config)
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            format="json"
        )
        
        result = llm_json_mode.invoke([
            SystemMessage(content=query_writer_instructions_formatted),
            HumanMessage(content="Generate a query for search:")
        ])
        
        query = json.loads(result.content)
        state["search_query"] = query['query']
        
        return state
    except Exception as e:
        logger.error(f"Error generating query: {e}")
        state.setdefault("_errors", []).append({
            "stage": "generate_query",
            "message": str(e)
        })
        state["search_query"] = state.get("research_topic", "")
        return state

@traceable
async def enterprise_research(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Search enterprise sources for information."""
    try:
        # Get query
        query = state.get("search_query", "")
        if not query:
            logger.warning("No search query available")
            return state
            
        # Search all sources
        results = await tool_manager.search_all_sources(query)
        
        # Update state
        state.setdefault("enterprise_research_results", []).extend(results)
        state.setdefault("sources_gathered", []).extend(results)
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
async def web_research(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Gather information from the web."""
    try:
        # Configure
        configurable = Configuration.from_runnable_config(config)
        search_api = configurable.search_api.value if hasattr(configurable.search_api, 'value') else configurable.search_api

        # Search the web
        if search_api == "tavily":
            search_results = await tavily_search(state["search_query"], include_raw_content=True, max_results=1)
            search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
        elif search_api == "perplexity":
            search_results = await perplexity_search(state["search_query"], state.get("research_loop_count", 0))
            search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
        elif search_api == "duckduckgo":
            search_results = await duckduckgo_search(state["search_query"], max_results=3, fetch_full_page=configurable.fetch_full_page)
            search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
        else:
            raise ValueError(f"Unsupported search API: {search_api}")

        # Update state
        state.setdefault("sources_gathered", []).append(format_sources(search_results))
        state["research_loop_count"] = state.get("research_loop_count", 0) + 1
        state.setdefault("web_research_results", []).append(search_str)
        
        return state
    except Exception as e:
        logger.error(f"Error in web research: {e}")
        state.setdefault("_errors", []).append({
            "stage": "web_research",
            "message": str(e)
        })
        return state

@traceable
async def summarize_sources(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Summarize the gathered sources."""
    try:
        # Get existing summary and sources
        existing_summary = state.get("running_summary", "")
        web_research_results = state.get("web_research_results", [])
        enterprise_results = state.get("enterprise_research_results", [])
        
        if not web_research_results and not enterprise_results:
            logger.info("No sources to summarize")
            return state
            
        # Combine sources
        all_sources = []
        if web_research_results:
            all_sources.append(web_research_results[-1])  # Most recent web results
        if enterprise_results:
            all_sources.extend(enterprise_results)  # All enterprise results
            
        combined_sources = "\n\n".join(all_sources)
        
        # Build the human message
        if existing_summary:
            human_message_content = (
                f"<User Input>\n{state.get('research_topic', '')}\n</User Input>\n\n"
                f"<Existing Summary>\n{existing_summary}\n</Existing Summary>\n\n"
                f"<New Sources>\n{combined_sources}\n</New Sources>"
            )
        else:
            human_message_content = (
                f"<User Input>\n{state.get('research_topic', '')}\n</User Input>\n\n"
                f"<Sources>\n{combined_sources}\n</Sources>"
            )
            
        # Run the LLM
        configurable = Configuration.from_runnable_config(config)
        llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0
        )
        
        result = llm.invoke([
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content)
        ])
        
        summary = result.content
        
        # Clean up the summary
        while "<think>" in summary and "</think>" in summary:
            start = summary.find("<think>")
            end = summary.find("</think>") + len("</think>")
            summary = summary[:start] + summary[end:]
            
        state["running_summary"] = summary
        return state
    except Exception as e:
        logger.error(f"Error summarizing sources: {e}")
        state.setdefault("_errors", []).append({
            "stage": "summarize_sources",
            "message": str(e)
        })
        return state

@traceable
async def reflect_on_findings(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Reflect on the research findings and generate follow-up queries."""
    try:
        # Get the current summary
        summary = state.get("running_summary", "")
        if not summary:
            logger.info("No summary to reflect on")
            return state
            
        # Generate reflection
        configurable = Configuration.from_runnable_config(config)
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            format="json"
        )
        
        result = llm_json_mode.invoke([
            SystemMessage(content=reflection_instructions.format(research_topic=state.get("research_topic", ""))),
            HumanMessage(content=f"Reflect on our findings and identify gaps: {summary}")
        ])
        
        try:
            reflection_data = json.loads(result.content)
            state["reflection_data"] = reflection_data
            
            # Extract alternative terms for next iteration
            if "alternative_search_terms" in reflection_data:
                state["alternative_terms"] = reflection_data["alternative_search_terms"]
                
            # Update search query if we have alternative terms
            if state.get("alternative_terms"):
                state["search_query"] = state["alternative_terms"][0]
                state["alternative_terms"] = state["alternative_terms"][1:]
            
            logger.info(f"Reflection complete: priority={reflection_data.get('priority', 'UNKNOWN')}")
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            state["reflection_data"] = {
                "key_findings": [],
                "identified_gaps": [],
                "next_steps": [],
                "priority": "LOW"
            }
            
        return state
    except Exception as e:
        logger.error(f"Error in reflection: {e}")
        state.setdefault("_errors", []).append({
            "stage": "reflect_on_findings",
            "message": str(e)
        })
        return state

@traceable
async def finalize_summary(state: ResearchState) -> ResearchState:
    """Finalize the research summary."""
    try:
        # Get existing summary
        summary = state.get("running_summary", "")
        topic = state.get("research_topic", "")
        
        if not summary:
            logger.info("No summary to finalize")
            state["running_summary"] = f"# Research on {topic}\n\nNo research findings available."
            return state
            
        # Add missing sections
        sections = ["## Overview", "## Key Findings", "## Technical Details", "## Security Implications", "## Recommendations"]
        existing_sections = {line.strip() for line in summary.split("\n") if line.startswith("## ")}
        
        for section in sections:
            if section not in existing_sections:
                summary += f"\n\n{section}\nNo information available."
                
        # Clean up formatting
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        
        # Add citations
        citations = state.get("citations", [])
        if citations and "## Sources" not in existing_sections:
            citation_section = "\n\n## Sources\n"
            for citation in citations:
                source_type = citation.get("source_type", "unknown")
                content = citation.get("content", "No content")
                url = citation.get("url", "#")
                citation_section += f"- [{source_type}]({url}): {content[:100]}...\n"
                
            summary += citation_section
            
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat(timespec='minutes')
        summary += f"\n\n*Generated on: {timestamp}*"
        
        state["running_summary"] = summary
        logger.info("Summary finalized successfully")
        
        return state
    except Exception as e:
        logger.error(f"Error in finalize_summary: {e}")
        state.setdefault("_errors", []).append({
            "stage": "finalize_summary",
            "message": str(e)
        })
        return state

@traceable
async def export_research(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Export the research results in the desired format."""
    try:
        # Get export format
        export_format = state.get("export_format", "markdown")
        
        # Get the summary
        summary = state.get("running_summary", "")
        if not summary:
            logger.info("No summary to export")
            state["exported_data"] = {"error": "No summary to export"}
            return state
            
        # Export based on format
        if export_format == "markdown":
            state["exported_data"] = {"content": summary, "format": "markdown"}
        elif export_format == "json":
            export_data = {
                "summary": summary,
                "key_points": state.get("key_points", []),
                "sources_used": state.get("sources_used", []),
                "technical_validation": state.get("technical_validation", {}),
                "security_findings": state.get("security_findings", []),
                "citations": state.get("citations", [])
            }
            state["exported_data"] = {"content": json.dumps(export_data, indent=2), "format": "json"}
        else:
            logger.warning(f"Unsupported export format: {export_format}")
            state["exported_data"] = {"error": f"Unsupported export format: {export_format}"}
            
        logger.info(f"Research exported in {export_format} format")
        return state
    except Exception as e:
        logger.error(f"Error in export_research: {e}")
        state.setdefault("_errors", []).append({
            "stage": "export_research",
            "message": str(e)
        })
        state["exported_data"] = {"error": str(e)}
        return state

def route_research(state: ResearchState, config: RunnableConfig) -> str:
    """Route the research flow based on state and configuration."""
    try:
        configurable = Configuration.from_runnable_config(config)
        
        # Check if we should continue research
        if state.get("research_loop_count", 0) <= configurable.max_research_loops:
            # Check if we have enterprise results
            if state.get("enterprise_research_results"):
                return "summarize_sources"
            else:
                return "web_research"
        else:
            return "finalize_summary"
    except Exception as e:
        logger.error(f"Error in route_research: {e}")
        return "finalize_summary"

def build_research_graph() -> StateGraph:
    """Create and compile the research graph."""
    # Create graph
    research_graph = StateGraph(ResearchState)
    
    # Add nodes
    research_graph.add_node("generate_query", generate_query)
    research_graph.add_node("enterprise_research", enterprise_research)
    research_graph.add_node("web_research", web_research)
    research_graph.add_node("summarize_sources", summarize_sources)
    research_graph.add_node("reflect_on_findings", reflect_on_findings)
    research_graph.add_node("finalize_summary", finalize_summary)
    research_graph.add_node("export_research", export_research)
    
    # Add edges
    research_graph.add_edge(START, "generate_query")
    research_graph.add_edge("generate_query", "enterprise_research")
    research_graph.add_conditional_edges(
        "enterprise_research",
        route_research,
        {
            "web_research": "web_research",
            "summarize_sources": "summarize_sources",
            "finalize_summary": "finalize_summary"
        }
    )
    research_graph.add_edge("web_research", "summarize_sources")
    research_graph.add_edge("summarize_sources", "reflect_on_findings")
    research_graph.add_conditional_edges(
        "reflect_on_findings",
        route_research,
        {
            "web_research": "web_research",
            "summarize_sources": "summarize_sources",
            "finalize_summary": "finalize_summary"
        }
    )
    research_graph.add_edge("finalize_summary", "export_research")
    research_graph.add_edge("export_research", END)
    
    return research_graph.compile()
