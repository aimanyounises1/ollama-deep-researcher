"""
Graph construction for research package.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from .types import ResearchState
from .tools import tool_manager
from .search import search_all_sources, search_web
from .error_handling import error_recovery
from .knowledge_graph import generate_knowledge_graph, extract_citations

logger = logging.getLogger(__name__)

def build_research_graph():
    """Create and compile the research graph."""
    # Create graph for research states
    research_graph = StateGraph(ResearchState)
    
    # Add nodes for each step in the research process
    research_graph.add_node("generate_query", generate_query)
    research_graph.add_node("enterprise_research", enterprise_research)
    research_graph.add_node("filter_irrelevant_data", filter_irrelevant_data)
    research_graph.add_node("check_retrieval_relevance", check_retrieval_relevance)
    research_graph.add_node("analyze_perforce_diffs", analyze_perforce_diffs)
    research_graph.add_node("perform_web_search", perform_web_search)
    research_graph.add_node("summarize_sources", summarize_sources)
    research_graph.add_node("analyze_security", analyze_security)
    research_graph.add_node("validate_technical_aspects", validate_technical_aspects)
    research_graph.add_node("verify_summary", verify_summary)
    research_graph.add_node("extract_citations", extract_citations)
    research_graph.add_node("generate_knowledge_graph", generate_knowledge_graph)
    research_graph.add_node("reflect_on_findings", reflect_on_findings)
    research_graph.add_node("finalize_summary", finalize_summary)
    research_graph.add_node("export_research", export_research)
    research_graph.add_node("error_recovery", error_recovery)
    research_graph.add_node("extract_jira_keys", extract_jira_keys)
    research_graph.add_node("get_jira_details", get_jira_details)
    research_graph.add_node("extract_perforce_paths", extract_perforce_paths)
    research_graph.add_node("get_perforce_changes", get_perforce_changes)
    
    # Define the graph edges
    research_graph.add_edge(START, "generate_query")
    research_graph.add_edge("generate_query", "enterprise_research")
    research_graph.add_edge("enterprise_research", "filter_irrelevant_data")
    research_graph.add_edge("filter_irrelevant_data", "extract_jira_keys")
    research_graph.add_edge("extract_jira_keys", "get_jira_details")
    research_graph.add_edge("get_jira_details", "extract_perforce_paths")
    research_graph.add_edge("extract_perforce_paths", "get_perforce_changes")
    research_graph.add_edge("get_perforce_changes", "check_retrieval_relevance")
    
    # Add conditional branching based on relevance check
    research_graph.add_conditional_edges(
        "check_retrieval_relevance",
        lambda state: state.get("_relevance_check", "relevant"),
        {
            "relevant": "analyze_perforce_diffs",
            "irrelevant": "perform_web_search"
        }
    )
    
    research_graph.add_edge("analyze_perforce_diffs", "analyze_security")
    research_graph.add_edge("perform_web_search", "analyze_security")
    research_graph.add_edge("analyze_security", "summarize_sources")
    research_graph.add_edge("summarize_sources", "validate_technical_aspects")
    research_graph.add_edge("validate_technical_aspects", "verify_summary")
    
    # Add error recovery path
    research_graph.add_conditional_edges(
        "verify_summary",
        lambda state: "has_errors" if state.get("_errors") else "clean",
        {
            "has_errors": "error_recovery",
            "clean": "extract_citations"
        }
    )
    
    research_graph.add_edge("error_recovery", "summarize_sources")
    research_graph.add_edge("extract_citations", "generate_knowledge_graph")
    research_graph.add_edge("generate_knowledge_graph", "reflect_on_findings")
    
    # Decide whether to continue research or finalize
    research_graph.add_conditional_edges(
        "reflect_on_findings",
        lambda state: "continue" if state.get("research_loop_count", 0) < 3 else "complete",
        {
            "continue": "enterprise_research",
            "complete": "finalize_summary"
        }
    )
    
    research_graph.add_edge("finalize_summary", "export_research")
    research_graph.add_edge("export_research", END)
    
    # Compile the graph
    return research_graph.compile()

async def extract_jira_keys(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Extract JIRA issue keys from research content."""
    try:
        # Extract from summary and sources
        content = state.get("running_summary", "")
        sources = state.get("sources_gathered", [])
        
        # Combine all sources into one string
        combined_content = content + "\n".join(sources)
        
        # Extract keys (simple regex approach)
        import re
        jira_keys = re.findall(r'[A-Z]+-\d+', combined_content)
        
        # Deduplicate
        unique_keys = list(set(jira_keys))
        
        # Store keys in state
        state["jira_keys"] = unique_keys
        logger.info(f"Extracted {len(unique_keys)} JIRA keys: {unique_keys}")
        
        return state
    except Exception as e:
        logger.error(f"Error extracting JIRA keys: {e}")
        state.setdefault("_errors", []).append({
            "stage": "extract_jira_keys",
            "message": str(e)
        })
        return state

async def get_jira_details(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Get detailed information for JIRA issues."""
    try:
        # Get JIRA keys from state
        jira_keys = state.get("jira_keys", [])
        if not jira_keys:
            return state
            
        # Use tool_manager to get details
        jira_details = []
        for key in jira_keys:
            try:
                details = await tool_manager.get_jira_details(key)
                if details:
                    jira_details.append(details)
            except Exception as e:
                logger.error(f"Error getting details for JIRA key {key}: {e}")
                
        # Store details in state
        state["filtered_jira_results"] = jira_details
        logger.info(f"Retrieved details for {len(jira_details)} JIRA issues")
        
        return state
    except Exception as e:
        logger.error(f"Error getting JIRA details: {e}")
        state.setdefault("_errors", []).append({
            "stage": "get_jira_details",
            "message": str(e)
        })
        return state

async def extract_perforce_paths(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Extract Perforce paths from research content."""
    try:
        # Extract from JIRA details and research content
        jira_details = state.get("filtered_jira_results", [])
        content = state.get("running_summary", "")
        
        paths = []
        
        # Extract from JIRA details
        for issue in jira_details:
            description = issue.get("description", "")
            # Look for file paths in description
            import re
            # Match common file path patterns
            path_matches = re.findall(r'(/[\w/\.-]+\.\w+|[\w/\.-]+\.\w+)', description)
            paths.extend(path_matches)
        
        # Extract from running summary
        if content:
            import re
            path_matches = re.findall(r'(/[\w/\.-]+\.\w+|[\w/\.-]+\.\w+)', content)
            paths.extend(path_matches)
        
        # Deduplicate
        unique_paths = list(set(paths))
        
        # Store paths in state
        state["perforce_paths"] = unique_paths
        logger.info(f"Extracted {len(unique_paths)} Perforce paths")
        
        return state
    except Exception as e:
        logger.error(f"Error extracting Perforce paths: {e}")
        state.setdefault("_errors", []).append({
            "stage": "extract_perforce_paths",
            "message": str(e)
        })
        return state

async def get_perforce_changes(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Get Perforce changes for extracted paths."""
    try:
        # Get paths from state
        paths = state.get("perforce_paths", [])
        if not paths:
            return state
            
        # Use tool_manager to get changes
        changes = []
        for path in paths:
            try:
                path_changes = await tool_manager.get_perforce_changes(path)
                changes.extend(path_changes)
            except Exception as e:
                logger.error(f"Error getting changes for path {path}: {e}")
                
        # Store changes in state
        state["perforce_diff_analysis"] = {
            "changes": changes,
            "total_changes": len(changes)
        }
        logger.info(f"Retrieved {len(changes)} Perforce changes")
        
        return state
    except Exception as e:
        logger.error(f"Error getting Perforce changes: {e}")
        state.setdefault("_errors", []).append({
            "stage": "get_perforce_changes",
            "message": str(e)
        })
        return state

async def analyze_perforce_diffs(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Analyze Perforce diffs for additional insights."""
    try:
        # Get Perforce diff analysis from state
        perforce_analysis = state.get("perforce_diff_analysis", {})
        changes = perforce_analysis.get("changes", [])
        
        if not changes:
            logger.info("No Perforce changes to analyze")
            return state
            
        # Use LLM to analyze code changes
        cfg = config.get("configurable", {})
        ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        local_llm = cfg.get("local_llm", "deepseek:latest")
        
        llm = ChatOllama(
            base_url=ollama_base_url,
            model=local_llm,
            temperature=0,
            format="json"
        )
        
        # Prepare changes for analysis
        change_descriptions = []
        for i, change in enumerate(changes[:20]):  # Limit to avoid token limits
            change_descriptions.append(
                f"Change #{i+1}:\n"
                f"File: {change.get('file', 'N/A')}\n"
                f"CL: {change.get('change_list', 'N/A')}\n"
                f"Developer: {change.get('developer', 'N/A')}\n"
                f"Description: {change.get('change_description', 'N/A')}\n"
                f"Code: {change.get('code', 'N/A')}\n"
            )
        
        # Create prompt for LLM
        joined_changes = "\n".join(change_descriptions)
        prompt = f"""Analyze the following Perforce code changes and return a JSON object with:
        1. "summary": A concise technical summary of the changes
        2. "findings": A list of objects with "file", "change_list", "developer", "change_description", and "analysis"

        Changes:
        {joined_changes}:"""
        
        try:
            # Call LLM for analysis
            response = await llm.ainvoke([
                SystemMessage(content="You are an expert code analyst. Provide clear, technical analysis of code changes."),
                HumanMessage(content=prompt)
            ])
            
            try:
                analysis_result = json.loads(response.content)
                state["perforce_diff_analysis"] = analysis_result
                logger.info(f"Successfully analyzed {len(changes)} Perforce changes")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                state["perforce_diff_analysis"] = {
                    "summary": "Error analyzing Perforce changes: Invalid JSON response.",
                    "findings": []
                }
        except Exception as e:
            logger.error(f"Error calling LLM for Perforce analysis: {e}")
            state["perforce_diff_analysis"] = {
                "summary": f"Error analyzing Perforce changes: {str(e)}",
                "findings": []
            }
        
        return state
    except Exception as e:
        logger.error(f"Error in analyze_perforce_diffs: {e}")
        state["perforce_diff_analysis"] = {
            "summary": f"Error analyzing Perforce changes: {str(e)}",
            "findings": []
        }
        return state

async def summarize_sources(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Summarize all gathered sources."""
    try:
        # Get sources to summarize
        sources = state.get("sources_gathered", [])
        if not sources:
            logger.info("No sources to summarize")
            return state
            
        # Use LLM to summarize
        cfg = config.get("configurable", {})
        ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        local_llm = cfg.get("local_llm", "deepseek:latest")
        
        llm = ChatOllama(
            base_url=ollama_base_url,
            model=local_llm,
            temperature=0,
            format="json"
        )
        
        # Get the research topic
        topic = state.get("research_topic", "")
        
        # Prepare sources for summarization
        combined_sources = "\n\n".join(sources)
        
        # Create prompt for LLM
        prompt = f"""Summarize the following research information for the topic "{topic}":

Sources:
{combined_sources[:4000]}  # Limit content to avoid token limits

Return a JSON object with:
1. "summary": A comprehensive summary of the findings
2. "key_points": Array of key points extracted from the sources
3. "sources_used": Array of source types used in the summary
"""
        
        try:
            # Call LLM for summarization
            response = await llm.ainvoke([
                SystemMessage(content="You are a research analyst. Provide clear, comprehensive summaries of research findings."),
                HumanMessage(content=prompt)
            ])
            
            try:
                summary_result = json.loads(response.content)
                state["running_summary"] = summary_result.get("summary", "")
                state["key_points"] = summary_result.get("key_points", [])
                state["sources_used"] = summary_result.get("sources_used", [])
                logger.info("Successfully summarized sources")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                state["running_summary"] = f"Error summarizing sources: Invalid JSON response."
        except Exception as e:
            logger.error(f"Error calling LLM for summarization: {e}")
            state["running_summary"] = f"Error summarizing sources: {str(e)}"
        
        return state
    except Exception as e:
        logger.error(f"Error in summarize_sources: {e}")
        state.setdefault("_errors", []).append({
            "stage": "summarize_sources",
            "message": str(e)
        })
        return state

async def validate_technical_aspects(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Validate technical aspects of the research."""
    try:
        # Get content to validate
        summary = state.get("running_summary", "")
        if not summary:
            logger.info("No content to validate for technical aspects")
            state["technical_validation"] = {"valid": True, "findings": []}
            return state
            
        # Use LLM to validate technical aspects
        cfg = config.get("configurable", {})
        ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        local_llm = cfg.get("local_llm", "deepseek:latest")
        
        llm = ChatOllama(
            base_url=ollama_base_url,
            model=local_llm,
            temperature=0,
            format="json"
        )
        
        # Get the research topic
        topic = state.get("research_topic", "")
        
        prompt = f"""Validate the technical aspects of the following research summary for the topic "{topic}":
{summary[:4000]}  # Limit content to avoid token limits

Return a JSON object with:
1. "valid": boolean indicating if the content is technically valid
2. "findings": array of technical findings/issues
3. "improvement_suggestions": array of suggestions to improve technical accuracy
"""
        
        try:
            response = await llm.ainvoke([
                SystemMessage(content="You are a technical validator. Assess the technical accuracy of the content."),
                HumanMessage(content=prompt)
            ])
            
            try:
                validation_result = json.loads(response.content)
                state["technical_validation"] = validation_result
                logger.info(f"Technical validation complete: valid={validation_result.get('valid', False)}")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                state["technical_validation"] = {"valid": True, "findings": [], "improvement_suggestions": []}
        except Exception as e:
            logger.error(f"Error calling LLM for technical validation: {e}")
            state["technical_validation"] = {"valid": True, "findings": [], "improvement_suggestions": []}
        
        return state
    except Exception as e:
        logger.error(f"Error in validate_technical_aspects: {e}")
        state.setdefault("_errors", []).append({
            "stage": "technical_validation",
            "message": str(e)
        })
        state["technical_validation"] = {"valid": True, "findings": [], "improvement_suggestions": []}
        return state

async def verify_summary(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Verify the quality and completeness of the summary."""
    try:
        # Get summary to verify
        summary = state.get("running_summary", "")
        if not summary:
            logger.info("No summary to verify")
            return state
            
        # Check for citations
        citation_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        citations = re.findall(citation_pattern, summary)
        
        if not citations:
            logger.warning("No citations found in summary")
            state.setdefault("_errors", []).append({
                "stage": "verification",
                "message": "No citations found in summary",
                "severity": "warning"
            })
        
        # Check for technical terms
        topic = state.get("research_topic", "")
        topic_terms = topic.lower().split()
        
        missing_terms = []
        for term in topic_terms:
            if len(term) > 3 and term not in summary.lower():
                missing_terms.append(term)
        
        if missing_terms:
            logger.warning(f"Summary missing key terms: {missing_terms}")
            state.setdefault("_errors", []).append({
                "stage": "verification",
                "message": f"Summary missing key terms: {missing_terms}",
                "severity": "warning"
            })
        
        return state
    except Exception as e:
        logger.error(f"Error in verify_summary: {e}")
        state.setdefault("_errors", []).append({
            "stage": "verification",
            "message": str(e),
            "severity": "error"
        })
        return state

async def reflect_on_findings(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Reflect on the research findings for additional insights."""
    try:
        # Get research content
        summary = state.get("running_summary", "")
        if not summary:
            logger.info("No summary to reflect on")
            state["reflection_data"] = {
                "key_findings": [],
                "identified_gaps": [],
                "next_steps": []
            }
            return state
            
        # Use LLM for reflection
        cfg = config.get("configurable", {})
        ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        local_llm = cfg.get("local_llm", "deepseek:latest")
        
        llm = ChatOllama(
            base_url=ollama_base_url,
            model=local_llm,
            temperature=0,
            format="json"
        )
        
        # Get the research topic
        topic = state.get("research_topic", "")
        
        prompt = f"""Reflect on the following research summary for the topic "{topic}":
{summary[:4000]}  # Limit content to avoid token limits

Return a JSON object with:
1. "key_findings": array of key findings
2. "identified_gaps": array of gaps in the research
3. "next_steps": array of suggested next steps
4. "alternative_search_terms": array of alternative search terms for further research
5. "priority": "HIGH", "MEDIUM", or "LOW" for continuing research
"""
        
        try:
            response = await llm.ainvoke([
                SystemMessage(content="You are a research analyst. Identify key findings, gaps, and next steps."),
                HumanMessage(content=prompt)
            ])
            
            try:
                reflection_data = json.loads(response.content)
                state["reflection_data"] = reflection_data
                
                # Extract alternative terms for next search iteration
                if "alternative_search_terms" in reflection_data:
                    state["alternative_terms"] = reflection_data["alternative_search_terms"]
                
                logger.info(f"Reflection complete: priority={reflection_data.get('priority', 'UNKNOWN')}")
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response")
                state["reflection_data"] = {
                    "key_findings": [],
                    "identified_gaps": [],
                    "next_steps": [],
                    "priority": "LOW"
                }
        except Exception as e:
            logger.error(f"Error calling LLM for reflection: {e}")
            state["reflection_data"] = {
                "key_findings": [],
                "identified_gaps": [],
                "next_steps": [],
                "priority": "LOW"
            }
        
        return state
    except Exception as e:
        logger.error(f"Error in reflect_on_findings: {e}")
        state.setdefault("_errors", []).append({
            "stage": "reflection",
            "message": str(e)
        })
        state["reflection_data"] = {
            "key_findings": [],
            "identified_gaps": [],
            "next_steps": [],
            "priority": "LOW"
        }
        return state

async def finalize_summary(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Finalize the research summary."""
    try:
        # Get existing summary
        summary = state.get("running_summary", "")
        topic = state.get("research_topic", "")
        
        if not summary:
            logger.info("No summary to finalize")
            state["running_summary"] = f"# Research on {topic}\n\nNo research findings available."
            return state
            
        # Add sections that might be missing
        sections = ["## Overview", "## Key Findings", "## Technical Details", "## Security Implications", "## Recommendations"]
        existing_sections = set()
        
        for line in summary.split("\n"):
            if line.startswith("## "):
                existing_sections.add(line.strip())
        
        missing_sections = []
        for section in sections:
            if section not in existing_sections:
                missing_sections.append(section)
        
        if missing_sections:
            # Append missing sections
            for section in missing_sections:
                summary += f"\n\n{section}\nNo information available."
        
        # Clean up the format
        summary = re.sub(r'\n{3,}', '\n\n', summary)  # Remove excessive newlines
        
        # Add citations section if there are citations
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
        
        # Store the finalized summary
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

async def export_research(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Export the research results in the desired format."""
    try:
        # Get the export format
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
            # Convert to JSON
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

# Import required functions from other modules
from .search import search_all_sources, search_web
from .error_handling import error_recovery
from .knowledge_graph import generate_knowledge_graph, extract_citations

# Define or import other node functions 
# Use direct function references, not from engine to avoid circular imports
async def analyze_security(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Analyze security implications in the research content."""
    try:
        # Get content to analyze
        content = state.get("running_summary", "")
        if not content:
            logger.info("No content to analyze for security")
            state["security_findings"] = []
            return state
            
        # Use security analyzer tool
        security_findings = await tool_manager.analyze_security(content)
        
        # Store findings in state
        state["security_findings"] = security_findings
        logger.info(f"Found {len(security_findings)} security findings")
        
        return state
    except Exception as e:
        logger.error(f"Error in analyze_security: {e}")
        state.setdefault("_errors", []).append({
            "stage": "security_analysis",
            "message": str(e)
        })
        state["security_findings"] = []
        return state

# Define other node functions (these are placeholders - integrate with actual implementations)
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

async def enterprise_research(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Search enterprise sources for information."""
    results = await search_all_sources(state["search_query"])
    state["enterprise_research_results"] = [results]
    state["sources_gathered"] = [results]
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    return state

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

async def perform_web_search(state: ResearchState, config: RunnableConfig = None) -> ResearchState:
    """Search the web for additional information."""
    web_results = await search_web(state["search_query"])
    state["web_search_results"] = web_results
    return state 