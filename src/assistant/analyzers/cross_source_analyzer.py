# src/assistant/analyzers/cross_source_analyzer.py

import logging
import asyncio
import json
import gc
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Union
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from src.assistant.core.types import ResearchState
from src.assistant.utils.langgraph_helpers import clean_llm_output, extract_identifiers, traceable

logger = logging.getLogger(__name__)

# LLM Configuration
LLM_CONFIG_PARAMS = {
    "temperature": 0.15,
    "num_ctx": 8192,
    "top_p": 0.7,
    "repeat_penalty": 1.15,
}

# Constants
MAX_LLM_INPUT_SIZE = 32000
MAX_CONFLUENCE_CONTENT_SIZE = 50000
MAX_JIRA_DESC_SIZE = 30000

@traceable
async def analyze_cross_source_connections(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze connections between different data sources, including:
    - Jira tickets referenced in Perforce changelists
    - Jira tickets referenced in Confluence docs
    - Code changes related to documentation
    
    Args:
        state: Dictionary with relevant data:
            - jira_data: List of Jira issue data
            - perforce_data: List of Perforce changelist data
            - confluence_data: List of Confluence page data
        
    Returns:
        Dictionary with analysis results
    """
    # Create empty connections list
    connections = []
    
    # Process data for cross-source connections
    try:
        # Check if we have any data to process
        jira_data = state.get("jira_data", [])
        perforce_data = state.get("perforce_data", [])
        confluence_data = state.get("confluence_data", [])
        
        if not jira_data and not perforce_data and not confluence_data:
            logger.warning("No data available for cross-source analysis")
            return {
                "text": "No data available for cross-source analysis.",
                "connections": []
            }
        
        # Log the data we're processing
        logger.info(f"Analyzing connections between {len(jira_data)} Jira issues, "
                   f"{len(perforce_data)} Perforce changelists, and "
                   f"{len(confluence_data)} Confluence pages")
        
        # --- Extract identifier mappings ---
        jira_keys = set()
        jira_mtv_mapping = {}  # Maps MTV IDs to Jira keys
        perforce_changes = set()
        confluence_ids = set()
        
        # Process Jira data to extract keys and MTV mappings
        for item in jira_data:
            if not isinstance(item, dict):
                continue
                
            key = item.get("key")
            if not key:
                continue
                
            jira_keys.add(key)
            
            # Extract MTV references from Jira data
            summary = item.get("summary", "") or item.get("fields", {}).get("summary", "")
            description = item.get("description", "") or item.get("fields", {}).get("description", "")
            
            # Find MTV references
            identifiers = extract_identifiers(f"{key} {summary} {description}")
            mtv_ids = identifiers.get("mtv_ids", [])
            
            for mtv_id in mtv_ids:
                if mtv_id not in jira_mtv_mapping:
                    jira_mtv_mapping[mtv_id] = []
                if key not in jira_mtv_mapping[mtv_id]:
                    jira_mtv_mapping[mtv_id].append(key)
        
        # Process Perforce data to extract changelist numbers
        for item in perforce_data:
            if not isinstance(item, dict):
                continue
                
            change = str(item.get("change", "") or item.get("changelist", "") or item.get("change_list", ""))
            if change:
                perforce_changes.add(change)
        
        # Process Confluence data to extract page IDs
        for i, item in enumerate(confluence_data):
            if not isinstance(item, dict):
                continue
                
            conf_id = str(item.get("id", "") or hash(item.get("url", "") or f"unknown_{i}"))
            confluence_ids.add(conf_id)
        
        # --- Find direct connections ---
        # Jira tickets referenced in Perforce descriptions
        perforce_to_jira = {}  # Maps CL numbers to Jira keys
        jira_to_perforce = {}  # Maps Jira keys to CL numbers
        
        for item in perforce_data:
            if not isinstance(item, dict):
                continue
                
            change = str(item.get("change", "") or item.get("changelist", "") or item.get("change_list", ""))
            if not change:
                continue
                
            description = str(item.get("description", "") or item.get("desc", ""))
            
            # Extract identifiers from description
            identifiers = extract_identifiers(description)
            jira_refs = identifiers.get("jira_keys", [])
            mtv_refs = identifiers.get("mtv_ids", [])
            
            # Initialize the connection lists if not already present
            if change not in perforce_to_jira:
                perforce_to_jira[change] = []
            
            # Add direct Jira key connections
            for jira_key in jira_refs:
                if jira_key in jira_keys and jira_key not in perforce_to_jira[change]:
                    perforce_to_jira[change].append(jira_key)
                    
                    # Update the reverse mapping
                    if jira_key not in jira_to_perforce:
                        jira_to_perforce[jira_key] = []
                    if change not in jira_to_perforce[jira_key]:
                        jira_to_perforce[jira_key].append(change)
                    
                    connections.append({
                        "source1_type": "perforce",
                        "source1_id": change,
                        "source2_type": "jira",
                        "source2_id": jira_key,
                        "connection_type": "direct_reference",
                        "confidence": 1.0,
                        "evidence": f"Perforce changelist {change} directly references Jira issue {jira_key}"
                    })
            
            # Add MTV-based connections
            for mtv_ref in mtv_refs:
                if mtv_ref in jira_mtv_mapping:
                    for jira_key in jira_mtv_mapping[mtv_ref]:
                        if jira_key not in perforce_to_jira[change]:
                            perforce_to_jira[change].append(jira_key)
                            
                            # Update the reverse mapping
                            if jira_key not in jira_to_perforce:
                                jira_to_perforce[jira_key] = []
                            if change not in jira_to_perforce[jira_key]:
                                jira_to_perforce[jira_key].append(change)
                            
                            connections.append({
                                "source1_type": "perforce",
                                "source1_id": change,
                                "source2_type": "jira",
                                "source2_id": jira_key,
                                "connection_type": "mtv_reference",
                                "confidence": 0.9,
                                "evidence": f"Perforce changelist {change} and Jira issue {jira_key} both reference MTV ID {mtv_ref}"
                            })
        
        # Jira tickets referenced in Confluence pages
        confluence_to_jira = {}  # Maps Confluence IDs to Jira keys
        jira_to_confluence = {}  # Maps Jira keys to Confluence IDs
        
        for item in confluence_data:
            if not isinstance(item, dict):
                continue
                
            conf_id = str(item.get("id", "") or hash(item.get("url", "") or ""))
            if not conf_id:
                continue
                
            # Handle nested 'content' dict from SolutionBookQuerier
            title = str(item.get("title", ""))
            content = str(item.get("content", "") or item.get("body", "") or item.get("text", "") or "")

            # If top-level only has 'content', dive deeper
            if not title or not content:
                nested = item.get("content") if isinstance(item.get("content"), dict) else None
                if nested:
                    # ID override if available
                    conf_id = str(item.get("id") or nested.get("id") or conf_id)
                    if not title:
                        title = str(nested.get("title", title))
                    # Try common body paths
                    body_txt = ""
                    if "body" in nested:
                        # Could be dict with storage.value
                        body_node = nested["body"]
                        if isinstance(body_node, dict):
                            body_txt = body_node.get("storage", {}).get("value", "") or body_node.get("view", {}).get("value", "")
                    if not body_txt and "excerpt" in nested:
                        body_txt = str(nested.get("excerpt"))
                    if not body_txt and "content" in nested:
                        body_txt = str(nested.get("content"))
                    content = f"{content} {body_txt}".strip()
            
            # Extract identifiers from title and content
            title_identifiers = extract_identifiers(title)
            content_identifiers = extract_identifiers(content)
            
            # Combine all identifiers
            jira_refs = list(set(title_identifiers.get("jira_keys", []) + content_identifiers.get("jira_keys", [])))
            mtv_refs = list(set(title_identifiers.get("mtv_ids", []) + content_identifiers.get("mtv_ids", [])))
            
            # Initialize the connection lists if not already present
            if conf_id not in confluence_to_jira:
                confluence_to_jira[conf_id] = []
            
            # Add direct Jira key connections
            for jira_key in jira_refs:
                if jira_key in jira_keys and jira_key not in confluence_to_jira[conf_id]:
                    confluence_to_jira[conf_id].append(jira_key)
                    
                    # Update the reverse mapping
                    if jira_key not in jira_to_confluence:
                        jira_to_confluence[jira_key] = []
                    if conf_id not in jira_to_confluence[jira_key]:
                        jira_to_confluence[jira_key].append(conf_id)
                    
                    connections.append({
                        "source1_type": "confluence",
                        "source1_id": conf_id,
                        "source2_type": "jira",
                        "source2_id": jira_key,
                        "connection_type": "direct_reference",
                        "confidence": 1.0,
                        "evidence": f"Confluence page {conf_id} directly references Jira issue {jira_key}"
                    })
            
            # Add MTV-based connections
            for mtv_ref in mtv_refs:
                if mtv_ref in jira_mtv_mapping:
                    for jira_key in jira_mtv_mapping[mtv_ref]:
                        if jira_key not in confluence_to_jira[conf_id]:
                            confluence_to_jira[conf_id].append(jira_key)
                            
                            # Update the reverse mapping
                            if jira_key not in jira_to_confluence:
                                jira_to_confluence[jira_key] = []
                            if conf_id not in jira_to_confluence[jira_key]:
                                jira_to_confluence[jira_key].append(conf_id)
                            
                            connections.append({
                                "source1_type": "confluence",
                                "source1_id": conf_id,
                                "source2_type": "jira",
                                "source2_id": jira_key,
                                "connection_type": "mtv_reference",
                                "confidence": 0.9,
                                "evidence": f"Confluence page {conf_id} and Jira issue {jira_key} both reference MTV ID {mtv_ref}"
                            })
        
        # --- Find indirect/transitive connections ---
        # Perforce changelists indirectly connected to Confluence pages via Jira tickets
        for jira_key in jira_keys:
            if jira_key in jira_to_perforce and jira_key in jira_to_confluence:
                perforce_cls = jira_to_perforce[jira_key]
                confluence_pages = jira_to_confluence[jira_key]
                
                for cl in perforce_cls:
                    for conf_id in confluence_pages:
                        connections.append({
                            "source1_type": "perforce",
                            "source1_id": cl,
                            "source2_type": "confluence",
                            "source2_id": conf_id,
                            "connection_type": "indirect_via_jira",
                            "confidence": 0.7,
                            "evidence": f"Perforce changelist {cl} and Confluence page {conf_id} are both linked to Jira issue {jira_key}"
                        })
        
        # Generate text analysis of connections
        analysis_text = []
        analysis_text.append("# Cross-Source Connection Analysis\n")
        
        # Add a summary section
        analysis_text.append("## Summary\n")
        analysis_text.append(f"- Analyzed {len(jira_data)} Jira issues, {len(perforce_data)} Perforce changelists, and {len(confluence_data)} Confluence pages\n")
        analysis_text.append(f"- Found {len(jira_keys)} unique Jira tickets, {len(perforce_changes)} unique changelists, and {len(confluence_ids)} unique Confluence pages\n")
        
        # Count different types of connections
        direct_jira_perforce = len([c for c in connections if c["connection_type"] == "direct_reference" and 
                                    c["source1_type"] == "perforce" and c["source2_type"] == "jira"])
        mtv_jira_perforce = len([c for c in connections if c["connection_type"] == "mtv_reference" and 
                                c["source1_type"] == "perforce" and c["source2_type"] == "jira"])
        direct_jira_confluence = len([c for c in connections if c["connection_type"] == "direct_reference" and 
                                    c["source1_type"] == "confluence" and c["source2_type"] == "jira"])
        mtv_jira_confluence = len([c for c in connections if c["connection_type"] == "mtv_reference" and 
                                c["source1_type"] == "confluence" and c["source2_type"] == "jira"])
        indirect_perforce_confluence = len([c for c in connections if c["connection_type"] == "indirect_via_jira" and 
                                        c["source1_type"] == "perforce" and c["source2_type"] == "confluence"])
        
        analysis_text.append(f"- Found {direct_jira_perforce} direct references from Perforce changelists to Jira tickets\n")
        analysis_text.append(f"- Found {mtv_jira_perforce} MTV-based connections between Perforce changelists and Jira tickets\n")
        analysis_text.append(f"- Found {direct_jira_confluence} direct references from Confluence pages to Jira tickets\n")
        analysis_text.append(f"- Found {mtv_jira_confluence} MTV-based connections between Confluence pages and Jira tickets\n")
        analysis_text.append(f"- Found {indirect_perforce_confluence} indirect connections between Perforce changelists and Confluence pages via Jira tickets\n")
        
        # Add a conclusion
        analysis_text.append("\n## Conclusion\n")
        if connections:
            analysis_text.append("The analysis found significant connections between code changes, documentation, and tickets, indicating proper cross-referencing between systems.\n")
            
            # Add insights based on results
            connected_jira = len(set([c["source2_id"] for c in connections if c["source2_type"] == "jira"]))
            connected_perforce = len(set([c["source1_id"] for c in connections if c["source1_type"] == "perforce"]))
            connected_confluence = len(set([c["source1_id"] for c in connections if c["source1_type"] == "confluence"]))
            
            if connected_jira < len(jira_keys) * 0.5:
                analysis_text.append(f"However, only {connected_jira} out of {len(jira_keys)} Jira tickets have cross-system connections. Consider improving linking practices.\n")
            
            if connected_perforce < len(perforce_changes) * 0.5:
                analysis_text.append(f"Only {connected_perforce} out of {len(perforce_changes)} Perforce changelists reference tickets. Consider enforcing ticket references in change descriptions.\n")
            
            if connected_confluence < len(confluence_ids) * 0.5:
                analysis_text.append(f"Only {connected_confluence} out of {len(confluence_ids)} Confluence pages reference tickets. Consider improving documentation linking practices.\n")
        else:
            analysis_text.append("No significant connections were found between code changes, documentation, and tickets. This suggests poor cross-referencing between systems.\n")
            analysis_text.append("Consider implementing practices to ensure proper linking between Jira tickets, code changes, and documentation.\n")
        
        return {
            "text": "\n".join(analysis_text),
            "connections": connections,
            "cross_source_connections": connections
        }
    
    except Exception as e:
        logger.error(f"Error in analyze_cross_source_connections: {e}", exc_info=True)
        return {
            "text": f"Error in cross-source analysis: {str(e)}",
            "connections": [],
            "cross_source_connections": []
        }

async def analyze_cross_sources(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Public wrapper for analyze_cross_source_connections function.
    Analyzes connections between different data sources.
    
    Args:
        state: Research state
        config: Optional configuration
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting cross-source analysis")
    
    # Copy relevant data from state into a format expected by analyze_cross_source_connections
    analysis_state = {}
    
    # Extract raw data from the nested state structure with better error handling
    try:
        # Extract Jira data from state - check all possible locations
        jira_data = []
        if "jira_data" in state and isinstance(state["jira_data"], dict) and "raw_results" in state["jira_data"]:
            jira_data = state["jira_data"]["raw_results"]
            logger.info(f"Found {len(jira_data)} Jira raw results in state.jira_data.raw_results")
        elif "raw_jira_results" in state and isinstance(state["raw_jira_results"], list):
            jira_data = state["raw_jira_results"]
            logger.info(f"Found {len(jira_data)} Jira raw results in state.raw_jira_results")
            
        # Extract Perforce data from state - check all possible locations
        perforce_data = []
        if "perforce_data" in state and isinstance(state["perforce_data"], dict) and "raw_results" in state["perforce_data"]:
            perforce_data = state["perforce_data"]["raw_results"]
            logger.info(f"Found {len(perforce_data)} Perforce raw results in state.perforce_data.raw_results")
        elif "raw_perforce_results" in state and isinstance(state["raw_perforce_results"], list):
            perforce_data = state["raw_perforce_results"]
            logger.info(f"Found {len(perforce_data)} Perforce raw results in state.raw_perforce_results")
            
        # Extract Confluence data from state - check all possible locations
        confluence_data = []
        if "confluence_data" in state and isinstance(state["confluence_data"], dict) and "raw_results" in state["confluence_data"]:
            confluence_data = state["confluence_data"]["raw_results"]
            logger.info(f"Found {len(confluence_data)} Confluence raw results in state.confluence_data.raw_results")
        elif "raw_confluence_results" in state and isinstance(state["raw_confluence_results"], list):
            confluence_data = state["raw_confluence_results"]
            logger.info(f"Found {len(confluence_data)} Confluence raw results in state.raw_confluence_results")
            
        # Add to analysis state with expected keys
        analysis_state["jira_data"] = jira_data
        analysis_state["perforce_data"] = perforce_data  
        analysis_state["confluence_data"] = confluence_data
        
        # Ensure we also have the jira_summary available for any functions that might need it
        # This prevents the "jira_summary is not defined" error
        if "jira_data" in state and isinstance(state["jira_data"], dict) and "summary" in state["jira_data"]:
            analysis_state["jira_summary"] = state["jira_data"]["summary"]
        else:
            # Provide a default value to avoid undefined variable errors
            analysis_state["jira_summary"] = "No Jira data summary available."
            
        # Similarly add perforce and confluence summaries to avoid potential undefined errors
        if "perforce_data" in state and isinstance(state["perforce_data"], dict) and "summary" in state["perforce_data"]:
            analysis_state["perforce_summary"] = state["perforce_data"]["summary"]
        else:
            analysis_state["perforce_summary"] = "No Perforce data summary available."
            
        if "confluence_data" in state and isinstance(state["confluence_data"], dict) and "summary" in state["confluence_data"]:
            analysis_state["confluence_summary"] = state["confluence_data"]["summary"]
        else:
            analysis_state["confluence_summary"] = "No Confluence data summary available."
        
        # Provide debug info
        total_items = len(jira_data) + len(perforce_data) + len(confluence_data)
        logger.info(f"Prepared analysis state with {total_items} total data items")
            
    except Exception as e:
        logger.error(f"Error preparing analysis state: {e}", exc_info=True)
        analysis_state["jira_data"] = []
        analysis_state["perforce_data"] = []
        analysis_state["confluence_data"] = []
        # Also set default summaries in case of exception
        analysis_state["jira_summary"] = "Error retrieving Jira data."
        analysis_state["perforce_summary"] = "Error retrieving Perforce data."
        analysis_state["confluence_summary"] = "Error retrieving Confluence data."
    
    # Provide detailed logging of the first item from each source if available
    for source_type, source_data in [
        ("jira_data", "Jira"), 
        ("perforce_data", "Perforce"), 
        ("confluence_data", "Confluence")
    ]:
        items = analysis_state.get(source_type, [])
        if items and isinstance(items, list) and items:
            example_item = items[0]
            if isinstance(example_item, dict):
                # Log a summary of keys rather than the full object which may be large
                logger.info(f"First {source_data} item has keys: {list(example_item.keys())}")
            else:
                logger.info(f"First {source_data} item is not a dictionary: {type(example_item)}")
    
    try:
        # Call the analysis function with the prepared analysis state
        analysis_result = await analyze_cross_source_connections(analysis_state)
        return analysis_result
    except Exception as e:
        logger.error(f"Error in analyze_cross_source_connections: {e}", exc_info=True)
        return {
            "text": f"Error analyzing cross-source connections: {str(e)}",
            "connections": []
        }

async def apply_cross_source_analysis(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Apply cross-source analysis to identify relationships between data sources."""
    # Extract enable_cross_source directly from config to avoid circular imports
    enable_cross_source = True
    if config and isinstance(config, dict) and "configurable" in config:
        configurable = config["configurable"]
        if isinstance(configurable, dict) and "enable_cross_source" in configurable:
            enable_cross_source = configurable["enable_cross_source"]
    
    if not enable_cross_source:
        logger.info("Cross-source analysis disabled. Skipping.")
        return {}
    
    try:
        # Log the state structure to help diagnose missing data
        logger.info("====== STATE DEBUG BEFORE CROSS-SOURCE ANALYSIS ======")
        logger.info(f"State keys: {list(state.keys())}")
        
        for key in ['jira_data', 'perforce_data', 'confluence_data', 'raw_jira_results', 'raw_perforce_results', 'raw_confluence_results']:
            if key in state:
                if isinstance(state[key], dict) and "raw_results" in state[key]:
                    count = len(state[key]["raw_results"]) if isinstance(state[key]["raw_results"], list) else "N/A"
                    logger.info(f"State contains {key} with {count} raw results")
                elif isinstance(state[key], list):
                    logger.info(f"State contains {key} as list with {len(state[key])} items")
                else:
                    logger.info(f"State contains {key} as {type(state[key])}")
            else:
                logger.info(f"State does NOT contain {key}")
        
        result = await analyze_cross_sources(state, config)
        
        # Debug log the result
        connection_count = len(result.get("cross_source_connections", []))
        logger.info(f"Cross-source analysis identified {connection_count} connections")
        
        return result
    except Exception as e:
        logger.error(f"Error in cross-source analysis: {e}", exc_info=True)
        return {
            "cross_source_connections": [],
            "cross_source_analysis": "Error during cross-source analysis: " + str(e)
        }
