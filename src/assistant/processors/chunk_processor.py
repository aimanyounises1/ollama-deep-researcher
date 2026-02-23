# src/assistant/processors/chunk_processor.py

import logging
import asyncio
from typing import Dict, Any, List, Set, Optional, Union
from collections import defaultdict

from langchain_core.runnables import RunnableConfig

from src.assistant.core.types import ResearchState

logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_PROCESSING_HOPS = 50
CONFLUENCE_CHUNK_SIZE = 5
JIRA_CHUNK_SIZE = 10
PERFORCE_CHUNK_SIZE = 10
CONFLUENCE_OVERLAP = 1
JIRA_OVERLAP = 2
PERFORCE_OVERLAP = 1

def _finish_source(state: Dict[str, Any], prefix: str) -> None:
    """Reset chunking state variables for a source (in-place)."""
    logger.debug(f"Resetting chunk state for source: {prefix}")
    state[f"{prefix}_chunks"] = []
    state[f"{prefix}_chunk_idx"] = 0
    state[f"{prefix}_total_chunks"] = 0
    # Reset stuck counters
    state[f"_last_{prefix}_chunk_idx"] = -1
    state[f"_stuck_{prefix}_count"] = 0

def reset_all_chunking(state: Dict[str, Any]) -> None:
    """Reset all chunking state variables for all sources (in-place)."""
    logger.debug("Resetting all chunk state for all sources")
    for prefix in ["confluence", "jira", "perforce"]:
        _finish_source(state, prefix)

def should_process_next_chunk(state: Dict[str, Any], source_type: str) -> bool:
    """
    Check if there are more chunks to process for a given source type.
    Includes safety checks against infinite loops.
    """
    # Safety Checks
    hop_counter = state.get("__hop_counter", 0)
    if hop_counter > MAX_CHUNK_PROCESSING_HOPS:
        logger.error(f"Chunk processing hop limit ({MAX_CHUNK_PROCESSING_HOPS}) exceeded for {source_type}, forcing completion.")
        _finish_source(state, source_type)
        return False

    chunk_idx = state.get(f"{source_type}_chunk_idx", 0)
    total_chunks = state.get(f"{source_type}_total_chunks", 0)
    chunks = state.get(f"{source_type}_chunks", [])

    # Validate indices and chunk presence
    if not isinstance(chunk_idx, int) or chunk_idx < 0:
        logger.warning(f"Invalid chunk index {chunk_idx} for {source_type}. Resetting.")
        _finish_source(state, source_type)
        return False
        
    if not isinstance(total_chunks, int) or total_chunks < 0:
        logger.warning(f"Invalid total chunks {total_chunks} for {source_type}. Resetting.")
        _finish_source(state, source_type)
        return False
        
    if total_chunks > 0 and not chunks:
        logger.warning(f"Inconsistent state: {total_chunks} {source_type} chunks expected, but none found. Resetting.")
        _finish_source(state, source_type)
        return False
        
    if chunks and chunk_idx >= len(chunks):
        logger.info(f"Chunk index {chunk_idx} reached end of available {source_type} chunks ({len(chunks)}). Finishing source.")
        _finish_source(state, source_type)
        return False

    # Stuck loop detection
    last_index = state.get(f"_last_{source_type}_chunk_idx", -1)
    if chunk_idx == last_index and chunk_idx >= 0:
        stuck_count = state.get(f"_stuck_{source_type}_count", 0) + 1
        state[f"_stuck_{source_type}_count"] = stuck_count
        if stuck_count > 3:
            logger.warning(f"Stuck processing {source_type} chunk {chunk_idx} for {stuck_count} iterations. Forcing completion.")
            _finish_source(state, source_type)
            return False
    else:
        state[f"_stuck_{source_type}_count"] = 0

    # Update last seen index for next check
    state[f"_last_{source_type}_chunk_idx"] = chunk_idx

    # Normal Check - Are there more chunks?
    has_next = chunk_idx < total_chunks

    # Clean up if finishing normally
    if not has_next and total_chunks > 0:
        logger.debug(f"No more {source_type} chunks to process ({chunk_idx}/{total_chunks}). Finishing source.")
        _finish_source(state, source_type)

    return has_next

async def process_generic_chunk(state: ResearchState, source_type: str) -> Dict[str, Any]:
    """
    Process the next chunk for the given source type with deduplication.
    Extracted from the original monolithic implementation for better modularity.
    """
    # Check if processing should occur
    if not should_process_next_chunk(state, source_type):
        logger.debug(f"No more chunks to process for {source_type} or processing skipped.")
        return {
            f"{source_type}_chunks": state.get(f"{source_type}_chunks", []),
            f"{source_type}_chunk_idx": state.get(f"{source_type}_chunk_idx", 0),
            f"{source_type}_total_chunks": state.get(f"{source_type}_total_chunks", 0),
            f"_last_{source_type}_chunk_idx": state.get(f"_last_{source_type}_chunk_idx", -1),
            f"_stuck_{source_type}_count": state.get(f"_stuck_{source_type}_count", 0),
        }

    await asyncio.sleep(0)  # Yield control

    # Get current chunk data
    chunk_idx = state.get(f"{source_type}_chunk_idx", 0)
    chunks = state.get(f"{source_type}_chunks", [])
    total_chunks = state.get(f"{source_type}_total_chunks", 0)

    if chunk_idx >= len(chunks):
        logger.warning(f"Chunk index {chunk_idx} out of bounds for {source_type}. Finishing source.")
        _finish_source(state, source_type)
        return {
            f"{source_type}_chunks": [],
            f"{source_type}_chunk_idx": 0,
            f"{source_type}_total_chunks": 0,
            f"_last_{source_type}_chunk_idx": -1,
            f"_stuck_{source_type}_count": 0,
        }

    current_chunk_items = chunks[chunk_idx]
    logger.info(f"--- Processing {source_type.capitalize()} Chunk {chunk_idx + 1}/{total_chunks} ({len(current_chunk_items)} items) ---")

    # Prepare for processing
    content_hashes = state.get("_content_hashes", set()).copy()
    jira_keys = state.get("_jira_keys_seen", set()).copy()
    perforce_cls = state.get("_perforce_cl_numbers", set()).copy()
    confluence_urls = state.get("_confluence_urls_seen", set()).copy()
    confluence_ids = state.get("_confluence_ids_seen", set()).copy()

    # Process items in the chunk
    unique_items_in_chunk = []
    new_ids_found_in_chunk = set()

    for item in current_chunk_items:
        is_unique = False
        item_id_for_tracking = None

        try:
            if source_type == "jira" and isinstance(item, dict):
                item_id_for_tracking = item.get("key")
                status = item.get("status", "").lower()
                if item_id_for_tracking and item_id_for_tracking not in jira_keys and status != "cancelled":
                    is_unique = True
                    jira_keys.add(item_id_for_tracking)
                    
            elif source_type == "perforce" and isinstance(item, dict):
                item_id_for_tracking = item.get('changelist') or item.get('change') or item.get('change_list')
                if item_id_for_tracking and str(item_id_for_tracking) not in perforce_cls:
                    is_unique = True
                    perforce_cls.add(str(item_id_for_tracking))
                    
            elif source_type == "confluence":
                if isinstance(item, dict):
                    url = item.get("url")
                    page_id = str(item.get("id", ""))
                    
                    if (url and url in confluence_urls) or (page_id and page_id in confluence_ids):
                        is_unique = False
                        item_id_for_tracking = url or f"id:{page_id}"
                        logger.debug(f"Skipping duplicate Confluence item: {item_id_for_tracking}")
                    else:
                        is_unique = True
                        if url:
                            confluence_urls.add(url)
                            item_id_for_tracking = url
                        if page_id:
                            confluence_ids.add(page_id)
                            item_id_for_tracking = item_id_for_tracking or f"id:{page_id}"
                            
                elif isinstance(item, str):
                    # Plain string results - check for URLs
                    import re
                    urls_in_text = re.findall(r'https?://[^\s"\'<>,)]+', item)
                    found_new_url_in_text = False
                    for url in urls_in_text:
                        norm_url = url.strip().rstrip('/')
                        if norm_url and norm_url not in confluence_urls:
                            confluence_urls.add(norm_url)
                            found_new_url_in_text = True
                    is_unique = found_new_url_in_text or not urls_in_text
                    item_id_for_tracking = "text_block"

            if is_unique:
                unique_items_in_chunk.append(item)
                if item_id_for_tracking:
                    new_ids_found_in_chunk.add(str(item_id_for_tracking))

        except Exception as item_proc_err:
            logger.error(f"Error processing item in {source_type} chunk {chunk_idx+1}: {item_proc_err}", exc_info=True)

    # Format and Add Unique Content Block
    formatted_block_content = None
    if unique_items_in_chunk:
        try:
            import json
            header = f"=== {source_type.capitalize()} Results (Chunk {chunk_idx + 1}/{total_chunks}) ===\n"
            
            # Safely convert items to string for JSON dump
            items_to_format = []
            for item in unique_items_in_chunk:
                if isinstance(item, dict):
                    try:
                        items_to_format.append(json.dumps(item, indent=2, default=str))
                    except Exception:
                        items_to_format.append("[Unserializable Dict]")
                elif isinstance(item, list):
                    try:
                        items_to_format.append(json.dumps(item, indent=2, default=str))
                    except Exception:
                        items_to_format.append("[Unserializable List]")
                else:
                    try:
                        items_to_format.append(str(item))
                    except Exception:
                        items_to_format.append("[Unserializable Item]")

            content_str = "\n\n---\n\n".join(items_to_format)
            formatted_block_content = f"{header}{content_str}"

        except Exception as format_err:
            logger.error(f"Error formatting {source_type} chunk {chunk_idx+1}: {format_err}", exc_info=True)
            formatted_block_content = f"=== {source_type.capitalize()} Results (Chunk {chunk_idx + 1}/{total_chunks}) ===\nError: Failed to format chunk content: {format_err}"

    # Add the formatted block if it's new
    block_to_add = None
    if formatted_block_content:
        block_hash = hash(formatted_block_content)
        if block_hash not in content_hashes:
            block_to_add = formatted_block_content
            content_hashes.add(block_hash)
            logger.info(f"Prepared unique content block for {source_type} chunk {chunk_idx + 1} ({len(unique_items_in_chunk)} unique items). New IDs: {len(new_ids_found_in_chunk)}")
        else:
            logger.info(f"Formatted block for {source_type} chunk {chunk_idx + 1} was already seen (hash collision?). Skipping add.")
    else:
        logger.info(f"No unique items found in {source_type} chunk {chunk_idx + 1}, no block generated.")

    # Prepare State Updates
    next_chunk_idx = chunk_idx + 1
    updates: Dict[str, Any] = {
        f"{source_type}_chunk_idx": next_chunk_idx,
        "_jira_keys_seen": jira_keys,
        "_perforce_cl_numbers": perforce_cls,
        "_confluence_urls_seen": confluence_urls,
        "_confluence_ids_seen": confluence_ids,
        "_content_hashes": content_hashes,
        "enterprise_research_results": [block_to_add] if block_to_add else [],
    }
    
    # Keep sources_gathered in sync if a block was added
    if block_to_add:
        updates["sources_gathered"] = [block_to_add]

    # Check if this source is now finished
    if next_chunk_idx >= total_chunks:
        logger.info(f"Finished processing all {total_chunks} chunks for {source_type}.")
        updates[f"{source_type}_chunks"] = []
        updates[f"{source_type}_total_chunks"] = 0
        updates[f"_last_{source_type}_chunk_idx"] = -1
        updates[f"_stuck_{source_type}_count"] = 0
    else:
        updates[f"{source_type}_chunks"] = chunks
        updates[f"{source_type}_total_chunks"] = total_chunks

    return updates

async def process_confluence_chunk(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Process the next chunk of Confluence results."""
    return await process_generic_chunk(state, "confluence")

async def process_jira_chunk(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Process the next chunk of Jira results."""
    return await process_generic_chunk(state, "jira")

async def process_perforce_chunk(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Process the next chunk of Perforce results."""
    logger.debug("Entering process_perforce_chunk node")
    result = await process_generic_chunk(state, "perforce")
    logger.debug("Exiting process_perforce_chunk node")
    return result

def route_after_research(state: Dict[str, Any]) -> str:
    """
    Decide the next step after enterprise_research or a chunk processing node.
    Prioritizes processing remaining chunks before moving to filtering/analysis.
    """
    # Safety Checks
    hop_counter = state.get("__hop_counter", 0) + 1
    state["__hop_counter"] = hop_counter
    
    if hop_counter > MAX_CHUNK_PROCESSING_HOPS:
        logger.error(f"Maximum graph hops ({MAX_CHUNK_PROCESSING_HOPS}) exceeded. Forcing summarization.")
        reset_all_chunking(state)
        return "finalize_summary"

    # Prioritize Chunk Processing
    if should_process_next_chunk(state, "confluence"):
        logger.debug(f"Routing to process_confluence_chunk: {state.get('confluence_chunk_idx', 0)}/{state.get('confluence_total_chunks', 0)}")
        return "process_confluence_chunk"
    elif should_process_next_chunk(state, "jira"):
        logger.debug(f"Routing to process_jira_chunk: {state.get('jira_chunk_idx', 0)}/{state.get('jira_total_chunks', 0)}")
        return "process_jira_chunk"
    elif should_process_next_chunk(state, "perforce"):
        logger.debug(f"Routing to process_perforce_chunk: {state.get('perforce_chunk_idx', 0)}/{state.get('perforce_total_chunks', 0)}")
        return "process_perforce_chunk"
    else:
        # All Chunk Processing Complete
        logger.info("All chunk processing complete. Moving to filter_irrelevant_data.")
        return "filter_irrelevant_data"

def enterprise_research(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Set up chunked processing of enterprise search results.
    Skip fresh search if chunk processing is already in progress.
    """
    # Check if chunk processing is ongoing
    if (should_process_next_chunk(state, "confluence") or
            should_process_next_chunk(state, "jira") or
            should_process_next_chunk(state, "perforce")):
        logger.debug("Chunk processing in progress – skipping fresh enterprise search setup.")
        return {}  # Return empty dict to keep state unchanged

    logger.info("--- Setting up Enterprise Research Chunks ---")
    
    # Increment loop count only when starting a new search cycle
    research_loop_count = state.get("research_loop_count", 0) + 1

    # Apply source-specific limits
    MAX_JIRA_ISSUES = 100
    MAX_CONFLUENCE_PAGES = 40
    MAX_PERFORCE_CLS = 60

    # Apply limits to raw results
    raw_jira = state.get("raw_jira_results", [])
    raw_confluence = state.get("raw_confluence_results", [])
    raw_perforce = state.get("raw_perforce_results", [])

    if len(raw_jira) > MAX_JIRA_ISSUES:
        logger.info(f"Limiting Jira results from {len(raw_jira)} to {MAX_JIRA_ISSUES}")
        raw_jira = raw_jira[:MAX_JIRA_ISSUES]

    if len(raw_confluence) > MAX_CONFLUENCE_PAGES:
        logger.info(f"Limiting Confluence results from {len(raw_confluence)} to {MAX_CONFLUENCE_PAGES}")
        raw_confluence = raw_confluence[:MAX_CONFLUENCE_PAGES]

    if len(raw_perforce) > MAX_PERFORCE_CLS:
        logger.info(f"Limiting Perforce results from {len(raw_perforce)} to {MAX_PERFORCE_CLS}")
        raw_perforce = raw_perforce[:MAX_PERFORCE_CLS]

    # Better Chunk Setup with Smaller Single Blocks
    new_chunk_setup = {
        "jira": (raw_jira, min(MAX_JIRA_ISSUES, 20)),
        "confluence": (raw_confluence, min(MAX_CONFLUENCE_PAGES, 10)),
        "perforce": (raw_perforce, min(MAX_PERFORCE_CLS, 15))
    }

    total_chunks_created = 0
    state_updates: Dict[str, Any] = {}

    for source_type, (results_list, chunk_size) in new_chunk_setup.items():
        if results_list and isinstance(results_list, list):
            # Use a single chunk when possible to avoid repetitive processing
            if len(results_list) <= chunk_size:
                chunks = [results_list]
                logger.info(f"Created single chunk for {source_type} ({len(results_list)} items)")
            else:
                # Create multiple chunks with minimal overlap
                chunks = []
                for i in range(0, len(results_list), chunk_size):
                    chunks.append(results_list[i:i + chunk_size])
                logger.info(f"Created {len(chunks)} chunks for {source_type} ({len(results_list)} items, size={chunk_size})")

            state_updates[f"{source_type}_chunks"] = chunks
            state_updates[f"{source_type}_chunk_idx"] = 0
            state_updates[f"{source_type}_total_chunks"] = len(chunks)
            total_chunks_created += len(chunks)
        else:
            # No results or invalid format
            state_updates[f"{source_type}_chunks"] = []
            state_updates[f"{source_type}_chunk_idx"] = 0
            state_updates[f"{source_type}_total_chunks"] = 0

    logger.info(f"Chunk setup complete. {total_chunks_created} total chunks created for processing.")

    # Ensure Correct Totals
    for source_type in new_chunk_setup.keys():
        if f"{source_type}_chunks" in state_updates and state_updates[f"{source_type}_chunks"]:
            state_updates[f"{source_type}_total_chunks"] = len(state_updates[f"{source_type}_chunks"])
        elif f"{source_type}_total_chunks" not in state_updates:
            state_updates[f"{source_type}_total_chunks"] = 0

    # Reset tracking fields
    state_updates.update({
        "enterprise_research_results": [],
        "sources_gathered": [],
        "_content_hashes": set(),
        "research_loop_count": research_loop_count,
        "__hop_counter": 0,
        "_last_confluence_chunk_idx": -1,
        "_last_jira_chunk_idx": -1,
        "_last_perforce_chunk_idx": -1,
        "_stuck_confluence_count": 0,
        "_stuck_jira_count": 0,
        "_stuck_perforce_count": 0,
        "raw_jira_results": raw_jira,
        "raw_confluence_results": raw_confluence,
        "raw_perforce_results": raw_perforce
    })

    return state_updates

# Add the missing function that graph_3.py is trying to import
async def process_chunks(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Process chunks from multiple sources sequentially, respecting source priorities.
    This function acts as a wrapper around the individual chunk processing functions.
    
    Args:
        state: The current research state
        config: Optional configuration
        
    Returns:
        Updated state dictionary with processed chunks
    """
    logger.info("Processing chunks from all sources")
    
    # First check if any source needs processing
    if not (should_process_next_chunk(state, "confluence") or 
            should_process_next_chunk(state, "jira") or 
            should_process_next_chunk(state, "perforce")):
        logger.info("No chunks to process from any source")
        return {}
    
    # Process in priority order: Jira, Confluence, Perforce
    if should_process_next_chunk(state, "jira"):
        logger.info("Processing Jira chunk")
        return await process_jira_chunk(state, config)
    
    if should_process_next_chunk(state, "confluence"):
        logger.info("Processing Confluence chunk")
        return await process_confluence_chunk(state, config)
    
    if should_process_next_chunk(state, "perforce"):
        logger.info("Processing Perforce chunk")
        return await process_perforce_chunk(state, config)
    
    # Shouldn't reach here due to the first check, but just in case
    return {}
