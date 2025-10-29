# src/assistant/graph_2.py (Refactored - Complete Code)

# --- Standard Imports ---
import asyncio
import gc
import inspect  # Keep inspect for checking function signatures if needed
import json
import logging
import operator
import os
import re
import sys
import urllib.parse  # Add this import at the top with other imports
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import List, TypedDict, Dict, Any, Optional, Union, Annotated, Set, cast, SupportsIndex
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from src.assistant.tools.PerforceTool import PerforceHelper

# --- Binary Content Processing Imports ---
try:
    from PIL import Image
    import io
    import base64
    import pytesseract
    import openpyxl
    import pdfplumber
    import numpy as np
    HAS_BINARY_PROCESSING = True
    logging.info("Binary content processing libraries loaded successfully.")
except ImportError:
    HAS_BINARY_PROCESSING = False
    # Create placeholder classes
    class ImagePlaceholder:
        open = lambda *args, **kwargs: None
    Image = ImagePlaceholder()
    pytesseract = type('pytesseract', (), {'image_to_string': lambda *args, **kwargs: ""})
    openpyxl = type('openpyxl', (), {'load_workbook': lambda *args, **kwargs: None})
    pdfplumber = type('pdfplumber', (), {'open': lambda *args, **kwargs: None})
    logging.warning("Binary content processing libraries not available. Install PIL, pytesseract, openpyxl, and pdfplumber to enable this functionality.")

# --- Helper Function for Type Casting ---
def ensure_index(value: int) -> SupportsIndex:
    """Helper function to ensure an integer value is properly cast to SupportsIndex"""
    return cast(SupportsIndex, value)

# --- Langchain/LangGraph Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph



# --- Memory Management Constants ---
MAX_SNIPPET_SIZE = 10000  # Maximum size for code snippets
MAX_CONFLUENCE_CONTENT_SIZE = 50000  # Maximum size for Confluence content
MAX_JIRA_DESC_SIZE = 30000  # Maximum size for Jira descriptions
MAX_LLM_INPUT_SIZE = 32000  # Maximum context size for LLM inputs
MAX_SAFE_RESULTS = 100  # Maximum number of results to process
# --- END MEMORY CONSTANTS ---

# --- ADDED: Disable LangSmith traceable decorator to speed up imports and prevent heavy dependencies ---
def traceable(func=None, **kwargs):
    """
    Modified decorator that implements size-limiting for LangSmith tracing.
    When tracing large payloads, this will truncate the data to stay under
    LangSmith's size limits.
    """
    if func is None:
        return lambda f: f

    async def wrapper(*args, **kwargs):
        # Execute the function normally without tracing
        result = await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)

        # Return result directly - Skip all LangSmith integration
        # This prevents the "Content length exceeds maximum size limit" error
        return result

    # For non-async functions, provide a sync version
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Return appropriate wrapper based on function type
    if inspect.iscoroutinefunction(func):
        return wrapper
    return sync_wrapper
# from langsmith import traceable  # Disabled LangSmith tracing

from src.assistant.configuration import Configuration


# --- Project-Specific Imports ---

# Configuration and State
# Assuming configuration.py defines Configuration class and ConfigurationError exception

# --- MODIFIED: ResearchState definition ---
# (Ensure this TypedDict accurately reflects ALL fields used in the graph)
class ResearchState(TypedDict, total=False):
    # --- Core state ---
    research_topic: str
    search_query: str
    running_summary: str # Will be assembled at the end now
    research_loop_count: int
    messages: Annotated[List[BaseMessage], operator.add] # Chat history

    # --- Raw Tool Results (Structured) ---
    raw_jira_results: List[Dict[str, Any]]
    raw_confluence_results: List[Union[Dict[str, Any], str]]
    raw_perforce_results: List[Dict[str, Any]]
    raw_web_search_results: List[Dict[str, Any]] # Added for consistency
    
    # --- Binary Content Processing ---
    raw_binary_attachments: List[Dict[str, Any]]
    processed_binary_content: Dict[str, List[Dict[str, Any]]]
    binary_content_summaries: Dict[str, str]
    image_analysis_results: List[Dict[str, Any]]
    excel_data_extracts: List[Dict[str, Any]]
    diagram_interpretations: List[Dict[str, Any]]

    # --- Chunking State ---
    jira_chunks: List[List[Dict[str, Any]]]
    jira_chunk_idx: int
    jira_total_chunks: int
    confluence_chunks: List[List[Union[Dict[str, Any], str]]]
    confluence_chunk_idx: int
    confluence_total_chunks: int
    perforce_chunks: List[List[Dict[str, Any]]]
    perforce_chunk_idx: int
    perforce_total_chunks: int

    # --- Processed Results (Formatted Strings for Summary/Display) ---
    enterprise_research_results: Annotated[List[str], operator.add]
    sources_gathered: Annotated[List[str], operator.add]

    # --- ADDED: Pre-processed Summarization Input & Individual Summaries ---
    pre_processed_summaries: Dict[str, List[str]] # Keys like '### JIRA', values are lists of bullet points
    jira_summary: str
    confluence_summary: str
    perforce_summary: str
    web_summary: str
    # --- END ADDED ---

    # --- Analysis Results ---
    security_findings: List[Dict[str, Any]]
    technical_validation: Dict[str, Any]
    reflection_data: Union[Dict[str, Any], str]
    perforce_diff_analysis: Dict[str, Any]
    verification_results: Dict[str, Any]
    citations: List[Dict[str, Any]]
    knowledge_graph: Dict[str, Any]
    # --- ADDED: Missing Analysis Keys ---
    cross_source_analysis_text: str # From analyze_cross_source_connections
    refined_summary_context: str # From refine_combined_summaries
    quiz_questions: List[str] # From answer_quiz_questions (topic parsing)
    quiz_answers: List[Dict[str, str]] # From answer_quiz_questions
    identified_corrections: List[str] # List of potential corrections/contradictions
    # --- END ADDED ---

    # --- Deduplication & Tracking ---
    _content_hashes: Set[int] # Hashes of formatted string blocks
    _jira_keys_seen: Set[str]
    _perforce_cl_numbers: Set[str]
    _confluence_urls_seen: Set[str] # Tracks URLs seen during chunk processing
    _confluence_ids_seen: Set[str] # Tracks IDs seen during chunk processing

    # --- Control Flow & Meta ---
    alternative_terms: List[str]
    _relevance_check: str # 'relevant' or 'irrelevant'
    __hop_counter: int # Routing hop counter for loop detection
    # --- Added missing keys for stuck loop detection ---
    _last_confluence_chunk_idx: int
    _last_jira_chunk_idx: int
    _last_perforce_chunk_idx: int
    _stuck_confluence_count: int
    _stuck_jira_count: int
    _stuck_perforce_count: int
    # --- End Added Keys ---
    # Error handling
    _errors: List[Dict[str, Any]]
    _recovery_actions: List[str]
    _recovery_metadata: Dict[str, Any]
    # Export
    export_format: Optional[str]
    exported_data: Dict[str, Any]
    # Runtime config from LangGraph
    config: Dict[str, Any]
    # --- ADDED: Topic Classification ---
    topic_classification: str # e.g., "Deep Research", "Quiz", "Simple QA"
    # --- END ADDED ---
# --- END ResearchState definition ---

# --- Tool Imports ---
# Import the actual tool functions/classes we intend to use
# Using the class-based tool for Jira
from src.assistant.tools.jira_tool import JiraProject, ConfigurationError
# Using the function-based tools for Confluence and Perforce
from src.assistant.tools.tool_confluence import search_confluence as tool_search_confluence # Renamed to avoid conflict
from src.assistant.tools.tool_perforce import search_perforce as tool_search_perforce # Renamed to avoid conflict
# Assuming SecurityAnalyzer exists and has the static method analyze_file_content
from src.assistant.tools.security_analyzer import SecurityAnalyzer

# --- Utility Imports ---
# Assuming utils directory is one level up from assistant
from src.utils.ssl_fix import patch_ssl_for_huggingface, with_ssl_disabled, initialize_ssl_fixes
# Assuming prompts exist in utils/prompts.py relative to src/
from src.utils.prompts import (
    security_analysis_instructions,
    technical_validation_instructions,
    # --- ADDED: Import reflection and cross-source prompts ---
    # reflection_prompt as default_reflection_prompt,
    # cross_source_connection_prompt as default_cross_source_prompt,
    # --- END ADDED ---
    # Add other necessary prompts here if they exist
    # e.g., kg_generation_prompt
)
# Import web search tool if used
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    DDGS = None # Define placeholder
    logging.warning("DuckDuckGo Search library not found. Web search functionality disabled.")

# Optional Trafilatura Import
try:
    import trafilatura # type: ignore
except ImportError:
    trafilatura = None
    logging.warning("Optional dependency 'trafilatura' not found; web content extraction will be limited.")

# Optional visualization libraries
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_VISUALIZATION = True
except ImportError:
    # (Keep placeholder setup as before)
    logging.warning("Visualization libraries (networkx, matplotlib, pandas) not available.")
    HAS_VISUALIZATION = False
    class NXPlaceholder:
        DiGraph = object
        spring_layout = lambda *args, **kwargs: {}
        draw_networkx_nodes = lambda *args, **kwargs: None
        draw_networkx_edges = lambda *args, **kwargs: None
        draw_networkx_labels = lambda *args, **kwargs: None
    class PLTPlaceholder:
        figure = lambda *args, **kwargs: None
        Line2D = lambda *args, **kwargs: object()
        cm = type('cm', (), {'tab10': lambda x: x})()
        legend = lambda *args, **kwargs: None
        title = lambda *args, **kwargs: None
        axis = lambda *args, **kwargs: None
        tight_layout = lambda *args, **kwargs: None
        show = lambda *args, **kwargs: None
        close = lambda *args, **kwargs: None
        savefig = lambda *args, **kwargs: None
    class PDPlaceholder:
        DataFrame = lambda *args, **kwargs: None
        json_normalize = lambda *args, **kwargs: None
        ExcelWriter = lambda *args, **kwargs: None
    nx = NXPlaceholder()
    plt = PLTPlaceholder()
    pd = PDPlaceholder()

# --- Environment & Config Setup ---
initialize_ssl_fixes()
# Load environment variables (consider moving to Configuration class)
from dotenv import load_dotenv
load_dotenv()

# Ensure Python path includes src directory if running directly
src_dir = Path(__file__).parent.parent.parent # Adjust based on actual file location relative to src
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Logging setup
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG', 'false').lower() == 'true' else logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Variables (Load from environment or config file)
OLLAMA_BASE_URL = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3:30b-a3b') # Default model
JIRA_SERVER = os.getenv('JIRA_SERVER')
JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN') # Prefer token over password
JIRA_CONNECTION_TIMEOUT = int(os.getenv('JIRA_CONNECTION_TIMEOUT', '60')) # Increased default timeout
CONFLUENCE_DOMAIN = os.getenv('SOLUTIONBOOK_DOMAIN', 'solutionbook.amdocs.com') # Example domain
# Add Perforce config vars if needed by tool_search_perforce directly
P4_PORT = os.getenv('P4PORT')
P4_USER = os.getenv('P4USER')
P4_PASSWD = os.getenv('P4PASSWD')
# Add other necessary config vars (proxies, SSL paths, etc.)

# Constants
CONFLUENCE_CHUNK_SIZE = 5 # Number of pages/items per chunk
JIRA_CHUNK_SIZE = 10 # Number of issues per chunk
PERFORCE_CHUNK_SIZE = 10 # Number of changes per chunk
# --- ADDED: Overlap Sizes ---
CONFLUENCE_OVERLAP = 1 # Number of items to overlap (e.g., pages)
JIRA_OVERLAP = 2       # Number of items to overlap (e.g., issues)
PERFORCE_OVERLAP = 1   # Number of items to overlap (e.g., CLs)
# --- END ADDED ---
MAX_CHUNK_PROCESSING_HOPS = 50 # Safety limit for chunk processing loops
MAX_TOTAL_HOPS = 100 # Overall safety limit for graph execution
MAX_SUMMARY_CONTEXT = 30000 # Max chars for summary LLM context

# --- ADDED: Central LLM Configuration ---
# Parameters optimized for DeepSeek-R1:14B report generation/summarization
# Note: num_ctx might need adjustment based on available VRAM
LLM_CONFIG_PARAMS = {
    "temperature": 0.15,       # Low temp for factual consistency
    "num_ctx": 8192,           # Increased context window (adjust based on hardware)
    "top_p": 0.7,              # Moderate top_p for some flexibility
    "repeat_penalty": 1.15,    # Helps prevent repetitive output
    # "top_k": 40,             # Usually set top_p OR top_k, not both
}
# --- END ADDED ---

# --- Helper Functions ---

def sanitize_query_for_jira(query: str) -> str:
    """Remove Ollama error messages and sanitize query for JIRA."""
    if not isinstance(query, str): return "" # Handle non-string input
    if "[Ollama Error:" in query:
        query = re.sub(r"(\[Ollama Error:[^]]*])", "", query)
    # Remove characters problematic for JQL text search
    query = re.sub(r'[\\/*?"<>|~^(){}\[\]!:]', ' ', query) # Expanded list
    # Escape JQL reserved words if they appear standalone (simple approach)
    # reserved = {'and', 'or', 'not', 'empty', 'null', 'order', 'by', 'asc', 'desc'}
    # query_parts = query.split()
    # query = " ".join(f'"{part}"' if part.lower() in reserved else part for part in query_parts)
    # Escape single quotes for JQL strings
    query = query.replace("'", "''")
    return query.strip()
def clean_llm_output(text: str) -> str:
    """Clean up special tags and formatting issues in LLM output"""
    if not isinstance(text, str): return "" # Handle non-string input
    try:
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove other common XML-like tags used for reasoning
        tags_to_remove = ['assistant', 'thinking', 'reasoning', 'context', 'analysis']
        for tag in tags_to_remove:
             text = re.sub(rf'</?{tag}[^>]*>', '', text)
        # Normalize markdown headings (ensure space after #)
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
             stripped_line = line.strip()
             if stripped_line.startswith('#'):
                  # Ensure space after hashes
                  line = re.sub(r'^(#+)([^#\s])', r'\1 \2', stripped_line)
             cleaned_lines.append(line)
        cleaned_text = '\n'.join(cleaned_lines).strip()

        # --- NEW: Strip enclosing triple-backtick fences if entire content is fenced ---
        if cleaned_text.startswith('```') and cleaned_text.rstrip().endswith('```'):
            # Remove the first line (``` or ```markdown) and the last line (```)
            content_lines = cleaned_text.splitlines()
            # Identify language spec in first line
            if len(content_lines) >= 2:
                # Drop first and last line
                cleaned_text = '\n'.join(content_lines[1:-1]).strip()

        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning LLM output: {e}")
        return text # Return original on error

def sanitize_markdown_links(text: str) -> str:
    """Clean common formatting issues in markdown links produced by LLMs."""
    if not isinstance(text, str): return "" # Handle non-string input
    # Regex to find markdown links, ensuring it doesn't capture nested brackets incorrectly
    # Making the label capture non-greedy and ensuring URL doesn't start with space
    # --- FIXED: Corrected unterminated string literal by removing trailing backslash ---
    link_pattern = re.compile(r"\[([^\]]+?)\]\(([^)\s][^)]*)\)")
    # --- END FIX ---

    def _clean_url(raw_url: str) -> str:
        # Trim leading/trailing whitespace and common stray characters
        cleaned = raw_url.strip().strip("'\" .,;:\\")
        # Remove potential leading/trailing brackets if they leaked in
        cleaned = cleaned.strip('()')

        # --- NEW: URL Encode parentheses within the path ---
        try:
            parsed = urllib.parse.urlparse(cleaned)
            # Only encode if it looks like a real URL (has scheme or netloc)
            if parsed.scheme or parsed.netloc:
                 path_encoded = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe='/%')
                 cleaned = urllib.parse.urlunparse((
                      parsed.scheme,
                      parsed.netloc,
                      path_encoded,
                      parsed.params,
                      parsed.query,
                      parsed.fragment
                 ))
            # Else: Assume it's a relative path or anchor, leave as is for now
            # (Further refinement might be needed for complex relative paths)
        except ValueError:
             logger.warning(f"Could not parse URL for encoding: {cleaned}")
             # Fallback to original cleaned version if parsing fails
        # --- END NEW ---

        # Re-check http/https prefix if missing after potential encoding
        if not cleaned.startswith(('http://', 'https://', '#', '/')):
             # If it looks like a domain, assume https
             if '.' in cleaned and '/' not in cleaned.split('.')[0]:
                  cleaned = 'https://' + cleaned
        return cleaned

    def _replacer(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        cleaned_url = _clean_url(url)
        # Ensure label doesn't contain nested markdown links inadvertently
        clean_label = re.sub(r'\[[^\]]*\]\([^)]*\)', '', label).strip()
        if not clean_label: clean_label = "link" # Use placeholder if label becomes empty
        return f"[{clean_label}]({cleaned_url})"

    try:
        return link_pattern.sub(_replacer, text)
    except Exception as e:
        logger.error(f"Error sanitizing markdown links: {e}")
        return text # Return original on error

# --- Refactored Tool Wrapper Functions ---

async def search_jira_wrapper(
        query: str,
        config: RunnableConfig | None = None,
        *,
        max_results: Optional[int] = None,
        get_hierarchy: bool = False,
        download_attachments: bool = True
) -> List[Dict[str, Any]]:
    """
    Wrapper for JIRA search using JiraProject. Standardizes output.

    Args:
        query (str): The search query.
        config (RunnableConfig | None): LangGraph runtime config (optional).
        max_results (Optional[int]): Max results to return. 0 or None implies unlimited (or default if underlying tool doesn't support unlimited).
        get_hierarchy (bool): If True, retrieves parent/child/epic relationships for direct issue keys
        download_attachments (bool): If True, downloads and processes attachment content

    Returns:
        List[Dict[str, Any]]: List of issue dictionaries, or empty list on error.
    """
    logger.debug(f"Executing JIRA search wrapper for query: {query}")

    # Use globally loaded env vars for simplicity here
    if not JIRA_SERVER or not JIRA_API_TOKEN:
        logger.error("JIRA_SERVER or JIRA_API_TOKEN not configured.")
        return []

    try:
        # Ensure URL schema
        server_url = JIRA_SERVER
        if not server_url.startswith(('http://', 'https://')):
            server_url = f'https://{server_url}'  # Default to https

        # Initialize the tool
        jira_tool = JiraProject(
            server=server_url,
            token=JIRA_API_TOKEN,
        )

        # Sanitize query before passing to tool
        clean_query = sanitize_query_for_jira(query)

        # Determine max_results for the JiraProject tool
        jira_max_results = 50  # Default value if not specified
        if max_results is not None:
            if max_results <= 0:
                jira_max_results = 100  # Use a larger number for 'unlimited'
            else:
                jira_max_results = max_results

        # Check if query is a specific JIRA key
        is_key = re.match(r'^[A-Z]+-\d+$', clean_query)

        # If using get_hierarchy option AND query is a specific issue key,
        # fetch the full hierarchy instead of doing a regular search
        if get_hierarchy and is_key:
            logger.info(f"Getting issue hierarchy for {clean_query}")
            loop = asyncio.get_running_loop()

            try:
                # Get hierarchy with attachments if requested
                hierarchy = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: jira_tool.get_issue_hierarchy(clean_query, download_attachments=download_attachments)
                    ),
                    timeout=JIRA_CONNECTION_TIMEOUT
                )

                # Convert hierarchy to a list of dicts (expected output format)
                result_list = []

                # Add the main issue
                main_issue = jira_tool.get_issue_details(clean_query, download_attachments=download_attachments)
                if main_issue and not main_issue.get("error"):
                    result_list.append(main_issue)

                # Add parent if exists
                if hierarchy.get("parent"):
                    parent_key = hierarchy["parent"].get("key")
                    if parent_key:
                        parent_details = jira_tool.get_issue_details(parent_key,
                                                                     download_attachments=download_attachments)
                        if parent_details and not parent_details.get("error"):
                            result_list.append(parent_details)

                # Add epic if exists and is different from parent
                if hierarchy.get("epic") and hierarchy["epic"].get("key") != hierarchy.get("parent", {}).get("key"):
                    epic_key = hierarchy["epic"].get("key")
                    if epic_key:
                        epic_details = jira_tool.get_issue_details(epic_key, download_attachments=download_attachments)
                        if epic_details and not epic_details.get("error"):
                            result_list.append(epic_details)

                # Add children and subtasks (limit to avoid excessive processing)
                MAX_CHILDREN = 5
                for child in (hierarchy.get("children", []) + hierarchy.get("subtasks", []))[:MAX_CHILDREN]:
                    if isinstance(child, dict) and child.get("key"):
                        child_key = child.get("key")
                        child_details = jira_tool.get_issue_details(child_key,
                                                                    download_attachments=download_attachments)
                        if child_details and not child_details.get("error"):
                            result_list.append(child_details)

                logger.info(f"Retrieved hierarchy with {len(result_list)} related issues for {clean_query}")
                return result_list

            except asyncio.TimeoutError:
                logger.error(f"Hierarchy retrieval timed out for {clean_query}")
                # Fall back to regular search if hierarchy retrieval fails
            except Exception as e:
                logger.error(f"Error getting hierarchy for {clean_query}: {e}", exc_info=True)
                # Fall back to regular search if hierarchy retrieval fails

        # Execute regular search if not using hierarchy or if hierarchy retrieval failed
        loop = asyncio.get_running_loop()
        issues_list = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: jira_tool.robust_search_issues(clean_query, max_results=jira_max_results)
            ),
            timeout=JIRA_CONNECTION_TIMEOUT
        )

        # Validate and return results
        if issues_list is None:
            logger.error("Jira tool returned None.")
            return []

        if not isinstance(issues_list, list):
            logger.error(f"Jira tool returned non-list type: {type(issues_list)}")
            # Attempt to handle if it's a dict containing issues
            if isinstance(issues_list, dict) and 'issues' in issues_list and isinstance(issues_list['issues'], list):
                issues_list = issues_list['issues']
            else:
                return []

        # If download_attachments is requested, get full details for each issue
        if download_attachments:
            detailed_results = []
            # Process a limited number to avoid excessive processing
            MAX_DETAILS_TO_FETCH = min(len(issues_list), 10)

            for i, issue in enumerate(issues_list[:MAX_DETAILS_TO_FETCH]):
                if isinstance(issue, dict) and issue.get("key"):
                    try:
                        # Get details including attachments
                        details = jira_tool.get_issue_details(issue["key"], download_attachments=True)
                        if details and not details.get("error"):
                            detailed_results.append(details)
                        else:
                            # Fall back to the original issue data
                            detailed_results.append(issue)
                    except Exception as e:
                        logger.error(f"Error getting details for {issue.get('key')}: {e}")
                        # Fall back to the original issue data
                        detailed_results.append(issue)
                else:
                    # Invalid issue format, add as is
                    detailed_results.append(issue)

            # Add remaining issues without downloading attachments
            for issue in issues_list[MAX_DETAILS_TO_FETCH:]:
                detailed_results.append(issue)

            issues_list = detailed_results

        # Ensure items in list are dicts
        validated_issues = [item for item in issues_list if isinstance(item, dict)]
        if len(validated_issues) != len(issues_list):
            logger.warning("Some items returned by Jira tool were not dictionaries.")

        logger.info(f"JIRA search wrapper successful, found {len(validated_issues)} valid issues.")
        return validated_issues

    except ConfigurationError as config_err:
        logger.error(f"JiraProject Configuration Error in wrapper: {config_err}")
        return []
    except asyncio.TimeoutError:
        logger.error(f"JIRA search wrapper timed out after {JIRA_CONNECTION_TIMEOUT} seconds.")
        return []
    except Exception as e:
        logger.error(f"Error in search_jira_wrapper: {e}", exc_info=True)
        return []


async def search_confluence_wrapper(
        query: str,
        config: RunnableConfig | None = None,
        *,
        max_results: Optional[int] = None,
        space_key: Optional[str] = None,
        include_attachments: bool = False
) -> List[Union[Dict[str, Any], str]]:
    """
    Wrapper for Confluence search. Standardizes output.

    Args:
        query (str): The search query.
        config (RunnableConfig | None): LangGraph runtime config (optional).
        max_results (Optional[int]): Max results to return. 0 or None implies unlimited (or default if underlying tool doesn't support unlimited).
        space_key (Optional[str]): Space key to limit search to (optional).
        include_attachments (bool): If True, download and process attachments

    Returns:
        List[Union[Dict[str, Any], str]]: List of results (dicts preferred), or list containing error string.
    """
    logger.debug(f"Executing Confluence search wrapper for query: {query}")

    try:
        # Create parameters object for tool_search_confluence
        tool_params = {"query": query}

        # Add config if expected
        sig = inspect.signature(tool_search_confluence)
        if 'config' in sig.parameters:
            tool_params["config"] = config

        # Add max_results if supported
        if 'max_results' in sig.parameters and max_results is not None:
            if max_results <= 0:
                tool_params["max_results"] = 40  # Use 40 as reasonable limit
            else:
                tool_params["max_results"] = max_results

        # Add space_key if supported and provided
        if 'space_key' in sig.parameters and space_key:
            tool_params["space_key"] = space_key

        # Add include_attachments if supported
        if 'include_attachments' in sig.parameters:
            tool_params["include_attachments"] = include_attachments

        # Determine if the imported tool function is async or sync
        if inspect.iscoroutinefunction(tool_search_confluence):
            logger.debug("Calling async tool_search_confluence")
            result = await asyncio.wait_for(tool_search_confluence(**tool_params), timeout=90)
        else:
            logger.debug("Calling sync tool_search_confluence in executor")
            loop = asyncio.get_running_loop()
            from functools import partial
            result = await asyncio.wait_for(
                loop.run_in_executor(None, partial(tool_search_confluence, **tool_params)),
                timeout=90
            )

        # Process result
        result_data = []
        if isinstance(result, list):
            result_data = result
        elif isinstance(result, dict):
            if 'results' in result and isinstance(result['results'], list):
                result_data = result['results']
            elif 'pages' in result and isinstance(result['pages'], list):
                result_data = result['pages']
            else:
                result_data = [result]
        elif isinstance(result, str):
            try:
                parsed_json = json.loads(result)
                if isinstance(parsed_json, list):
                    result_data = parsed_json
                elif isinstance(parsed_json, dict):
                    if 'results' in parsed_json and isinstance(parsed_json['results'], list):
                        result_data = parsed_json['results']
                    elif 'pages' in parsed_json and isinstance(parsed_json['pages'], list):
                        result_data = parsed_json['pages']
                    else:
                        result_data = [parsed_json]
                else:
                    result_data = [result]
            except json.JSONDecodeError:
                if result.startswith("Error:") or "No results found" in result:
                    result_data = [result]
                else:
                    result_data = [{"content": result, "url": None, "title": "Text Result"}]
        elif result is None:
            logger.warning("Confluence tool returned None.")
            result_data = []
        else:
            logger.warning(f"Unexpected Confluence result type: {type(result)}")
            result_data = [f"Error: Unexpected result type {type(result)}"]

        # Validate items in the list
        validated_data = [item for item in result_data if isinstance(item, (dict, str))]
        if len(validated_data) != len(result_data):
            logger.warning("Some items returned by Confluence tool were not dictionaries or strings.")

        logger.info(f"Confluence search wrapper successful, obtained {len(validated_data)} result items.")
        return validated_data

    except asyncio.TimeoutError:
        logger.error("Confluence search wrapper timed out.")
        return ["Error: Confluence search timed out."]
    except Exception as e:
        logger.error(f"Error in search_confluence_wrapper: {e}", exc_info=True)
        return [f"Error searching Confluence: {str(e)}"]
@traceable # Optional: Add tracing
@with_ssl_disabled # Apply SSL disabling if needed by the underlying tool
async def search_perforce_wrapper(query: str, config: RunnableConfig | None = None, *, max_results: Optional[int] = None, force_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Wrapper for Perforce search. Standardizes output to List[Dict].

    Args:
        query (str): The search query.
        config (RunnableConfig | None): LangGraph runtime config (optional).
        max_results (Optional[int]): Max results to return. 0 or None implies unlimited (or default if underlying tool doesn't support true unlimited).
        force_path (Optional[str]): If specified, limit the search to this branch path.

    Returns:
        List[Dict[str, Any]]: List of change dictionaries, or empty list on error.
    """
    logger.debug(f"Executing Perforce search wrapper for query: '{query}' with max_results={max_results}, force_path={force_path}")
    patch_ssl_for_huggingface()

    perforce_helper = None
    result_data: List[Dict[str, Any]] = []  # Default empty result

    try:
        # Initialize Perforce helper for search operation
        logger.debug("Initializing Perforce helper")
        perforce_helper = PerforceHelper()
        loop = asyncio.get_running_loop()

        # --- ENHANCED: Enhanced logging of P4 environment ---
        env_info = {
            "P4CLIENT": os.environ.get("P4CLIENT", "Not set"),
            "P4PORT": os.environ.get("P4PORT", "Not set"),
            "P4USER": os.environ.get("P4USER", "Not set"),
            "DEPOT_PATH": os.environ.get("DEPOT_PATH", "Not set"),
            "P4_TARGET_PATHS": os.environ.get("P4_TARGET_PATHS", "Not set"),
        }
        logger.info(f"Perforce environment: {env_info}")
        # --- END ENHANCED ---

        # --- MODIFIED: Add hard safety limits for result size ---
        # Determine override for max_results: Use a large number for 0/None, else use the provided positive value.
        # P4 usually defaults if None is passed, so we need a specific large number for 'all'.
        UNLIMITED_P4_MARKER = 100  # Lower this to a safer value (was 9999)
        if max_results is None or max_results <= 0:
            override_max = UNLIMITED_P4_MARKER
            logger.debug(f"Treating max_results={max_results} as 'unlimited', using marker: {override_max}")
        else:
            override_max = min(max_results, UNLIMITED_P4_MARKER)  # Apply safety cap
        # --- END MODIFICATION ---

        # --- ENHANCED: Try multiple search strategies ---
        # First, try the original query
        logger.debug(f"Searching Perforce for query: '{query}' with max_results_override={override_max}")

        # --- NEW: Add timeout for Perforce operations ---
        try:
            raw_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    perforce_helper.search_changelists_by_keyword,
                    query,
                    None,  # status
                    force_path,  # Use the provided force_path for limiting search scope
                    override_max  # Use the determined max_results override with safety limit
                ),
                timeout=60  # Add 60-second timeout to prevent hanging
            )
        except asyncio.TimeoutError:
            logger.warning(f"Perforce search timed out after 60 seconds for query: '{query}'")
            raw_result = []  # Set empty result on timeout
        # --- END NEW ---

        # Rest of the search logic...
        # [... existing search implementation ...]

        # If no results, try different naming conventions automatically
        if not raw_result or len(raw_result) == 0:
            # Generate possible naming variations for MTV patterns
            variations = []
            mtv_match = re.match(r'^(MTV)(\d{4,})$', query, re.IGNORECASE)
            if mtv_match:
                prefix, digits = mtv_match.groups()
                variations = [
                    f"{prefix.upper()}{digits}",  # MTV####
                    f"{prefix.title()}{digits}",  # Mtv####
                    f"{prefix.lower()}{digits}",  # mtv####
                    f"{prefix[0].upper()}{digits}",  # M####
                    f"{prefix[0].lower()}{digits}",  # m####
                ]
                # Ensure original is included if it was different case
                if query not in variations:
                    variations.append(query)

                logger.info(f"No results found for '{query}'. Trying variations: {variations}")

                # Try each variation
                for variation in variations:
                    if variation == query:
                        continue  # Skip the original query we already tried

                    logger.debug(f"Trying variation: '{variation}'")
                    try:
                        variation_result = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                perforce_helper.search_changelists_by_keyword,
                                variation,
                                None,
                                force_path,
                                override_max
                            ),
                            timeout=30  # Shorter timeout for variations
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Perforce search timed out for variation: '{variation}'")
                        variation_result = []
                        continue

                    if variation_result and len(variation_result) > 0:
                        logger.info(f"Found {len(variation_result)} results for variation '{variation}'")
                        raw_result = variation_result
                        break
        # --- END ENHANCED ---

        # --- IMPROVED: Smarter token search strategy ---
        # If no results and the query contains multiple tokens, try each keyword individually but with low max
        if (not raw_result or len(raw_result) == 0) and isinstance(query, str) and ' ' in query:
            logger.info("No results for combined query; splitting into individual keywords for targeted search.")
            tokens = [tok.strip() for tok in set(query.split()) if len(tok.strip()) > 2]
            aggregated: List[Dict[str, Any]] = []
            seen_cls: Set[str] = set()

            # Limit tokens to avoid excessive searching
            MAX_TOKENS_TO_SEARCH = 10
            if len(tokens) > MAX_TOKENS_TO_SEARCH:
                logger.info(f"Too many tokens ({len(tokens)}), limiting to {MAX_TOKENS_TO_SEARCH} most specific tokens")
                # Sort tokens by length - longer tokens are usually more specific
                tokens.sort(key=len, reverse=True)
                tokens = tokens[:MAX_TOKENS_TO_SEARCH]

            # Reduced max results per token search
            TOKEN_SEARCH_MAX = 10  # Limit results per token

            for tok in tokens:
                try:
                    # Add timeout for each token search
                    partial = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            perforce_helper.search_changelists_by_keyword,
                            tok,
                            None,
                            force_path,
                            TOKEN_SEARCH_MAX  # Use reduced limit per token
                        ),
                        timeout=15  # Short timeout for individual token searches
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Perforce keyword split search timed out for '{tok}'")
                    continue
                except Exception as etok:
                    logger.warning(f"Perforce keyword split search failed for '{tok}': {etok}")
                    continue

                if not isinstance(partial, list):
                    continue

                # Force garbage collection after each token search to manage memory
                if len(partial) > 5:
                    gc.collect()

                for item in partial:
                    cl_id = str(item.get('changelist') or item.get('change') or item.get('change_list') or '')
                    if cl_id and cl_id not in seen_cls:
                        seen_cls.add(cl_id)
                        aggregated.append(item)

                # Safety limit on total results from token searches
                if len(aggregated) >= 50:
                    logger.info(f"Reached safety limit of 50 results from token searches. Stopping early after {tokens.index(tok)+1}/{len(tokens)} tokens.")
                    break

            raw_result = aggregated
            logger.info(f"Aggregated {len(raw_result)} total CLs from targeted keyword searches.")

            # Force garbage collection after token search phase
            gc.collect()
        # --- END IMPROVED ---

        # --- ENHANCED: Try fallback search ---
        # If still no results, try a fallback search without force_path
        if not raw_result or len(raw_result) == 0:
            if force_path:
                logger.info(f"No results found with force_path='{force_path}'. Trying fallback search without path restriction.")

                try:
                    # Try without force_path, with tight limits
                    fallback_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            perforce_helper.search_changelists_by_keyword,
                            query,
                            None,
                            None,  # No force_path
                            min(20, override_max)  # Even stricter limit for unrestricted search
                        ),
                        timeout=30  # Add timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Fallback Perforce search timed out")
                    fallback_result = []

                if fallback_result and len(fallback_result) > 0:
                    logger.info(f"Fallback search found {len(fallback_result)} results without path restriction")
                    raw_result = fallback_result
        # --- END ENHANCED ---

        # --- NEW: Process and filter results to avoid memory bloat ---
        if isinstance(raw_result, list) and raw_result:
            logger.info(f"Processing {len(raw_result)} raw Perforce results")

            # Limit total results if necessary
            MAX_SAFE_RESULTS = 100
            if len(raw_result) > MAX_SAFE_RESULTS:
                logger.warning(f"Too many Perforce results ({len(raw_result)}), limiting to {MAX_SAFE_RESULTS}")
                raw_result = raw_result[:MAX_SAFE_RESULTS]

            # Process results with size limits
            result_data = []
            for item in raw_result:
                if not isinstance(item, dict):
                    continue

                # Create a memory-efficient copy with only essential fields
                filtered_item = {
                    'change': item.get('change') or item.get('changelist') or item.get('change_list', ''),
                    'description': item.get('description', '') or item.get('desc', ''),
                    'user': item.get('user', 'Unknown'),
                    'time': item.get('time', ''),
                    'date': item.get('date', ''),
                    'dateFormatted': item.get('dateFormatted', ''),
                }

                # Add code snippets with size control
                if 'code_snippets' in item and isinstance(item['code_snippets'], list):
                    MAX_SNIPPETS_PER_CL = 5

                    filtered_snippets = []
                    for i, snippet in enumerate(item['code_snippets']):
                        if i >= MAX_SNIPPETS_PER_CL:
                            logger.info(f"Limiting snippets to {MAX_SNIPPETS_PER_CL} for CL {filtered_item['change']}")
                            break

                        if isinstance(snippet, dict):
                            # Limit snippet size to prevent memory issues
                            if 'snippet' in snippet and isinstance(snippet['snippet'], str):
                                snippet_content = snippet['snippet']
                                if len(snippet_content) > MAX_SNIPPET_SIZE:
                                    logger.warning(f"Large snippet ({len(snippet_content)} bytes) for CL {filtered_item['change']}, truncating to {MAX_SNIPPET_SIZE}")
                                    snippet['snippet'] = snippet_content[:MAX_SNIPPET_SIZE] + "... [TRUNCATED DUE TO SIZE]"

                                    # Force garbage collection for very large snippets
                                    if len(snippet_content) > 500000:  # 500KB
                                        gc.collect()

                            filtered_snippets.append(snippet)

                    # Set the filtered snippets
                    filtered_item['code_snippets'] = filtered_snippets

                result_data.append(filtered_item)
        else:
            # Handle potential string error messages
            if isinstance(raw_result, str) and raw_result.startswith("Error:"):
                 logger.error(f"Perforce tool returned an error string: {raw_result}")
            else:
                 logger.warning(f"Unexpected Perforce result type: {type(raw_result)}")
            result_data = [] # Return empty list for non-list/dict or error strings
        # --- END NEW ---

        # --- MODIFIED: Enhanced logging for results ---
        if not result_data:
            logger.info(f"Perforce search wrapper completed, obtained 0 result items for query '{query}'")

            # Dump more diagnostic info in debug mode
            logger.debug(f"Search parameters: query='{query}', force_path={force_path}, max_results_override={override_max}")
            logger.debug(f"Environment: P4CLIENT={env_info['P4CLIENT']}, DEPOT_PATH={env_info['DEPOT_PATH']}")
        else:
            logger.info(f"Perforce search wrapper successful, obtained {len(result_data)} result items")

            # Log more details about first few results at debug level
            if len(result_data) > 0:
                for i, item in enumerate(result_data[:2]):  # Log first 2 items
                    logger.debug(f"Result {i+1}: CL={item.get('change')}, User={item.get('user')}")
                    desc = item.get('desc', '')[:100] + '...' if len(item.get('desc', '')) > 100 else item.get('desc', '')
                    logger.debug(f"Description: {desc}")

            # Force garbage collection after processing large result sets
            if len(result_data) > 10:
                gc.collect()
        # --- END MODIFIED ---

    except asyncio.TimeoutError:
        logger.error("Perforce search wrapper timed out.")
    except Exception as e:
        logger.error(f"Error in search_perforce_wrapper: {e}", exc_info=True)
    finally:
        # --- IMPROVED: Safe cleanup in finally block ---
        # Ensure cleanup always happens, but with robust error handling
        if perforce_helper:
            try:
                # Use a separate try-except just for the cleanup
                logger.debug("Cleaning up Perforce helper resources")
                perforce_helper.disconnect_all()
                logger.debug("Perforce helper cleanup completed")
            except Exception as cleanup_err:
                # Log but continue - never let cleanup errors prevent returning results
                logger.warning(f"Non-fatal error during Perforce cleanup: {cleanup_err}")
            finally:
                # Clear the reference to help with garbage collection
                perforce_helper = None

        # Force garbage collection before returning
        gc.collect()

    # Always return the result (empty list if error occurred)
    return result_data


@traceable
async def search_all_sources(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Search all configured enterprise sources concurrently using wrappers.
    Uses a targeted query for Perforce based on keywords/IDs.
    Updates the state with raw results under 'raw_*_results'.

    Args:
        state (ResearchState): The current graph state.
        config (RunnableConfig | None): LangGraph runtime config.

    Returns:
        Dict[str, Any]: An update dictionary for the state, primarily adding raw results.
                        Returns an empty dict on error.
    """
    logger.info("Running search_all_sources")

    try:
        # Get the research topic
        orig_topic = state.get("research_topic", "")
        if not orig_topic or not isinstance(orig_topic, str):
            logger.warning("No research topic provided to search_all_sources. Skipping.")
            return {}

        # --- First, run JIRA and Confluence searches (these use the full topic) ---
        logger.info("Starting JIRA and Confluence searches")

        # --- MODIFIED: Include enhanced parameters for attachment handling ---
        # Get config from state if not passed directly
        cfg = Configuration.from_runnable_config(config) if config else Configuration()

        # Determine if we should download attachments based on config
        download_attachments = bool(getattr(cfg, "download_attachments", True))
        logger.info(f"Attachment downloading is {'enabled' if download_attachments else 'disabled'}")

        jira_task = search_jira_wrapper(orig_topic, config=config,
                                        max_results=100,  # Limit to 100
                                        get_hierarchy=True,  # Get parent/child relationships
                                        download_attachments=download_attachments)  # Process attachments

        conf_task = search_confluence_wrapper(orig_topic, config=config,
                                              max_results=40,  # Limit to 40
                                              include_attachments=download_attachments)  # Process attachments
        # --- END MODIFIED ---

        # Wait for initial search tasks to complete
        jira_results, conf_results = await asyncio.gather(jira_task, conf_task)

        # --- Next, prepare enhanced Perforce search query ---
        # [... Same Perforce search logic as before ...]
        identifiers = []
        # MTV identifiers - handle various formats (MTV1234, MTV-1234, M1234)
        mtv_patterns = [
            re.findall(r'\b(MTV?\d{4,})\b', orig_topic, re.IGNORECASE),
            re.findall(r'\b(MTV-\d{4,})\b', orig_topic, re.IGNORECASE),
            re.findall(r'\bM-(\d{4,})\b', orig_topic, re.IGNORECASE)
        ]
        for pattern_matches in mtv_patterns:
            identifiers.extend(pattern_matches)

        # JIRA/ticket IDs (VIT-123, ABC-456)
        jira_ids = re.findall(r'\b([A-Z]+-\d+)\b', orig_topic)
        identifiers.extend(jira_ids)

        # Other ID formats (potential identifiers like API_123)
        other_ids = re.findall(r'\b([A-Z]{2,5}_[A-Z0-9]+)\b', orig_topic)
        identifiers.extend(other_ids)

        # Extract technical keywords from topic
        technical_terms = []
        tech_pattern = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]{2,}(?:\.[a-zA-Z][a-zA-Z0-9_]{2,})*)\b', orig_topic)
        technical_terms.extend([term for term in tech_pattern if len(term) > 3])

        # Extract additional identifiers and keywords from JIRA results
        jira_keywords = set()
        if jira_results:
            logger.info(f"Extracting additional search terms from {len(jira_results)} JIRA results")
            for jira_item in jira_results:
                if not isinstance(jira_item, dict):
                    continue

                # Extract the VIT ticket ID if not already in identifiers
                jira_key = jira_item.get("key", "")
                if jira_key and jira_key not in identifiers:
                    identifiers.append(jira_key)

                # Extract keywords from summary and description
                summary = jira_item.get("summary", "") or jira_item.get("fields", {}).get("summary", "")
                description = jira_item.get("description", "") or jira_item.get("fields", {}).get("description", "")

                # Extract potential technical terms
                if summary:
                    summary_terms = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]{2,}(?:\.[a-zA-Z][a-zA-Z0-9_]{2,})*)\b',
                                               summary)
                    jira_keywords.update([term for term in summary_terms if len(term) > 3])
                    summary_ids = re.findall(r'\b(MTV?\d{4,}|[A-Z]+-\d+)\b', summary, re.IGNORECASE)
                    identifiers.extend([id for id in summary_ids if id not in identifiers])

                if description:
                    desc_sample = description[:1000] if len(description) > 1000 else description
                    desc_terms = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]{2,}(?:\.[a-zA-Z][a-zA-Z0-9_]{2,})*)\b',
                                            desc_sample)
                    jira_keywords.update([term for term in desc_terms if len(term) > 3])
                    desc_ids = re.findall(r'\b(MTV?\d{4,}|[A-Z]+-\d+)\b', desc_sample, re.IGNORECASE)
                    identifiers.extend([id for id in desc_ids if id not in identifiers])

        # Combine and filter search terms
        unique_identifiers = []
        id_set = set()
        for id_val in identifiers:
            if id_val.lower() not in id_set:
                unique_identifiers.append(id_val)
                id_set.add(id_val.lower())

        all_tech_terms = set(technical_terms) | jira_keywords
        sorted_tech_terms = sorted(all_tech_terms, key=len, reverse=True)
        selected_tech_terms = sorted_tech_terms[:5]

        # Construct optimized perforce search query
        search_components = []
        if unique_identifiers:
            search_components.append(" ".join(unique_identifiers))
        if selected_tech_terms:
            search_components.append(" ".join(selected_tech_terms))

        perforce_query = " ".join(search_components).strip()
        if not perforce_query:
            perforce_query = orig_topic[:100]
            if len(orig_topic) > 100:
                perforce_query += "..."

        logger.info(f"Enhanced Perforce search query: '{perforce_query}'")

        # Execute Perforce search with enhanced query - limit to 60 CLs
        p4_results = await search_perforce_wrapper(perforce_query, config=config, max_results=60)

        # --- ADDED: Extract Binary Attachments ---
        # Extract binary attachments from JIRA and Confluence results
        binary_attachments = []

        # Process JIRA attachments
        for item in jira_results:
            if not isinstance(item, dict):
                continue

            # Look for explicit attachments field
            att_list = item.get("attachments", [])
            if att_list:
                for att in att_list:
                    if isinstance(att, dict):
                        binary_attachments.append({
                            "source": "jira",
                            "source_id": item.get("key", ""),
                            "filename": att.get("filename", ""),
                            "content_type": att.get("mime_type", ""),
                            "size": att.get("size", 0),
                            "url": att.get("url", ""),
                            "content": att.get("content_text", ""),
                            "file_type": att.get("mime_type", "").split("/")[0] if "/" in att.get("mime_type",
                                                                                                  "") else "unknown"
                        })

            # Also check fields.attachment (Jira API format)
            fields_attachments = item.get("fields", {}).get("attachment", [])
            if isinstance(fields_attachments, list):
                for att in fields_attachments:
                    # Skip if not a dict or already processed
                    if not isinstance(att, dict):
                        continue

                    # Create attachment entry
                    binary_attachments.append({
                        "source": "jira",
                        "source_id": item.get("key", ""),
                        "filename": att.get("filename", ""),
                        "content_type": att.get("mimeType", ""),
                        "size": att.get("size", 0),
                        "url": att.get("content", ""),
                        "file_type": att.get("mimeType", "").split("/")[0] if "/" in att.get("mimeType",
                                                                                             "") else "unknown"
                    })

        # Process Confluence attachments
        for item in conf_results:
            if not isinstance(item, dict):
                continue

            # Look for attachments in result
            att_list = item.get("attachments", [])
            if att_list:
                for att in att_list:
                    if isinstance(att, dict):
                        binary_attachments.append({
                            "source": "confluence",
                            "source_id": item.get("id", ""),
                            "filename": att.get("title", ""),
                            "content_type": att.get("mediaType", ""),
                            "size": att.get("size", 0),
                            "url": f"{cfg.configurable.get('confluence_url', 'https://confluence.example.com')}{att.get('downloadLink', '')}",
                            "content": att.get("content", ""),
                            "file_type": att.get("mediaType", "").split("/")[0] if "/" in att.get("mediaType",
                                                                                                  "") else "unknown"
                        })

        logger.info(f"Extracted {len(binary_attachments)} binary attachments from search results")
        # --- END ADDED ---

        # Store results in dictionary
        results = {
            "raw_jira_results": jira_results or [],
            "raw_confluence_results": conf_results or [],
            "raw_perforce_results": p4_results or [],
            "raw_binary_attachments": binary_attachments or []  # Add binary attachments
        }

        # Log result counts for debugging
        logger.info(
            f"Search results: JIRA={len(jira_results or [])}, Confluence={len(conf_results or [])}, Perforce={len(p4_results or [])}, Attachments={len(binary_attachments or [])}")

        # Save raw results to logs for debugging (if needed)
        try:
            os.makedirs('./logs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'./logs/search_results_{timestamp}.json', 'w') as f:
                json.dump(results, f, default=str, indent=2)
            logger.debug(f"Raw search results saved to ./logs/search_results_{timestamp}.json")
        except Exception as save_err:
            logger.warning(f"Could not save raw results to log: {save_err}")

        return results
    except Exception as e:
        logger.error(f"Error in search_all_sources: {e}", exc_info=True)
        return {}
@traceable # Optional
async def check_retrieval_relevance(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Checks if the retrieved content (formatted blocks) is relevant to the research topic.
    Updates state["_relevance_check"] to either "relevant" or "irrelevant".

    Args:
        state (ResearchState): The current graph state.
        config (RunnableConfig | None): LangGraph runtime config (optional).

    Returns:
        Dict[str, Any]: State update dictionary with the relevance check result.
    """
    logger.info("--- Node: check_retrieval_relevance ---")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    topic = state.get("research_topic", "")
    relevance_result = "irrelevant" # Default to irrelevant if no content or error

    # Use the formatted blocks generated by chunk processing
    formatted_blocks = state.get("enterprise_research_results", [])

    if not topic:
        logger.warning("No research topic found for relevance check.")
        return {"_relevance_check": relevance_result} # Return default

    if not formatted_blocks:
        logger.info("No formatted research results to check for relevance.")
        return {"_relevance_check": relevance_result} # Return default

    # Create a sample of the content for the LLM
    sample_content = ""
    max_sample_len = 2000 # Limit sample size for LLM
    current_len = 0
    for block in formatted_blocks[:5]: # Sample first 5 blocks max
        if isinstance(block, str):
            block_sample = block[:max_sample_len - current_len]
            sample_content += block_sample + "\n...\n"
            current_len += len(block_sample) + 5 # Account for added ellipsis
            if current_len >= max_sample_len:
                break
        else:
             logger.warning(f"Non-string block found in enterprise_research_results during relevance check: {type(block)}")


    if not sample_content.strip():
         logger.warning("Could not generate a content sample for relevance check.")
         return {"_relevance_check": relevance_result} # Return default

    relevance_prompt = (
        f"Determine if the following retrieved content sample is relevant to the research topic: '{topic}'\n\n"
        f"Content sample:\n{sample_content}\n\n"
        "Reply with a single word only: 'relevant' or 'irrelevant'"
    )

    try:
        # Ensure ollama_base_url and local_llm are available in cfg
        if not hasattr(cfg, 'ollama_base_url') or not hasattr(cfg, 'local_llm'):
             raise ValueError("LLM configuration (ollama_base_url, local_llm) not found in config.")

        # --- MODIFIED: Use central config ---
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            **LLM_CONFIG_PARAMS # Unpack shared parameters
            # Override temperature for this specific classification task
            # temperature=0
        )
        # --- END MODIFICATION ---

        result = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=relevance_prompt)]),
            timeout=120
        )

        response = result.content.strip().lower()

        # Check the LLM response
        if "relevant" in response and not response.startswith("ir"):
            relevance_result = "relevant"
            logger.info("Content deemed relevant to research topic.")
        else:
            relevance_result = "irrelevant"
            logger.info("Content deemed irrelevant to research topic.")

    except Exception as e:
        logger.error(f"Error checking content relevance: {e}", exc_info=True)
        # Default to relevant on error to avoid stopping unnecessarily? Or irrelevant? Let's default relevant.
        relevance_result = "relevant"
        logger.warning("Defaulting relevance check to 'relevant' due to error.")

    # Return dict for state update
    return {"_relevance_check": relevance_result}


# --- NEW NODE: Pre-process data for individual summarizers ---
@traceable
async def preprocess_for_summary(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Pre-processes the filtered 'enterprise_research_results' into a structured
    dictionary ('pre_processed_summaries') with bullet points for each source,
    ready for individual summarization nodes. Limits items per source.
    ENHANCED: Added memory management for large content processing.
    """
    logger.info("--- Node: preprocess_for_summary (Memory-Optimized Processing) ---")

    # Define batch size constant for processing
    BATCH_SIZE = 5

    src_blocks = state.get("enterprise_research_results", [])
    source_map = {
        "=== Jira Results": "### JIRA",
        "=== Confluence Results": "### Confluence",
        "=== Perforce Results": "### Perforce",
        "=== Web Search Results": "### Web"
    }

    pre_processed_data: Dict[str, List[str]] = defaultdict(list)
    items_processed_count = defaultdict(int)
    dedup_tracker = defaultdict(set) # Local dedupe for this pass

    # Maximum items per source type to prevent memory overflow
    MAX_ITEMS_PER_SOURCE = 50

    # --- Initialize Text Splitter for Confluence Content ---
    confluence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Target size for content chunks
        chunk_overlap=150,  # Overlap between chunks
        length_function=len,
    )

    if not src_blocks:
        logger.warning("No enterprise research results found to pre-process for summary.")
        return {"pre_processed_summaries": {}}

    # Process blocks in batches to manage memory
    total_blocks = len(src_blocks)
    logger.info(f"Processing {total_blocks} blocks in batches of {BATCH_SIZE}")

    for batch_start in range(0, total_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_blocks)
        logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(total_blocks + BATCH_SIZE - 1)//BATCH_SIZE}: blocks {batch_start+1}-{batch_end}")

        for block_idx in range(batch_start, batch_end):
            block = src_blocks[block_idx]
            if not isinstance(block, str):
                continue

            block_lines = block.strip().splitlines()
            if not block_lines:
                continue

            header_line = block_lines[0].strip()
            matched_heading = None
            for header_prefix, summary_heading in source_map.items():
                if header_line.startswith(header_prefix):
                    matched_heading = summary_heading
                    break

            if not matched_heading:
                logger.warning(f"Could not match header: {header_line}")
                continue # Skip blocks without a recognized header

            # Skip if we've already processed max items for this source
            if items_processed_count[matched_heading] >= MAX_ITEMS_PER_SOURCE:
                logger.info(f"Reached maximum items ({MAX_ITEMS_PER_SOURCE}) for {matched_heading}, skipping additional content")
                continue

            content_part = "\n".join(block_lines[1:]).strip().lstrip("---").rstrip("---").strip()
            if not content_part:
                continue

            items_str = "" # Initialize item_str before try block
            try:
                # The block content is expected to be a string representing a list of JSON objects, separated by '\n\n---\n\n'
                # Replace the separator to form a valid JSON list string
                json_string = content_part.replace('\n\n---\n\n', ',')
                # Handle potential trailing comma
                if json_string.endswith(','):
                    json_string = json_string[:ensure_index(-1)]
                items_str = f"[{json_string}]"

                items = json.loads(items_str)

                # Process each item within the parsed list
                for item in items:
                    # Skip processing if we've reached max items for this source
                    if items_processed_count[matched_heading] >= MAX_ITEMS_PER_SOURCE:
                        break

                    if not isinstance(item, dict):
                        continue

                    entry = None
                    entries_to_add = [] # Use list for potential multiple entries per item
                    summary_heading = matched_heading  # Store matched_heading in a local variable

                    try:
                        if summary_heading == "### JIRA":
                            # ... (JIRA processing - generates one entry) ...
                            summary = item.get('summary') or item.get('fields',{}).get('summary', '')
                            key = item.get('key')
                            url = item.get('url')
                            status = item.get('status') or item.get('fields',{}).get('status',{}).get('name')
                            assignee_info = item.get('fields', {}).get('assignee')
                            assignee = assignee_info.get('displayName', 'Unassigned') if assignee_info else 'Unassigned'
                            reporter_info = item.get('fields', {}).get('reporter')
                            reporter = reporter_info.get('displayName', 'N/A') if reporter_info else 'N/A'
                            created = item.get('fields', {}).get('created', '')[:10]
                            updated = item.get('fields', {}).get('updated', '')[:10]
                            if key and summary:
                                 key_str = str(key); url_str = str(url) if url else ""
                                 key_text = f"**[{key_str}]({url_str})**" if url_str else f"**{key_str}**"
                                 entry = f"- {key_text} ({status or 'N/A'}): {summary} (Reported by: {reporter} on {created}, Assigned to: {assignee}, Last Updated: {updated})"
                                 if entry:
                                     entries_to_add.append(entry)

                        elif summary_heading == "### Confluence":
                            title = item.get('title')
                            relative_url = item.get('url')
                            excerpt = item.get('excerpt', '')
                            content = item.get('content', '') # Get full content
                            history = item.get('history', {})
                            author_info = history.get('createdBy', {})
                            author = author_info.get('displayName', 'Unknown Author')
                            last_modified_date = history.get('createdDate', '')[:10]

                            full_url = ""
                            if relative_url and CONFLUENCE_DOMAIN:
                                scheme = 'https://'
                                domain = CONFLUENCE_DOMAIN
                                if domain.startswith(('http://', 'https://')):
                                    scheme = ''
                                full_url = f"{scheme}{domain.rstrip('/')}{relative_url}"
                            else:
                                full_url = relative_url

                            MIN_CONTENT_LEN_FOR_CHUNK = 500
                            if title and content and len(content) > MIN_CONTENT_LEN_FOR_CHUNK:
                                logger.debug(f"Chunking Confluence content for: {title}")
                                # Controlled Truncation with Size Limits
                                MAX_CONFLUENCE_CHUNK_ENTRY_LEN = 1500 # Limit length per entry

                                # Limit total chunks per Confluence page
                                MAX_CHUNKS_PER_PAGE = 3

                                # Apply overall size limit before chunking to avoid excessive memory usage
                                if len(content) > MAX_CONFLUENCE_CONTENT_SIZE:
                                    logger.info(f"Limiting Confluence content from {len(content)} to {MAX_CONFLUENCE_CONTENT_SIZE} chars before chunking")
                                    content = content[:MAX_CONFLUENCE_CONTENT_SIZE] + "... [TRUNCATED DUE TO SIZE]"
                                    # Force GC for very large content
                                    gc.collect()

                                text_chunks = confluence_splitter.split_text(content)
                                num_chunks = min(len(text_chunks), MAX_CHUNKS_PER_PAGE)
                                for i in range(num_chunks):
                                    chunk = text_chunks[i]
                                    # Apply truncation here
                                    truncated_chunk = chunk[:MAX_CONFLUENCE_CHUNK_ENTRY_LEN]
                                    if len(chunk) > MAX_CONFLUENCE_CHUNK_ENTRY_LEN:
                                        truncated_chunk += "... [TRUNCATED] ..." # Add indicator if truncated
                                    # Use truncated_chunk in the entry string
                                    chunk_entry = f"- **[{title}]({full_url})** (Content Snippet {i+1}/{num_chunks}): {truncated_chunk} (Author: {author}, Date: {last_modified_date})"
                                    entries_to_add.append(chunk_entry)

                                # Run garbage collection after processing large Confluence content
                                if len(content) > 100000:  # If original content was very large
                                    gc.collect()
                            elif title:
                                excerpt_entry = f"- **[{title}]({full_url})**" + (f": {excerpt}" if excerpt else "") + f" (Author: {author}, Date: {last_modified_date})"
                                entries_to_add.append(excerpt_entry) # Ensure excerpt handling is preserved if needed

                        elif summary_heading == "### Perforce":
                            # --- MODIFICATION: Include Snippets in Bullets with Size Limits ---
                            cl = item.get('changelist') or item.get('change')
                            desc = item.get('description', '')
                            user = item.get('user')
                            date = item.get('dateFormatted')
                            # Ensure user and date are extracted reliably
                            if not user or not isinstance(user, str):
                                user = "Unknown User"
                            date_val = item.get('dateFormatted')
                            if not date_val:
                                time_val = item.get('time')
                                if isinstance(time_val, (int, str)) and str(time_val).isdigit():
                                    try:
                                        date_val = datetime.fromtimestamp(int(time_val)).strftime("%Y-%m-%d %H:%M:%S")
                                    except Exception:
                                        date_val = "Invalid Timestamp"
                                else:
                                    date_val = "Unknown Date"
                            elif not isinstance(date_val, str):
                                date_val = "Unknown Date"
                            date = date_val

                            swarm_url_base = os.getenv("P4_SWARM_URL", "")
                            change_url = f"{swarm_url_base.rstrip('/')}/changes/{cl}" if swarm_url_base and cl else ""

                            if cl:
                                # First, add the main changelist entry
                                cl_link_text = f"**CL [{cl}]({change_url})**" if change_url else f"**CL {cl}**"
                                # Limit description length to avoid excessively long entries
                                MAX_DESC_LENGTH = 250
                                if desc and len(desc) > MAX_DESC_LENGTH:
                                    desc = desc[:MAX_DESC_LENGTH] + "..."
                                cl_entry = f"- {cl_link_text} by **{user}** on {date}: {desc}"
                                entries_to_add.append(cl_entry)

                                # Now, iterate through snippets and add them as separate bullet points with size limits
                                snippets_list = item.get('code_snippets')
                                if isinstance(snippets_list, list) and snippets_list:
                                    # Limit number of snippets per changelist
                                    MAX_SNIPPETS_PER_CL = 3
                                    snippet_count = 0

                                    for snippet_data in snippets_list:
                                        if not isinstance(snippet_data, dict) or snippet_count >= MAX_SNIPPETS_PER_CL:
                                            continue

                                        # Skip placeholder snippet entries
                                        raw_snippet = snippet_data.get("snippet", "").strip()
                                        if raw_snippet.startswith("[Could not retrieve diff"):
                                            logger.debug(f"Skipping placeholder snippet for CL {cl} in preprocess_for_summary")
                                            continue

                                        file_path = snippet_data.get("file", "Unknown File")
                                        language = snippet_data.get("language", "text") or "text"
                                        action = snippet_data.get("action", "edit")

                                        # Controlled Truncation with memory management
                                        MAX_PERFORCE_SNIPPET_LEN = 3000 # Limit length per snippet
                                        if len(raw_snippet) > MAX_PERFORCE_SNIPPET_LEN:
                                            logger.info(f"Limiting Perforce snippet from {len(raw_snippet)} to {MAX_PERFORCE_SNIPPET_LEN} chars")
                                            snippet_text = raw_snippet[:MAX_PERFORCE_SNIPPET_LEN] + "... [TRUNCATED] ..."
                                            # Force GC for very large snippets
                                            if len(raw_snippet) > 100000:  # 100KB
                                                gc.collect()
                                        else:
                                            snippet_text = raw_snippet

                                        # Format as a nested bullet point with code block
                                        snippet_entry = (
                                            f"  - Snippet from `{file_path}` (Action: {action}):\\n"
                                            f"    ```\\n{snippet_text}\\n    ```"
                                        )
                                        entries_to_add.append(snippet_entry)
                                        snippet_count += 1
                                # End Perforce snippet loop

                        elif summary_heading == "### Web":
                            # ... (Web processing remains the same) ...
                            title = item.get('title')
                            url = item.get('url')
                            snippet = item.get('snippet')
                            if title:
                                entry = f"- **{title}**" + (f" ({url})" if url else "") + (f": {snippet}" if snippet else "")
                                if entry:
                                    entries_to_add.append(entry)

                    except Exception as e_item:
                        logger.error(f"Error processing item for {summary_heading}: {e_item}", exc_info=True)
                        error_entry = f"- Error processing item: {item.get('key') or item.get('title') or 'Unknown'}"
                        entries_to_add.append(error_entry)

                    # Add all generated entries for this item, checking local dedupe
                    for gen_entry in entries_to_add:
                        # Ensure matched_heading is valid before using as key
                        if matched_heading and gen_entry and gen_entry not in dedup_tracker[matched_heading]:
                            pre_processed_data[matched_heading].append(gen_entry)
                            dedup_tracker[matched_heading].add(gen_entry)

                    # Increment count only once per original item processed
                    if entries_to_add: # Only increment if we actually added something
                        if matched_heading: # Check again
                            items_processed_count[matched_heading] += 1

            except json.JSONDecodeError as e:
                # Ensure matched_heading is valid before using as key
                if matched_heading:
                    error_context = items_str[max(0, e.pos-20):min(len(items_str), e.pos+20)]
                    logger.warning(f"JSONDecodeError processing block for {matched_heading}: {e} - Context: '{error_context}'")
                    # Attempt to add raw block content as fallback if JSON parsing fails
                    raw_content_entry = f"- Raw Block Content (JSON Parse Failed): {content_part[:ensure_index(500)]}..."
                    if raw_content_entry not in dedup_tracker[matched_heading]:
                        pre_processed_data[matched_heading].append(raw_content_entry)
                        dedup_tracker[matched_heading].add(raw_content_entry)
                        items_processed_count[matched_heading] += 1
                else:
                    logger.error(f"JSONDecodeError encountered but matched_heading was None. Block: {block_lines[0]}...")

            except Exception as e_gen:
                logger.error(f"Unexpected error preprocessing block for {matched_heading}: {e_gen}", exc_info=True)
                # Ensure matched_heading is valid before using as key
                if matched_heading:
                    error_entry = f"- Unexpected Error Processing Block: {content_part[:ensure_index(500)] if isinstance(content_part, str) else str(content_part)[:ensure_index(500)]}..."
                    if error_entry not in dedup_tracker[matched_heading]:
                        pre_processed_data[matched_heading].append(error_entry)
                        dedup_tracker[matched_heading].add(error_entry)
                        items_processed_count[matched_heading] += 1
                else:
                    logger.error(f"Unexpected error encountered but matched_heading was None. Block: {block_lines[0]}...")

        # Force garbage collection after each batch
        gc.collect()
        logger.info(f"Finished processing batch {batch_start//BATCH_SIZE + 1}, items processed so far: {dict(items_processed_count)}")

    logger.info(f"Pre-processing complete. Total items per source: {dict(items_processed_count)}")
    return {"pre_processed_summaries": dict(pre_processed_data)}

@traceable
async def summarize_sources(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    (DEPRECATED - Logic moved to individual summarizers and preprocess_for_summary)
    Create/refine the running summary using processed research results (formatted blocks)
    and Perforce diff analysis. Includes auto-citation.
    """
    logger.warning("DEPRECATED summarize_sources node called. This logic should be handled by individual summarizers.")
    # Return minimal update to avoid breaking graph if accidentally called
    return {"running_summary": state.get("running_summary", "ERROR: Deprecated summarizer called.")}


# Function to implement source-specific limits
def enterprise_research(state: ResearchState, config: RunnableConfig | None = None) -> ResearchState:
    """
    Launches search across sources (via search_all_sources node update)
    and sets up results for chunked processing.
    Skips fresh search if chunk processing is already in progress.

    Args:
        state (ResearchState): The current graph state.
        config (RunnableConfig | None): LangGraph runtime config (optional).

    Returns:
        ResearchState: The updated state dictionary.
    """
    # --- Check if chunk processing is ongoing ---
    if (should_process_next_chunk(state, "confluence") or
            should_process_next_chunk(state, "jira") or
            should_process_next_chunk(state, "perforce")):
        logger.debug("Chunk processing in progress – skipping fresh enterprise search setup.")
        # Return state without modifications as chunking will proceed
        return {}  # LangGraph merges this empty dict, state remains unchanged

    logger.info("--- Setting up Enterprise Research Chunks ---")
    # Increment loop count only when starting a new search cycle
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    # --- ADDED: Apply source-specific limits to raw results ---
    # Define maximum items per source type
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

    # Update state with limited results
    state["raw_jira_results"] = raw_jira
    state["raw_confluence_results"] = raw_confluence
    state["raw_perforce_results"] = raw_perforce
    # --- END ADDED ---

    # --- Initialize/Reset State for New Cycle ---
    state["enterprise_research_results"] = []  # Clear formatted results for this cycle
    state["sources_gathered"] = []  # Clear gathered sources for this cycle
    state["_content_hashes"] = set()  # Reset hashes for formatted blocks of this cycle
    # Accumulate seen IDs across loops
    state["_jira_keys_seen"] = state.get("_jira_keys_seen", set())
    state["_perforce_cl_numbers"] = state.get("_perforce_cl_numbers", set())
    state["_confluence_urls_seen"] = state.get("_confluence_urls_seen", set())
    state["_confluence_ids_seen"] = state.get("_confluence_ids_seen", set())
    # Reset hop counter for chunk processing phase
    state["__hop_counter"] = 0
    state["_last_confluence_chunk_idx"] = -1
    state["_last_jira_chunk_idx"] = -1
    state["_last_perforce_chunk_idx"] = -1
    state["_stuck_confluence_count"] = 0
    state["_stuck_jira_count"] = 0
    state["_stuck_perforce_count"] = 0

    # --- MODIFIED: Better Chunk Setup with Smaller Single Blocks ---
    # Make chunks smaller, use single block for each source when possible
    new_chunk_setup = {
        "jira": (raw_jira, min(MAX_JIRA_ISSUES, 20)),  # Smaller chunk size (was JIRA_CHUNK_SIZE)
        "confluence": (raw_confluence, min(MAX_CONFLUENCE_PAGES, 10)),  # Smaller chunk size
        "perforce": (raw_perforce, min(MAX_PERFORCE_CLS, 15))  # Smaller chunk size
    }

    total_chunks_created = 0
    state_updates: Dict[str, Any] = {}  # Collect updates to return

    for source_type, (results_list, chunk_size) in new_chunk_setup.items():
        if results_list and isinstance(results_list, list):
            # Use a single chunk when possible to avoid repetitive processing
            if len(results_list) <= chunk_size:
                chunks = [results_list]  # One chunk containing all items
                logger.info(f"Created single chunk for {source_type} ({len(results_list)} items)")
            else:
                # Create multiple chunks with minimal overlap
                chunks = []
                for i in range(0, len(results_list), chunk_size):
                    chunks.append(results_list[i:i + chunk_size])
                logger.info(
                    f"Created {len(chunks)} chunks for {source_type} ({len(results_list)} items, size={chunk_size})")

            state_updates[f"{source_type}_chunks"] = chunks
            state_updates[f"{source_type}_chunk_idx"] = 0
            state_updates[f"{source_type}_total_chunks"] = len(chunks)
            total_chunks_created += len(chunks)
        else:
            # No results or invalid format, ensure chunk state is reset
            state_updates[f"{source_type}_chunks"] = []
            state_updates[f"{source_type}_chunk_idx"] = 0
            state_updates[f"{source_type}_total_chunks"] = 0
    # --- END MODIFIED ---

    logger.info(f"Chunk setup complete. {total_chunks_created} total chunks created for processing.")

    # --- Ensure Correct Totals are Returned ---
    for source_type in new_chunk_setup.keys():
        if f"{source_type}_chunks" in state_updates and state_updates[f"{source_type}_chunks"]:
            state_updates[f"{source_type}_total_chunks"] = len(state_updates[f"{source_type}_chunks"])
        elif f"{source_type}_total_chunks" not in state_updates:
            state_updates[f"{source_type}_total_chunks"] = 0

    # Return only the fields that were updated
    # Also include the reset fields
    state_updates["enterprise_research_results"] = []
    state_updates["sources_gathered"] = []
    state_updates["_content_hashes"] = set()
    state_updates["research_loop_count"] = state["research_loop_count"]  # Ensure loop count update is returned
    state_updates["__hop_counter"] = 0
    state_updates["_last_confluence_chunk_idx"] = -1
    state_updates["_last_jira_chunk_idx"] = -1
    state_updates["_last_perforce_chunk_idx"] = -1
    state_updates["_stuck_confluence_count"] = 0
    state_updates["_stuck_jira_count"] = 0
    state_updates["_stuck_perforce_count"] = 0

    # Return raw results with limits applied
    state_updates["raw_jira_results"] = raw_jira
    state_updates["raw_confluence_results"] = raw_confluence
    state_updates["raw_perforce_results"] = raw_perforce

    return state_updates
@traceable # Optional
async def process_generic_chunk(state: ResearchState, source_type: str) -> Dict[str, Any]:
    """
    Processes the *next* chunk for the given source type.
    Performs deduplication based on IDs and adds formatted *unique* content
    from the chunk to 'enterprise_research_results'.

    Args:
        state (ResearchState): The current graph state.
        source_type (str): 'jira', 'confluence', or 'perforce'.

    Returns:
        Dict[str, Any]: Dictionary of state updates.
    """
    # --- Check if processing should occur ---
    if not should_process_next_chunk(state, source_type):
        logger.debug(f"No more chunks to process for {source_type} or processing skipped.")
        # Return empty dict as no state change occurred related to chunk processing
        # However, should_process_next_chunk might have called _finish_source,
        # so we need to return the potentially modified state fields from that.
        return {
             f"{source_type}_chunks": state.get(f"{source_type}_chunks", []),
             f"{source_type}_chunk_idx": state.get(f"{source_type}_chunk_idx", 0),
             f"{source_type}_total_chunks": state.get(f"{source_type}_total_chunks", 0),
             f"_last_{source_type}_chunk_idx": state.get(f"_last_{source_type}_chunk_idx", -1),
             f"_stuck_{source_type}_count": state.get(f"_stuck_{source_type}_count", 0),
        }

    await asyncio.sleep(0) # Yield control

    # --- Get current chunk data ---
    chunk_idx = state.get(f"{source_type}_chunk_idx", 0)
    chunks = state.get(f"{source_type}_chunks", [])
    total_chunks = state.get(f"{source_type}_total_chunks", 0)

    if chunk_idx >= len(chunks): # Double check index validity
        logger.warning(f"Chunk index {chunk_idx} out of bounds for {source_type} (available: {len(chunks)}). Finishing source.")
        _finish_source(state, source_type)
        # Return the reset state fields
        return {
             f"{source_type}_chunks": [], f"{source_type}_chunk_idx": 0, f"{source_type}_total_chunks": 0,
             f"_last_{source_type}_chunk_idx": -1, f"_stuck_{source_type}_count": 0,
        }

    current_chunk_items = chunks[chunk_idx]
    logger.info(f"--- Processing {source_type.capitalize()} Chunk {chunk_idx + 1}/{total_chunks} ({len(current_chunk_items)} items) ---")

    # --- Prepare for processing ---
    # Get existing tracking sets (use copies for modification)
    content_hashes = state.get("_content_hashes", set()).copy()
    jira_keys = state.get("_jira_keys_seen", set()).copy()
    perforce_cls = state.get("_perforce_cl_numbers", set()).copy()
    confluence_urls = state.get("_confluence_urls_seen", set()).copy()
    confluence_ids = state.get("_confluence_ids_seen", set()).copy()
    # Removed direct copy of enterprise_research_results as we only return the new block

    # --- Process items within the chunk ---
    unique_items_in_chunk = []
    new_ids_found_in_chunk = set() # Track new IDs found specifically in this chunk

    for item in current_chunk_items:
        is_unique = False
        item_id_for_tracking = None # ID used for deduplication tracking

        try:
            if source_type == "jira" and isinstance(item, dict):
                item_id_for_tracking = item.get("key")
                # Also check status here if desired
                status = item.get("status", "").lower()
                if item_id_for_tracking and item_id_for_tracking not in jira_keys and status != "cancelled":
                    is_unique = True
                    jira_keys.add(item_id_for_tracking) # Add to updated tracking set
            elif source_type == "perforce" and isinstance(item, dict):
                item_id_for_tracking = item.get('changelist') or item.get('change') or item.get('change_list')
                if item_id_for_tracking and str(item_id_for_tracking) not in perforce_cls:
                    is_unique = True
                    perforce_cls.add(str(item_id_for_tracking)) # Add to updated tracking set
            elif source_type == "confluence":
                if isinstance(item, dict):
                    url = item.get("url")
                    page_id = str(item.get("id", ""))
                    # --- MODIFIED: Rigorous check against seen URLs and IDs ---
                    if (url and url in confluence_urls) or (page_id and page_id in confluence_ids):
                        is_unique = False # Explicitly mark as not unique if URL or ID already seen
                        item_id_for_tracking = url or f"id:{page_id}" # Still useful for logging maybe
                        logger.debug(f"Skipping duplicate Confluence item: {item_id_for_tracking}")
                    else:
                        # Only mark as unique if neither URL nor ID has been seen
                        is_unique = True
                        if url:
                            confluence_urls.add(url)
                            item_id_for_tracking = url
                        if page_id:
                            confluence_ids.add(page_id)
                            # Prefer URL for tracking if both exist, otherwise use ID
                            item_id_for_tracking = item_id_for_tracking or f"id:{page_id}"
                    # --- END MODIFICATION ---
                elif isinstance(item, str):
                    # For plain string results, check for new URLs within the text
                    urls_in_text = re.findall(r'https?://[^\s"\'<>,)]+', item)
                    found_new_url_in_text = False
                    for url in urls_in_text:
                         # Normalize URL slightly before checking/adding
                         norm_url = url.strip().rstrip('/')
                         if norm_url and norm_url not in confluence_urls:
                              confluence_urls.add(norm_url)
                              found_new_url_in_text = True
                    # Consider unique if new URL found or if it's just text with no URLs
                    is_unique = found_new_url_in_text or not urls_in_text
                    item_id_for_tracking = "text_block" # Placeholder

            if is_unique:
                unique_items_in_chunk.append(item)
                if item_id_for_tracking: new_ids_found_in_chunk.add(str(item_id_for_tracking))

        except Exception as item_proc_err:
            logger.error(f"Error processing item in {source_type} chunk {chunk_idx+1}: {item_proc_err}", exc_info=True)
            # Optionally add error placeholder to unique_items_in_chunk if needed

    # --- Format and Add Unique Content Block ---
    formatted_block_content = None
    if unique_items_in_chunk:
        try:
            header = f"=== {source_type.capitalize()} Results (Chunk {chunk_idx + 1}/{total_chunks}) ===\n"
            # Safely convert items to string for JSON dump or join
            items_to_format = []
            for item in unique_items_in_chunk:
                if isinstance(item, dict):
                    # Always include full item data for Perforce items, including code_snippets
                    item_to_serialize = item
                    try:
                        items_to_format.append(json.dumps(item_to_serialize, indent=2, default=str))
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

    # Add the formatted block if it's new content
    block_to_add = None # Initialize as None
    if formatted_block_content:
        block_hash = hash(formatted_block_content)
        if block_hash not in content_hashes:
            block_to_add = formatted_block_content # Store the block to add
            content_hashes.add(block_hash) # Add to updated hash set
            logger.info(f"Prepared unique content block for {source_type} chunk {chunk_idx + 1} ({len(unique_items_in_chunk)} unique items). New IDs: {len(new_ids_found_in_chunk)}")
        else:
             logger.info(f"Formatted block for {source_type} chunk {chunk_idx + 1} was already seen (hash collision?). Skipping add.")
    else:
        logger.info(f"No unique items found in {source_type} chunk {chunk_idx + 1}, no block generated.")

    # --- Prepare State Updates ---
    next_chunk_idx = chunk_idx + 1
    updates: Dict[str, Any] = {
        f"{source_type}_chunk_idx": next_chunk_idx,
        # Always return the potentially updated tracking sets
        "_jira_keys_seen": jira_keys,
        "_perforce_cl_numbers": perforce_cls,
        "_confluence_urls_seen": confluence_urls,
        "_confluence_ids_seen": confluence_ids,
        # --- MODIFICATION: Always return _content_hashes and enterprise_research_results ---
        "_content_hashes": content_hashes, # Return the working copy of the hash set
        "enterprise_research_results": [block_to_add] if block_to_add else [], # Return list with block or empty list
        # --- END MODIFICATION ---
    }
    # Keep sources_gathered in sync if a block was added
    if block_to_add:
        updates["sources_gathered"] = [block_to_add]
    # else: sources_gathered remains unchanged for this step

    # Check if this source is now finished
    if next_chunk_idx >= total_chunks:
        logger.info(f"Finished processing all {total_chunks} chunks for {source_type}.")
        # Apply cleanup updates from _finish_source
        updates[f"{source_type}_chunks"] = []
        # updates[f"{source_type}_chunk_idx"] = 0 # Index already updated above
        updates[f"{source_type}_total_chunks"] = 0
        updates[f"_last_{source_type}_chunk_idx"] = -1
        updates[f"_stuck_{source_type}_count"] = 0
    else:
         # If not finished, still return the current chunking state
         updates[f"{source_type}_chunks"] = chunks # Pass back the original chunks list
         updates[f"{source_type}_total_chunks"] = total_chunks

    return updates


# --- Specific Chunk Processors (Call the generic one) ---

async def process_confluence_chunk(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Processes the next chunk of Confluence results."""
    # Pass config if process_generic_chunk needs it (it currently doesn't)
    return await process_generic_chunk(state, "confluence")

async def process_jira_chunk(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Processes the next chunk of Jira results."""
    return await process_generic_chunk(state, "jira")

async def process_perforce_chunk(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Processes the next chunk of Perforce results."""
    logger.debug("Entering process_perforce_chunk node") # Added entry log
    result = await process_generic_chunk(state, "perforce")
    logger.debug("Exiting process_perforce_chunk node") # Added exit log
    return result


# --- Helper: Finish Source Processing ---
def _finish_source(state: Dict[str, Any], prefix: str) -> None:
    """
    Resets chunking state variables for a source.
    Modifies the state dictionary *in-place*.
    """
    logger.debug(f"Resetting chunk state for source: {prefix}")
    state[f"{prefix}_chunks"] = []
    state[f"{prefix}_chunk_idx"] = 0 # Reset index to 0
    state[f"{prefix}_total_chunks"] = 0
    # Reset stuck counters as well
    state[f"_last_{prefix}_chunk_idx"] = -1
    state[f"_stuck_{prefix}_count"] = 0

def reset_all_chunking(state: Dict[str, Any]) -> None:
    """
    Resets all chunking state variables for all sources.
    Modifies the state dictionary *in-place*.
    """
    logger.debug("Resetting all chunk state for all sources")
    for prefix in ["confluence", "jira", "perforce"]:
        _finish_source(state, prefix)

# --- Helper: Check if More Chunks Exist ---
def should_process_next_chunk(state: Dict[str, Any], source_type: str) -> bool:
    """
    Generic helper to check if there are more chunks to process for a given source type.
    Includes safety checks against infinite loops. Modifies state *in-place* for stuck detection.
    """
    # --- Safety Checks ---
    hop_counter = state.get("__hop_counter", 0)
    if hop_counter > MAX_CHUNK_PROCESSING_HOPS: # Use constant
        logger.error(f"Chunk processing hop limit ({MAX_CHUNK_PROCESSING_HOPS}) exceeded for {source_type}, forcing completion.")
        _finish_source(state, source_type) # Modify state in-place
        return False

    chunk_idx = state.get(f"{source_type}_chunk_idx", 0)
    total_chunks = state.get(f"{source_type}_total_chunks", 0)
    chunks = state.get(f"{source_type}_chunks", [])

    # Validate indices and chunk presence
    if not isinstance(chunk_idx, int) or chunk_idx < 0:
        logger.warning(f"Invalid chunk index {chunk_idx} for {source_type}. Resetting.")
        _finish_source(state, source_type) # Modify state in-place
        return False
    if not isinstance(total_chunks, int) or total_chunks < 0:
         logger.warning(f"Invalid total chunks {total_chunks} for {source_type}. Resetting.")
         _finish_source(state, source_type) # Modify state in-place
         return False
    # If total_chunks is > 0 but chunks list is empty, state is inconsistent
    if total_chunks > 0 and not chunks:
        logger.warning(f"Inconsistent state: {total_chunks} {source_type} chunks expected, but none found. Resetting.")
        _finish_source(state, source_type) # Modify state in-place
        return False
    # If index points beyond the actual available chunks list
    if chunks and chunk_idx >= len(chunks):
         logger.info(f"Chunk index {chunk_idx} reached end of available {source_type} chunks ({len(chunks)}). Finishing source.")
         _finish_source(state, source_type) # Modify state in-place
         return False

    # Stuck loop detection (modifies state in-place)
    last_index = state.get(f"_last_{source_type}_chunk_idx", -1)
    if chunk_idx == last_index and chunk_idx >= 0: # Check >= 0 to avoid triggering on initial state
        stuck_count = state.get(f"_stuck_{source_type}_count", 0) + 1
        state[f"_stuck_{source_type}_count"] = stuck_count # Modify state
        if stuck_count > 3:
            logger.warning(f"Stuck processing {source_type} chunk {chunk_idx} for {stuck_count} iterations. Forcing completion.")
            _finish_source(state, source_type) # Modify state in-place
            return False
    else:
        state[f"_stuck_{source_type}_count"] = 0 # Modify state (reset stuck count)

    # Update last seen index for next check (modifies state in-place)
    state[f"_last_{source_type}_chunk_idx"] = chunk_idx

    # --- Normal Check ---
    # Are there more chunks based on index and total count?
    has_next = chunk_idx < total_chunks

    # If finishing normally, ensure state is cleaned up
    if not has_next and total_chunks > 0:
        logger.debug(f"No more {source_type} chunks to process ({chunk_idx}/{total_chunks}). Finishing source.")
        _finish_source(state, source_type) # Modify state in-place

    return has_next

# --- Specific Checkers (Call the generic one) ---

def should_process_next_confluence_chunk(state: Dict[str, Any]) -> bool:
    """Checks if there are more Confluence chunks to process."""
    return should_process_next_chunk(state, "confluence")

def should_process_next_jira_chunk(state: Dict[str, Any]) -> bool:
    """Checks if there are more Jira chunks to process."""
    return should_process_next_chunk(state, "jira")

def should_process_next_perforce_chunk(state: Dict[str, Any]) -> bool:
    """Checks if there are more Perforce chunks to process."""
    return should_process_next_chunk(state, "perforce")

# --- Refactored Routing and Filtering ---

def route_after_research(state: Dict[str, Any]) -> str:
    """
    Decide the next step after enterprise_research or a chunk processing node.
    Prioritizes processing remaining chunks before moving to filtering/analysis.
    Modifies state for hop counting.
    """
    # --- Safety Checks (modifies state) ---
    hop_counter = state.get("__hop_counter", 0) + 1
    state["__hop_counter"] = hop_counter # Modify state
    if hop_counter > MAX_TOTAL_HOPS: # Use overall hop limit
        logger.error(f"Maximum graph hops ({MAX_TOTAL_HOPS}) exceeded. Forcing summarization.")
        reset_all_chunking(state) # Modify state
        return "finalize_summary" # Go directly to final summary

    # --- Prioritize Chunk Processing ---
    # Check functions modify state for stuck detection but return boolean
    if should_process_next_confluence_chunk(state):
        logger.debug(f"Routing to process_confluence_chunk: {state.get('confluence_chunk_idx', 0)}/{state.get('confluence_total_chunks', 0)}")
        return "process_confluence_chunk"
    elif should_process_next_jira_chunk(state):
        logger.debug(f"Routing to process_jira_chunk: {state.get('jira_chunk_idx', 0)}/{state.get('jira_total_chunks', 0)}")
        return "process_jira_chunk"
    elif should_process_next_perforce_chunk(state):
        logger.debug(f"Routing to process_perforce_chunk: {state.get('perforce_chunk_idx', 0)}/{state.get('perforce_total_chunks', 0)}")
        return "process_perforce_chunk"
    else:
        # --- All Chunk Processing Complete ---
        logger.info("All chunk processing complete. Moving to filter_irrelevant_data.")
        # Hop counter continues for subsequent phases
        return "filter_irrelevant_data" # Move to the next phase

@traceable # Optional
async def filter_irrelevant_data(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Performs final filtering/cleanup *after* all chunk processing.
    - Removes empty blocks.
    - Filters blocks based on direct relevance to the research_topic.
    - Updates sources_gathered.

    Args:
        state (ResearchState): The current graph state.
        config (RunnableConfig | None): LangGraph runtime config.

    Returns:
        Dict[str, Any]: Dictionary containing updated state fields.
    """
    logger.info("--- Entering filter_irrelevant_data --- ")
    state_updates: Dict[str, Any] = {}

    current_blocks = state.get("enterprise_research_results", [])
    research_topic = state.get("research_topic", "").strip()

    if not current_blocks:
        logger.info("No research results to filter.")
        state_updates["sources_gathered"] = []
        return state_updates

    if not research_topic:
        logger.warning("No research topic found, skipping relevance filtering.")
        # Filter only for empty/error blocks if no topic
        filtered_blocks = [
            b for b in current_blocks
            if isinstance(b, str) and b.strip() and "Error: Failed to format chunk content" not in b
        ]
        removed_count = len(current_blocks) - len(filtered_blocks)
        logger.info(f"Removed {removed_count} empty/error blocks (topic-based filtering skipped).")
        state_updates["enterprise_research_results"] = filtered_blocks
        state_updates["sources_gathered"] = filtered_blocks
        return state_updates

    # --- Start Relevance Filtering Logic (Conservative) ---
    # We will only remove blocks that are *clearly* unrelated; unclear blocks will be kept to avoid data loss.
    logger.info(f"Filtering results for relevance to topic: '{research_topic}' (conservative mode)")

    relevant_blocks: List[str] = []
    possibly_relevant_blocks: List[str] = []
    topic_lower = research_topic.lower()
    # Extract potential key identifiers from the topic (e.g., MTVxxxx, JIRA-KEY)
    topic_keys = set(re.findall(r'\b(MTV\d{4,}|[A-Z]+-\d+)\b', research_topic, re.IGNORECASE))
    logger.debug(f"Topic keys extracted for filtering: {topic_keys}")

    for block in current_blocks:
        if not isinstance(block, str) or not block.strip() or "Error: Failed to format chunk content" in block:
            continue  # Skip empty/error blocks

        block_content_lower = block.lower()

        # Obvious relevance check
        if topic_lower in block_content_lower:
            relevant_blocks.append(block)
            continue

        # Check for identifier overlap
        if topic_keys and any(k.lower() in block_content_lower for k in topic_keys):
            relevant_blocks.append(block)
            continue

        # If we reach this point, the block might still be useful (e.g., indirect references).
        possibly_relevant_blocks.append(block)

    # Decide whether to keep possibly relevant blocks
    # If we removed more than 30% of blocks, re-add the ambiguous ones to avoid losing context.
    removed_count_initial = len(current_blocks) - len(relevant_blocks)
    if removed_count_initial > 0 and (removed_count_initial / len(current_blocks)) > 0.3:
        logger.warning("More than 30% of blocks flagged as irrelevant – retaining ambiguous blocks to avoid data loss.")
        relevant_blocks.extend(possibly_relevant_blocks)

    # If still nothing, fall back to original list to avoid empty context
    if not relevant_blocks:
        logger.warning("Relevance filtering produced 0 blocks – reverting to unfiltered set to avoid missing context.")
        relevant_blocks = current_blocks
    # --- End Relevance Filtering Logic ---

    final_blocks = relevant_blocks # Use the relevant blocks
    removed_count = len(current_blocks) - len(final_blocks)

    logger.info(f"Filtering complete. Kept {len(final_blocks)} relevant blocks, removed {removed_count}. Topic: '{research_topic}'")

    # Update the state only if changes were made
    if removed_count > 0:
        state_updates["enterprise_research_results"] = final_blocks
        state_updates["sources_gathered"] = final_blocks
    else:
        # Ensure sources_gathered is consistent even if no filtering occurred
        state_updates["sources_gathered"] = current_blocks

    logger.info("--- Exiting filter_irrelevant_data --- ")
    return state_updates


# --- Analysis and Processing Nodes ---
# (Include implementations for generate_query, analyze_perforce_diffs,
#  perform_web_search, analyze_security, validate_technical_aspects,
#  verify_summary, extract_citations, generate_knowledge_graph,
#  reflect_on_findings, finalize_summary, export_research, error_recovery,
#  check_content_quality, _verification_status, try_determine_continuation
#  These should be adapted to use the correct state fields, e.g.,
#  raw_*_results for structured data, enterprise_research_results for formatted blocks)

# --- Placeholder Implementations for Missing Nodes ---
# Replace these with actual logic from the original file, adapted for the new state structure

@traceable
async def generate_query(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Generate an optimized search query based on the research topic."""
    logger.info("--- Node: generate_query ---")
    topic = state.get('research_topic', '')
    if not topic:
        logger.warning("No research topic found.")
        return {"search_query": ""} # Return update dict

    # Simple query generation: use topic directly (can be enhanced with LLM)
    # Sanitize potentially for safety, although search wrappers also sanitize
    query = topic # Basic example
    logger.info(f"Generated search query: {query}")
    # Return dict for state update
    return {"search_query": query, "alternative_terms": []} # Include alternative_terms if generated

@traceable
async def analyze_perforce_diffs(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Analyze Perforce diffs using raw Perforce results."""
    logger.info("--- Node: analyze_perforce_diffs ---")
    # Default analysis state
    analysis_result = {
        "summary": "No Perforce diff analysis performed.",
        "findings": [], "languages": [], "total_changes": "unknown",
        "high_impact_files": [], "technical_debt_indicators": []
    }

    # Use the raw structured data
    perforce_results = state.get("raw_perforce_results", [])
    # If no raw Perforce tool results, fallback to parsing formatted enterprise_research_results
    if not perforce_results:
        logger.info("No raw Perforce results found to analyze. Falling back to enterprise_research_results.")
        perforce_results = []
        formatted_blocks = state.get("enterprise_research_results", []) or []
        for block in formatted_blocks:
            if isinstance(block, str) and block.startswith("=== Perforce Results"):
                lines = block.splitlines()[1:]
                for line in lines:
                    # Expect lines like 'CL 12345 - description'
                    m = re.match(r"CL\s*(\d+)\s*-\s*(.*)", line)
                    if m:
                        perforce_results.append({
                            'changelist': m.group(1),
                            'description': m.group(2)
                        })
        if not perforce_results:
            logger.info("No formatted Perforce result blocks found to analyze.")
            return {"perforce_diff_analysis": analysis_result}

    # --- Logic to analyze diffs ---
    # This logic would involve iterating through perforce_results (list of dicts),
    # extracting diff information (e.g., file paths, code snippets if available),
    # potentially calling an LLM or a dedicated code analysis tool.
    # --- MODIFIED: Enhance analysis by aggregating code_summary data ---
    num_changes = len(perforce_results)
    total_added = 0
    total_removed = 0
    languages = set()
    key_files = defaultdict(int)
    processed_cls = set()

    for item in perforce_results:
        cl = item.get('changelist') or item.get('change')
        if not cl or cl in processed_cls:
            continue
        processed_cls.add(cl)

        # Check for code_summary potentially added by the PerforceTool wrapper/enhancer
        # Note: This assumes the wrapper adds a 'code_summary' dict to each item
        code_summary = item.get("code_summary")
        if isinstance(code_summary, dict):
            total_added += code_summary.get("added_lines", 0)
            total_removed += code_summary.get("removed_lines", 0)
            languages.update(code_summary.get("languages", []))
            # Count files mentioned in this summary
            for f_path in code_summary.get("files", []): # Assuming code_summary might list files?
                 if isinstance(f_path, str):
                      key_files[f_path.split('/')[-1]] += 1 # Count by filename
        else:
             # Fallback: Count files directly from the item if no code_summary
             files_list = item.get('files', [])
             if isinstance(files_list, list):
                  for file_detail in files_list:
                       if isinstance(file_detail, dict) and file_detail.get('path'):
                            key_files[file_detail['path'].split('/')[-1]] += 1
                            if file_detail.get('language'):
                                 languages.add(file_detail['language'])

    analysis_summary_parts = [f"Found {num_changes} relevant Perforce changes."]
    if total_added > 0 or total_removed > 0:
        analysis_summary_parts.append(f"Estimated code impact: ~{total_added} lines added, ~{total_removed} lines removed.")
    if languages:
        analysis_summary_parts.append(f"Primary languages involved: {', '.join(filter(None, languages))}.")

    top_files = sorted(key_files.items(), key=lambda item: item[1], reverse=True)[:3]
    if top_files:
         analysis_summary_parts.append(f"Most frequently modified files include: {', '.join([f[0] for f in top_files])}.")

    analysis_result["summary"] = " ".join(analysis_summary_parts)
    analysis_result["total_changes"] = num_changes
    analysis_result["languages"] = list(filter(None, languages))
    analysis_result["added_lines"] = total_added
    analysis_result["removed_lines"] = total_removed
    # --- END ENHANCED ANALYSIS ---

    # Example: Extract CL numbers for findings (keep this for now)
    findings = []

    logger.info(f"Perforce diff analysis complete. Summary: {analysis_result['summary']}")
    # Return dict for state update
    return {"perforce_diff_analysis": analysis_result}

@traceable
async def perform_web_search(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Perform web search using DuckDuckGo."""
    logger.info("--- Node: perform_web_search ---")
    if not HAS_DDGS:
        logger.warning("DDGS not available, skipping web search.")
        return {"raw_web_search_results": []} # Return update dict

    query = state.get("search_query", state.get("research_topic", ""))
    if not query:
        logger.error("No query available for web search")
        return {"raw_web_search_results": []}

    web_results_list = []
    try:
        logger.info(f"Performing web search for: {query}")
        with DDGS() as ddgs:
            # Use text search, limit results
            results_iterator = ddgs.text(query, max_results=5)
            for result in results_iterator:
                web_results_list.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "web_search"
                })
        logger.info(f"Web search found {len(web_results_list)} results.")

        # Optional: Enhance with Trafilatura if available
        if trafilatura and web_results_list:
             logger.info("Attempting to extract full text for top web results...")
             # (Add Trafilatura extraction logic here if needed, modifying web_results_list)
             pass # Placeholder for extraction logic

    except Exception as e:
        logger.error(f"Error during web search: {e}", exc_info=True)
        # Add error to state? For now, just return empty results.

    # Return dict for state update
    return {"raw_web_search_results": web_results_list}

@traceable
async def analyze_security(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Analyze security implications using static analysis and LLM."""
    logger.info("--- Node: analyze_security ---")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    findings = []

    # --- Static Scan ---
    try:
        # Combine content from formatted results for scanning
        # Use a sample to avoid excessive length
        content_to_scan = "\n\n".join(state.get("enterprise_research_results", [])[:5]) # Scan first 5 blocks
        if len(content_to_scan) > 10000: content_to_scan = content_to_scan[:10000] # Limit length

        if content_to_scan:
             static_findings = SecurityAnalyzer.analyze_file_content(content_to_scan)
             if static_findings:
                  logger.info(f"Static security scan found {len(static_findings)} potential issues.")
                  findings.extend(static_findings)
        else:
             logger.info("No content available for static security scan.")
    except Exception as static_err:
        logger.error(f"Static security analysis failed: {static_err}", exc_info=True)

    # --- LLM Analysis ---
    summary_for_llm = state.get("running_summary", "")
    if summary_for_llm:
         try:
             # --- MODIFIED: Use central config, add format="json" ---
             llm = ChatOllama(
                 base_url=cfg.ollama_base_url,
                 model=cfg.local_llm,
                 format="json", # Request JSON output for easier parsing
                 **LLM_CONFIG_PARAMS # Unpack shared parameters
             )
             # --- END MODIFICATION ---
             prompt = security_analysis_instructions
             human_msg = f"Analyze security implications of:\n{summary_for_llm[:MAX_SUMMARY_CONTEXT]}"

             result = await asyncio.wait_for(
                 llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=human_msg)]),
                 timeout=300 # Adjust timeout
             )
             try:
                 llm_parsed = json.loads(result.content)
                 llm_found = []
                 if isinstance(llm_parsed, list):
                     llm_found = llm_parsed
                 elif isinstance(llm_parsed, dict):
                     llm_found = llm_parsed.get("findings", []) or llm_parsed.get("issues", [])
                 # Validate findings format
                 validated_llm = [f for f in llm_found if isinstance(f, dict)]
                 if validated_llm:
                      logger.info(f"LLM security analysis identified {len(validated_llm)} potential issues.")
                      findings.extend(validated_llm)
             except json.JSONDecodeError as decode_err:
                 logger.error(f"Failed to parse LLM security findings JSON: {decode_err}")
         except Exception as llm_err:
             logger.error(f"LLM security analysis failed: {llm_err}", exc_info=True)

    # --- Deduplicate Findings ---
    deduped_findings: List[Dict[str, Any]] = []
    seen_keys: Set[tuple] = set()
    for item in findings:
        # Create a unique key based on type, location (if available), and description/match
        key = (
            item.get("type", "unknown").lower(),
            item.get("line_number"), # Can be None
            (item.get("match") or item.get("description", ""))[:50] # Use first 50 chars of description/match
        )
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_findings.append(item)

    logger.info(f"Security analysis complete. Total unique findings: {len(deduped_findings)}")
    # Return dict for state update
    return {"security_findings": deduped_findings}

@traceable
async def validate_technical_aspects(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Validate technical aspects using LLM."""
    logger.info("--- Node: validate_technical_aspects ---")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    summary = state.get("running_summary", "")
    validation_result = {} # Default empty dict

    if not summary:
        logger.warning("No summary available for technical validation.")
        return {"technical_validation": validation_result}

    try:
        # --- MODIFIED: Use central config, add format="json" ---
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            format="json", # Request JSON output
            **LLM_CONFIG_PARAMS # Unpack shared parameters
        )
        # --- END MODIFICATION ---
        prompt = technical_validation_instructions
        human_msg = f"Validate technical aspects of:\n{summary[:MAX_SUMMARY_CONTEXT]}"

        result = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=human_msg)]),
            timeout=300 # Adjust timeout
        )
        try:
            validation_result = json.loads(result.content)
            if not isinstance(validation_result, dict):
                 logger.error("Technical validation did not return a JSON object.")
                 validation_result = {"error": "Invalid format received from LLM"}
        except json.JSONDecodeError as decode_err:
            logger.error(f"Failed to parse technical validation JSON: {decode_err}")
            validation_result = {"error": f"JSON parsing failed: {decode_err}"}

    except Exception as e:
        logger.error(f"Error in technical validation: {e}", exc_info=True)
        validation_result = {"error": f"Validation failed: {e}"}

    logger.info("Technical validation complete.")
    # Return dict for state update
    return {"technical_validation": validation_result}

@traceable
async def verify_summary(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Verify that the research summary is accurate, complete, and addresses the topic."""
    logger.info("--- Node: verify_summary ---")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    # --- MODIFIED: Use individual summaries + cross-source for verification context ---
    jira_summary = state.get("jira_summary", "")
    confluence_summary = state.get("confluence_summary", "")
    perforce_summary = state.get("perforce_summary", "")
    web_summary = state.get("web_summary", "")
    cross_source_analysis = state.get("cross_source_analysis_text", "")

    verification_context_parts = [
        "## JIRA Analysis", jira_summary if jira_summary else "_Not generated._", "\n", # Fixed newline placement
        "## Confluence Analysis", confluence_summary if confluence_summary else "_Not generated._", "\n", # Fixed newline placement
        "## Perforce Analysis", perforce_summary if perforce_summary else "_Not generated._", "\n", # Fixed newline placement
        "## Web Search Analysis", web_summary if web_summary else "_Not generated or N/A._", "\n", # Fixed newline placement
        cross_source_analysis if cross_source_analysis else "_Not generated._", "\n", # Fixed newline placement
    ]
    verification_context = "\n".join(filter(None, verification_context_parts))
    # --- END MODIFICATION ---

    topic = state.get("research_topic", "Unknown Topic")
    verification_results = { # Default structure
        "accurate": True, "complete": True, "issues": [], "suggestions": []
    }

    # --- MODIFIED: Check verification_context ---
    if not verification_context.strip() or "_Not generated._" in verification_context : # Basic check if context is empty/default
        # Skip verification gracefully if summary is not yet available.
        logger.warning("No summary components available for verification – skipping this step for now.")
        verification_results = {
            "accurate": True,    # Treat as accurate to avoid error route
            "complete": False,   # Cannot assess completeness yet
            "issues": [],        # No issues recorded because verification was skipped
            "skipped": True      # Additional flag
        }
        return {"verification_results": verification_results}
    # --- END MODIFICATION ---

    verification_prompt = (
        f"Please verify the accuracy and completeness of these combined research summary sections on the topic: '{topic}'\n\n"
        # --- MODIFIED: Use verification_context ---
        f"Summary Sections:\n{verification_context[:MAX_SUMMARY_CONTEXT]}\n\n" # Limit context
        # --- END MODIFICATION ---
        "Provide your verification as a JSON object with these keys:\n"
        "- 'accurate': boolean (do the sections accurately reflect the topic based on typical knowledge?)\n"
        "- 'complete': boolean (do the sections seem reasonably complete for a brief overview?)\n"
        "- 'issues': array of strings listing specific factual inaccuracies or major omissions detected.\n"
        "- 'suggestions': array of strings for specific improvements (e.g., 'Clarify section on X', 'Add details about Y')."
    )

    try:
        # --- MODIFIED: Use central config, add format="json" ---
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            format="json", # Request JSON output
            **LLM_CONFIG_PARAMS # Unpack shared parameters
        )
        # --- END MODIFICATION ---
        result = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=verification_prompt)]),
            timeout=300 # Adjust timeout
        )
        try:
            parsed_results = json.loads(result.content)
            # Validate structure and types
            if isinstance(parsed_results, dict) and \
               isinstance(parsed_results.get('accurate'), bool) and \
               isinstance(parsed_results.get('complete'), bool) and \
               isinstance(parsed_results.get('issues'), list) and \
               isinstance(parsed_results.get('suggestions'), list):
                verification_results = parsed_results
            else:
                 logger.error("Verification LLM returned invalid JSON structure.")
                 verification_results["issues"].append("LLM returned invalid verification format.")
        except json.JSONDecodeError as decode_err:
            logger.error(f"Failed to parse verification results JSON: {decode_err}")
            verification_results["issues"].append(f"JSON parsing failed: {decode_err}")

    except Exception as e:
        logger.error(f"Error during summary verification: {e}", exc_info=True)
        verification_results["issues"].append(f"Verification process failed: {e}")

    logger.info(f"Summary verification complete. Accurate: {verification_results['accurate']}, Complete: {verification_results['complete']}, Issues: {len(verification_results['issues'])}")
    # Return dict for state update
    return {"verification_results": verification_results}


@traceable
async def assemble_intermediate_summary(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Assembles an intermediate summary from all the individual source summaries.
    This is useful for preparing the final summary structure before going through
    citation extraction and other finishing steps.
    """
    logger.info("--- Node: assemble_intermediate_summary ---")
    topic = state.get("research_topic", "Unknown Topic")

    # Get all the individual summaries
    jira_summary = state.get("jira_summary", "")
    confluence_summary = state.get("confluence_summary", "")
    perforce_summary = state.get("perforce_summary", "")
    web_summary = state.get("web_summary", "")
    cross_source_analysis = state.get("cross_source_analysis_text", "")

    # Get all quiz answers if available
    quiz_questions = state.get("quiz_questions", [])
    quiz_answers = state.get("quiz_answers", [])

    # Format the quiz answers if they exist
    quiz_section = ""
    if quiz_answers:
        formatted_qa = _format_quiz_answers(quiz_answers)
        quiz_section = f"""
## Quiz Answers
{formatted_qa}
"""

    # Combine all summaries into a structured report
    intermediate_summary = f"""# Research Summary: {topic}

## Executive Summary
{topic} has been researched across multiple data sources including JIRA, Confluence, and Perforce repositories.

## JIRA Findings
{jira_summary}

## Confluence Information
{confluence_summary}

## Perforce Code Analysis
{perforce_summary}

## Web Research
{web_summary}

## Cross-Source Analysis
{cross_source_analysis}
{quiz_section}
"""

    # Calculate basic summary stats
    word_count = len(intermediate_summary.split())
    section_count = intermediate_summary.count("##")

    logger.info(f"Assembled intermediate summary with {word_count} words and {section_count} sections")

    # Update the running summary with this intermediate version
    return {
        "running_summary": intermediate_summary,
        "intermediate_summary_stats": {
            "word_count": word_count,
            "section_count": section_count,
            "has_jira": bool(jira_summary),
            "has_confluence": bool(confluence_summary),
            "has_perforce": bool(perforce_summary),
            "has_web": bool(web_summary),
            "has_cross_source": bool(cross_source_analysis),
            "has_quiz_answers": bool(quiz_answers)
        }
    }

@traceable
async def extract_citations(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Extract and standardize citations from the research summary."""
    logger.info("--- Node: extract_citations ---")
    summary = state.get("running_summary", "")
    citations = [] # Default empty list

    if not summary:
        logger.warning("No summary available for citation extraction.")
        return {"citations": citations}

    # Use regex to find markdown links: [text](url)
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    citations_dict: Dict[str, Dict] = {} # url -> citation info

    try:
        for match in link_pattern.finditer(summary):
            text, raw_url = match.groups()
            url = raw_url.strip().strip("'\" ") # Basic cleaning

            # Skip invalid or placeholder URLs
            if not url or url == "#" or url.startswith("about:") or url == "https://example.com":
                continue

            # Normalize URL slightly (remove trailing slash) for better matching
            norm_url = url.rstrip('/')

            # Determine source ID (e.g., domain name)
            try:
                domain = urlparse(norm_url).netloc or "Unknown Source"
            except ValueError:
                domain = "Invalid URL" # Handle potential parsing errors

            # Update citation count or add new entry
            if norm_url not in citations_dict:
                citations_dict[norm_url] = {
                    "source_id": domain,
                    "url": url, # Store original URL for display
                    "text": text.strip(),
                    "count": 1
                }
            else:
                citations_dict[norm_url]["count"] += 1
                # Optionally update text if a longer version is found?
                if len(text.strip()) > len(citations_dict[norm_url]["text"]):
                     citations_dict[norm_url]["text"] = text.strip()

        # Convert dict to list and sort (e.g., by count descending)
        citations = sorted(citations_dict.values(), key=lambda x: x["count"], reverse=True)

        logger.info(f"Extracted {len(citations)} unique citations.")

    except Exception as e:
        logger.error(f"Error during citation extraction: {e}", exc_info=True)
        # Return empty list on error, maybe add error to state?

    # Return dict for state update
    return {"citations": citations}

@traceable
async def generate_knowledge_graph(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Generate a knowledge graph from the research data."""
    logger.info("--- Node: generate_knowledge_graph ---")
    kg_state_update = {"knowledge_graph": {"status": "skipped", "reason": "Not implemented"}} # Default

    if not HAS_VISUALIZATION:
        logger.warning("Skipping knowledge graph generation - visualization libraries not available")
        kg_state_update["knowledge_graph"]["reason"] = "Visualization libraries missing"
        return kg_state_update

    summary = state.get("running_summary", "")
    if not summary:
        logger.warning("No summary available for knowledge graph generation.")
        kg_state_update["knowledge_graph"]["reason"] = "No summary available"
        return kg_state_update

    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()

    try:
        # --- LLM Call to Extract Entities/Relationships ---
        # (Keep the LLM call logic as in the original file, using appropriate prompt)
        kg_prompt = (
            "Extract key entities (people, projects, technologies, concepts) and their relationships "
            "from this research summary. Return a JSON object with 'entities': [{'id', 'name', 'type'}] "
            "and 'relationships': [{'source', 'target', 'type'}]. Use simple relationship types like "
            "'related_to', 'uses', 'part_of', 'developed_by'.\n\n"
            f"Summary:\n{summary[:5000]}" # Limit context
        )
        # --- MODIFIED: Use central config, add format="json" ---
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            format="json", # Request JSON output
            **LLM_CONFIG_PARAMS # Unpack shared parameters
        )
        # --- END MODIFICATION ---
        result = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=kg_prompt)]), timeout=300
        )
        kg_data = json.loads(result.content)

        # --- Validate and Build Graph ---
        if not isinstance(kg_data, dict) or "entities" not in kg_data or "relationships" not in kg_data:
             logger.warning(f"Knowledge graph data missing required fields or invalid format. Type: {type(kg_data)}")
             
             # Initialize to dict if not already
             if not isinstance(kg_data, dict):
                 kg_data = {}
             
             # Add missing required fields with empty lists
             if "entities" not in kg_data:
                 kg_data["entities"] = []
                 
             if "relationships" not in kg_data:
                 kg_data["relationships"] = []
                 
             logger.info("Created fallback empty knowledge graph structure")
             # Continue with fallback structure instead of raising exception

        G = nx.DiGraph()
        node_map = {} # Map entity name/id to graph node id
        for entity in kg_data.get("entities", []):
            node_id = entity.get("id", entity.get("name")) # Use name as fallback id
            if not node_id: continue
            node_map[node_id] = node_id # Store mapping
            G.add_node(node_id, name=entity.get("name", node_id), type=entity.get("type", "unknown"))

        for rel in kg_data.get("relationships", []):
            source_id = rel.get("source")
            target_id = rel.get("target")
            # Use mapped IDs if available
            source_node = node_map.get(source_id)
            target_node = node_map.get(target_id)
            rel_type = rel.get("type", "related_to")
            if source_node and target_node and G.has_node(source_node) and G.has_node(target_node):
                G.add_edge(source_node, target_node, type=rel_type)

        kg_state_update["knowledge_graph"] = {
            "data": kg_data, # Store raw LLM output
            # "graph": G, # Storing NetworkX object in state might be problematic for serialization
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "status": "success"
        }
        logger.info(f"Generated knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # --- Generate Visualization (Optional) ---
        # (Keep visualization logic as in the original file, using G)
        if cfg.configurable.get("generate_visualizations", False) and G.number_of_nodes() > 0:
             try:
                  viz_id = uuid4().hex[:8]
                  filename = f"knowledge_graph_{viz_id}.png"
                  output_path = Path("./outputs")
                  output_path.mkdir(exist_ok=True)
                  full_path = output_path / filename

                  plt.figure(figsize=(14, 11)) # Adjusted size
                  pos = nx.spring_layout(G, k=0.6, iterations=50) # Adjust layout params

                  node_types = nx.get_node_attributes(G, 'type')
                  unique_types = sorted(list(set(node_types.values())))
                  colors = plt.cm.get_cmap('tab10', len(unique_types)) # Use tab10 colormap
                  color_map = {ntype: colors(i) for i, ntype in enumerate(unique_types)}
                  node_colors = [color_map.get(node_types.get(n, 'unknown'), '#cccccc') for n in G.nodes()]

                  nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.9)
                  nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True, edge_color='gray')
                  labels = {n: data.get("name", n) for n, data in G.nodes(data=True)}
                  nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

                  # Create legend
                  legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=ntype,
                                          markerfacecolor=color_map[ntype], markersize=10)
                                   for ntype in unique_types]
                  plt.legend(handles=legend_elements, title="Entity Types", loc='upper left')

                  plt.title(f"Knowledge Graph: {state.get('research_topic', 'Research')}", fontsize=16)
                  plt.axis('off')
                  plt.tight_layout()
                  plt.savefig(full_path, dpi=150) # Adjust DPI if needed
                  plt.close()

                  kg_state_update["knowledge_graph"]["visualization_path"] = str(full_path)
                  logger.info(f"Saved knowledge graph visualization to {full_path}")
             except Exception as viz_err:
                  logger.error(f"Failed to generate knowledge graph visualization: {viz_err}", exc_info=True)
                  kg_state_update["knowledge_graph"]["visualization_status"] = f"Error: {viz_err}"


    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse knowledge graph data as JSON: {e}")
        kg_state_update["knowledge_graph"] = {"status": "error", "reason": f"JSON parsing failed: {e}"}
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {e}", exc_info=True)
        kg_state_update["knowledge_graph"] = {"status": "error", "reason": str(e)}

    # Return dict for state update
    return kg_state_update

def _format_quiz_answers(quiz_answers: List[Dict[str, str]]) -> str:
    """Format quiz answers into a readable string for reflection or display."""
    if not quiz_answers:
        return "_No quiz answers available._"

    formatted_parts = []
    for i, qa in enumerate(quiz_answers):
        question = qa.get("question", f"Question {i+1}")
        answer = qa.get("answer", "No answer provided")
        justification = qa.get("justification", "No justification provided")

        formatted_parts.append(f"**Question {i+1}:** {question}\n**Answer:** {answer}\n**Justification:** {justification}")

    return "\n\n---\n\n".join(formatted_parts)

@traceable
async def reflect_on_findings(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Reflect on the research findings and generate insights using LLM."""
    logger.info("--- Node: reflect_on_findings ---")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    # --- MODIFIED: Use assembled running_summary + QA for reflection context ---
    intermediate_summary = state.get("running_summary", "") # Use the assembled summary
    quiz_answers_str = _format_quiz_answers(state.get("quiz_answers", [])) # Format QA

    reflection_context_parts = [
        "## Assembled Research Summary",
        intermediate_summary if intermediate_summary and "_No summary components" not in intermediate_summary else "_No summary was assembled._\n",
        "\n",
        "## Quiz Answers",
        quiz_answers_str if quiz_answers_str and "_No quiz answers" not in quiz_answers_str else "_No quiz answers provided or generated._\n",
        "\n",
    ]
    reflection_context = "\n".join(filter(None, reflection_context_parts))
    # --- END MODIFICATION ---

    reflection_data = "Reflection skipped: No summary/QA components available." # Default
    # --- ADDED: Initialize corrections list by retrieving existing ones ---
    existing_corrections = state.get("identified_corrections", [])
    new_corrections = [] # Corrections found in this step
    # --- END ADDED ---

    # Skip reflection if chunk processing is still somehow active (safety check)
    if (should_process_next_chunk(state, "confluence") or
        should_process_next_chunk(state, "jira") or
        should_process_next_chunk(state, "perforce")):
        logger.info("Skipping reflection – chunk processing seems active.")
        # Return existing corrections unmodified if skipped
        return {"reflection_data": "Reflection skipped: Chunk processing active.", "identified_corrections": existing_corrections}

    # --- MODIFIED: Check reflection_context and Limit Context ---
    if reflection_context.strip() and "_No summary components" not in reflection_context and "_No quiz answers" not in reflection_context: # Check if *any* content is available
        # Apply context limit
        current_context_len = len(reflection_context)
        if current_context_len > MAX_SUMMARY_CONTEXT:
             logger.warning(f"Reflection context truncated from {current_context_len} to {MAX_SUMMARY_CONTEXT} chars.")
             reflection_context = reflection_context[:MAX_SUMMARY_CONTEXT] + "... [CONTEXT TRUNCATED] ..."

        # --- MODIFIED: Refined Reflection Prompt with Correction Identification ---
        reflection_prompt = (
            "Critically analyze the following research outputs (summaries, cross-source analysis, quiz answers). Identify:\\n"
            "1. **Key Findings:** What are the most important takeaways from the data?\\n"
            "2. **Gaps/Contradictions:** Are there obvious gaps in information, unclear points, or contradictions between sources or answers?\\n"
            # --- ADDED: Explicit instruction for corrections --- 
            "3. **Potential Corrections Needed:** Based ONLY on the Gaps/Contradictions identified in point 2, list any specific statements or findings from the summaries that seem inaccurate or need correction/clarification. Use a clear list under the heading `### Potential Corrections Needed`. If none, state 'None needed'.\\n"
            # --- END ADDED ---
            "4. **Next Steps/Questions:** Based *only* on the analysis, suggest 1-2 concrete next steps or follow-up questions. If the research seems complete or no obvious gaps exist, state that. *If gaps were identified, explicitly suggest if another research loop might be beneficial.*\\n\\n"
            f"Research Outputs:\\n{reflection_context}\\n\\n"
            "Provide your reflection as concise bullet points under headings: Findings, Gaps/Contradictions, Potential Corrections Needed, Next Steps."
        )
        # --- END MODIFICATION ---

        try:
            # --- MODIFIED: Use central config ---
            llm = ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.local_llm,
                **LLM_CONFIG_PARAMS # Unpack shared parameters
                # Allow slightly higher temp if needed for reflection? Overridden by central dict for now.
                # temperature=0.2
            )
            # --- END MODIFICATION ---
            result = await asyncio.wait_for(
                llm.ainvoke([SystemMessage(content=reflection_prompt)]), timeout=300
            )
            reflection_data_raw = result.content
            reflection_data = clean_llm_output(reflection_data_raw)
            logger.info("Reflection completed successfully.")

            # --- ADDED: Extract corrections from reflection output ---
            correction_marker = "### Potential Corrections Needed"
            if correction_marker in reflection_data_raw: # Check raw output before cleaning might remove it
                try:
                    correction_section = reflection_data_raw.split(correction_marker, 1)[1]
                    # Extract lines after marker, split potential further headings
                    if "### Next Steps" in correction_section:
                         correction_section = correction_section.split("### Next Steps", 1)[0]

                    potential_corrections = [line.strip().lstrip('*- ') for line in correction_section.split('\n') if line.strip()]
                    # Filter out placeholders and very short lines
                    new_corrections = [c for c in potential_corrections if len(c) > 10 and "none needed" not in c.lower() and not c.lower().startswith(("list", "note", "identify"))]
                    if new_corrections:
                         logger.info(f"Extracted {len(new_corrections)} potential corrections from reflection.")
                except Exception as ex_err:
                     logger.error(f"Error extracting corrections from reflection: {ex_err}")
            # --- END ADDED ---

        except Exception as e:
            logger.error(f"Error during reflection LLM call: {e}", exc_info=True)
            reflection_data = f"Reflection generation failed: {e}"
            new_corrections = [] # Ensure empty on error
    else:
        logger.warning("No summary components available for reflection.")
    # --- END MODIFICATION ---

    # --- MODIFIED: Combine existing and new corrections and return ---
    final_corrections = existing_corrections + new_corrections
    # Optional: Deduplicate if needed
    # final_corrections = list(dict.fromkeys(final_corrections))

    return {
        "reflection_data": reflection_data,
        "identified_corrections": final_corrections
    }
    # --- END MODIFICATION ---


@traceable
async def finalize_summary(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Produce the final human-readable research summary by assembling pre-computed summaries
    and generating an executive summary using an LLM.
    """
    logger.info("--- Node: finalize_summary (LLM Enhanced Assembler) ---")
    cfg = Configuration.from_runnable_config(config)
    topic = state.get("research_topic", "Unknown Topic")

    # --- MODIFIED: Use refined summary context OR assemble CONCISE context ---
    refined_context = state.get("refined_summary_context")
    llm_context_base = ""
    if refined_context:
        logger.info("Using refined summary context for final report generation.")
        # Use the already condensed summary from the refinement step
        llm_context_base = refined_context
    else:
        logger.info("Assembling CONCISE context for final report (no refinement needed/failed).")
        # Assemble context manually, but potentially truncate individual summaries
        jira_summary = state.get("jira_summary", "")
        confluence_summary = state.get("confluence_summary", "")
        perforce_summary = state.get("perforce_summary", "")
        # web_summary = state.get("web_summary", "") # OMITTED
        cross_source_analysis = state.get("cross_source_analysis_text", "")
        
        # --- ADDED: Include binary content summaries ---
        binary_summaries = state.get("binary_content_summaries", {})
        binary_integrated_summary = binary_summaries.get("integrated", "")
        # --- END ADDED ---

        # --- ADDED: Truncate individual summaries for concise context ---
        MAX_LEN_PER_SOURCE = 4000  # Max characters per source summary in final prompt context

        def truncate_summary(text, max_len):
            if len(text) > max_len:
                return text[:max_len] + "... [TRUNCATED] ..."
            return text

        # --- MODIFIED: Handle potentially empty/default summaries ---
        jira_summary_text = truncate_summary(jira_summary,
                                             MAX_LEN_PER_SOURCE) if jira_summary and "_No relevant" not in jira_summary and "_Error summarizing" not in jira_summary else "_No relevant JIRA information found._"
        confluence_summary_text = truncate_summary(confluence_summary,
                                                   MAX_LEN_PER_SOURCE) if confluence_summary and "_No relevant" not in confluence_summary and "_Error summarizing" not in confluence_summary else "_No relevant Confluence information found._"
        perforce_summary_text = truncate_summary(perforce_summary,
                                                 MAX_LEN_PER_SOURCE) if perforce_summary and "_No relevant" not in perforce_summary and "_Error summarizing" not in perforce_summary else "_No relevant Perforce information found._"
        cross_source_text = truncate_summary(cross_source_analysis,
                                             MAX_LEN_PER_SOURCE) if cross_source_analysis and "_No data available" not in cross_source_analysis and "_Cross-source analysis was not performed" not in cross_source_analysis and "_Error during analysis" not in cross_source_analysis else "_No cross-source analysis was performed._"
        
        # --- ADDED: Include binary content summary if available ---
        binary_content_text = truncate_summary(binary_integrated_summary, 
                                              MAX_LEN_PER_SOURCE) if binary_integrated_summary and "_No relevant binary content found" not in binary_integrated_summary else "_No relevant binary content found._"
        # --- END ADDED ---

        combined_parts = [
            f"## JIRA Analysis\n{jira_summary_text}",
            f"## Confluence Analysis\n{confluence_summary_text}",
            f"## Perforce Analysis\n{perforce_summary_text}",
            # Web search omitted
            f"{cross_source_text}" if cross_source_text != "_No cross-source analysis was performed._" else "",
            # --- ADDED: Binary content section ---
            f"## Binary Content Analysis\n{binary_content_text}" if binary_content_text != "_No relevant binary content found._" else "",
            # --- END ADDED ---
        ]
        # --- END MODIFICATION ---
        llm_context_base = "\n\n---\n\n".join(filter(None, combined_parts))
        # --- END ADDED ---
    # --- END MODIFICATION ---

    # --- Extract and Format Code Snippets ---
    code_snippet_section = ""
    code_snippet_blocks: List[str] = []
    try:
        raw_p4_results = state.get("raw_perforce_results", []) or []
        snippets_extracted_count = 0
        # --- MODIFIED: Increase maximum snippets displayed ---
        max_snippets_to_show = 20  # Increased from 5 to 20 to show more code snippets in the report
        # --- END MODIFICATION ---

        # --- START FIX: Correct loop for snippets attached to CL dict ---
        for res in raw_p4_results:
            if snippets_extracted_count >= max_snippets_to_show:
                break
            # Check if snippets exist directly on the result dictionary
            if isinstance(res, dict) and "code_snippets" in res and res["code_snippets"]:
                cl_num = res.get("change", "Unknown CL")
                cl_url = res.get("url", "")
                # Iterate through the snippets list attached to this CL
                for snippet_data in res["code_snippets"]:
                    if snippets_extracted_count >= max_snippets_to_show:
                        break
                    if not isinstance(snippet_data, dict):
                        continue  # Skip non-dict items in list

                    file_path = snippet_data.get("file", "Unknown File")
                    snippet_text = snippet_data.get("snippet", "[No snippet text]")
                    language = snippet_data.get("language", "text") or "text"
                    action = snippet_data.get("action", "edit")

                    file_link_text = f" (CL [{cl_num}]({cl_url}) - Action: {action})" if cl_url else f" (CL {cl_num} - Action: {action})"
                    header = f"**{file_path}**{file_link_text}"

                    max_snippet_len = 1000  # Increased length slightly
                    if len(snippet_text) > max_snippet_len:
                        snippet_text = snippet_text[:max_snippet_len] + "... [TRUNCATED] ..."

                    # Force GC for very large snippets to prevent memory issues during report generation
                    if len(snippet_text) > 5000:  # If original snippet was large
                        gc.collect()

                    code_block = f"{header}\n```{language}\n{snippet_text}\n```"
                    code_snippet_blocks.append(code_block)
                    snippets_extracted_count += 1
        # --- END FIX ---

        if code_snippet_blocks:
            logger.info(f"Extracted {len(code_snippet_blocks)} code snippets for the final report.")
            code_snippet_section = "\n## Representative Code Snippets\n" + "\n\n".join(code_snippet_blocks)
        else:
            logger.info("No code snippets found or extracted for the final report.")
            code_snippet_section = "\n## Representative Code Snippets\n_No relevant code snippets were found or extracted._"

    except Exception as snippet_err:
        logger.error(f"Failed to extract code snippets for summary: {snippet_err}", exc_info=True)
        code_snippet_section = "\n## Representative Code Snippets\n_Error extracting code snippets._"
        
    # --- ADDED: Extract and Format Binary Content Thumbnails ---
    binary_content_section = ""
    try:
        processed_content = state.get("processed_binary_content", {})
        
        # Get image thumbnails
        image_thumbs = []
        images = processed_content.get("images", [])
        diagrams = processed_content.get("diagrams", [])
        
        # Combine images and diagrams for thumbnail display
        all_visual_content = images + diagrams
        
        # Limit the number of thumbnails
        max_thumbs = 5
        thumb_count = 0
        
        for visual in all_visual_content:
            if thumb_count >= max_thumbs:
                break
                
            thumbnail = visual.get("thumbnail")
            if thumbnail:
                filename = visual.get("filename", f"Image {thumb_count+1}")
                is_diagram = visual.get("is_diagram", False)
                content_type = "Diagram" if is_diagram else "Image"
                
                # Create thumbnail entry
                thumb_entry = f"**{content_type}: {filename}**\n\n![{filename}]({thumbnail})\n"
                image_thumbs.append(thumb_entry)
                thumb_count += 1
        
        # Create binary content section if thumbnails exist
        if image_thumbs:
            binary_content_section = "\n## Binary Content Previews\n" + "\n\n".join(image_thumbs)
        
    except Exception as binary_err:
        logger.error(f"Failed to extract binary content previews: {binary_err}", exc_info=True)
        binary_content_section = ""
    # --- END ADDED ---

    # --- Assemble Final Context for LLM (Base + Snippets + Reflection + Binary Content) ---
    final_context_parts_for_llm = [
        f"# Research Topic: {topic}\n",
        llm_context_base,
        code_snippet_section,
        binary_content_section,  # Added binary content section
    ]
    reflection = state.get("reflection_data", "")
    if isinstance(reflection, str) and reflection and "skipped" not in reflection.lower():
        final_context_parts_for_llm.extend(["\n## Reflection", reflection, "\n"])

    llm_context = "\n".join(filter(None, final_context_parts_for_llm))
    # Apply final overall limit if needed (though individual truncation helps)
    max_final_context_len = MAX_SUMMARY_CONTEXT
    if len(llm_context) > max_final_context_len:
        logger.warning(f"Final LLM context truncated from {len(llm_context)} to {max_final_context_len} chars.")
        llm_context = llm_context[:max_final_context_len] + "... [CONTEXT TRUNCATED FOR FINAL LLM] ..."
        # Force garbage collection after truncating large content
        gc.collect()

    # --- ADDED: Extract original quiz questions for the prompt ---
    original_quiz_questions = state.get("quiz_questions", []) # Get from state if QA node ran
    if not original_quiz_questions and "quiz" in topic.lower():
        # Fallback: Re-parse from topic if not in state (e.g., if QA node was skipped)
        logger.info("Re-parsing original quiz questions from topic for final report.")
        # Simplified parsing logic based on answer_quiz_questions
        parsed_questions = []
        quiz_marker = "Here is the quiz:"
        quiz_content = topic.split(quiz_marker, 1)[-1] if quiz_marker in topic else topic
        # Combine patterns for better matching
        patterns = [
            re.compile(r'\d+\.\s+([^?]+\?)\s*(\n|$)'), # Simple question ending with ?
            re.compile(r'\d+\.\s+(.*?)(?=\n\s*\d+\.|\n\s*$)') # Question until next number or end
        ]
        # Use finditer to get all matches robustly
        extracted_texts = set() # Use a set to avoid duplicates from multiple patterns
        for pattern in patterns:
            for match in pattern.finditer(quiz_content):
                 question_text = match.group(1).strip()
                 # Basic cleanup of options if present
                 question_text = re.sub(r'\n\s*[A-D][).].*', '', question_text, flags=re.MULTILINE).strip()
                 if question_text and len(question_text) > 5: # Basic filter for meaningful questions
                     extracted_texts.add(question_text)
        # Sort for consistent ordering if needed
        original_quiz_questions = sorted(list(extracted_texts))
        logger.info(f"Parsed {len(original_quiz_questions)} questions from topic for prompt.")

    # Format Quiz Q&A for prompt context
    formatted_quiz_qa = "_No quiz answers available or questions could not be parsed._"
    quiz_answers_data = state.get("quiz_answers", [])
    # Ensure we have both questions and answers before formatting
    if original_quiz_questions and quiz_answers_data:
        qa_parts = []
        num_questions = len(original_quiz_questions)
        num_answers = len(quiz_answers_data)

        # Handle potential mismatch in numbers
        max_items = min(num_questions, num_answers)
        if num_questions != num_answers:
            logger.warning(f"Mismatch between parsed questions ({num_questions}) and generated answers ({num_answers}). Formatting based on {max_items} pairs.")

        for i in range(max_items):
            q = original_quiz_questions[i]
            ans_data = quiz_answers_data[i]
            a = ans_data.get('answer', 'N/A')
            j = ans_data.get('justification', '_No justification provided_')
            # Ensure justification cites evidence if available
            if 'evidence:' not in j.lower() and 'not found' not in j.lower() and '[' in j and ']' in j: # Basic check for links
                j += " (Evidence should be explicitly cited here based on retrieved context)" # Add reminder if missing
            qa_parts.append(f"**Question {i+1}:** {q}\n**Answer:** {a}\n**Justification:** {j}")

        if qa_parts:
            formatted_quiz_qa = "\\n\\n---\\n\\n".join(qa_parts)
        # Add notes about any mismatches
        if num_questions > max_items:
            formatted_quiz_qa += "\\n\\n_Note: More questions were parsed than answers generated._"
        elif num_answers > max_items:
             formatted_quiz_qa += "\\n\\n_Note: More answers were generated than questions parsed._"

    elif original_quiz_questions: # Only questions parsed
        formatted_quiz_qa = "**Original Questions Found:**\\n- " + "\\n- ".join(original_quiz_questions) + "\\n\\n_Answers could not be generated or retrieved._"
    # --- END ADDED ---


    # --- Final Report Generation Prompt ---
    # --- ADDED: Get identified corrections from state ---
    corrections_list = state.get("identified_corrections", [])
    corrections_prompt_part = ""
    if corrections_list:
        corrections_prompt_part = (
                "\\n\\n**IMPORTANT CORRECTIONS/CONTRADICTIONS IDENTIFIED:** During analysis, the following potential issues were noted. "
                "Review these points and ensure the final report corrects or clarifies these issues where possible, referencing the more reliable information:\\n"
                "- " + "\\n- ".join(corrections_list) + "\\n"
        )
    # --- END ADDED ---
    # --- MODIFIED: Enhance Final Prompt Structure & Handling of Missing Sections ---
    final_prompt = (
            f"Objective: Synthesize provided analysis sections into a final, comprehensive, multi-page research report on: '{topic}'. Aim for detailed, insightful content.\\n\\n"
            "Input: Pre-summarized sections for JIRA, Confluence, Perforce, Cross-Source Analysis, Binary Content Analysis, Code Snippets, Reflection, and Formatted Quiz Q&A.\\n\\n"
            "Instructions:\\n"
            "1. **Structure:** Adhere strictly to this report structure. **Include ALL numbered sections**, even if the corresponding input data is missing or empty. State clearly within the section if no information was found or generated.\\n"
            f"   - `# Research Report: {topic}`\\n"
            "   - `## 1. Executive Summary`: Concise (1-2 paras). **Must** synthesize critical findings *and note gaps* from the detailed sections below.\\n"
            # --- MODIFIED: Updated Instruction for Quiz Section ---
            "   - `## 2. Quiz Question Analysis`: **Accurately reproduce** the questions, answers, and justifications exactly as provided in the 'Formatted Quiz Q&A' input section below. Use the provided **Question #, Answer:, Justification:** format for each item. If the input indicates no Q&A is available, state that clearly.\\n"
            # --- END MODIFIED ---
            "   - `## 3. Detailed Findings`: Combine the provided JIRA, Confluence, and Perforce summaries into a flowing narrative. **If a summary is empty or indicates no information was found, state that clearly under the relevant sub-heading.** Rewrite/synthesize where appropriate.\\n"
            "     - `### JIRA Analysis`: Based *only* on the provided 'JIRA Analysis' input. Include key personnel mentioned.\\n"
            "     - `### Confluence Analysis`: Based *only* on the provided 'Confluence Analysis' input.\\n"
            "     - `### Perforce Analysis`: Based *only* on the provided 'Perforce Analysis' input.\\n"
            # --- ADDED: Binary Content Analysis Section ---
            "     - `### Binary Content Analysis`: Based *only* on the provided 'Binary Content Analysis' input. Describe images, diagrams, spreadsheet data, and other non-text content that was analyzed.\\n"
            # --- END ADDED ---
            "   - `## 4. Cross-Source Analysis & Justification`: Include the provided 'Cross-Source Analysis' input section *verbatim*. State if no analysis was performed.\\n"
            "   - `## 5. Representative Code Snippets`: Include the provided 'Code Snippets' section. State clearly if no snippets were found/extracted.\\n"
            "   - `## 6. Reflection`: Include the provided 'Reflection' section *verbatim*. State clearly if reflection was skipped or failed.\\n"
            "   - `## 7. Sources/Citations`: After composing the report, gather *all* unique Markdown links you referenced inline (one per unique URL) and list them here.\\n"
            # --- MODIFIED: Re-emphasized Inline Citation Instruction ---
            "\\n2. **Inline Citations & Evidence (MANDATORY):** For the 'Detailed Findings' section (Jira, Confluence, Perforce), **EVERY** factual statement or finding MUST be *immediately* followed by an inline citation using the relevant Markdown link(s) from the input context (e.g., `This requires parameter X [VIT-12345]` or `The process involves Y [Page Title](URL)`). If multiple sources support a statement, cite all applicable links separated by commas. Do **NOT** make assumptions or state facts without direct evidence and its corresponding inline citation. Adherence is critical.\\n"
            # --- END MODIFIED ---
            "3. **Source Integration:** Prefer analysis that can be cross-validated by *multiple* source types (e.g., JIRA *and* Confluence). If only one source provides evidence, still cite it, but note the single-source nature succinctly.\\n"
            "\\n4. **Incorporate Corrections:** Review the 'IMPORTANT CORRECTIONS/CONTRADICTIONS IDENTIFIED' section below (if present) and ensure the final report reflects these corrections or addresses the identified ambiguities.\\n"
            "\\n5. **Length:** Aim for a comprehensive report.\\n"
            f"{corrections_prompt_part}"
            f"\\n\\n--- INPUT SECTIONS START HERE ---\\n\\n" # Added explicit marker
            # --- MODIFIED: Pass Formatted Q&A to prompt ---
            f"**Formatted Quiz Q&A:**\\n{formatted_quiz_qa}\\n\\n---\\n\\n" # Added section markers
            # --- END MODIFIED ---
            f"**Analysis Context (Jira, Confluence, Perforce, Cross-Source, Snippets, Reflection):**\\n{llm_context}\\n\\n--- INPUT SECTIONS END HERE ---\\n\\n" # Added explicit marker
            "Generate the final, synthesized research report below, adhering strictly to the structure and instructions:"
    )
    # --- END MODIFICATION of prompt ---

    # --- Invoke LLM for Final Report ---
    final_summary_content = "_Error generating final report._"
    try:
        # ... (LLM instantiation using central config) ...
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            timeout=480,
            **LLM_CONFIG_PARAMS
        )
        messages = [SystemMessage(content=final_prompt)]
        logger.info(f"Invoking LLM for final report synthesis (context length: {len(llm_context)} chars)")
        out = await llm.ainvoke(messages)
        final_summary_content = clean_llm_output(out.content)
        logger.info("Final report synthesis complete.")

        # --- MODIFICATION: Extract citations from LLM output ---
        final_citations = []
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        citations_dict: Dict[str, Dict] = {}
        # Search within the LLM-generated content for links
        for match in link_pattern.finditer(final_summary_content):
            text, raw_url = match.groups()
            url = raw_url.strip().strip("'\"")
            if not url or url == "#" or url.startswith("about:"):
                continue
            norm_url = url.rstrip('/')
            domain = urlparse(norm_url).netloc or "Unknown Source"
            if norm_url not in citations_dict:
                citations_dict[norm_url] = {"source_id": domain, "url": url, "text": text.strip(), "count": 1}
            else:
                citations_dict[norm_url]["count"] += 1
        final_citations = sorted(citations_dict.values(), key=lambda x: x["count"], reverse=True)
        logger.info(f"Extracted {len(final_citations)} unique citations from final report.")
        # --- END MODIFICATION ---

    except Exception as e:
        logger.error(f"Error during final report LLM call: {e}", exc_info=True)
        final_summary_content = f"# Research Report: {topic}\n\n_Error generating final report content: {e}_"
        final_citations = state.get("citations", [])

    final_summary_content = sanitize_markdown_links(final_summary_content)

    # --- MODIFIED: Add Appendix *after* LLM call ---
    appendix_parts = ["\n\n## Appendix: Pre-processed Data Points\n"]
    pre_processed_data = state.get("pre_processed_summaries", {})
    if pre_processed_data:
        for source_key, bullets in pre_processed_data.items():
            if bullets:
                appendix_parts.append(f"\n**{source_key.replace('### ', '')} Points:**\n")
                appendix_parts.extend(bullets)
    else:
        appendix_parts.append("_No pre-processed data was generated._")
    final_summary_with_appendix = final_summary_content + "\n".join(appendix_parts)
    # --- END MODIFICATION ---

    # Prepare Export Data (using summary *with* appendix)
    export_payload = {
        "research_topic": topic,
        "summary": final_summary_with_appendix,  # Use summary with appendix for export
        "timestamp": datetime.now().isoformat(),
        "citations": final_citations,
        "security_findings": state.get("security_findings", []),
        "technical_validation": state.get("technical_validation", {}),
        "perforce_diff_analysis": state.get("perforce_diff_analysis", {}),
        "verification_results": state.get("verification_results", {}),
        "knowledge_graph_data": state.get("knowledge_graph", {}).get("data"),
        "knowledge_graph_visualization": state.get("knowledge_graph", {}).get("visualization_path"),
        "pre_processed_data": state.get("pre_processed_summaries", {})
    }
    logger.info("Final summary synthesized and export data prepared.")

    return {
        "running_summary": final_summary_with_appendix,
        "exported_data": export_payload
    }


@traceable
async def export_research(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Export the research findings in the requested format."""
    logger.info("--- Node: export_research ---")
    export_format = state.get("export_format", "markdown").lower() # Default to markdown
    export_data = state.get("exported_data", {}) # Get data prepared by finalize_summary
    research_topic = export_data.get("research_topic", "Research")
    summary = export_data.get("summary", "No summary available.")

    if not export_data:
         logger.error("No data available for export (finalize_summary might have failed).")
         return {"exported_data": {"error": "No data to export"}} # Return error in state

    # Generate filename
    safe_topic = re.sub(r'[^\w\-]+', '_', research_topic)[:50] # Limit length
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{safe_topic}_{timestamp}"
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)

    export_info = {"format": export_format} # Store info about the export attempt

    try:
        if export_format == "markdown" or export_format == "md":
            md_path = output_dir / f"{filename_base}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(summary)
            export_info["export_path"] = str(md_path)
            logger.info(f"Exported research as markdown to {md_path}")

        elif export_format == "json":
            json_path = output_dir / f"{filename_base}.json"
            # Dump the prepared export_data dictionary
            with open(json_path, "w", encoding="utf-8") as f:
                # Use default=str for safety with complex objects like datetime
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            export_info["export_path"] = str(json_path)
            logger.info(f"Exported research as JSON to {json_path}")

        elif export_format == "csv":
            if not HAS_VISUALIZATION: # Needs pandas
                 raise ImportError("CSV export requires pandas library.")
            # Export citations and findings to separate CSVs
            csv_paths = []
            if export_data.get("citations"):
                citations_df = pd.DataFrame(export_data["citations"])
                citations_csv = output_dir / f"{filename_base}_citations.csv"
                citations_df.to_csv(citations_csv, index=False)
                csv_paths.append(str(citations_csv))
            if export_data.get("security_findings"):
                 findings_df = pd.DataFrame(export_data["security_findings"])
                 findings_csv = output_dir / f"{filename_base}_security.csv"
                 findings_df.to_csv(findings_csv, index=False)
                 csv_paths.append(str(findings_csv))
            # Could add more CSVs (e.g., tech validation)
            if csv_paths:
                 export_info["export_paths"] = csv_paths
                 logger.info(f"Exported research data as CSV files in {output_dir}")
            else:
                 logger.info("No data suitable for CSV export found.")
                 export_info["message"] = "No data for CSV export."

        elif export_format == "html":
            try:
                import markdown
                # Basic HTML structure with CSS
                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{research_topic} - Research Summary</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; max-width: 900px; margin: 20px auto; padding: 15px; border: 1px solid #ddd; box-shadow: 2px 2px 5px #eee; }}
        h1, h2, h3 {{ color: #333; }} h1 {{ border-bottom: 2px solid #eee; }} h2 {{ border-bottom: 1px solid #eee; }}
        a {{ color: #007bff; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
        pre {{ background-color: #f8f9fa; padding: 10px; border: 1px solid #dee2e6; border-radius: 4px; overflow-x: auto; font-size: 0.9em; }}
        code {{ background-color: #e9ecef; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; }}
        blockquote {{ border-left: 4px solid #ccc; padding-left: 15px; color: #666; margin-left: 0; }}
        ul {{ padding-left: 20px; }} li {{ margin-bottom: 5px; }}
        .timestamp {{ color: #777; font-size: 0.8em; text-align: right; margin-top: 20px; }}
    </style>
</head>
<body>
    {markdown.markdown(summary, extensions=['tables', 'fenced_code'])}
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body>
</html>"""
                html_path = output_dir / f"{filename_base}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                export_info["export_path"] = str(html_path)
                logger.info(f"Exported research as HTML to {html_path}")
            except ImportError:
                markdown = None  # Define markdown in except block
                logger.error("HTML export requires markdown library.")

        elif export_format == "pdf":
             try:
                  from weasyprint import HTML
                  import markdown
                  # Generate HTML first (similar to above)
                  html_for_pdf = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{research_topic}</title>
<style>
    body {{ font-family: sans-serif; line-height: 1.5; }} h1, h2, h3 {{ color: #333; }}
    a {{ color: #007bff; text-decoration: none; }} pre {{ background-color: #f8f9fa; padding: 10px; border: 1px solid #dee2e6; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }}
    code {{ font-size: 0.9em; }} blockquote {{ border-left: 4px solid #ccc; padding-left: 15px; color: #666; margin-left: 0; }}
    @page {{ margin: 1.5cm; }}
</style></head><body>{markdown.markdown(summary, extensions=['tables', 'fenced_code'])}</body></html>"""
                  pdf_path = output_dir / f"{filename_base}.pdf"
                  HTML(string=html_for_pdf).write_pdf(pdf_path)
                  export_info["export_path"] = str(pdf_path)
                  logger.info(f"Exported research as PDF to {pdf_path}")
             except ImportError:
                  HTML = None  # Define HTML in except block
                  markdown = None  # Define markdown in except block
                  logger.error("PDF export requires weasyprint and markdown libraries.")

        else:
            logger.warning(f"Unsupported export format: {export_format}. Defaulting to markdown.")
            export_info["error"] = f"Unsupported format: {export_format}"
            # Optionally default to markdown export here
            md_path = output_dir / f"{filename_base}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(summary)
            export_info["export_path"] = str(md_path)
            export_info["actual_format"] = "markdown"

    except ImportError as imp_err:
         logger.error(f"Export failed due to missing library: {imp_err}")
         export_info["error"] = f"Export failed: Missing library {imp_err}"
    except Exception as e:
        logger.error(f"Error during export (format: {export_format}): {e}", exc_info=True)
        export_info["error"] = f"Export failed: {str(e)}"

    # Update state with export attempt info (even if failed)
    # We replace exported_data with just the status/path info
    return {"exported_data": export_info}


@traceable
def error_recovery(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Attempt to recover from errors recorded in the state."""
    logger.warning("--- Node: error_recovery ---")
    errors = state.get('_errors', [])
    if not errors:
        logger.info("No errors found in state, proceeding.")
        # Decide where to go if called erroneously - maybe finalize?
        return {"_recovery_actions": ["finalize_summary"]} # Default recovery

    # Log errors
    for error in errors:
        logger.error(f"Error encountered in stage '{error.get('stage', 'unknown')}': {error.get('message', 'No message')}")

    # Simple recovery: just log and proceed to finalize
    # More complex logic could be added here based on error type/stage
    logger.warning("Proceeding to finalize summary after encountering errors.")

    # Clear errors after logging/handling attempt
    state_updates = {
        "_errors": [], # Clear errors
        "_recovery_actions": ["finalize_summary"] # Set action to finalize
    }
    return state_updates

# --- Conditional Edge Logic ---

def check_content_quality(state: ResearchState) -> str:
    """Determine if the current research quality is sufficient or needs enhancement."""
    logger.debug("--- Edge Logic: check_content_quality ---")
    # Default route if checks pass or aren't applicable
    next_node = "proceed_to_verify" # Default to verification if quality seems ok

    try:
        # Get config safely
        cfg = Configuration.from_runnable_config(state.get('config', {}))
        include_web_search = cfg.configurable.get('include_web_search', False)

        # --- MODIFIED: Check individual summary fields ---
        jira_summary = state.get('jira_summary', "")
        confluence_summary = state.get('confluence_summary', "")
        perforce_summary = state.get('perforce_summary', "")
        web_summary = state.get('web_summary', "") # Check even if disabled, might exist from previous loop
        cross_source = state.get("cross_source_analysis_text", "")

        # Combine relevant summaries to check total length
        combined_summary_length = sum(len(s) for s in [jira_summary, confluence_summary, perforce_summary, web_summary, cross_source] if isinstance(s, str))
        # --- END MODIFICATION ---

        research_loop_count = state.get('research_loop_count', 0)
        has_web_results = bool(state.get("raw_web_search_results")) # Check raw results

        # Check combined length, only trigger web search on first loop if enabled and not done
        min_summary_length = 150
        # --- MODIFIED: Use combined_summary_length ---
        if combined_summary_length < min_summary_length and research_loop_count <= 1: # Allow one loop even if short
            if include_web_search and not has_web_results and not web_summary: # Also check if web summary already generated
                logger.info("Quality check: Combined summary short, initiating web search.")
                next_node = "needs_web_search"
            else:
                 # Summary short, but web search done or disabled. Move on.
                 logger.info(f"Quality check: Combined summary short ({combined_summary_length} chars), but proceeding (web search done/disabled or already attempted). Routing to reflection.")
                 next_node = "reflect_now" # Go to reflection to potentially identify gaps
        # --- END MODIFICATION ---

        # Add checks for specific analysis based on topic (optional)
        # elif not state.get('security_findings') and "security" in state.get('research_topic', '').lower():
        #     next_node = "needs_security_analysis"
        # elif not state.get('technical_validation') and "technical" in state.get('research_topic', '').lower():
        #     next_node = "needs_technical_validation"

    except Exception as e:
        logger.error(f"Error in check_content_quality: {e}", exc_info=True)
        # Default to proceeding on error
        next_node = "proceed_to_verify"

    logger.debug(f"Quality check result: routing to '{next_node}'")
    return next_node


def _verification_status(state: ResearchState) -> str:
    """Determine routing after summary verification."""
    logger.debug("--- Edge Logic: _verification_status ---")
    verification = state.get("verification_results", {})
    # Default to clean if verification didn't run or failed unexpectedly
    is_accurate = verification.get("accurate", True)
    issues = verification.get("issues", [])

    if not is_accurate or (isinstance(issues, list) and len(issues) > 0):
        logger.warning(f"Verification found issues (Accurate: {is_accurate}, Issues: {issues}). Routing to error recovery.")
        # Add verification issues to main error list?
        state.setdefault("_errors", []).append({
             'stage': 'verification',
             'message': f"Verification failed. Accurate={is_accurate}, Issues={issues}",
             'timestamp': datetime.now().isoformat()
        })
        return "has_errors" # Route to error handler to decide next step
    else:
        logger.debug("Verification clean. Proceeding.")
        return "clean" # Proceed normally

def try_determine_continuation(state: ResearchState) -> str:
    """Decide whether to start another research loop or finish."""
    logger.debug("--- Edge Logic: try_determine_continuation ---")
    # Check for errors first
    if state.get("_errors"):
        logger.warning("Errors detected, routing to error_recovery from continuation check.")
        return "has_errors"

    # Check if chunk processing is somehow still active (shouldn't be at this stage)
    if (should_process_next_chunk(state, "confluence") or
        should_process_next_chunk(state, "jira") or
        should_process_next_chunk(state, "perforce")):
        logger.error("Chunk processing detected during continuation check - potential loop issue. Forcing completion.")
        reset_all_chunking(state) # Reset chunk state
        return "complete" # Force completion

    # Use the helper function to decide based on iterations, etc.
    if should_continue_research(state):
        logger.info("Continuation check: Starting new research loop.")
        return "continue" # Route back to enterprise_research setup
    else:
        logger.info("Continuation check: Completing research.")
        return "complete" # Route to finalize_summary

def should_continue_research(state: ResearchState) -> bool:
    """Heuristic to determine if another research iteration is needed.
    MODIFIED: Reverted to original logic (max 1 forced loop for shortness, check reflection)
    """
    iterations = state.get("research_loop_count", 0)
    max_iterations = 3 # Keep max loops at 3 for safety

    # --- REVERTED: Remove forced loops ---
    # MIN_LOOPS = 2
    # if iterations < MIN_LOOPS:
    #     logger.info(f"Forcing continuation: Iteration {iterations+1}/{MIN_LOOPS} minimum loops.")
    #     return True
    # --- END REVERT ---

    if iterations >= max_iterations:
        logger.info(f"Reached maximum iteration count ({max_iterations}), completing research.")
        return False

    # Check combined length of actual summaries
    jira_summary = state.get('jira_summary', "")
    confluence_summary = state.get('confluence_summary', "")
    perforce_summary = state.get('perforce_summary', "")
    web_summary = state.get('web_summary', "")
    cross_source = state.get("cross_source_analysis_text", "")
    combined_summary_length = sum(len(s) for s in [jira_summary, confluence_summary, perforce_summary, web_summary, cross_source] if isinstance(s, str))

    # --- REVERTED: Original check (allow one extra loop if short) ---
    min_summary_for_stop = 500 # Original threshold
    if combined_summary_length < min_summary_for_stop and iterations < max_iterations - 1:
        logger.info(f"Combined summary length ({combined_summary_length}) is less than threshold ({min_summary_for_stop}), continuing research (iteration {iterations+1}/{max_iterations}).")
        return True
    # --- END REVERT ---

    # Check reflection for explicit need for more info
    reflection = state.get("reflection_data", "")
    # --- REVERTED: Original check ---
    if isinstance(reflection, str) and iterations < max_iterations - 1:
        needs_more_patterns = ["further research", "more information needed", "insufficient data", "gap", "unclear", "needs clarification"]
        if any(pattern in reflection.lower() for pattern in needs_more_patterns):
            logger.info(f"Reflection suggests more research is needed, continuing (iteration {iterations+1}/{max_iterations}).")
            return True
    # --- END REVERT ---

    logger.info(f"Iteration {iterations+1}: Thresholds met or max iterations reached, completing research.")
    return False

# --- NEW Individual Summarizer Nodes (Defined BEFORE build_research_graph) ---

async def _summarize_single_source(source_key: str, source_name: str, state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Helper function to summarize pre-processed bullet points for a single source."""
    logger.info(f"--- Node: Summarizing {source_name} ---")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    topic = state.get("research_topic", "Unknown Topic")
    pre_processed_bullets = state.get("pre_processed_summaries", {}).get(source_key, [])

    output_key = f"{source_name.lower()}_summary"
    # --- MODIFIED: Consistent default message ---
    summary_output = f"_No relevant {source_name} information was found or processed._" # Default output
    # --- END MODIFICATION ---

    if not pre_processed_bullets:
        logger.info(f"No pre-processed bullets found for {source_name}. Skipping summarization.")
        return {output_key: summary_output}

    # --- MODIFIED: Implement Iterative Summarization Logic ---
    MAX_CHARS_PER_SUMMARIZE_CALL = 10000  # Max characters per single LLM call (adjust as needed)
    # Calculate total length of bullet points combined
    total_bullet_chars = sum(len(b) + 1 for b in pre_processed_bullets) # +1 for newline

    try:
        if not hasattr(cfg, 'ollama_base_url') or not hasattr(cfg, 'local_llm'):
            raise ValueError("LLM configuration missing")

        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            timeout=180,
            **LLM_CONFIG_PARAMS
        )

        if total_bullet_chars <= MAX_CHARS_PER_SUMMARIZE_CALL:
            # --- Original Logic (Single LLM Call) ---
            logger.info(f"{source_name} bullet points fit within limit ({total_bullet_chars}/{MAX_CHARS_PER_SUMMARIZE_CALL}). Using single LLM call.")
            bullet_string = "\n".join(pre_processed_bullets)
            prompt = _get_summarization_prompt(source_key, source_name, topic, bullet_string, state.get("perforce_diff_analysis", {}))

            messages = [SystemMessage(content=prompt)]
            out = await llm.ainvoke(messages)
            cleaned_summary = clean_llm_output(out.content)

            if len(cleaned_summary) < 30 or "based on the following" in cleaned_summary.lower() or "no specific findings" in cleaned_summary.lower():
                logger.warning(f"LLM for {source_name} returned a short or generic summary, using default.")
            else:
                summary_output = cleaned_summary
                logger.info(f"{source_name} summarization complete (single call).")

        else:
            # --- Iterative Logic ---
            logger.info(f"{source_name} bullet points exceed limit ({total_bullet_chars}/{MAX_CHARS_PER_SUMMARIZE_CALL}). Using iterative LLM calls.")
            batches = []
            current_batch_chars = 0
            current_batch = []

            for bullet in pre_processed_bullets:
                bullet_len = len(bullet) + 1
                if current_batch_chars + bullet_len > MAX_CHARS_PER_SUMMARIZE_CALL and current_batch:
                    batches.append("\n".join(current_batch))
                    current_batch = [bullet]
                    current_batch_chars = bullet_len
                else:
                    current_batch.append(bullet)
                    current_batch_chars += bullet_len
            if current_batch: # Add the last batch
                batches.append("\n".join(current_batch))

            logger.info(f"Split {source_name} bullets into {len(batches)} batches for LLM.")
            running_summary = ""
            for i, batch_str in enumerate(batches):
                logger.info(f"Processing batch {i + 1}/{len(batches)} for {source_name}...")
                # --- Get the appropriate prompt, potentially modifying it for iterative calls ---
                prompt = _get_summarization_prompt(
                    source_key, source_name, topic, batch_str,
                    state.get("perforce_diff_analysis", {}),
                    is_iterative=True, previous_summary=running_summary, batch_num=i + 1, total_batches=len(batches)
                )

                messages = [SystemMessage(content=prompt)]
                out = await llm.ainvoke(messages)
                # --- Update running summary with the latest output ---
                # We expect the LLM to integrate the new batch into the previous summary
                running_summary = clean_llm_output(out.content)
                logger.debug(f"Batch {i + 1} summary (running): {running_summary[:100]}...")

            # Final summary is the result from the last iteration
            if not running_summary.strip() or "_Error summarizing" in running_summary:
                 logger.error(f"Iterative summarization for {source_name} failed or produced empty result.")
                 summary_output = f"_Error during iterative summarization for {source_name}._"
            else:
                summary_output = running_summary
                logger.info(f"{source_name} iterative summarization complete ({len(batches)} calls).")

    except Exception as e:
        logger.error(f"{source_name} summarization failed: {e}", exc_info=True)
        summary_output = f"_Error summarizing {source_name} information: {e}_"

    return {output_key: summary_output.strip()}

# --- ADDED: Helper function to generate prompts (handles iterative logic) ---
def _get_summarization_prompt(source_key: str, source_name: str, topic: str, bullet_string: str, p4_analysis: dict, is_iterative: bool = False, previous_summary: str = "", batch_num: int = 0, total_batches: int = 0) -> str:
    """Generates the correct LLM prompt for summarization, handling iterative refinement."""

    # Base instructions common to most sources
    base_instructions = (
        f"Objective: Generate a detailed, narrative analysis section covering '{source_name}' findings relevant to the research topic: '{topic}'."
        "\nInput: Bullet points below, containing key info, metadata, links, and potentially code snippets."
        "\nInstructions:"
        "\n1. Synthesize findings into narrative. Don't just list points."
        "\n2. Weave in metadata (authors, dates, statuses) for context."
        "\n3. Ensure source links (Jira keys, Confluence URLs, Perforce CLs/Swarm) are included as Markdown links `[Text](URL)`."
        # Modified instruction 4 placeholder below for Perforce
        "\n4. If code snippets are included in the bullets, incorporate their relevance into the analysis."
        "\n5. Use clear Markdown (paragraphs, bullets). Aim for substantial contribution."
        f"\n6. Base output *only* on provided bullets relevant to '{topic}'. Do not infer information not present."
    )

    # Default prompt uses base instructions
    prompt = base_instructions + f"\n\nBullet Points from {source_name}:\\n{bullet_string}\\n\\nGenerate the comprehensive '{source_name}' summary section below:"

    # Specific prompt modifications/overrides per source
    if source_key == "### JIRA":
        jira_personnel_instruction = "\n7. **Mention Key Personnel:** Briefly mention the reporter or assignee if they appear associated with significant issues relevant to the topic."
        prompt = base_instructions + jira_personnel_instruction + f"\n\nBullet Points from {source_name}:\\n{bullet_string}\\n\\nGenerate the comprehensive '{source_name}' summary section below:"

    elif source_key == "### Perforce":
        diff_summary = p4_analysis.get("summary", "") if isinstance(p4_analysis, dict) else ""
        has_diff = diff_summary and "No Perforce diff" not in diff_summary and not diff_summary.startswith("Error")

        # Refined instruction 4 for Perforce snippets
        snippet_instruction = "\n4. Bullet points may include code snippets associated with a Change List (CL). Analyze these snippets and incorporate their relevance into the overall narrative for that CL."

        perforce_instruction = "\n7. Incorporate this overall Perforce code analysis summary where relevant: " + diff_summary if has_diff else ""

        # Combine base (with replaced instruction 4), snippet instruction, and overall analysis summary instruction
        prompt = base_instructions.replace(
            "\n4. If code snippets are included in the bullets, incorporate their relevance into the analysis.",
            snippet_instruction
        ) + perforce_instruction + f"\n\nBullet Points from {source_name}:\\n{bullet_string}\\n\\nGenerate the comprehensive '{source_name}' summary section below:"

    elif source_key == "### Confluence":
        # Use the user-provided detailed prompt for Confluence
        confluence_prompt_refined = (
            f"Objective: Generate a DETAILED '{source_name}' analysis section for a research report on '{topic}'." # Use source_name and topic
            f"Input: Multiple bullet points below. Each starting `- **[Page Title](URL)** (Content Snippet N/M): ...` represents a CHUNK of content from the linked Confluence page. Some snippets contain prose, others contain table-like data describing system interactions.\n"
            "Instructions:\n"
            "1. **Synthesize Deeply:** DO NOT just list pages/excerpts. Write a comprehensive narrative integrating information from ALL relevant snippets. Explain *what* the documentation reveals about '{topic}' and the associated APIX upgrade.\n" # Use topic
            "2. **Extract ALL Specifics:** Meticulously extract and incorporate specific details. **List or reference ALL mentioned API/Interface IDs (e.g., I1950_1, I1881_2, I1902_1, etc.)**, component names (OSB, APIX, NEXT ESB, SM-DP+, Amdocs Ordering, etc.), API types (REST, SOAP), statuses ('Not Started'), and configuration requirements (IPs, hostnames, SSL, LLD phase). Summarize key interactions described in table-like snippets explicitly mentioning the interacting components and the service.\n"
            "3. **Structure Logically:** Organize findings by sub-topic (e.g., System Interactions, Configuration, Specific API Details, Operational Status) or by summarizing each major document/page, ensuring smooth transitions.\n"
            "4. **Incorporate Metadata:** Include author/date if available. Mention the context provided by identifiers like 'R1.0 PI28'. Explain the relevance of statuses like 'Not Started'.\n"
            "5. **Use Links:** Ensure page titles are Markdown links `[Title](URL)`.\n"
            "6. **Length & Detail:** Aim for a thorough, multi-paragraph section accurately reflecting the technical depth of the provided snippets.\n"
            f"7. **Strict Relevance:** Base output *only* on provided snippets clearly related to '{topic}'. Do not infer information not present.\n" # Use topic
            f"\nInput Bullet Points/Snippets from {source_name}:\n{bullet_string}\n\n" # Use source_name and bullet_string
            f"Generate the comprehensive, technically detailed '{source_name}' analysis section below:" # Use source_name
        )
        prompt = confluence_prompt_refined # Override the default prompt

    # --- Modify prompt for iterative calls ---
    if is_iterative:
        if batch_num == 1:
            # First batch: Instruct LLM to create initial summary, aware more is coming
            iterative_instruction = f"\n\n**IMPORTANT:** This is the first batch (1/{total_batches}) of bullet points for {source_name}. Generate a summary based on this batch, but be prepared to integrate information from subsequent batches."
            prompt += iterative_instruction
        else:
            # Subsequent batches: Instruct LLM to integrate new info with previous summary
            iterative_instruction = (
                f"\n\n**IMPORTANT:** This is batch {batch_num}/{total_batches} for {source_name}. You MUST integrate the information from the 'New Bullet Points' below with the 'Previous Summary'. Update and refine the summary to incorporate the new details cohesively. Output ONLY the *complete, updated summary* incorporating all information processed so far."
                f"\n\n**Previous Summary (from batches 1 to {batch_num - 1}):**\\n{previous_summary}\\n\\n**New Bullet Points (batch {batch_num}):**\\n{bullet_string}\\n\\nGenerate the *complete, updated* {source_name} summary section below:"
            )
            # The core prompt (instructions + objective) is replaced entirely for iterative steps after the first
            base_objective = f"Objective: Refine and extend the existing research summary for '{source_name}' (related to '{topic}') using new information."
            prompt = base_objective + iterative_instruction

    return prompt
# --- END ADDED HELPER ---

@traceable
async def summarize_jira_results(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Summarizes the pre-processed Jira bullet points."""
    return await _summarize_single_source("### JIRA", "Jira", state, config)

@traceable
async def summarize_confluence_results(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Summarizes the pre-processed Confluence bullet points."""
    return await _summarize_single_source("### Confluence", "Confluence", state, config)

@traceable
async def summarize_perforce_results(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Summarizes the pre-processed Perforce bullet points and incorporates diff analysis."""
    return await _summarize_single_source("### Perforce", "Perforce", state, config)

@traceable
async def analyze_cross_source_connections(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Analyze connections between data from different sources (Jira, Confluence, Perforce).
    Creates explicit data structures for connections between sources and generates
    a comprehensive analysis of relationships between requirements, documentation, and code.

    Enhanced to fully process all connection data first, then apply memory optimization to stored content.
    """
    logger.info("--- Node: analyze_cross_source_connections (Enhanced) ---")

    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    topic = state.get("research_topic", "Unknown Topic")

    try:
        # --- 1. First pass: Extract all identifiers and connection information without storing full content ---

        # Initialize connection maps
        jira_to_perforce = defaultdict(list)  # Jira key -> [CL numbers]
        perforce_to_jira = defaultdict(list)  # CL number -> [Jira keys]
        jira_to_confluence = defaultdict(list)  # Jira key -> [Confluence IDs]
        confluence_to_jira = defaultdict(list)  # Confluence ID -> [Jira keys]
        perforce_to_confluence = defaultdict(list)  # CL number -> [Confluence IDs]
        confluence_to_perforce = defaultdict(list)  # Confluence ID -> [CL numbers]

        # Track all unique IDs
        jira_keys = set()
        perforce_changes = set()
        confluence_ids = set()

        # Helper function to extract identifiers from text
        def extract_identifiers(text):
            if not isinstance(text, str):
                return [], []

            # Extract Jira keys (e.g., VIT-1234)
            jira_pattern = re.compile(r'\b([A-Z]+-\d+)\b')
            jira_matches = jira_pattern.findall(text)

            # Extract MTV IDs (e.g., MTV1234, MTV-1234)
            mtv_pattern = re.compile(r'\b(MTV[-\s]?\d{4,})\b', re.IGNORECASE)
            mtv_matches = [m.upper().replace(' ', '').replace('-', '') for m in mtv_pattern.findall(text)]

            return jira_matches, mtv_matches

        # --- First pass: Extract Jira keys and collect minimal metadata ---
        logger.info("First pass: Extracting Jira keys and minimal metadata")
        jira_minimal_data = {}  # key -> {title, status} (minimal data)

        for item in state.get("raw_jira_results", []):
            if not isinstance(item, dict):
                continue

            key = item.get("key")
            if not key:
                continue

            jira_keys.add(key)
            jira_minimal_data[key] = {
                "key": key,
                "title": item.get("summary") or item.get("fields", {}).get("summary", ""),
                "status": item.get("status") or item.get("fields", {}).get("status", {}).get("name", ""),
            }

        # --- First pass: Extract Perforce CL numbers and collect minimal metadata ---
        logger.info("First pass: Extracting Perforce CL numbers and minimal metadata")
        perforce_minimal_data = {}  # change -> {description} (minimal data)

        for item in state.get("raw_perforce_results", []):
            if not isinstance(item, dict):
                continue

            change = str(item.get("change") or item.get("changelist") or item.get("change_list", ""))
            if not change:
                continue

            perforce_changes.add(change)
            description = item.get("description", "") or item.get("desc", "")
            if not isinstance(description, str):
                description = str(description) if description is not None else ""

            perforce_minimal_data[change] = {
                "change": change,
                "description": description,
            }

            # Extract Jira references from description immediately
            jira_refs, mtv_refs = extract_identifiers(description)

            # Add connections for Jira references
            for jira_key in jira_refs:
                if jira_key in jira_keys:
                    if jira_key not in perforce_to_jira[change]:
                        perforce_to_jira[change].append(jira_key)
                    if change not in jira_to_perforce[jira_key]:
                        jira_to_perforce[jira_key].append(change)

            # Check if any Jira issues contain the MTV references
            for mtv_ref in mtv_refs:
                for jira_key in jira_keys:
                    jira_title = jira_minimal_data.get(jira_key, {}).get("title", "")

                    # Look for MTV reference in Jira key or title
                    if mtv_ref in jira_key or (jira_title and mtv_ref in jira_title.upper()):
                        if jira_key not in perforce_to_jira[change]:
                            perforce_to_jira[change].append(jira_key)
                        if change not in jira_to_perforce[jira_key]:
                            jira_to_perforce[jira_key].append(change)

        # --- First pass: Extract Confluence IDs and collect minimal metadata ---
        logger.info("First pass: Extracting Confluence IDs and connection data")
        confluence_minimal_data = {}  # id -> {title, url} (minimal data)

        for item in state.get("raw_confluence_results", []):
            if not isinstance(item, dict):
                continue

            # Get ID or use URL hash as fallback ID
            conf_id = str(item.get("id") or hash(item.get("url", "") or ""))
            confluence_ids.add(conf_id)

            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content", "") or item.get("body", "") or item.get("text", "") or item.get("excerpt", "")

            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            confluence_minimal_data[conf_id] = {
                "id": conf_id,
                "title": title,
                "url": url,
            }

            # Check both title and content for references
            for text in [title, content]:
                jira_refs, mtv_refs = extract_identifiers(text)

                # Add connections for Jira references
                for jira_key in jira_refs:
                    if jira_key in jira_keys:
                        if jira_key not in confluence_to_jira[conf_id]:
                            confluence_to_jira[conf_id].append(jira_key)
                        if conf_id not in jira_to_confluence[jira_key]:
                            jira_to_confluence[jira_key].append(conf_id)

                # Check if any Jira issues contain the MTV references
                for mtv_ref in mtv_refs:
                    for jira_key in jira_keys:
                        jira_title = jira_minimal_data.get(jira_key, {}).get("title", "")

                        # Look for MTV reference in Jira key or title
                        if mtv_ref in jira_key or (jira_title and mtv_ref in jira_title.upper()):
                            if jira_key not in confluence_to_jira[conf_id]:
                                confluence_to_jira[conf_id].append(jira_key)
                            if conf_id not in jira_to_confluence[jira_key]:
                                jira_to_confluence[jira_key].append(conf_id)

            # Look for CL numbers directly mentioned in content
            cl_pattern = re.compile(r'\bCL\s+(\d+)\b')
            cl_matches = cl_pattern.findall(content)
            for cl in cl_matches:
                if cl in perforce_changes:
                    if cl not in confluence_to_perforce[conf_id]:
                        confluence_to_perforce[conf_id].append(cl)
                    if conf_id not in perforce_to_confluence[cl]:
                        perforce_to_confluence[cl].append(conf_id)

        logger.info(f"First pass completed: Found {len(jira_keys)} Jira issues, {len(perforce_changes)} Perforce changes, {len(confluence_ids)} Confluence pages")
        logger.info(f"Connections found: {sum(len(v) for v in jira_to_perforce.values())} Jira→Perforce, {sum(len(v) for v in jira_to_confluence.values())} Jira→Confluence")

        # --- 2. Second pass: Load full data with size limits, now that we have all connections ---

        # Initialize full data containers
        jira_data = {}
        perforce_data = {}
        confluence_data = {}

        # --- Second pass: Load Jira full data with size limits ---
        logger.info("Second pass: Loading Jira full data with size limits")
        for item in state.get("raw_jira_results", []):
            if not isinstance(item, dict):
                continue

            key = item.get("key")
            if not key or key not in jira_keys:
                continue

            # Get description with size limit
            description = item.get("description") or item.get("fields", {}).get("description", "")
            if isinstance(description, str) and len(description) > MAX_JIRA_DESC_SIZE:
                logger.info(f"Limiting Jira {key} description from {len(description)} to {MAX_JIRA_DESC_SIZE} chars")
                description = description[:MAX_JIRA_DESC_SIZE] + "... [TRUNCATED]"

            jira_data[key] = {
                "key": key,
                "title": item.get("summary") or item.get("fields", {}).get("summary", ""),
                "description": description,
                "status": item.get("status") or item.get("fields", {}).get("status", {}).get("name", ""),
                "assignee": (lambda a: a.get("displayName", "Unknown") if isinstance(a, dict) else "Unknown")(item.get("assignee") or item.get("fields", {}).get("assignee", {})),
                "reporter": (lambda r: r.get("displayName", "Unknown") if isinstance(r, dict) else "Unknown")(item.get("reporter") or item.get("fields", {}).get("reporter", {})),
                "created": item.get("created") or item.get("fields", {}).get("created", ""),
                "updated": item.get("updated") or item.get("fields", {}).get("updated", ""),
                "url": f"https://deljira/browse/{key}" if key else ""
            }

            # Add connection data (already processed)
            jira_data[key]["perforce_cls"] = jira_to_perforce.get(key, [])
            jira_data[key]["confluence_pages"] = jira_to_confluence.get(key, [])

        # --- Second pass: Load Perforce full data with size limits ---
        logger.info("Second pass: Loading Perforce full data with size limits")
        for item in state.get("raw_perforce_results", []):
            if not isinstance(item, dict):
                continue

            change = str(item.get("change") or item.get("changelist") or item.get("change_list", ""))
            if not change or change not in perforce_changes:
                continue

            perforce_data[change] = {
                "change": change,
                "description": perforce_minimal_data[change]["description"],
                "user": item.get("user", "Unknown"),
                "date": item.get("dateFormatted", "") or item.get("date", ""),
                "files": [],
                "code_snippets": []
            }

            # Extract files and code snippets with size limits
            if "code_snippets" in item and isinstance(item["code_snippets"], list):
                file_count = 0
                snippet_count = 0
                MAX_FILES_PER_CHANGE = 20
                MAX_SNIPPETS_PER_CHANGE = 5

                for snippet in item["code_snippets"]:
                    if not isinstance(snippet, dict):
                        continue

                    file_path = snippet.get("file", "")
                    if file_path and file_path not in perforce_data[change]["files"]:
                        if file_count < MAX_FILES_PER_CHANGE:
                            perforce_data[change]["files"].append(file_path)
                            file_count += 1

                    # Store snippet with size limit
                    if snippet_count >= MAX_SNIPPETS_PER_CHANGE:
                        continue

                    snippet_content = snippet.get("snippet", "")
                    if isinstance(snippet_content, str) and snippet_content:
                        # Limit snippet size
                        if len(snippet_content) > MAX_SNIPPET_SIZE:
                            logger.info(f"Limiting snippet for CL {change} from {len(snippet_content)} to {MAX_SNIPPET_SIZE} chars")
                            snippet_content = snippet_content[:MAX_SNIPPET_SIZE] + "... [TRUNCATED]"

                        perforce_data[change]["code_snippets"].append({
                            "file": file_path,
                            "content": snippet_content,
                            "action": snippet.get("action", "")
                        })
                        snippet_count += 1

            # Add connection data (already processed)
            perforce_data[change]["jira_issues"] = perforce_to_jira.get(change, [])
            perforce_data[change]["confluence_pages"] = perforce_to_confluence.get(change, [])

        # --- Second pass: Load Confluence full data with size limits ---
        logger.info("Second pass: Loading Confluence full data with size limits")
        for item in state.get("raw_confluence_results", []):
            if not isinstance(item, dict):
                continue

            conf_id = str(item.get("id") or hash(item.get("url", "") or ""))
            if conf_id not in confluence_ids:
                continue

            # Get content with size limit
            content = item.get("content", "") or item.get("body", "") or item.get("text", "") or item.get("excerpt", "")
            if isinstance(content, str) and len(content) > MAX_CONFLUENCE_CONTENT_SIZE:
                logger.info(f"Limiting Confluence content from {len(content)} to {MAX_CONFLUENCE_CONTENT_SIZE} chars")
                content = content[:MAX_CONFLUENCE_CONTENT_SIZE] + "... [TRUNCATED]"

            confluence_data[conf_id] = {
                "id": conf_id,
                "title": item.get("title", "Untitled"),
                "content": content,
                "url": item.get("url", ""),
                "author": item.get("author", "") or
                         item.get("history", {}).get("createdBy", {}).get("displayName", "Unknown"),
                "date": item.get("history", {}).get("createdDate", "")
            }

            # Add connection data (already processed)
            confluence_data[conf_id]["jira_issues"] = confluence_to_jira.get(conf_id, [])
            confluence_data[conf_id]["perforce_cls"] = confluence_to_perforce.get(conf_id, [])

        # Clean up temporary data and force garbage collection after second pass
        perforce_minimal_data.clear()
        jira_minimal_data.clear()
        confluence_minimal_data.clear()
        gc.collect()

        logger.info("Second pass completed: All data loaded with appropriate size limits")

        # --- 3. Identify implicit connections through transitive relationships ---
        logger.info("Identifying implicit connections between sources")

        # Track new connections
        new_connections = []

        # Find Confluence -> Perforce connections via Jira
        for conf_id, jira_keys_list in confluence_to_jira.items():
            for jira_key in jira_keys_list:
                # Get Perforce CLs connected to this Jira
                cl_list = jira_to_perforce.get(jira_key, [])

                for cl in cl_list:
                    # Create Confluence <-> Perforce connection if not already exists
                    if cl not in confluence_to_perforce[conf_id]:
                        new_connections.append(("conf_to_p4", conf_id, cl))
                    if conf_id not in perforce_to_confluence[cl]:
                        new_connections.append(("p4_to_conf", cl, conf_id))

        # Apply new connections
        for conn_type, from_id, to_id in new_connections:
            if conn_type == "conf_to_p4":
                confluence_to_perforce[from_id].append(to_id)
                if "perforce_cls" in confluence_data.get(from_id, {}):
                    if to_id not in confluence_data[from_id]["perforce_cls"]:
                        confluence_data[from_id]["perforce_cls"].append(to_id)
            elif conn_type == "p4_to_conf":
                perforce_to_confluence[from_id].append(to_id)
                if "confluence_pages" in perforce_data.get(from_id, {}):
                    if to_id not in perforce_data[from_id]["confluence_pages"]:
                        perforce_data[from_id]["confluence_pages"].append(to_id)

        logger.info(f"Identified {len(new_connections)} implicit connections")

        # --- 4. Format connections for readable output ---
        logger.info("Generating cross-source analysis text")
        connection_text = []
        connection_text.append("# Cross-Source Connection Analysis\n")

        # Identify connected Jira tickets for detailed reporting
        connected_jira_keys = sorted([k for k in jira_keys
                                     if (k in jira_to_perforce and jira_to_perforce[k])
                                     or (k in jira_to_confluence and jira_to_confluence[k])])

        # Add Jira connections summary
        connection_text.append("## Jira Ticket Connections\n")

        if connected_jira_keys:
            # Process in batches to manage memory
            BATCH_SIZE = 10
            for i in range(0, len(connected_jira_keys), BATCH_SIZE):
                batch = connected_jira_keys[i:i+BATCH_SIZE]

                for jira_key in batch:
                    jira_title = jira_data.get(jira_key, {}).get("title", "")
                    jira_status = jira_data.get(jira_key, {}).get("status", "")

                    connection_text.append(f"### {jira_key}: {jira_title} ({jira_status})\n")

                    # Perforce connections
                    if jira_key in jira_to_perforce and jira_to_perforce[jira_key]:
                        connection_text.append("**Perforce Changes:**\n")
                        for cl in jira_to_perforce[jira_key]:
                            cl_data = perforce_data.get(cl, {})
                            cl_desc = cl_data.get("description", "")
                            cl_desc_short = cl_desc[:100] + "..." if len(cl_desc) > 100 else cl_desc
                            connection_text.append(f"- CL {cl}: {cl_desc_short}\n")

                    # Confluence connections
                    if jira_key in jira_to_confluence and jira_to_confluence[jira_key]:
                        connection_text.append("**Confluence Pages:**\n")
                        for conf_id in jira_to_confluence[jira_key]:
                            conf_title = confluence_data.get(conf_id, {}).get("title", "")
                            conf_url = confluence_data.get(conf_id, {}).get("url", "")
                            connection_text.append(f"- [{conf_title}]({conf_url})\n")

                    connection_text.append("\n")

                # Apply regular garbage collection during batch processing
                if i + BATCH_SIZE < len(connected_jira_keys):
                    gc.collect()
        else:
            connection_text.append("*No connected Jira tickets found.*\n\n")

        # Add summary for unconnected entities
        orphan_jira = [k for k in jira_keys
                      if (k not in jira_to_perforce or not jira_to_perforce[k])
                      and (k not in jira_to_confluence or not jira_to_confluence[k])]
        if orphan_jira:
            connection_text.append("## Jira Tickets Without Connections\n")
            for jira_key in sorted(orphan_jira)[:10]:  # Limit to 10
                jira_title = jira_data.get(jira_key, {}).get("title", "")
                jira_status = jira_data.get(jira_key, {}).get("status", "")
                connection_text.append(f"- {jira_key}: {jira_title} ({jira_status})\n")

            if len(orphan_jira) > 10:
                connection_text.append(f"- ... and {len(orphan_jira)-10} more unconnected Jira tickets\n")

            connection_text.append("\n")

        # Add Perforce changes without Jira tickets
        orphan_cls = [cl for cl in perforce_changes if cl not in perforce_to_jira or not perforce_to_jira[cl]]
        if orphan_cls:
            connection_text.append("## Perforce Changes Without Jira References\n")
            for cl in sorted(orphan_cls)[:10]:  # Limit to 10
                cl_desc = perforce_data.get(cl, {}).get("description", "")
                cl_desc_short = cl_desc[:100] + "..." if len(cl_desc) > 100 else cl_desc
                connection_text.append(f"- CL {cl}: {cl_desc_short}\n")

            if len(orphan_cls) > 10:
                connection_text.append(f"- ... and {len(orphan_cls)-10} more changes without Jira references\n")

            connection_text.append("\n")

        # Add orphaned Confluence pages
        orphan_conf = [conf_id for conf_id in confluence_ids
                      if (conf_id not in confluence_to_jira or not confluence_to_jira[conf_id])
                      and (conf_id not in confluence_to_perforce or not confluence_to_perforce[conf_id])]
        if orphan_conf:
            connection_text.append("## Confluence Pages Without Connections\n")
            for conf_id in sorted(orphan_conf)[:10]:  # Limit to 10
                conf_title = confluence_data.get(conf_id, {}).get("title", "")
                conf_url = confluence_data.get(conf_id, {}).get("url", "")
                connection_text.append(f"- [{conf_title}]({conf_url})\n")

            if len(orphan_conf) > 10:
                connection_text.append(f"- ... and {len(orphan_conf)-10} more pages without connections\n")

            connection_text.append("\n")

        # Add summary metrics
        connection_text.append("## Connection Statistics\n")
        connection_text.append(f"- Total Jira tickets: {len(jira_keys)}\n")
        connection_text.append(f"- Total Perforce changes: {len(perforce_changes)}\n")
        connection_text.append(f"- Total Confluence pages: {len(confluence_ids)}\n")
        connection_text.append(f"- Jira tickets with code changes: {sum(1 for k in jira_keys if k in jira_to_perforce and jira_to_perforce[k])}\n")
        connection_text.append(f"- Jira tickets with documentation: {sum(1 for k in jira_keys if k in jira_to_confluence and jira_to_confluence[k])}\n")
        connection_text.append(f"- Perforce changes with Jira references: {sum(1 for cl in perforce_changes if cl in perforce_to_jira and perforce_to_jira[cl])}\n")
        connection_text.append(f"- Confluence pages with Jira references: {sum(1 for cid in confluence_ids if cid in confluence_to_jira and confluence_to_jira[cid])}\n")

        # --- 5. Use LLM to add higher-level analysis ---

        # Join data for base report
        base_cross_source_analysis = "".join(connection_text)

        # Limit size for LLM input to prevent memory issues
        if len(base_cross_source_analysis) > MAX_LLM_INPUT_SIZE:
            logger.warning(f"Cross-source analysis text too large ({len(base_cross_source_analysis)} chars), truncating")
            # Keep first part and a portion of the end to maintain structure
            first_part = base_cross_source_analysis[:MAX_LLM_INPUT_SIZE//2]
            last_part = base_cross_source_analysis[-MAX_LLM_INPUT_SIZE//4:]
            base_cross_source_analysis = (first_part +
                                        "\n\n... [CONTENT TRUNCATED DUE TO SIZE] ...\n\n" +
                                        last_part)
            # Force garbage collection after truncating large content
            gc.collect()

        # Use the LLM to enhance and find deeper insights
        enhanced_analysis = base_cross_source_analysis
        corrections = []

        if hasattr(cfg, 'ollama_base_url') and hasattr(cfg, 'local_llm'):
            try:
                # Define LLM prompt for enhanced analysis
                llm_analysis_prompt = (
                    f"Analyze the following cross-source connections report for research on '{topic}'."
                    " Identify insights, patterns, and potential contradictions across the connected data sources (Jira, Perforce, Confluence)."
                    " Add a section at the end named '## Higher-Level Insights' with your analytical findings."
                    " Also add a section named '## Contradictions and Inconsistencies' if you identify any contradictory information."
                    " Focus on technical implementation details, requirement changes, and documentation accuracy."
                    f"\n\nREPORT:\n{base_cross_source_analysis}\n\n"
                    "Provide your enhanced analysis with the new sections added at the end:"
                )

                # Invoke LLM
                llm = ChatOllama(
                    base_url=cfg.ollama_base_url,
                    model=cfg.local_llm,
                    timeout=120,
                    **LLM_CONFIG_PARAMS
                )
                messages = [SystemMessage(content=llm_analysis_prompt)]
                logger.info("Invoking LLM for enhanced cross-source analysis")
                out = await llm.ainvoke(messages)
                enhanced_text = clean_llm_output(out.content)

                # Update analysis with LLM output
                enhanced_analysis = enhanced_text

                # Extract contradictions section for state update
                contradiction_marker = "## Contradictions and Inconsistencies"
                if contradiction_marker in enhanced_text:
                    try:
                        contradiction_section = enhanced_text.split(contradiction_marker, 1)[1]
                        next_section = re.search(r'\n## ', contradiction_section)
                        if next_section:
                            contradiction_section = contradiction_section[:next_section.start()]

                        # Extract individual contradiction points
                        contradiction_lines = [line.strip() for line in contradiction_section.split('\n')
                                             if line.strip() and line.strip().startswith('-')]

                        if contradiction_lines:
                            logger.info(f"Extracted {len(contradiction_lines)} contradictions from analysis")
                            corrections = contradiction_lines
                    except Exception as e_parse:
                        logger.error(f"Error parsing contradictions section: {e_parse}")

            except Exception as e_llm:
                logger.error(f"Error enhancing cross-source analysis with LLM: {e_llm}")
                # Fall back to base analysis

        # --- 6. Create connection metadata for state ---

        connection_metadata = {
            "jira_to_perforce": dict(jira_to_perforce),
            "perforce_to_jira": dict(perforce_to_jira),
            "jira_to_confluence": dict(jira_to_confluence),
            "confluence_to_jira": dict(confluence_to_jira),
            "perforce_to_confluence": dict(perforce_to_confluence),
            "confluence_to_perforce": dict(confluence_to_perforce),
            "jira_count": len(jira_keys),
            "perforce_count": len(perforce_changes),
            "confluence_count": len(confluence_ids),
            "connected_jira": len(connected_jira_keys),
            "orphan_jira": len(orphan_jira),
            "orphan_perforce": len(orphan_cls),
            "orphan_confluence": len(orphan_conf),
            "timestamp": datetime.now().isoformat()
        }

        # Cleanup to free memory
        try:
            logger.info("Cleaning up memory in cross-source analysis")
            # Clear all large data containers
            jira_data.clear()
            perforce_data.clear()
            confluence_data.clear()
            connection_text.clear()
            jira_to_perforce.clear()
            perforce_to_jira.clear()
            jira_to_confluence.clear()
            confluence_to_jira.clear()
            perforce_to_confluence.clear()
            confluence_to_perforce.clear()

            # Force garbage collection
            gc.collect()
        except Exception as cleanup_err:
            logger.warning(f"Non-critical error during memory cleanup: {cleanup_err}")

        # Return both the structured metadata and the formatted analysis text
        return {
            "cross_source_analysis_text": enhanced_analysis,
            "connection_metadata": connection_metadata,
            "identified_corrections": corrections
        }

    except Exception as e:
        logger.error(f"Error in analyze_cross_source_connections: {e}", exc_info=True)
        return {
            "cross_source_analysis_text": "_Cross-source analysis was not performed due to an error._",
            "connection_metadata": {},
            "identified_corrections": []
        }

@traceable
async def refine_combined_summaries(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Refine the combined summaries by analyzing contradictions and inconsistencies."""
    logger.info("--- Node: refine_combined_summaries ---")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    topic = state.get("research_topic", "Unknown Topic")

    # Extract the cross-source analysis text
    cross_source_analysis = state.get("cross_source_analysis_text", "")

    # Extract the individual summaries
    jira_summary = state.get("jira_summary", "")
    confluence_summary = state.get("confluence_summary", "")
    perforce_summary = state.get("perforce_summary", "")
    web_summary = state.get("web_summary", "")

    # Combine all summaries into a single text
    all_summaries = [
        f"## JIRA Analysis\n{jira_summary}",
        f"## Confluence Analysis\n{confluence_summary}",
        f"## Perforce Analysis\n{perforce_summary}",
        f"## Web Search Analysis\n{web_summary}",
        f"{cross_source_analysis}"
    ]

    # Join summaries into a single string
    combined_summaries = "\n\n---\n\n".join(all_summaries)

    # Use the LLM to analyze the combined summaries
    llm_prompt = (
        f"Analyze the following combined research summaries for contradictions and inconsistencies: '{topic}'.\n\n"
        "Instructions:\n"
        "1. Identify any contradictions or inconsistencies between the summaries.\n"
        "2. Explain why these contradictions or inconsistencies exist.\n"
        "3. Suggest ways to resolve or clarify these contradictions or inconsistencies.\n\n"
        "Combined Summaries:\n{combined_summaries}\n\n"
        "Provide your analysis below:"
    )

    try:
        # Ensure LLM config is available
        if hasattr(cfg, 'ollama_base_url') and hasattr(cfg, 'local_llm'):
            # Use the LLM to analyze the combined summaries
            llm = ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.local_llm,
                **LLM_CONFIG_PARAMS
            )
            messages = [SystemMessage(content=llm_prompt)]
            result = await llm.ainvoke(messages)
            refined_summary = clean_llm_output(result.content)

            # Return the analysis
            return {"refined_summary_context": refined_summary}
        else:
            logger.warning("LLM configuration not available for refine_combined_summaries")
            return {"refined_summary_context": "Refinement could not be performed due to missing LLM configuration."}

    except Exception as e:
        logger.error(f"Error in refine_combined_summaries: {e}", exc_info=True)
        return {"refined_summary_context": f"Error during refinement: {e}"}

# Function to check if the available tools are aligned with what the code expects
def check_tool_alignment() -> bool:
    """Check if the required tools are available and correctly configured."""
    try:
        # Check if essential tools are available
        tools_ok = True

        # Check JIRA connection
        if not JIRA_SERVER or not JIRA_API_TOKEN:
            logger.warning("JIRA tool not properly configured (missing server or token)")
            tools_ok = False

        # Check Confluence
        if not CONFLUENCE_DOMAIN:
            logger.warning("Confluence tool not properly configured (missing domain)")
            tools_ok = False

        # Check Perforce
        if not P4_PORT or not P4_USER:
            logger.warning("Perforce tool not properly configured (missing P4PORT or P4USER)")
            tools_ok = False

        # Check LLM
        if not OLLAMA_BASE_URL or not LLM_MODEL:
            logger.warning("LLM not properly configured (missing endpoint or model)")
            tools_ok = False

        return tools_ok
    except Exception as e:
        logger.error(f"Error checking tool alignment: {e}")
        return False

@traceable
async def classify_research_topic(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Advanced research topic classifier that uses a multi-faceted approach to determine:
    1. Research type (Deep Research, Quiz, Simple Question, Code Generation)
    2. Intent classification (Causality, Comparison, Timeline, Technical Details)
    3. Required data sources (JIRA, Confluence, Perforce, Web, etc.)

    This enhanced classifier enables targeted research strategies tailored to the
    specific information needs expressed in the query.
    """
    logger.info("--- Node: classify_research_topic ---")
    topic = state.get("research_topic", "")
    if not topic:
        return {"topic_classification": "Unknown", "research_intent": "Unknown", "required_sources": []}

    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()

    # Multi-dimensional classification
    classification = {
        "topic_classification": "Deep Research",  # Default
        "research_intent": "Informational",  # Default
        "required_sources": ["jira", "confluence", "perforce"]  # Default all sources
    }

    # Pattern-based classification (primary layer)
    if "quiz" in topic.lower() or "Here is the quiz:" in topic or re.search(r'\d+\.\s+.+\?', topic):
        classification["topic_classification"] = "Quiz"
        classification["research_intent"] = "Assessment"
    elif len(topic.split()) < 8 and "?" in topic:
        classification["topic_classification"] = "Simple Question"
        classification["research_intent"] = "Factual"
    elif any(code_term in topic.lower() for code_term in ["generate code", "create a function", "write a program"]):
        classification["topic_classification"] = "Code Generation"
        classification["research_intent"] = "Implementation"

    # Intent detection (secondary layer)
    if "compare" in topic.lower() or "versus" in topic.lower() or " vs " in topic.lower():
        classification["research_intent"] = "Comparison"
    elif any(time_term in topic.lower() for time_term in ["history", "timeline", "evolution", "progress"]):
        classification["research_intent"] = "Timeline"
    elif any(cause_term in topic.lower() for cause_term in ["why", "cause", "reason", "explain why"]):
        classification["research_intent"] = "Causality"
    elif any(tech_term in topic.lower() for tech_term in ["how to", "implementation", "architecture"]):
        classification["research_intent"] = "Technical"

    # Source requirement analysis (tertiary layer)
    required_sources = []

    # Check for project/MTV patterns requiring JIRA
    if re.search(r'(MTV\d{4,}|[A-Z]+-\d+)', topic, re.IGNORECASE):
        required_sources.append("jira")

    # Check for terms suggesting documentation needs
    if any(doc_term in topic.lower() for doc_term in ["document", "documentation", "guide", "manual", "spec"]):
        required_sources.append("confluence")

    # Check for code/implementation focus
    if any(code_term in topic.lower() for code_term in ["code", "implementation", "function", "class", "method"]):
        required_sources.append("perforce")

    # Check for external knowledge needs
    if any(web_term in topic.lower() for web_term in ["external", "market", "competitor", "industry", "standard"]):
        required_sources.append("web")

    # If specific sources were detected, use them; otherwise, use the default all-sources approach
    if required_sources:
        classification["required_sources"] = list(set(required_sources))  # Deduplicate

    # Try to use LLM for more nuanced classification if available
    try:
        if hasattr(cfg, 'ollama_base_url') and hasattr(cfg, 'local_llm'):
            prompt = (
                "Analyze the following research topic and classify it into these categories:\\n\\n"
                "1. Topic Type (exactly ONE of):\\n"
                "   - 'Quiz': Contains multiple questions to be answered\\n"
                "   - 'Deep Research': Requires comprehensive analysis across multiple sources\\n"
                "   - 'Simple Question': A straightforward question needing a concise answer\\n"
                "   - 'Code Generation': Primarily requesting code to be written\\n\\n"
                "2. Research Intent (exactly ONE of):\\n"
                "   - 'Factual': Seeks specific facts or data points\\n"
                "   - 'Comparison': Seeks to compare multiple items\\n"
                "   - 'Timeline': Seeks historical progression\\n"
                "   - 'Causality': Seeks reasons or explanations\\n"
                "   - 'Technical': Seeks implementation details\\n"
                "   - 'Assessment': Tests knowledge on a topic\\n"
                "   - 'Informational': General information gathering\\n\\n"
                "3. Required Sources (multiple possible):\\n"
                "   - 'jira': Issue tracking data\\n"
                "   - 'confluence': Documentation\\n"
                "   - 'perforce': Code repositories\\n"
                "   - 'web': External information\\n\\n"
                f"Topic: {topic}\\n\\n"
                "Reply in JSON format with these three keys only."
            )

            llm = ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.local_llm,
                format="json",
                **LLM_CONFIG_PARAMS
            )

            result = await asyncio.wait_for(
                llm.ainvoke([SystemMessage(content=prompt)]),
                timeout=30
            )

            try:
                llm_classification = json.loads(result.content)
                if isinstance(llm_classification, dict):
                    # Apply LLM classifications where available
                    for key in ["topic_classification", "research_intent", "required_sources"]:
                        if key in llm_classification and llm_classification[key]:
                            # Validate required_sources is a list
                            if key == "required_sources" and not isinstance(llm_classification[key], list):
                                logger.warning(f"LLM returned non-list for required_sources: {llm_classification[key]}. Ignoring.")
                                continue
                            classification[key] = llm_classification[key]

                    logger.info(f"Using LLM classification: {classification}")
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM classification JSON")

    except Exception as e:
        logger.error(f"Error using LLM for classification: {e}")

    logger.info(f"Final topic classification: {classification}")
    return classification

@traceable
async def answer_quiz_questions(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Extracts quiz questions from the research topic and attempts to answer them
    based on the retrieved research results. Returns structured Q&A pairs.
    """
    logger.info("--- Node: answer_quiz_questions ---")
    topic = state.get("research_topic", "")
    # Handle the case when config is None
    cfg = Configuration.from_runnable_config(config) if config else Configuration()

    # Step 1: Extract questions from the topic
    extracted_questions = []

    # Check if topic is a quiz
    if "quiz" in topic.lower() or "Here is the quiz:" in topic:
        logger.info("Detected quiz format in research topic, extracting questions")

        # Extract the quiz content
        quiz_marker = "Here is the quiz:"
        quiz_content = topic.split(quiz_marker, 1)[-1] if quiz_marker in topic else topic

        # Try different patterns to extract questions
        patterns = [
            re.compile(r'\d+\.\s+([^?]+\?)\s*(\n|$)'),  # Simple question ending with ?
            re.compile(r'\d+\.\s+(.*?)(?=\n\s*\d+\.|\n\s*$)')  # Question until next number or end
        ]

        # Use finditer to get all matches robustly
        extracted_texts = set()  # Use a set to avoid duplicates from multiple patterns
        for pattern in patterns:
            for match in pattern.finditer(quiz_content):
                question_text = match.group(1).strip()
                # Basic cleanup of options if present
                question_text = re.sub(r'\n\s*[A-D][).].*', '', question_text, flags=re.MULTILINE).strip()
                if question_text and len(question_text) > 5:  # Basic filter for meaningful questions
                    extracted_texts.add(question_text)

        # Convert to list and sort
        extracted_questions = sorted(list(extracted_texts))
        logger.info(f"Extracted {len(extracted_questions)} questions from topic")

    # If no questions were found, handle gracefully
    if not extracted_questions:
        logger.info("No quiz questions found in topic, proceeding with normal research flow")
        return {
            "quiz_questions": [],
            "quiz_answers": []
        }

    # Step 2: Use the information gathered to answer each question
    quiz_answers = []

    # Get all the research results
    source_materials = []

    # Include individual summaries as context if available
    jira_summary = state.get("jira_summary", "")
    if jira_summary:
        source_materials.append(f"## JIRA Summary\n{jira_summary}")

    confluence_summary = state.get("confluence_summary", "")
    if confluence_summary:
        source_materials.append(f"## Confluence Summary\n{confluence_summary}")

    perforce_summary = state.get("perforce_summary", "")
    if perforce_summary:
        source_materials.append(f"## Perforce Summary\n{perforce_summary}")

    # Add formatted research results if summaries aren't available
    if not source_materials:
        logger.info("No summaries available yet, using raw research results")
        research_results = state.get("enterprise_research_results", [])
        source_materials.extend(research_results)

    # Combine all source materials into a single context
    context = "\n\n---\n\n".join(source_materials)

    # Process questions in batches to avoid exceeding context limits
    MAX_QUESTIONS_PER_BATCH = 5
    MAX_CONTEXT_SIZE = 8000  # Limit context size

    if hasattr(cfg, 'ollama_base_url') and hasattr(cfg, 'local_llm'):
        for i in range(0, len(extracted_questions), MAX_QUESTIONS_PER_BATCH):
            batch_questions = extracted_questions[i:i+MAX_QUESTIONS_PER_BATCH]

            # Truncate context if needed
            truncated_context = context
            if len(truncated_context) > MAX_CONTEXT_SIZE:
                logger.info(f"Truncating context from {len(context)} to {MAX_CONTEXT_SIZE} chars")
                truncated_context = context[:MAX_CONTEXT_SIZE] + "... [TRUNCATED]"

            try:
                # Create a prompt for batch question answering
                combined_questions = "\n".join([f"{j+1}. {q}" for j, q in enumerate(batch_questions)])

                qa_prompt = (
                    f"Answer the following questions about: {topic}\n\n"
                    "Instructions:\n"
                    "1. For each question, provide both an answer and a justification based ONLY on the provided context.\n"
                    "2. If the context doesn't contain information to answer a question, say 'Unable to answer from provided context' and explain why.\n"
                    "3. Include specific evidence from the context in your justifications, using direct quotes or paraphrases with clear citations/references.\n"
                    "4. Format each answer as a JSON object with 'question', 'answer', and 'justification' fields.\n"
                    "5. CRITICAL: You MUST wrap all answers in a JSON array, even if there's only one question.\n"
                    "6. Your response MUST be properly formatted JSON that can be parsed directly, starting with '[' and ending with ']'.\n"
                    "7. DO NOT include explanations before or after the JSON array.\n\n"
                    f"Questions:\n{combined_questions}\n\n"
                    f"Context:\n{truncated_context}\n\n"
                    "Return your answers in this exact JSON format:\n"
                    """[
                        {"question": "Question text here", "answer": "Your answer here", "justification": "Your detailed justification with evidence here"}
                    ]"""
                )

                # Use modified parameters for more reliable JSON formatting
                params = LLM_CONFIG_PARAMS.copy()
                params["temperature"] = 0.1  # Lower temperature
                params["top_p"] = 0.5  # More focused token selection

                llm = ChatOllama(
                    base_url=cfg.ollama_base_url,
                    model=cfg.local_llm,
                    format="json",
                    **params
                )

                result = await asyncio.wait_for(
                    llm.ainvoke([SystemMessage(content=qa_prompt)]),
                    timeout=180  # Longer timeout for complex Q&A
                )

                # Parse the JSON response
                try:
                    batch_answers = json.loads(result.content)
                    if isinstance(batch_answers, list):
                        quiz_answers.extend(batch_answers)
                    else:
                        logger.error(f"LLM returned non-list JSON: {batch_answers}")
                        # Try to handle empty JSON object or single object response
                        if isinstance(batch_answers, dict) and batch_answers.get("question") and batch_answers.get("answer"):
                            # Handle case where LLM returned a single answer object instead of an array
                            logger.info("Found single answer object outside array, wrapping it")
                            quiz_answers.append(batch_answers)
                        else:
                            # Create fallback answers
                            for q in batch_questions:
                                quiz_answers.append({
                                    "question": q,
                                    "answer": "Error processing response",
                                    "justification": "The system encountered an error processing this question."
                                })
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {result.content}")
                    # Create fallback answers
                    for q in batch_questions:
                        quiz_answers.append({
                            "question": q,
                            "answer": "Error parsing response",
                            "justification": "The system encountered an error parsing the answer to this question."
                        })

            except Exception as e:
                logger.error(f"Error answering batch of questions: {e}")
                # Create fallback answers for this batch
                for q in batch_questions:
                    quiz_answers.append({
                        "question": q,
                        "answer": "Error",
                        "justification": f"The system encountered an error: {str(e)}"
                    })
    else:
        logger.warning("LLM not available for quiz answering")
        # Create placeholders for all questions
        for q in extracted_questions:
            quiz_answers.append({
                "question": q,
                "answer": "Could not process",
                "justification": "LLM not available to process this question."
            })

    logger.info(f"Processed {len(quiz_answers)}/{len(extracted_questions)} questions")

    return {
        "quiz_questions": extracted_questions,
        "quiz_answers": quiz_answers
    }
def categorize_file_type(file_ext: str, content_type: str) -> str:
    """Categorize file type based on extension and MIME type."""
    file_ext = file_ext.lower() if file_ext else ""
    content_type = content_type.lower() if content_type else ""

    # Image types
    if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'] or 'image/' in content_type:
        return "image"

    # Excel files
    if file_ext in ['.xlsx', '.xls', '.csv'] or 'spreadsheet' in content_type or 'excel' in content_type:
        return "excel"

    # Diagram files (could be image files with specific content)
    if file_ext in ['.vsdx', '.drawio', '.dia'] or 'diagram' in content_type:
        return "diagram"

    # PDF files
    if file_ext == '.pdf' or 'pdf' in content_type:
        return "pdf"

    # Other document types
    if file_ext in ['.doc', '.docx', '.ppt', '.pptx']:
        return "document"

    # Default
    return "other"


async def download_binary_attachments(attachments: List[Dict[str, Any]], max_size: int = 10 * 1024 * 1024,
                                      max_files: int = 20) -> None:
    """
    Download binary attachments from their URLs.

    Args:
        attachments: List of attachment dictionaries
        max_size: Maximum file size in bytes
        max_files: Maximum number of files to download
    """
    import aiohttp

    # Apply limits
    filtered_attachments = []
    for attachment in attachments:
        size = attachment.get("size", 0)
        if size > max_size:
            logger.warning(f"Skipping large attachment: {attachment.get('filename')} ({size} bytes > {max_size} bytes)")
            continue
        filtered_attachments.append(attachment)

    # Limit number of files
    if len(filtered_attachments) > max_files:
        logger.warning(f"Too many attachments ({len(filtered_attachments)}), limiting to {max_files}")
        filtered_attachments = filtered_attachments[:max_files]

    # Download files
    async with aiohttp.ClientSession() as session:
        download_tasks = []
        for attachment in filtered_attachments:
            url = attachment.get("url")
            if not url:
                continue

            # Create download task
            download_tasks.append(download_attachment(session, attachment, url))

        # Execute all download tasks concurrently
        if download_tasks:
            await asyncio.gather(*download_tasks)


async def download_attachment(session, attachment: Dict[str, Any], url: str) -> None:
    """Download a single attachment."""
    try:
        # Some APIs require auth tokens, try to handle common patterns
        source = attachment.get("source", "")
        headers = {}

        # For Jira, might need to add auth headers
        if source == "jira" and JIRA_API_TOKEN:
            # Depends on auth method (basic, bearer, etc.)
            pass

        # For Confluence, might need auth
        if source == "confluence":
            # Add auth if needed
            pass

        timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds timeout
        async with session.get(url, headers=headers, timeout=timeout) as response:
            if response.status == 200:
                content = await response.read()
                attachment["content"] = content
                attachment["downloaded"] = True
                attachment["size"] = len(content)
                logger.info(f"Downloaded {attachment.get('filename')} ({len(content)} bytes)")
            else:
                attachment["error"] = f"HTTP error: {response.status}"
                logger.warning(f"Failed to download {attachment.get('filename')}: HTTP {response.status}")
    except Exception as e:
        attachment["error"] = str(e)
        logger.error(f"Error downloading {attachment.get('filename')}: {e}")

# python
@traceable
async def extract_binary_attachments(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Extract binary attachments from Confluence and Jira results.
    Supports images, Excel files, diagrams, PDFs, etc.
    """
    logger.info("--- Node: extract_binary_attachments ---")

    if not HAS_BINARY_PROCESSING:
        logger.warning("Binary processing libraries not available. Skipping binary content extraction.")
        return {"raw_binary_attachments": []}

    # Get existing results
    jira_results = state.get("raw_jira_results", [])
    confluence_results = state.get("raw_confluence_results", [])

    binary_attachments = []

    # Process Jira attachments
    for item in jira_results:
        if not isinstance(item, dict):
            continue
        # Extract attachments from Jira
        attachments = item.get("fields", {}).get("attachment", [])
        if isinstance(attachments, list):
            for attachment in attachments:
                binary_attachments.append(attachment)

    # Process Confluence attachments
    for item in confluence_results:
        if isinstance(item, dict):
            attachments_obj = item.get("attachments", {})
            # Check if attachments_obj is a list or dict
            if isinstance(attachments_obj, list):
                attachments = attachments_obj
            else:
                attachments = attachments_obj.get("results", [])
            for attachment in attachments:
                binary_attachments.append({
                    "source": "confluence",
                    "source_id": item.get("id", ""),
                    "attachment_id": attachment.get("id"),
                    "filename": attachment.get("title", ""),
                    "content_type": attachment.get("mediaType", ""),
                    "size": attachment.get("fileSize", 0),
                    "url": attachment.get("_links", {}).get("download", ""),
                    "file_type": categorize_file_type(os.path.splitext(attachment.get("title", ""))[1].lower(), attachment.get("mediaType", "")),
                    "content": None
                })
        elif isinstance(item, list):
            for sub_item in item:
                attachments_obj = sub_item.get("attachments", {})
                if isinstance(attachments_obj, list):
                    attachments = attachments_obj
                else:
                    attachments = attachments_obj.get("results", [])
                for attachment in attachments:
                    binary_attachments.append({
                        "source": "confluence",
                        "source_id": sub_item.get("id", ""),
                        "attachment_id": attachment.get("id"),
                        "filename": attachment.get("title", ""),
                        "content_type": attachment.get("mediaType", ""),
                        "size": attachment.get("fileSize", 0),
                        "url": attachment.get("_links", {}).get("download", ""),
                        "file_type": categorize_file_type(os.path.splitext(attachment.get("title", ""))[1].lower(), attachment.get("mediaType", "")),
                        "content": None
                    })

    if binary_attachments:
        try:
            await download_binary_attachments(binary_attachments, max_size=10 * 1024 * 1024, max_files=20)
        except Exception as e:
            logger.error(f"Error downloading attachments: {e}")

    logger.info(f"Extracted {len(binary_attachments)} binary attachments")
    return {"raw_binary_attachments": binary_attachments}
@traceable
async def process_binary_content(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Process different types of binary content using specialized processors."""
    logging.info("--- Node: process_binary_content ---")

    if not HAS_BINARY_PROCESSING:
        logging.warning("Binary processing libraries not available. Skipping binary content processing.")
        return {"processed_binary_content": {}}

    attachments = state.get("raw_binary_attachments", [])
    if not attachments:
        logging.info("No binary attachments to process")
        return {"processed_binary_content": {}}

    processed_results = {
        "images": [],
        "excel": [],
        "diagrams": [],
        "pdf": [],
        "other": []
    }

    # Process each attachment based on its type
    for attachment in attachments:
        file_type = attachment.get("file_type", "unknown")
        content = attachment.get("content")
        filename = attachment.get("filename", "")

        if not content:
            logging.debug(f"Skipping {filename}: No content available")
            continue

        try:
            # Create a base processed item with metadata
            processed_item = {
                "source": attachment.get("source"),
                "source_id": attachment.get("source_id"),
                "filename": filename,
                "file_type": file_type,
                "size": len(content) if content else 0,
                "content_extracted": False
            }

            # Process images (including diagrams marked as images)
            if file_type == "image" or (
                    file_type == "diagram" and attachment.get("content_type", "").startswith("image/")):
                try:
                    # Convert binary data to PIL Image
                    image = Image.open(io.BytesIO(content))

                    # Store image attributes
                    processed_item["width"] = image.width
                    processed_item["height"] = image.height
                    processed_item["format"] = image.format
                    processed_item["mode"] = image.mode

                    # Extract text using OCR if available
                    if pytesseract and pytesseract.image_to_string:
                        extracted_text = pytesseract.image_to_string(image)
                        if extracted_text and len(extracted_text.strip()) > 5:  # Only keep meaningful text
                            processed_item["extracted_text"] = extracted_text.strip()
                            processed_item["content_extracted"] = True
                            logging.info(f"Extracted {len(extracted_text)} chars of text from image: {filename}")

                    # Basic image classification for diagrams
                    is_diagram = False
                    if image.width > 400 and image.height > 400:  # Reasonable size for diagram
                        # Convert to numpy array for analysis if possible
                        if hasattr(np, 'array'):
                            img_array = np.array(image.convert('RGB'))
                            # Check color distribution (diagrams often have fewer colors)
                            unique_colors = len(set(tuple(map(tuple, img_array.reshape(-1, 3)))))
                            if unique_colors < 50:  # Diagrams typically use few colors
                                is_diagram = True

                    processed_item["is_diagram"] = is_diagram

                    # Save to appropriate category
                    if is_diagram:
                        processed_results["diagrams"].append(processed_item)
                    else:
                        processed_results["images"].append(processed_item)

                except Exception as img_err:
                    logging.error(f"Error processing image {filename}: {img_err}")
                    processed_item["error"] = str(img_err)
                    processed_results["images"].append(processed_item)

            # Process Excel files
            elif file_type == "excel":
                try:
                    if openpyxl and openpyxl.load_workbook:
                        # Load workbook from binary content
                        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
                        sheet_data = {}

                        # Process each sheet
                        for sheet_name in wb.sheetnames:
                            sheet = wb[sheet_name]
                            rows = []

                            # Extract data (limited to first 100 rows, 20 columns)
                            for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
                                if row_idx >= 100:  # Limit rows
                                    break
                                row_data = []
                                for col_idx, cell in enumerate(row):
                                    if col_idx >= 20:  # Limit columns
                                        break
                                    # Convert cell value to string
                                    cell_str = str(cell) if cell is not None else ""
                                    row_data.append(cell_str)
                                rows.append(row_data)

                            if rows:
                                sheet_data[sheet_name] = rows

                        if sheet_data:
                            processed_item["sheet_data"] = sheet_data
                            processed_item["sheet_count"] = len(sheet_data)
                            processed_item["content_extracted"] = True
                            logging.info(f"Extracted data from {len(sheet_data)} sheets in Excel file: {filename}")

                    processed_results["excel"].append(processed_item)

                except Exception as excel_err:
                    logging.error(f"Error processing Excel file {filename}: {excel_err}")
                    processed_item["error"] = str(excel_err)
                    processed_results["excel"].append(processed_item)

            # Process PDF files
            elif file_type == "pdf":
                try:
                    if pdfplumber:
                        with pdfplumber.open(io.BytesIO(content)) as pdf:
                            # Get basic PDF info
                            processed_item["page_count"] = len(pdf.pages)

                            # Extract text from first few pages
                            text_parts = []
                            for i, page in enumerate(pdf.pages[:5]):  # Limit to first 5 pages
                                page_text = page.extract_text()
                                if page_text:
                                    text_parts.append(page_text)

                                # Stop if we've collected a reasonable amount of text
                                if sum(len(t) for t in text_parts) > 10000:
                                    break

                            if text_parts:
                                processed_item["extracted_text"] = "\n\n".join(text_parts)
                                processed_item["content_extracted"] = True
                                logging.info(
                                    f"Extracted {len(processed_item['extracted_text'])} chars from PDF: {filename}")

                    processed_results["pdf"].append(processed_item)

                except Exception as pdf_err:
                    logging.error(f"Error processing PDF file {filename}: {pdf_err}")
                    processed_item["error"] = str(pdf_err)
                    processed_results["pdf"].append(processed_item)

            # Process other file types (minimal handling)
            else:
                processed_results["other"].append(processed_item)

        except Exception as e:
            logging.error(f"Error processing attachment {attachment.get('filename')}: {e}")
            # Add to appropriate category with error info
            error_item = {
                "source": attachment.get("source"),
                "source_id": attachment.get("source_id"),
                "filename": attachment.get("filename", ""),
                "file_type": file_type,
                "size": len(content) if content else 0,
                "error": str(e)
            }
            processed_results["other"].append(error_item)

    # Log summary statistics
    logging.info(f"Processed {len(attachments)} attachments: "
                 f"{len(processed_results['images'])} images, "
                 f"{len(processed_results['diagrams'])} diagrams, "
                 f"{len(processed_results['excel'])} Excel files, "
                 f"{len(processed_results['pdf'])} PDFs, "
                 f"{len(processed_results['other'])} other files")

    return {"processed_binary_content": processed_results}


@traceable
async def summarize_binary_content(state: ResearchState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Generate summaries of the processed binary content."""
    logging.info("--- Node: summarize_binary_content ---")

    processed_content = state.get("processed_binary_content", {})
    if not processed_content:
        logging.info("No binary content to summarize")
        return {"binary_content_summaries": {"integrated": "_No relevant binary content found._"}}

    # Get the LLM config if available
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    topic = state.get("research_topic", "Unknown Topic")

    summaries: Dict[str, str] = {}

    # Images summary
    images = processed_content.get("images", [])
    if images:
        image_count = len(images)
        image_names = [img.get("filename", "Unnamed") for img in images[:5]]
        image_texts = [
            img.get("extracted_text", "").strip()
            for img in images
            if img.get("content_extracted")
        ]

        summary_text = (
                f"Found {image_count} images: {', '.join(image_names)}"
                + ("..." if image_count > 5 else "")
        )

        if any(image_texts):
            summary_text += "\n\nText extracted from images:\n"
            for i, text in enumerate(image_texts[:3]):
                if text:
                    text_sample = text[:300] + "..." if len(text) > 300 else text
                    summary_text += f"- Image {i + 1}: {text_sample}\n"

        summaries["images"] = summary_text

    # Excel summary
    excel_files = processed_content.get("excel", [])
    if excel_files:
        excel_count = len(excel_files)
        excel_names = [ex.get("filename", "Unnamed") for ex in excel_files[:5]]

        summary_text = (
                f"Found {excel_count} Excel files: {', '.join(excel_names)}"
                + ("..." if excel_count > 5 else "")
        )

        for excel in excel_files[:2]:
            if excel.get("content_extracted"):
                sheet_data = excel.get("sheet_data", {})
                if sheet_data:
                    summary_text += f"\n\nData from {excel.get('filename')}:\n"
                    for sheet_name, rows in list(sheet_data.items())[:1]:
                        summary_text += f"- Sheet '{sheet_name}':\n"
                        if rows:
                            headers = rows[0]
                            summary_text += (
                                    f"  - Headers: {', '.join(headers[:5])}"
                                    + ("..." if len(headers) > 5 else "")
                                    + "\n"
                            )
                            if len(rows) > 1:
                                summary_text += (
                                    f"  - Sample data ({min(len(rows) - 1, 2)} rows):\n"
                                )
                                for row in rows[1: min(len(rows), 3)]:
                                    summary_text += (
                                            f"    - {', '.join(row[:5])}"
                                            + ("..." if len(row) > 5 else "")
                                            + "\n"
                                    )

        summaries["excel"] = summary_text

    # Diagrams summary
    diagrams = processed_content.get("diagrams", [])
    if diagrams:
        diagram_count = len(diagrams)
        diagram_names = [d.get("filename", "Unnamed") for d in diagrams[:5]]
        extracted_texts = [
            d.get("extracted_text", "").strip()
            for d in diagrams
            if d.get("content_extracted")
        ]

        summary_text = (
                f"Found {diagram_count} diagrams: {', '.join(diagram_names)}"
                + ("..." if diagram_count > 5 else "")
        )

        if any(extracted_texts):
            summary_text += "\n\nText extracted from diagrams:\n"
            for i, text in enumerate(extracted_texts[:2]):
                if text:
                    text_sample = text[:300] + "..." if len(text) > 300 else text
                    summary_text += f"- Diagram {i + 1}: {text_sample}\n"

        summaries["diagrams"] = summary_text

    # PDF summary
    pdfs = processed_content.get("pdf", [])
    if pdfs:
        pdf_count = len(pdfs)
        pdf_names = [p.get("filename", "Unnamed") for p in pdfs[:5]]

        summary_text = (
                f"Found {pdf_count} PDF files: {', '.join(pdf_names)}"
                + ("..." if pdf_count > 5 else "")
        )

        pdf_texts = [
            p.get("extracted_text", "").strip()
            for p in pdfs
            if p.get("content_extracted")
        ]
        if any(pdf_texts):
            summary_text += "\n\nText samples from PDFs:\n"
            for i, text in enumerate(pdf_texts[:2]):
                if text:
                    text_lines = text.split("\n")
                    sample = "\n".join(text_lines[:5])
                    if len(text_lines) > 5:
                        sample += "\n..."
                    summary_text += f"- PDF {i + 1}:\n{sample}\n"

        summaries["pdf"] = summary_text

    # Use LLM to create an integrated summary if available
    integrated_summary = ""
    if summaries and hasattr(cfg, "ollama_base_url") and hasattr(cfg, "local_llm"):
        try:
            combined = ""
            for ctype, summ in summaries.items():
                combined += f"## {ctype.capitalize()}\n{summ}\n\n"

            prompt = (
                f"Analyze and summarize the following information about binary content "
                f"related to '{topic}'.\n\n"
                f"Content details:\n{combined}\n\n"
                "Create a concise (300-500 words) summary that:\n"
                "1. Describes the types and quantity of binary content found\n"
                "2. Extracts and analyzes the most relevant information\n"
                "3. Explains how this content relates to the research topic\n"
                "4. Highlights any notable patterns or insights\n\n"
                "If little seems relevant, acknowledge why.\n"
            )

            llm = ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.local_llm,
                **LLM_CONFIG_PARAMS
            )
            res = await llm.ainvoke([SystemMessage(content=prompt)])
            integrated_summary = clean_llm_output(res.content)

        except Exception as e:
            logger.error(f"Error generating integrated binary content summary: {e}")
            integrated_summary = "Binary Content Analysis:\n\n"
            for ctype, summ in summaries.items():
                first_line = summ.splitlines()[0]
                integrated_summary += f"- {ctype.capitalize()}: {first_line}\n"

    else:
        if summaries:
            integrated_summary = "Binary Content Analysis:\n\n"
            for ctype, summ in summaries.items():
                first_line = summ.splitlines()[0]
                integrated_summary += f"- {ctype.capitalize()}: {first_line}\n"
        else:
            integrated_summary = "_No relevant binary content found._"

    summaries["integrated"] = integrated_summary

    # Create thumbnail references for visualizing in the final report
    try:
        image_thumbs = []
        # Process the first few images and diagrams to create thumbnail references
        all_visuals = processed_content.get("images", []) + processed_content.get("diagrams", [])
        for i, img in enumerate(all_visuals[:5]):
            if img.get("filename"):
                # In a real implementation, this would save thumbnails to disk
                # Here we're creating placeholder references
                img_id = f"img_{i}_{uuid4().hex[:8]}"
                thumb_path = f"./outputs/thumbnails/{img_id}.png"
                img["thumbnail"] = thumb_path
                image_thumbs.append(thumb_path)
    except Exception as thumb_err:
        logger.error(f"Error creating thumbnail references: {thumb_err}")

    return {"binary_content_summaries": summaries}

def build_research_graph() -> CompiledStateGraph:
    """Builds the LangGraph state machine for the research process."""
    g = StateGraph(ResearchState, config_schema=Configuration)

    # --- Add Nodes ---
    g.add_node("generate_query", generate_query)
    g.add_node("classify_research_topic", classify_research_topic)
    g.add_node("search_all_sources", search_all_sources)
    g.add_node("enterprise_research", enterprise_research)
    g.add_node("process_confluence_chunk", process_confluence_chunk)
    g.add_node("process_jira_chunk", process_jira_chunk)
    g.add_node("process_perforce_chunk", process_perforce_chunk)
    g.add_node("route_after_research", lambda s, c=None: s)
    g.add_node("filter_irrelevant_data", filter_irrelevant_data)
    g.add_node("check_retrieval_relevance", check_retrieval_relevance)
    g.add_node("analyze_perforce_diffs", analyze_perforce_diffs)
    g.add_node("perform_web_search", perform_web_search)
    g.add_node("preprocess_for_summary", preprocess_for_summary)
    g.add_node("summarize_jira_results", summarize_jira_results)
    g.add_node("summarize_confluence_results", summarize_confluence_results)
    g.add_node("summarize_perforce_results", summarize_perforce_results)
    g.add_node("analyze_security", analyze_security)
    g.add_node("validate_technical_aspects", validate_technical_aspects)
    g.add_node("verify_summary", verify_summary)
    g.add_node("extract_citations", extract_citations)
    g.add_node("generate_knowledge_graph", generate_knowledge_graph)
    g.add_node("reflect_on_findings", reflect_on_findings)
    g.add_node("analyze_cross_source_connections", analyze_cross_source_connections)
    g.add_node("refine_combined_summaries", refine_combined_summaries)
    g.add_node("finalize_summary", finalize_summary)
    g.add_node("export_research", export_research)
    g.add_node("error_recovery", error_recovery)
    g.add_node("answer_quiz_questions", answer_quiz_questions)
    g.add_node("assemble_intermediate_summary", assemble_intermediate_summary)

    # --- Add Binary Content Processing Nodes ---
    g.add_node("extract_binary_attachments", extract_binary_attachments)
    g.add_node("process_binary_content", process_binary_content)
    g.add_node("summarize_binary_content", summarize_binary_content)

    # --- Define Edges ---
    g.add_edge(START, "generate_query")
    g.add_edge("generate_query", "classify_research_topic")

    # Define classification routes function
    def route_after_classification(state: ResearchState) -> str:
        classification = state.get("topic_classification", "Deep Research")
        logger.debug(f"Routing based on classification: {classification}")
        return "search_all_sources"

    g.add_conditional_edges(
        "classify_research_topic",
        route_after_classification,
        {
            "search_all_sources": "search_all_sources",
        }
    )

    g.add_edge("search_all_sources", "enterprise_research")
    g.add_edge("enterprise_research", "route_after_research")

    g.add_conditional_edges(
        "route_after_research",
        route_after_research,
        {
            "process_confluence_chunk": "process_confluence_chunk",
            "process_jira_chunk": "process_jira_chunk",
            "process_perforce_chunk": "process_perforce_chunk",
            "filter_irrelevant_data": "filter_irrelevant_data",
            "finalize_summary": "finalize_summary"
        }
    )

    g.add_edge("process_confluence_chunk", "route_after_research")
    g.add_edge("process_jira_chunk", "route_after_research")
    g.add_edge("process_perforce_chunk", "route_after_research")

    g.add_edge("filter_irrelevant_data", "check_retrieval_relevance")

    g.add_conditional_edges("check_retrieval_relevance",
                            lambda s, c=None: s.get("_relevance_check", "irrelevant"),
                            {
                                "relevant": "analyze_perforce_diffs",
                                "irrelevant": "finalize_summary",
                            })

    g.add_edge("analyze_perforce_diffs", "preprocess_for_summary")
    g.add_edge("preprocess_for_summary", "summarize_jira_results")
    g.add_edge("summarize_jira_results", "summarize_confluence_results")
    g.add_edge("summarize_confluence_results", "summarize_perforce_results")
    g.add_edge("summarize_perforce_results", "analyze_cross_source_connections")

    # Add binary content processing to the flow
    g.add_edge("analyze_cross_source_connections", "extract_binary_attachments")
    g.add_edge("extract_binary_attachments", "process_binary_content")
    g.add_edge("process_binary_content", "summarize_binary_content")
    g.add_edge("summarize_binary_content", "answer_quiz_questions")

    # Conditional edge now originates from QA node
    g.add_conditional_edges("answer_quiz_questions",
                            check_content_quality,
                            {
                                "needs_security_analysis": "analyze_security",
                                "needs_technical_validation": "validate_technical_aspects",
                                "proceed_to_verify": "verify_summary",
                                "reflect_now": "reflect_on_findings",
                            })

    g.add_edge("analyze_security", "verify_summary")
    g.add_edge("validate_technical_aspects", "verify_summary")

    g.add_conditional_edges("verify_summary",
                            _verification_status,
                            {
                                "has_errors": "error_recovery",
                                "clean": "assemble_intermediate_summary",
                            })
    g.add_edge("assemble_intermediate_summary", "extract_citations")

    g.add_edge("extract_citations", "generate_knowledge_graph")
    g.add_edge("generate_knowledge_graph", "reflect_on_findings")

    g.add_conditional_edges("reflect_on_findings",
                            try_determine_continuation,
                            {
                                "continue": "generate_query",
                                "has_errors": "error_recovery",
                                "complete": "finalize_summary",
                            })

    g.add_conditional_edges("error_recovery",
                            lambda s, c=None: "default_recovery",
                            {
                                "default_recovery": "finalize_summary"
                            })

    g.add_edge("finalize_summary", "export_research")
    g.add_edge("export_research", END)

    logger.info("Compiling the research graph...")
    return g.compile()
# Initialize graph here to avoid unresolved references
graph = build_research_graph()

# --- Example Main Execution Block ---
if __name__ == "__main__":
    logger.info("Running graph_2.py directly for testing...")

    # Now perform checks *before* invoking
    if not graph:
        logger.critical("Graph object is None after build attempt. Cannot run research. Exiting.")
        sys.exit(1)
    if not check_tool_alignment():
        logger.warning("Tools are not aligned. Research may fail.")

    # Example research topic
    topic_to_research = "What is MTV2005?"

    # Initial state definition
    initial_state: ResearchState = {
        "research_topic": topic_to_research,
        "messages": [HumanMessage(content=f"Please research: {topic_to_research}")],
        # --- ADDED: Initialize new state field ---
        "topic_classification": "",
        # --- END ADDED ---
        "research_loop_count": 0,
        "enterprise_research_results": [],
        "sources_gathered": [],
        "_errors": [],
        "_jira_keys_seen": set(),
        "_perforce_cl_numbers": set(),
        "_confluence_urls_seen": set(),
        "_confluence_ids_seen": set(),
        "_content_hashes": set(),
        "jira_chunk_idx": 0, "jira_total_chunks": 0, "jira_chunks": [],
        "confluence_chunk_idx": 0, "confluence_total_chunks": 0, "confluence_chunks": [],
        "perforce_chunk_idx": 0, "perforce_total_chunks": 0, "perforce_chunks": [],
        "raw_jira_results": [],
        "raw_confluence_results": [],
        "raw_perforce_results": [],
        "raw_web_search_results": [],
        "running_summary": "",
        "search_query": "",
        "security_findings": [],
        "technical_validation": {},
        "reflection_data": "",
        "perforce_diff_analysis": {},
        "verification_results": {},
        "citations": [],
        "knowledge_graph": {},
        "alternative_terms": [],
        "_relevance_check": "",
        "__hop_counter": 0,
        "_last_confluence_chunk_idx": -1,
        "_last_jira_chunk_idx": -1,
        "_last_perforce_chunk_idx": -1,
        "_stuck_confluence_count": 0,
        "_stuck_jira_count": 0,
        "_stuck_perforce_count": 0,
        "_recovery_actions": [],
        "_recovery_metadata": {},
        "export_format": "markdown",
        "exported_data": {},
        #"quiz_questions": [], # Initialize new state field
        "quiz_answers": [],   # Initialize new state field
    }

    # Runtime configuration definition
    runtime_config = {
        "configurable": {
            "ollama_base_url": OLLAMA_BASE_URL,
            "local_llm": LLM_MODEL,
            "embedding_model": os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest'),
            "include_web_search": False,
            "generate_visualizations": HAS_VISUALIZATION,
        "export_format": "markdown",
        },
        "recursion_limit": MAX_TOTAL_HOPS + 10
    }
    logger.info(f"Starting research with topic: '{topic_to_research}'")
    logger.info(f"Runtime config: {runtime_config['configurable']}")

    final_result_state = None
    try:
        # Invoke the graph
        final_result_state = graph.invoke(initial_state, config=runtime_config)
        logger.info("Research process finished.")

        print("\n" + "="*30 + " Research Complete " + "="*30)
        if final_result_state:
            print("\n--- Final Research Summary ---")
            final_summary = final_result_state.get("running_summary", "No final summary generated.")
            print(final_summary)

            export_info = final_result_state.get("exported_data", {})
            print("\n--- Export Status ---")
            if isinstance(export_info, dict):
                 print(f"Format: {export_info.get('format', 'N/A')}")
                 if export_info.get("error"):
                      print(f"Error: {export_info['error']}")
                 elif export_info.get("export_path"):
                      print(f"Saved to: {export_info['export_path']}")
                 elif export_info.get("export_paths"):
                      print(f"Saved CSVs to: {', '.join(export_info['export_paths'])}")
                 else:
                      print("Export details not found.")
            else:
                 print("Export information is not in the expected format.")

            final_errors = final_result_state.get("_errors", [])
            if final_errors:
                print("\n--- Errors Encountered (Final State) ---")
                for error in final_errors:
                     print(f"- Stage: {error.get('stage', 'unknown')}, Message: {error.get('message', 'N/A')}")
            else:
                 print("\nNo errors recorded in the final state.")
        else:
             print("\nError: Final state not captured.")

    except Exception as e:
        logger.critical(f"Critical error during graph execution: {e}", exc_info=True)
        print(f"\n--- Critical Execution Error ---")
        print(f"Research execution failed: {e}")

    print("\n" + "="*70)

# --- Binary Content Processing Functions ---

