#!/usr/bin/env python3
# run_graph_3.py - Streamlined version
"""
Enhanced run script for graph_3.py with robust error handling and anti-hallucination features.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import ssl
import urllib3
import socket
import gc
import signal
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from src.assistant import state

# NEW: Load environment variables from `.env` automatically
try:
    from dotenv import load_dotenv

    # Search for a .env file starting from the project root
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except ModuleNotFoundError:
    # python-dotenv is optional; proceed if not installed
    pass

# Import the research graph from the assistant module
from src.assistant.graph_3 import create_research_graph

# Configure proxy settings
os.environ["HTTP_PROXY"] = os.environ.get("HTTP_PROXY", "")
os.environ["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY", "")

# Disable SSL verification for internal development
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['NODE_TLS_REJECT_UNAUTHORIZED'] = '0'
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HUGGINGFACE_DISABLE_SSL'] = 'True'

# Modify Python's SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Ensure LangSmith traffic bypasses corporate proxy -------------------
# Decide whether to bypass the proxy for LangSmith.
# In many corporate environments outbound traffic *must* go through the
# configured HTTP(S)_PROXY, so blindly adding the LangSmith host to
# NO_PROXY breaks DNS / connectivity (exactly the error we just saw).
#
# New rule:
#   • If the user already has HTTP(S)_PROXY set, we assume those proxies
#     are required and *do not* add LangSmith to NO_PROXY.
#   • Otherwise, we keep the original behaviour (bypass when no proxy).

_no_proxy = os.environ.get("NO_PROXY", "")
_langsmith_domain = "api.smith.langchain.com"
_proxy_vars_present = any(os.environ.get(v) for v in [
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"
])

if not _proxy_vars_present and _langsmith_domain not in _no_proxy:
    # Safely append without leading/trailing comma issues
    os.environ["NO_PROXY"] = (f"{_no_proxy},{_langsmith_domain}" if _no_proxy else _langsmith_domain)
    logging.info("Added '%s' to NO_PROXY to allow direct LangSmith access", _langsmith_domain)
else:
    logging.info(
        "Keeping LangSmith traffic on configured proxy (HTTP(S)_PROXY present)"
    )

# --- Pre-check DNS for critical internal services (Jira, Confluence) --------
_internal_services_to_check = {
    "Jira": "deljira",
    "Confluence (SolutionBook)": ""
}
for service_name, service_host in _internal_services_to_check.items():
    try:
        socket.gethostbyname(service_host)
        logging.info("Successfully resolved %s (%s). Should be reachable.", service_name, service_host)
    except socket.gaierror: # More specific exception for DNS failures
        logging.warning(
            "Could not resolve %s (%s) via DNS. "
            "The service may be temporarily unavailable or there might be a network/DNS misconfiguration. "
            "The application will attempt to proceed, but %s functionality might be impaired.",
            service_name, service_host, service_name
        )
    except Exception as e: # Catch any other socket related error during gethostbyname
        logging.warning(
            "An unexpected error occurred while trying to resolve %s (%s): %s. "
            "The application will attempt to proceed, but %s functionality might be impaired.",
            service_name, service_host, str(e), service_name
        )
# ------------------------------------------------------------------------

# --- Ensure a tool-compatible LLM model -------------------------------
# DeepSeek-R1 models currently do NOT support the LangChain tool calling
# interface and will cause 400 "does not support tools" errors when the
# assistant attempts to use them via Ollama.  To prevent repeated failures
# we automatically fall back to the qwen3 model family when a DeepSeek-R1
# model is detected in the environment.
_llm_env = os.environ.get("LLM_MODEL", "").lower()
if _llm_env.startswith("deepseek-r1"):
    logging.warning(
        "Detected unsupported DeepSeek-R1 model (%s). Falling back to "
        "'qwen3:30b-a3b' which is tool-compatible.", _llm_env
    )
    os.environ["LLM_MODEL"] = "qwen3:30b-a3b"
# ---------------------------------------------------------------------

# Initialize global variables to track data sources
GLOBAL_QUERY = "Unknown"
GLOBAL_JIRA_COUNT = 0
GLOBAL_PERFORCE_COUNT = 0
GLOBAL_CONFLUENCE_COUNT = 0
GLOBAL_SUMMARY = ""
TERMINATION_REQUESTED = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('run_graph_3.log')
    ]
)
logger = logging.getLogger(__name__)

# Import dependencies
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

# Import local modules
from src.assistant.graph_3 import create_research_graph

# Define an async timeout decorator with proper cancellation
def async_timeout(seconds):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logging.warning(f"Function {func.__name__} timed out after {seconds} seconds")
                gc.collect()
                raise
        return wrapper
    return decorator

# Setup signal handlers for graceful termination
def setup_signal_handlers():
    global TERMINATION_REQUESTED
    
    def signal_handler(sig, frame):
        global TERMINATION_REQUESTED
        if not TERMINATION_REQUESTED:
            print("\nTermination signal received, initiating graceful shutdown...")
            TERMINATION_REQUESTED = True
        else:
            print("\nSecond termination signal received, exiting immediately")
            sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Add a new helper function to make messages JSON serializable
def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    serialized_messages = []
    for msg in messages:
        if isinstance(msg, BaseMessage): # Check if it's a Langchain message object
            serialized_messages.append({
                "type": msg.type, # 'ai', 'human', 'system'
                "content": msg.content,
                # Add other relevant attributes if necessary, e.g., 'name', 'additional_kwargs'
            })
        elif isinstance(msg, dict) and "type" in msg and "content" in msg:
            serialized_messages.append(msg) # Already a serializable dict
        else:
            # For unknown types, try to convert to string or skip
            try:
                serialized_messages.append(str(msg)) 
            except:
                serialized_messages.append("[UnserializableMessageObject]")
    return serialized_messages

# Custom progress callback
class ProgressCallback:
    def __init__(self):
        """Initialize with empty node times"""
        self.node_times = {}
        self.current_node = None
        self.start_time = None
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Track start of a node execution"""
        try:
            if serialized is None:
                self.current_node = "Unknown"
            elif isinstance(serialized, dict):
                # Try several likely dict keys in order of preference
                for key in ("name", "id", "node_name", "type"):
                    val = serialized.get(key)
                    if val:
                        self.current_node = str(val)
                        break
                else:
                    # Fallback to first key or generic Unknown
                    try:
                        self.current_node = str(next(iter(serialized.keys())))
                    except Exception:
                        self.current_node = "Unknown"
            else:
                # Non-dict object – use class name or string representation
                self.current_node = getattr(serialized, "__name__", str(serialized))

            self.start_time = time.time()
            logger.info(f"Starting node: {self.current_node}")
        except Exception as e:
            logger.warning(f"Error in on_chain_start: {e}")
    
    def on_chain_end(self, outputs, **kwargs):
        """Track end of a node execution"""
        try:
            if self.current_node and self.start_time:
                elapsed = time.time() - self.start_time
                self.node_times[self.current_node] = self.node_times.get(self.current_node, 0) + elapsed
                logger.info(f"Completed node: {self.current_node} in {elapsed:.2f}s (total: {self.node_times[self.current_node]:.2f}s)")
                
                # Log slow nodes for performance monitoring
                if elapsed > 5:
                    logger.warning(f"Slow node detected: {self.current_node} took {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"Error in on_chain_end: {e}")
    
    def on_chain_error(self, error, **kwargs):
        """Track errors in node execution"""
        try:
            if self.current_node:
                logger.error(f"Error in node {self.current_node}: {error}")
        except Exception as e:
            logger.warning(f"Error in on_chain_error: {e}")

# Custom filter to track data from log messages
class DataTrackingLogFilter(logging.Filter):
    def filter(self, record):
        global GLOBAL_QUERY, GLOBAL_JIRA_COUNT, GLOBAL_PERFORCE_COUNT, GLOBAL_CONFLUENCE_COUNT
        
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg) if hasattr(record, 'msg') else ""
        
        # Extract query
        if "Searching" in msg and "for:" in msg:
            parts = msg.split("for:")
            if len(parts) > 1:
                GLOBAL_QUERY = parts[1].strip()
                
        # Track Jira results (various log formats)
        if ("Found" in msg and "Jira issues" in msg) or ("Extracted" in msg and "issues" in msg):
            try:
                parts = msg.split("Found ") if "Found" in msg else msg.split("Extracted ")
                if len(parts) > 1:
                    count_part = parts[1].split(" ")[0]
                    if count_part.isdigit():
                        GLOBAL_JIRA_COUNT = int(count_part)
            except Exception:
                pass
                    
        # Track Perforce results
        if "Found" in msg and "Perforce changes" in msg:
            try:
                parts = msg.split("Found ")
                if len(parts) > 1:
                    count_part = parts[1].split(" ")[0]
                    if count_part.isdigit():
                        GLOBAL_PERFORCE_COUNT = int(count_part)
            except Exception:
                pass
            
        # Track Confluence results
        if ("Found" in msg and "Confluence" in msg) or ("Extracted" in msg and "Confluence" in msg):
            try:
                parts = msg.split("Found ") if "Found" in msg else msg.split("Extracted ")
                if len(parts) > 1:
                    count_part = parts[1].split(" ")[0]
                    if count_part.isdigit():
                        GLOBAL_CONFLUENCE_COUNT = int(count_part)
            except Exception:
                pass
                    
        return True

def configure_logging():
    """Configure logging with custom filter"""
    root_logger = logging.getLogger()
    data_tracking_filter = DataTrackingLogFilter()
    root_logger.addFilter(data_tracking_filter)

    # Ensure INFO level logging so that standard tracking messages are processed
    # Do not override more verbose settings such as DEBUG set earlier via --debug.
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

# Default configuration
DEFAULT_CONFIG = {
    "enable_rag": True,
    "enable_verification": True,
    "enable_cot": True,
    "enable_cove": True,
    "enable_map_rerank": True,
    "enable_cok": True,
    "enable_multi_agent": True,
    "enable_cross_source": True,
    "enable_hybrid_retrieval": True,
    "enable_multi_query": True,
    "enable_recursive_retrieval": True,
    "temperature": 0.15,
    "max_recursive_depth": 2,
    "llm_timeout": 60,
    "api_timeout": 120,
    "node_timeout": 180,
    "cross_source_timeout": 600,
    "max_memory_mb": 2000,
}

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the enhanced research system with anti-hallucination features."
    )
    
    # Allow the query to be passed either positionally or with --query for convenience.
    # The positional argument is now optional (nargs='?') so that the flag can be
    # used instead.  We reconcile the two after parsing.

    parser.add_argument(
        "query",
        nargs="?",  # optional positional; will be validated after parsing
        type=str,
        help="The research query to process (positional)."
    )

    parser.add_argument(
        "--query",
        dest="query_flag",
        metavar="QUERY",
        type=str,
        help="The research query to process (optional flag, alternative to positional)."
    )
    
    # Core options
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no-verification", action="store_true", help="Disable verification")
    parser.add_argument("--no-cross-source", action="store_true", help="Disable Cross-Source Analysis")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file path")
    
    # Timeout options
    parser.add_argument("--timeout", type=int, default=None, help="Overall timeout in seconds")
    parser.add_argument("--max-runtime", type=int, default=1800, help="Maximum runtime in seconds (default: 1800)")
    
    # Add a maximum iterations parameter with increased default
    parser.add_argument("--max-iterations", type=int, default=150,
                       help="Maximum number of iterations in the graph (default: 150)")
    
    # Fast mode for quick testing
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast mode: limit search results")
    
    args = parser.parse_args()

    # Reconcile positional vs flag.  Prefer the positional if both supplied.
    if args.query is None:
        if args.query_flag is None:
            parser.error("A research query must be supplied either positionally or via --query")
        args.query = args.query_flag
    elif args.query_flag is not None and args.query != args.query_flag:
        logger.warning("Both positional query ('%s') and --query flag ('%s') supplied; using positional.", args.query, args.query_flag)

    return args

# Configure based on command line arguments
def configure_from_args(args):
    config = DEFAULT_CONFIG.copy()
    
    if args.no_rag:
        config["enable_rag"] = False
    if args.no_verification:
        config["enable_verification"] = False
    if args.no_cross_source:
        config["enable_cross_source"] = False
        config["cross_source_disabled"] = True
    
    # Enable fast mode if specified
    if args.fast_mode:
        logger.info("Fast mode enabled: limiting search results and skipping slow operations")
        config["enable_cross_source"] = False
        config["cross_source_disabled"] = True
        config["enable_map_rerank"] = False
        config["enable_cok"] = False
        config["enable_multi_agent"] = False
        config["fast_mode"] = True
        # Reduce timeouts
        config["llm_timeout"] = 30
        config["api_timeout"] = 60
        config["node_timeout"] = 90
    
    return config

# Function to save partial results when needed
def save_partial_results(result, args, reason="partial"):
    """Save partial results in case of timeout or interruption"""
    global GLOBAL_QUERY, GLOBAL_JIRA_COUNT, GLOBAL_PERFORCE_COUNT, GLOBAL_CONFLUENCE_COUNT
    
    logger.info(
        f"RUN_GRAPH_3 (save_partial_results ENTER): reason='{reason}', Output File='{args.output or 'default'}', "
        f"GLOBAL_JIRA_COUNT={GLOBAL_JIRA_COUNT}, "
        f"GLOBAL_PERFORCE_COUNT={GLOBAL_PERFORCE_COUNT}, "
        f"GLOBAL_CONFLUENCE_COUNT={GLOBAL_CONFLUENCE_COUNT}"
    )
    try:
        if not result:
            return
        
        serializable_result = {}
        
        # Determine current query
        current_query = result.get("research_topic", result.get("search_query", GLOBAL_QUERY)) if isinstance(result, dict) else getattr(args, "query", GLOBAL_QUERY)

        # Extract data from the result
        if isinstance(result, dict):
            for k in ["search_query", "research_topic", "running_summary", "final_summary", 
                    "jira_data", "perforce_data", "confluence_data"]:
                if k in result:
                    serializable_result[k] = result[k]
                    
            # Special handling for max_iterations_exceeded
            if reason == "max_iterations_exceeded":
                # Ensure all collected data is included
                if "jira_data" in result and isinstance(result["jira_data"], dict):
                    serializable_result["jira_data"] = result["jira_data"]
                if "perforce_data" in result and isinstance(result["perforce_data"], dict):
                    serializable_result["perforce_data"] = result["perforce_data"]
                if "confluence_data" in result and isinstance(result["confluence_data"], dict):
                    serializable_result["confluence_data"] = result["confluence_data"]
                
                # If there's a running_summary but no final_summary, use running_summary
                if "running_summary" in result and (not "final_summary" in result or not result["final_summary"]):
                    serializable_result["final_summary"] = result["running_summary"]
        
        # Add metadata about data counts
        serializable_result["_data_counts"] = {
            "jira_count": GLOBAL_JIRA_COUNT,
            "perforce_count": GLOBAL_PERFORCE_COUNT, 
            "confluence_count": GLOBAL_CONFLUENCE_COUNT,
            "total_count": GLOBAL_JIRA_COUNT + GLOBAL_PERFORCE_COUNT + GLOBAL_CONFLUENCE_COUNT,
            "has_jira_data": GLOBAL_JIRA_COUNT > 0,
            "has_perforce_data": GLOBAL_PERFORCE_COUNT > 0,
            "has_confluence_data": GLOBAL_CONFLUENCE_COUNT > 0
        }
                
        # Ensure a final_summary exists
        if "final_summary" not in serializable_result or not serializable_result.get("final_summary"):
            running_summary = serializable_result.get("running_summary", "")
            if running_summary:
                serializable_result["final_summary"] = running_summary
            else:
                serializable_result["final_summary"] = (
                    f"Research on '{current_query}' was incomplete due to {reason}. "
                    f"Collected data from {GLOBAL_JIRA_COUNT} Jira issues, {GLOBAL_PERFORCE_COUNT} Perforce changes, "
                    f"and {GLOBAL_CONFLUENCE_COUNT} Confluence pages."
                )
        
        serializable_result["completed_research"] = True
            
        # Special handling for max_iterations_exceeded - create better summaries from raw data
        if reason == "max_iterations_exceeded":
            # Get all processed results across data sources
            jira_data = serializable_result.get("jira_data", {})
            perforce_data = serializable_result.get("perforce_data", {})
            confluence_data = serializable_result.get("confluence_data", {})
            
            # Create better summaries if needed
            if not jira_data.get("summary") and GLOBAL_JIRA_COUNT > 0:
                jira_data["summary"] = f"Found {GLOBAL_JIRA_COUNT} Jira issues related to the query."
                
            if not perforce_data.get("summary") and GLOBAL_PERFORCE_COUNT > 0:
                perforce_data["summary"] = f"Found {GLOBAL_PERFORCE_COUNT} Perforce changelists related to the query."
                
            if not confluence_data.get("summary") and GLOBAL_CONFLUENCE_COUNT > 0:
                confluence_data["summary"] = f"Found {GLOBAL_CONFLUENCE_COUNT} Confluence pages related to the query."
            
            # Create a better final summary
            data_items = []
            
            if GLOBAL_JIRA_COUNT > 0:
                data_items.append(f"Jira: {jira_data.get('summary', f'Found {GLOBAL_JIRA_COUNT} issues.')}")
                    
            if GLOBAL_PERFORCE_COUNT > 0:
                data_items.append(f"Perforce: {perforce_data.get('summary', f'Found {GLOBAL_PERFORCE_COUNT} changelists.')}")
                    
            if GLOBAL_CONFLUENCE_COUNT > 0:
                data_items.append(f"Confluence: {confluence_data.get('summary', f'Found {GLOBAL_CONFLUENCE_COUNT} pages.')}")
            
            # Make a more descriptive final summary
            if data_items:
                improved_summary = (
                    f"Research on '{current_query}' reached the maximum iterations limit, but collected "
                    f"significant data from multiple sources. Here's a summary of what was found:\n\n"
                )
                improved_summary += "\n".join(data_items)
                
                running_summary = serializable_result.get("running_summary", "")
                if running_summary:
                    improved_summary += f"\n\nPartial research summary:\n{running_summary}"
                
                serializable_result["final_summary"] = improved_summary

        # Track metadata about the save
        serializable_result["_partial_save"] = {
            "reason": reason,
            "timestamp": time.time(),
            "query": current_query
        }
        
        # Print progress report to console
        progress_report = (
            "\n=== RESEARCH PROGRESS REPORT ===\n"
            f"Query: {current_query}\n"
            f"Data sources retrieved:\n"
            f"  - Jira: {GLOBAL_JIRA_COUNT} issues\n"
            f"  - Perforce: {GLOBAL_PERFORCE_COUNT} changes\n"
            f"  - Confluence: {GLOBAL_CONFLUENCE_COUNT} pages\n"
            f"Total items collected: {GLOBAL_JIRA_COUNT + GLOBAL_PERFORCE_COUNT + GLOBAL_CONFLUENCE_COUNT}\n"
        )
        
        # Add summary preview if available
        if "final_summary" in serializable_result and serializable_result["final_summary"]:
            summary = serializable_result["final_summary"]
            summary_preview = summary[:500] + "..." if len(summary) > 500 else summary
            progress_report += f"\nCurrent summary:\n{summary_preview}\n"
            
        print(progress_report)
        
        # Ensure 'messages' in serializable_result is processed
        if "messages" in serializable_result and isinstance(serializable_result["messages"], list):
            serializable_result["messages"] = _serialize_messages(serializable_result["messages"])
        elif "messages" in state and isinstance(state.get("messages"), list): # Fallback to state if result didn't have it
             serializable_result["messages"] = _serialize_messages(state["messages"])

        # Save to file
        timestamp = int(time.time())
        output_file = args.output or f"graph_3_result_{reason}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=_json_fallback)
            
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving partial results: {str(e)}", exc_info=True)

# Main async function
async def main():
    """Main entry point for the application"""
    global GLOBAL_QUERY, GLOBAL_JIRA_COUNT, GLOBAL_PERFORCE_COUNT, GLOBAL_CONFLUENCE_COUNT
    
    # Load LangSmith tracing configuration from environment variables
    # Set these in your .env file if you want to enable LangSmith tracing
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        logger.info("LangSmith tracing enabled via environment variables")
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize GLOBAL_QUERY
    GLOBAL_QUERY = args.query or args.query_flag
    logger.info(f"Setting initial GLOBAL_QUERY to: {GLOBAL_QUERY}")
    
    # Configure logging with data tracking
    configure_logging()
    
    # Configure the system
    config = configure_from_args(args)
    
    # Create a thread ID for tracking in logs
    thread_id = f"research_{os.getpid()}"
    logger.info(f"Starting research with thread ID: {thread_id}")
    logger.info(f"Query: {args.query or args.query_flag}")
    logger.info(f"Timeout setting: {args.timeout or 'None (infinite)'}")
    
    # Set signal handlers for graceful termination
    setup_signal_handlers()
    
    # Setup runtime limits
    max_runtime = args.max_runtime or 1800  # 30 minutes default
    logger.info(f"Running with max runtime of {max_runtime} seconds")
    
    # Generate callback
    progress_callback = ProgressCallback()
    
    # Create a thread-local checkpoint saver
    thread_checkpointer = InMemorySaver()
    
    # Create the graph
    graph = create_research_graph()

    # Configure research parameters
    config_obj = {
        "configurable": {
            "enable_jira": True,
            "enable_confluence": True,
            "enable_perforce": True,
            "enable_web": False,
            "local_llm": "qwen3:30b-a3b",  # Force using qwen3, which supports tools
            "fast_mode": getattr(args, 'fast_mode', False),
            "enable_map_rerank": True,
            "enable_cok": True,
            "enable_multi_agent": True,
            "enable_cross_source": not getattr(args, 'no_cross_source', False),
            "enable_verification": True,
            "max_iterations": args.max_iterations
        }
    }
    
    # Create the initial state
    state = {
        "search_query": args.query or args.query_flag,
        "research_topic": args.query or args.query_flag,
        "running_summary": "",
        "research_loop_count": 0,
        "messages": [HumanMessage(content=args.query or args.query_flag)],
        "jira_data": {},
        "confluence_data": {},
        "perforce_data": {},
        "web_data": {},
        "retrieved_context": [],
        "verification_results": [],
        "map_rerank_results": None,
        "chain_of_knowledge_results": None,
        "multi_agent_verification_results": None,
        "cross_source_analysis": {},
        "cross_source_connections": [],
        "technical_validation": {},
        "security_findings": [],
        "final_summary": "",
        "completed_research": False,
        "hallucination_score": 0.5
    }
    
    result = None
    
    try:
        # Track start time
        start_time = time.time()
        
        # Create callbacks
        class ProgressHandler(BaseCallbackHandler):
            def __init__(self, callback):
                self.callback = callback
            
            def on_chain_start(self, serialized, inputs, **kwargs):
                try:
                    self.callback.on_chain_start(serialized, inputs, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in handler.on_chain_start: {e}")
            
            def on_chain_end(self, outputs, **kwargs):
                try:
                    self.callback.on_chain_end(outputs, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in handler.on_chain_end: {e}")
            
            def on_chain_error(self, error, **kwargs):
                try:
                    self.callback.on_chain_error(error, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in handler.on_chain_error: {e}")
        
        # Create callbacks with our handler
        progress_handler = ProgressHandler(progress_callback)
        callbacks = {"callbacks": [progress_handler]}
        
        # Pass callbacks to configurable
        combined_config = {"configurable": config_obj["configurable"]}
        combined_config.update(callbacks)
        
        # Add thread_id to configurable for checkpointer
        combined_config["configurable"]["thread_id"] = thread_id
        combined_config["configurable"]["max_iterations"] = args.max_iterations
        
        # Run with timeout if specified
        if args.timeout:
            @async_timeout(args.timeout if args.timeout else None)  # Use None for no timeout
            async def run_with_timeout():
                return await graph.ainvoke(state, combined_config)
                
            try:
                result = await run_with_timeout()
            except asyncio.TimeoutError:
                logger.warning(f"Research timed out after {args.timeout} seconds")
                save_partial_results(state, args, "timeout")
                return
        else:
            # Create tracker for max_iterations
            iteration_count = 0
            last_state = None
            last_log_time = time.time()
            
            # Run graph with streaming but enforce max_iterations
            astream = graph.astream(state, combined_config)
            try:
                async for step_output in astream:
                    # Update the last_state for each new step
                    last_state = step_output
                    
                    # Update memory tracking and elapsed time
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    # Log progress every 10 seconds
                    if iteration_count == 0 or (current_time - last_log_time >= 10):
                        last_log_time = current_time
                        logger.info(f"Iteration {iteration_count+1}/{args.max_iterations}, elapsed time: {elapsed_time:.1f}s/{max_runtime}s")
                        
                        # Check for iteration limit
                        if iteration_count >= args.max_iterations:
                            logger.warning(f"Research exceeded maximum iterations ({args.max_iterations}), forcing completion")
                            
                            # Mark the research as completed
                            if isinstance(last_state, dict):
                                last_state["completed_research"] = True
                                
                                # Ensure we have a running summary
                                if "running_summary" not in last_state or not last_state["running_summary"]:
                                    last_state["running_summary"] = f"Research on {args.query or args.query_flag} found data from multiple sources."
                                
                                # Define default summary values
                                jira_summary = "No Jira data found."
                                perforce_summary = "No Perforce data found."
                                confluence_summary = "No Confluence data found."
                                
                                # Extract summaries if available
                                if "jira_data" in last_state and isinstance(last_state["jira_data"], dict):
                                    if "summary" in last_state["jira_data"]:
                                        jira_summary = last_state["jira_data"]["summary"]
                                    elif "raw_results" in last_state["jira_data"] and last_state["jira_data"]["raw_results"]:
                                        jira_summary = f"Found {len(last_state['jira_data']['raw_results'])} Jira issues."
                                
                                if "perforce_data" in last_state and isinstance(last_state["perforce_data"], dict):
                                    if "summary" in last_state["perforce_data"]:
                                        perforce_summary = last_state["perforce_data"]["summary"]
                                    elif "raw_results" in last_state["perforce_data"] and last_state["perforce_data"]["raw_results"]:
                                        perforce_summary = f"Found {len(last_state['perforce_data']['raw_results'])} Perforce changelists."
                                
                                if "confluence_data" in last_state and isinstance(last_state["confluence_data"], dict):
                                    if "summary" in last_state["confluence_data"]:
                                        confluence_summary = last_state["confluence_data"]["summary"]
                                    elif "raw_results" in last_state["confluence_data"] and last_state["confluence_data"]["raw_results"]:
                                        confluence_summary = f"Found {len(last_state['confluence_data']['raw_results'])} Confluence pages."
                                
                                # Generate final summary
                                running_summary = last_state.get("running_summary", "")
                                final_summary = (
                                    f"Research on '{args.query or args.query_flag}' reached the maximum iterations limit. "
                                    f"Here's what was discovered:\n\n"
                                )
                                
                                final_summary += f"JIRA FINDINGS:\n{jira_summary}\n\n"
                                final_summary += f"PERFORCE CODE FINDINGS:\n{perforce_summary}\n\n" 
                                final_summary += f"CONFLUENCE DOCUMENTATION FINDINGS:\n{confluence_summary}\n\n"
                                
                                if running_summary:
                                    final_summary += f"OVERALL ANALYSIS:\n{running_summary}"
                                
                                # Update the state with the final summary
                                last_state["final_summary"] = final_summary
                            
                            # Update global trackers
                            if "perforce_data" in last_state and isinstance(last_state["perforce_data"], dict):
                                perforce_data = last_state["perforce_data"]
                                if "raw_results" in perforce_data:
                                    GLOBAL_PERFORCE_COUNT = max(GLOBAL_PERFORCE_COUNT, len(perforce_data["raw_results"]))

                            if "jira_data" in last_state and isinstance(last_state["jira_data"], dict):
                                jira_data = last_state["jira_data"]
                                if "raw_results" in jira_data:
                                    GLOBAL_JIRA_COUNT = max(GLOBAL_JIRA_COUNT, len(jira_data["raw_results"]))

                            if "confluence_data" in last_state and isinstance(last_state["confluence_data"], dict):
                                confluence_data = last_state["confluence_data"]
                                if "raw_results" in confluence_data:
                                    GLOBAL_CONFLUENCE_COUNT = max(GLOBAL_CONFLUENCE_COUNT, len(confluence_data["raw_results"]))
                            
                            # Save partial results
                            save_partial_results(last_state, args, "max_iterations_exceeded")
                            
                            # Break out of the loop
                            break
                    
                    # Increment iteration counter
                    iteration_count += 1
            finally:
                # Ensure the async generator is properly closed to avoid
                # "async generator ignored GeneratorExit" warnings.
                try:
                    await astream.aclose()
                except Exception as e:
                    logger.debug(f"Error closing async generator: {e}")
            
            # Use the last state as our result
            result = last_state
    
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        # Handle the savepoint
        save_partial_results(state, args, "interrupted")
        
    except Exception as e:
        logger.error(f"Error running research: {e}", exc_info=True)
        # Handle the savepoint if possible
        if state:
            save_partial_results(state, args, "error")
            
    # Save the final result
    if result:
        logger.info(
            f"RUN_GRAPH_3 (final save ENTER): Output File='{args.output or 'default'}', "
            f"GLOBAL_JIRA_COUNT={GLOBAL_JIRA_COUNT}, "
            f"GLOBAL_PERFORCE_COUNT={GLOBAL_PERFORCE_COUNT}, "
            f"GLOBAL_CONFLUENCE_COUNT={GLOBAL_CONFLUENCE_COUNT}"
        )
        try:
            # Check if result is a dictionary we can save
            if isinstance(result, dict):
                # Prioritize counts from the result state, fallback to global log-scraped counts
                final_jira_count = len(result.get("jira_data", {}).get("raw_results", []))
                logger.info(f"RUN_GRAPH_3 (final save): initial final_jira_count from result state: {final_jira_count}")
                if final_jira_count == 0 and GLOBAL_JIRA_COUNT > 0: # If state has none, but filter found some
                    logger.info(f"RUN_GRAPH_3 (final save): Jira count 0 in state, but GLOBAL_JIRA_COUNT is {GLOBAL_JIRA_COUNT}. Using global.")
                    final_jira_count = GLOBAL_JIRA_COUNT

                final_perforce_count = len(result.get("perforce_data", {}).get("raw_results", []))
                logger.info(f"RUN_GRAPH_3 (final save): initial final_perforce_count from result state: {final_perforce_count}")
                if final_perforce_count == 0 and GLOBAL_PERFORCE_COUNT > 0:
                    logger.info(f"RUN_GRAPH_3 (final save): Perforce count 0 in state, but GLOBAL_PERFORCE_COUNT is {GLOBAL_PERFORCE_COUNT}. Using global.")
                    final_perforce_count = GLOBAL_PERFORCE_COUNT

                final_confluence_count = len(result.get("confluence_data", {}).get("raw_results", []))
                logger.info(f"RUN_GRAPH_3 (final save): initial final_confluence_count from result state: {final_confluence_count}")
                if final_confluence_count == 0 and GLOBAL_CONFLUENCE_COUNT > 0:
                    logger.info(f"RUN_GRAPH_3 (final save): Confluence count 0 in state, but GLOBAL_CONFLUENCE_COUNT is {GLOBAL_CONFLUENCE_COUNT}. Using global.")
                    final_confluence_count = GLOBAL_CONFLUENCE_COUNT

                result["_data_counts"] = {
                    "jira_count": final_jira_count,
                    "perforce_count": final_perforce_count,
                    "confluence_count": final_confluence_count,
                    "total_count": final_jira_count + final_perforce_count + final_confluence_count,
                    "has_jira_data": final_jira_count > 0,
                    "has_perforce_data": final_perforce_count > 0,
                    "has_confluence_data": final_confluence_count > 0
                }
                
                # Ensure a final_summary exists
                if "final_summary" not in result or not result["final_summary"]:
                    running_summary = result.get("running_summary", "")
                    if running_summary:
                        result["final_summary"] = running_summary
                    else:
                        # Generate a basic summary using global counters
                        result["final_summary"] = (
                            f"Research on '{args.query or args.query_flag}' collected data from {final_jira_count} Jira issues, "
                            f"{final_perforce_count} Perforce changes, and {final_confluence_count} Confluence pages."
                        )
                
                # Mark as completed
                result["completed_research"] = True
                
                # Format messages if necessary
                if "messages" in result and isinstance(result["messages"], list):
                    # Always process with _serialize_messages if it's a list
                    # This will handle BaseMessage objects and pass through already serializable dicts.
                    print_content = None
                    if result["messages"]: # Check if list is not empty
                        last_msg = result["messages"][-1]
                        if hasattr(last_msg, 'content'):
                            print_content = last_msg.content
                        elif isinstance(last_msg, dict) and 'content' in last_msg:
                            print_content = last_msg['content']
                        
                    result["messages"] = _serialize_messages(result["messages"])
                    
                    if print_content: # Print after serialization if content was found
                        print("\n" + str(print_content)) 

                elif "final_summary" in result: # If no messages field but final_summary exists
                    result["messages"] = [{"type": "AIMessage", "content": result.get("final_summary")}]
                else: # Ensure messages field exists even if empty or final_summary is not there
                    result["messages"] = [{"type": "AIMessage", "content": "Research complete."}]

                # Save to file
                output_file = args.output or "graph_3_result.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=_json_fallback)
                
                logger.info(f"Results saved to {output_file}")
            else:
                logger.warning(f"Result is not a dictionary, cannot save to JSON: {type(result)}")
        except Exception as e:
            logger.error(f"Error saving result: {e}", exc_info=True)
    
    # Force garbage collection
    gc.collect()
    logger.info("Research completed.")
    
    return 0

# ---------------------------------------------------------------------
# Helper: make arbitrary objects JSON-serialisable for final output
# ---------------------------------------------------------------------
# Even after we normalise the top-level `messages` list, there may still be
# LangChain `BaseMessage` (e.g. `AIMessage`, `HumanMessage`) objects lurking in
# nested structures such as verification traces or tool outputs.  Python's
# `json` module will raise `TypeError: Object of type AIMessage is not JSON
# serializable` when it encounters these.  We register a lightweight fallback
# encoder that converts them (and any other unknown types) to a safe
# representation.

def _json_fallback(obj: Any):  # pragma: no cover – utility function
    """Fallback encoder for `json.dump` to handle LangChain message objects."""
    try:
        from langchain_core.messages import BaseMessage as _LCBaseMessage

        if isinstance(obj, _LCBaseMessage):
            return {
                "type": getattr(obj, "type", obj.__class__.__name__),
                "content": getattr(obj, "content", str(obj)),
            }
    except Exception:
        # Import failures or other edge cases fall through to generic handling
        pass

    # Generic fallback – stringify the object so we never fail serialisation
    try:
        return str(obj)
    except Exception:
        return f"<{obj.__class__.__name__} unserialisable>"

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
