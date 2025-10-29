# src/assistant/utils/langgraph_helpers.py
"""
Helper utilities for working with LangGraph and processing research data
"""

import asyncio
import logging
import re
import inspect
from typing import List, Dict, Any, Callable, Awaitable, Optional, Union, TypeVar

# Setup logging
logger = logging.getLogger(__name__)

# Type variables for better type hinting
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def traceable(func: Optional[F] = None, **kwargs) -> F:
    """
    Simplified decorator that replaces LangSmith's traceable.
    Simply passes through the function without actual tracing to avoid
    dependencies and token limit issues.
    """
    if func is None:
        return lambda f: f  # type: ignore

    async def wrapper(*args, **kwargs):
        # Execute the function normally without tracing
        if inspect.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return result

    # For non-async functions, provide a sync version
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Return appropriate wrapper based on function type
    if inspect.iscoroutinefunction(func):
        return wrapper  # type: ignore
    return sync_wrapper  # type: ignore

async def process_in_batches(
    items: List[T], 
    batch_size: int, 
    processor_func: Callable[[List[T]], Awaitable[Any]]
) -> List[Any]:
    """
    Process a list of items in batches using an async processor function.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        processor_func: Async function to process each batch
        
    Returns:
        List of results from processing each batch
    """
    if not items:
        return []
        
    # Create batches
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    logger.info(f"Processing {len(items)} items in {len(batches)} batches of size {batch_size}")
    
    # Process each batch
    results = []
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")
        batch_result = await processor_func(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
        
    return results

async def map_reduce_text(
    texts: List[str],
    map_func: Callable[[str], Awaitable[str]],
    reduce_func: Callable[[List[str]], Awaitable[str]],
    max_batch_tokens: int = 4000,
    token_estimator: Callable[[str], int] = lambda x: len(x) // 4
) -> str:
    """
    Process a list of texts using the map-reduce pattern.
    
    Args:
        texts: List of text documents to process
        map_func: Function to apply to each text document
        reduce_func: Function to reduce/combine mapped results
        max_batch_tokens: Maximum tokens per batch
        token_estimator: Function to estimate tokens in a text
        
    Returns:
        Final reduced result
    """
    if not texts:
        return ""
    
    # Apply map function to each text
    mapped_results = []
    for text in texts:
        try:
            result = await map_func(text)
            mapped_results.append(result)
        except Exception as e:
            logger.error(f"Error mapping text: {e}")
    
    # If only one result, no need to reduce
    if len(mapped_results) == 1:
        return mapped_results[0]
    
    # Split into batches based on token count
    batches = []
    current_batch = []
    current_tokens = 0
    
    for result in mapped_results:
        result_tokens = token_estimator(result)
        if current_tokens + result_tokens > max_batch_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [result]
            current_tokens = result_tokens
        else:
            current_batch.append(result)
            current_tokens += result_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    # Process batches recursively
    if len(batches) == 1:
        # Only one batch, reduce directly
        return await reduce_func(batches[0])
    else:
        # Multiple batches, reduce each batch then combine
        batch_results = []
        for batch in batches:
            result = await reduce_func(batch)
            batch_results.append(result)
        
        # Final reduction of batch results
        return await reduce_func(batch_results)

def extract_identifiers(text: str) -> Dict[str, List[str]]:
    """
    Extract various identifiers from text (Jira keys, CL numbers, etc.)
    
    Args:
        text: Text to extract identifiers from
        
    Returns:
        Dictionary of identifier types to lists of identifiers
    """
    if not text or not isinstance(text, str):
        return {"jira_keys": [], "cls": [], "mtv_ids": []}
    
    # Extract Jira keys (e.g., PROJ-123)
    jira_pattern = r'[A-Z]+-\d+'
    jira_keys = re.findall(jira_pattern, text)
    
    # Extract CL numbers (various formats)
    cl_patterns = [
        r'CL\s*(\d+)',  # CL 12345
        r'Change\s*(?:List)?\s*(\d+)',  # Change List 12345
        r'changelist\s*(\d+)'  # changelist 12345
    ]
    
    cls = []
    for pattern in cl_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        cls.extend(matches)
    
    # Extract MTV IDs (e.g., MTV-123 or MTV123)
    mtv_pattern = r'MTV[-_]?\d+'
    mtv_ids = re.findall(mtv_pattern, text, flags=re.IGNORECASE)
    
    return {
        "jira_keys": jira_keys,
        "cls": cls,
        "mtv_ids": mtv_ids
    }

def clean_llm_output(text: str) -> str:
    """
    Clean up LLM output by removing common artifacts.
    
    Args:
        text: Raw LLM output text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove markdown code block markers if present
    if text.startswith("```") and text.endswith("```"):
        # Check if it has a language specifier
        lines = text.split("\n", 1)
        if len(lines) > 1:
            first_line = lines[0].strip()
            if first_line.startswith("```") and len(first_line) > 3:
                # Language specified, remove first and last lines
                inner_text = lines[1]
                if inner_text.endswith("```"):
                    inner_text = inner_text[:-3].rstrip()
                return inner_text
    
    # Remove common prefixes
    prefixes = [
        "Here's the summary:",
        "Here is the summary:",
        "Summary:",
        "Based on the provided information,"
    ]
    
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    
    return text

class MemoryOptimizer:
    """
    Utility for optimizing memory usage during processing
    """
    @staticmethod
    def truncate_large_content(content: Dict[str, Any], max_size: int = 100000) -> Dict[str, Any]:
        """
        Truncate large string values in a dictionary to save memory
        
        Args:
            content: Dictionary with potentially large string values
            max_size: Maximum string length to allow
            
        Returns:
            Dictionary with truncated string values
        """
        if not isinstance(content, dict):
            return content
            
        result = {}
        for key, value in content.items():
            if isinstance(value, str) and len(value) > max_size:
                result[key] = value[:max_size] + "... [TRUNCATED]"
            elif isinstance(value, dict):
                result[key] = MemoryOptimizer.truncate_large_content(value, max_size)
            elif isinstance(value, list):
                result[key] = [
                    MemoryOptimizer.truncate_large_content(item, max_size) if isinstance(item, dict)
                    else (item[:max_size] + "... [TRUNCATED]" if isinstance(item, str) and len(item) > max_size else item)
                    for item in value
                ]
            else:
                result[key] = value
                
        return result
        
    @staticmethod
    def optimize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize memory usage in the state dictionary
        
        Args:
            state: State dictionary to optimize
            
        Returns:
            Optimized state dictionary
        """
        # Apply targeted optimizations based on known state structure
        if "raw_jira_results" in state and len(state["raw_jira_results"]) > 20:
            # Keep only essential fields from Jira results
            raw_jira = state["raw_jira_results"]
            optimized_jira = []
            
            for item in raw_jira:
                if isinstance(item, dict):
                    essential_fields = {
                        "key": item.get("key", ""),
                        "summary": item.get("summary", "") or 
                                  item.get("fields", {}).get("summary", ""),
                        "status": item.get("status", "") or 
                                 item.get("fields", {}).get("status", {}).get("name", "")
                    }
                    optimized_jira.append(essential_fields)
            
            state["raw_jira_results"] = optimized_jira
            
        # Add similar optimizations for other data sources
        
        # Apply general content truncation
        return MemoryOptimizer.truncate_large_content(state)