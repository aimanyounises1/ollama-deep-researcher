"""Optimization utilities for LangGraph execution.
"""
import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Type variables for better type hints
T = TypeVar('T')
TState = TypeVar('TState', bound=Dict[str, Any])

# Global caches with appropriate TTL values
QUERY_CACHE = TTLCache(maxsize=100, ttl=3600)  # 1 hour
RETRIEVAL_CACHE = TTLCache(maxsize=50, ttl=1800)  # 30 minutes

def cached(cache: TTLCache, key_function: Optional[Callable] = None):
    """Cache decorator with optional key function for LangGraph nodes."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(state: TState, *args, **kwargs):
            # Generate cache key from state and/or args
            if key_function:
                key = key_function(state, *args, **kwargs)
            else:
                # Default: use query or research_topic from state
                if 'search_query' in state and state['search_query']:
                    key = f"sq:{state['search_query']}"
                elif 'research_topic' in state and state['research_topic']:
                    key = f"rt:{state['research_topic']}"
                else:
                    # Fall back to function name and arg length as a non-cachable marker
                    key = f"{func.__name__}:{len(args)}:{len(kwargs)}"
            
            # Check cache first
            if key in cache:
                logger.info(f"Cache hit for {func.__name__}, key: {str(key)[:30]}...")
                return cache[key]
            
            # Call the function if not in cache
            result = await func(state, *args, **kwargs)
            
            # Cache the result
            cache[key] = result
            return result
        return wrapper
    return decorator

def timed_execution(func):
    """Decorator to measure execution time of async LangGraph nodes."""
    @functools.wraps(func)
    async def wrapper(state: TState, *args, **kwargs):
        start_time = time.time()
        try:
            result = await func(state, *args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Node {func.__name__} executed in {execution_time:.2f}s")
            
            # Add timing to state metadata for monitoring
            metadata = state.setdefault('metadata', {})
            timings = metadata.setdefault('node_timings', {})
            timings[func.__name__] = execution_time
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Node {func.__name__} failed after {execution_time:.2f}s: {e}")
            
            # Record error in state metadata
            metadata = state.setdefault('metadata', {})
            errors = metadata.setdefault('errors', [])
            errors.append({
                'node': func.__name__,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': execution_time
            })
            
            raise
    return wrapper

async def run_in_parallel(funcs_with_states: List[Tuple[Callable, TState]]) -> Dict[str, Any]:
    """Run multiple state-updating functions in parallel and merge results.
    
    Args:
        funcs_with_states: List of (func, state) tuples to execute in parallel
        
    Returns:
        Merged state dictionary
    """
    async def run_func(func, state):
        try:
            return await func(state)
        except Exception as e:
            logger.error(f"Parallel execution of {func.__name__} failed: {e}")
            # Return original state on error
            return state
    
    # Create tasks
    tasks = [run_func(func, state) for func, state in funcs_with_states]
    
    # Run in parallel and collect results
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    # Merge all result states
    merged_state = {}
    for state_dict in results:
        if state_dict:
            merged_state.update(state_dict)
    
    return merged_state

def prune_state(state: TState) -> TState:
    """Prune state to prevent memory bloat during long runs."""
    # Make a working copy
    pruned_state = dict(state)
    
    # Truncate long lists to reasonable sizes
    if 'enterprise_research_results' in pruned_state and len(pruned_state['enterprise_research_results']) > 3:
        # Keep only the most recent 3 results
        pruned_state['enterprise_research_results'] = pruned_state['enterprise_research_results'][-3:]
    
    # Limit sources gathered to most recent or most relevant
    if 'sources_gathered' in pruned_state and len(pruned_state['sources_gathered']) > 10:
        # Strategy: keep the first (original query) and the most recent N-1
        first = pruned_state['sources_gathered'][0]
        recent = pruned_state['sources_gathered'][-9:]  # Keep 9 most recent
        pruned_state['sources_gathered'] = [first] + recent
    
    # Don't keep huge running summaries
    if 'running_summary' in pruned_state and isinstance(pruned_state['running_summary'], str):
        if len(pruned_state['running_summary']) > 10000:
            # Keep only the first 10K chars - this should be plenty for context
            pruned_state['running_summary'] = pruned_state['running_summary'][:10000]
    
    return pruned_state 