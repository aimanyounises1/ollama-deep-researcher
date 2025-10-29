# src/assistant/utils/helpers.py

import re
import inspect
import logging
import urllib.parse
from typing import Callable, Any, Dict, TypeVar, Optional
from urllib.parse import urlparse

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

def clean_llm_output(text: str) -> str:
    """Clean up special tags and formatting issues in LLM output"""
    if not isinstance(text, str): 
        return ""  # Handle non-string input
    
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

        # Strip enclosing triple-backtick fences if entire content is fenced
        if cleaned_text.startswith('```') and cleaned_text.rstrip().endswith('```'):
            # Remove the first line (``` or ```markdown) and the last line (```)
            content_lines = cleaned_text.splitlines()
            if len(content_lines) >= 2:
                # Drop first and last line
                cleaned_text = '\n'.join(content_lines[1:-1]).strip()

        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error cleaning LLM output: {e}")
        return text  # Return original on error

def sanitize_markdown_links(text: str) -> str:
    """Clean common formatting issues in markdown links produced by LLMs."""
    if not isinstance(text, str): 
        return ""  # Handle non-string input
        
    # Regex to find markdown links, ensuring it doesn't capture nested brackets incorrectly
    link_pattern = re.compile(r"\[([^\]]+?)\]\(([^)\s][^)]*)\)")

    def _clean_url(raw_url: str) -> str:
        # Trim leading/trailing whitespace and common stray characters
        cleaned = raw_url.strip().strip("'\" .,;:\\")
        # Remove potential leading/trailing brackets if they leaked in
        cleaned = cleaned.strip('()')

        # URL Encode parentheses within the path
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
        except ValueError:
            logger.warning(f"Could not parse URL for encoding: {cleaned}")

        # Re-check http/https prefix if missing after potential encoding
        if not cleaned.startswith(('http://', 'https://', '#', '/')):
            # If it looks like a domain, assume https
            if '.' in cleaned and '/' not in cleaned.split('.')[0]:
                cleaned = 'https://' + cleaned
        return cleaned

    def _replacer(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        cleaned_url = _clean_url(url)
        # Ensure label doesn't contain nested markdown links
        clean_label = re.sub(r'\[[^\]]*\]\([^)]*\)', '', label).strip()
        if not clean_label: 
            clean_label = "link"  # Use placeholder if label becomes empty
        return f"[{clean_label}]({cleaned_url})"

    try:
        return link_pattern.sub(_replacer, text)
    except Exception as e:
        logger.error(f"Error sanitizing markdown links: {e}")
        return text  # Return original on error

def ensure_index(value: int) -> int:
    """Helper function to ensure an integer value is properly cast"""
    return value

def hash_content(content: str) -> int:
    """Generate a consistent hash for content to enable deduplication"""
    return hash(content)

def extract_identifiers(text: str) -> Dict[str, list]:
    """Extract different types of identifiers from text (Jira keys, MTV IDs, etc.)"""
    if not isinstance(text, str):
        return {"jira_keys": [], "mtv_ids": []}
        
    # Extract Jira keys (e.g., VIT-1234)
    jira_pattern = re.compile(r'\b([A-Z]+-\d+)\b')
    jira_matches = jira_pattern.findall(text)

    # Extract MTV IDs (e.g., MTV1234, MTV-1234)
    mtv_pattern = re.compile(r'\b(MTV[-\s]?\d{4,})\b', re.IGNORECASE)
    mtv_matches = [m.upper().replace(' ', '').replace('-', '') for m in mtv_pattern.findall(text)]
    
    # Extract CL numbers (e.g., CL 12345)
    cl_pattern = re.compile(r'\bCL\s+(\d+)\b')
    cl_matches = cl_pattern.findall(text)
    
    return {
        "jira_keys": jira_matches,
        "mtv_ids": mtv_matches,
        "cls": cl_matches
    }