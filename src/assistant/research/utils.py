"""
Utility functions for the research package.
"""

import logging
import re
import requests
import ssl
import os
from typing import List, Tuple, Dict, Any
from urllib3.exceptions import InsecureRequestWarning
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.utils.ssl_fix import with_ssl_disabled
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG', 'false').lower() == 'true' else logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
OLLAMA_BASE_URL = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
if not OLLAMA_BASE_URL.endswith('/'):
    OLLAMA_BASE_URL += '/'

def check_ollama_server(url=OLLAMA_BASE_URL):
    """Check if Ollama server is running and accessible."""
    try:
        session = requests.Session()
        session.verify = False
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        
        response = session.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model.get("name") for model in response.json().get("models", [])]
            logger.info(f"Ollama server is running at {url}. Available models: {models}")
            return True
        else:
            logger.warning(f"Ollama server is accessible but returned status code {response.status_code}")
            return False
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logger.warning(f"Ollama server not accessible at {url}")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama server: {e}")
        return False

async def async_input(prompt: str) -> str:
    """Safely get input in asyncio context."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))

def sanitize_query_for_jira(query: str) -> str:
    """Sanitize a query for JIRA search to avoid special character issues."""
    sanitized = re.sub(r'[+\-&|!(){}[\]^~*?:\\]', ' ', query)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized if sanitized else "information"

def report_deduplication(result_text: str, unique_parts: List[str]) -> None:
    """Report statistics about content deduplication."""
    if not result_text or not unique_parts:
        return

    try:
        orig_length = len(result_text.split())
        dedup_length = sum(len(part.split()) for part in unique_parts)
        
        if orig_length > 0:
            unique_percent = (dedup_length / orig_length) * 100
            removed_content = orig_length - dedup_length
            if removed_content > 10:
                logger.info(f"Content deduplication removed {removed_content} words (~{100-unique_percent:.1f}%)")
    except Exception as e:
        logger.error(f"Error reporting deduplication stats: {e}")

def extract_confluence_content(text: str) -> Tuple[str, str, bool]:
    """Extract and split Confluence content if it exceeds the size limit."""
    CHUNK_SIZE = 20000
    
    if not text or len(text) <= CHUNK_SIZE:
        return text, "", False
        
    # Find a good splitting point
    split_point = text.rfind("\n", 0, CHUNK_SIZE)
    if split_point == -1:
        split_point = CHUNK_SIZE
        
    return text[:split_point], text[split_point:], True

def deduplicate_lines(existing_content: List[str], new_content: str) -> str:
    """Remove duplicate lines from new content based on existing content."""
    if not new_content:
        return ""
        
    new_lines = new_content.split("\n")
    unique_lines = []
    
    for line in new_lines:
        if line.strip() and line not in existing_content:
            unique_lines.append(line)
            
    return "\n".join(unique_lines)

def sanitize_markdown_links(text: str) -> str:
    """Clean and validate markdown links in text."""
    def _clean_url(raw_url: str) -> str:
        """Clean and validate a URL."""
        try:
            url = raw_url.strip()
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'
            return url
        except Exception:
            return '#'

    def _replacer(match: re.Match) -> str:
        """Replace matched markdown links with cleaned versions."""
        text = match.group(1)
        url = _clean_url(match.group(2))
        return f'[{text}]({url})'

    # Replace markdown links with cleaned versions
    pattern = r'\[(.*?)\]\((.*?)\)'
    return re.sub(pattern, _replacer, text)

def clean_llm_output(text: str) -> str:
    """Clean and format LLM output."""
    if not text:
        return ""
        
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove markdown code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove inline code
    text = re.sub(r'`.*?`', '', text)
    
    # Clean markdown links
    text = sanitize_markdown_links(text)
    
    return text 