# tool_confluence.py
import json
import logging
import os
import re
import asyncio
import io
import mimetypes
import time
from typing import List, Dict, Any, Optional, Union

import requests
from dotenv import load_dotenv
from langchain.agents import Tool

from src.assistant.tools.SolutionBookTool import SolutionBookQuerier
from src.utils.content_cleaner import clean_confluence_content
from src.utils import proxy_helper
from src.assistant.utils.response_formatter import format_tool_response, format_error_response

logger = logging.getLogger(__name__)


async def search_confluence(query: str, space_key: Optional[str] = None, include_versions: bool = False, max_results: int = 10) -> Dict[str, Any]:
    """
    Search Confluence for pages and attachments related to the query.
    
    Args:
        query: Text to search for in Confluence
        space_key: Optional space key to restrict search
        include_versions: Whether to include version history (slower)
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results and metadata
    """
    try:
        # Environment variables should already be loaded in run_graph_3.py
        from src.assistant.utils.response_formatter import format_tool_response, format_error_response
        
        confluence_domain = os.getenv('CONFLUENCE_URL') or os.getenv('CONFLUENCE_DOMAIN')
        confluence_username = os.getenv('CONFLUENCE_USERNAME')
        confluence_token = os.getenv('CONFLUENCE_API_TOKEN')
        
        if not confluence_domain or not confluence_token or not confluence_username:
            message = "Confluence credentials missing: CONFLUENCE_URL, CONFLUENCE_USERNAME, or CONFLUENCE_API_TOKEN not found"
            logger.error(message)
            return format_error_response(ValueError(message), "confluence")
            
        # Add schema if missing
        if not confluence_domain.startswith(('http://', 'https://')):
            confluence_domain = f'https://{confluence_domain}'
            
        logger.info(f"Searching Confluence at: {confluence_domain}")
        
        # Build the search query URL
        search_url = f"{confluence_domain}/rest/api/content/search"
        
        # Build the query parameters
        params = {
            'cql': f'text ~ "{query}"' + (f' AND space="{space_key}"' if space_key else ''),
            'limit': max_results,
            'expand': 'body.storage,metadata.labels,history,version,ancestors'
        }
        
        auth = (confluence_username, confluence_token)
        
        # Make the search request
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params, auth=aiohttp.BasicAuth(confluence_username, confluence_token)) as response:
                if response.status != 200:
                    error_msg = f"Confluence search failed with status {response.status}"
                    logger.error(error_msg)
                    return format_error_response(ValueError(error_msg), "confluence")
                    
                result = await response.json()
                
        # Process the results
        pages = []
        for page in result.get('results', []):
            page_id = page.get('id')
            page_title = page.get('title', 'Untitled')
            page_type = page.get('type', 'unknown')
            
            # Process page content
            content = ''
            if 'body' in page and 'storage' in page['body'] and 'value' in page['body']['storage']:
                # Get HTML content
                html_content = page['body']['storage']['value']
                
                # Convert HTML to text
                content = _html_to_text(html_content)
                
            # Get page URL
            # Build canonical URL based on page type and ID
            if page_type == 'page':
                page_url = f"{confluence_domain}/pages/viewpage.action?pageId={page_id}"
            elif page_type == 'blogpost':
                page_url = f"{confluence_domain}/display/~{page.get('_links', {}).get('webui', '')}"
            else:
                page_url = f"{confluence_domain}/content/{page_id}"
                
            # Get ancestors (breadcrumb trail)
            ancestors = []
            for ancestor in page.get('ancestors', []):
                ancestors.append({
                    'id': ancestor.get('id'),
                    'title': ancestor.get('title', 'Untitled'),
                    'type': ancestor.get('type', 'unknown')
                })
                
            # Get labels/tags
            labels = []
            if 'metadata' in page and 'labels' in page['metadata'] and 'results' in page['metadata']['labels']:
                for label in page['metadata']['labels']['results']:
                    labels.append(label.get('name', ''))
                    
            # Get version information
            version_info = None
            if include_versions and 'version' in page:
                version_info = {
                    'number': page['version'].get('number', 1),
                    'by': page['version'].get('by', {}).get('displayName', 'Unknown'),
                    'when': page['version'].get('when', '')
                }
                
            # Get space information
            space_info = None
            if 'space' in page:
                space_info = {
                    'key': page['space'].get('key', ''),
                    'name': page['space'].get('name', 'Unknown Space')
                }
                
            # Compile page information
            page_info = {
                'id': page_id,
                'title': page_title,
                'type': page_type,
                'url': page_url,
                'content': content,
                'space': space_info,
                'ancestors': ancestors,
                'labels': labels
            }
            
            if version_info:
                page_info['version'] = version_info
                
            # If the page is a blogpost, try to get the author
            if page_type == 'blogpost' and 'history' in page and 'createdBy' in page['history']:
                page_info['author'] = page['history']['createdBy'].get('displayName', 'Unknown')
                
            # Add excerpt generation for search context
            excerpt = _generate_excerpt(content, query, 250)
            page_info['excerpt'] = excerpt
            
            # Add the page to our results
            pages.append(page_info)
            
        # Extract any MTV IDs found in the content
        mtv_ids = set()
        for page in pages:
            content = page.get('content', '')
            if content:
                # Find MTV IDs using regex pattern
                mtv_matches = re.finditer(r'MTV\d{4,}', content, re.IGNORECASE)
                for match in mtv_matches:
                    mtv_ids.add(match.group(0).upper())  # Store in uppercase
                    
        # Prepare the final result
        final_result = {
            'query': query,
            'space_key': space_key,
            'total_results': len(pages),
            'pages': pages,
            'extracted_mtv_ids': list(mtv_ids)
        }
        
        return format_tool_response(final_result, "confluence")
        
    except Exception as e:
        logger.error(f"Error searching Confluence: {e}", exc_info=True)
        return format_error_response(e, "confluence")


def extract_tables_from_html(html_content: str) -> List[Dict[str, Any]]:
    """
    Extract tables from HTML content and convert them to structured format.
    
    Args:
        html_content: HTML content containing tables
        
    Returns:
        List of extracted tables as dictionaries
    """
    if not html_content:
        return []
        
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            return []
            
        result = []
        
        for i, table in enumerate(tables):
            # Extract table headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                th_elements = header_row.find_all('th')
                headers = [th.get_text(strip=True) for th in th_elements]
            
            # If no headers found in thead, try first row of tbody
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    th_elements = first_row.find_all(['th', 'td'])
                    headers = [th.get_text(strip=True) for th in th_elements]
            
            # Extract rows
            rows = []
            for row in table.find_all('tr')[1:] if headers else table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    rows.append([cell.get_text(strip=True) for cell in cells])
            
            # Create structured table data
            table_data = {
                "table_index": i + 1,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "column_count": len(headers) if headers else (len(rows[0]) if rows else 0)
            }
            
            result.append(table_data)
        
        return result
    except Exception as e:
        logger.warning(f"Error extracting tables from HTML: {e}")
        return []


def _html_to_text(html_content: str) -> str:
    """
    Convert HTML content to plain text.
    
    Args:
        html_content: The HTML content to convert
        
    Returns:
        Plain text content with HTML tags removed
    """
    try:
        from bs4 import BeautifulSoup
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Handle whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except ImportError:
        # Fallback if BeautifulSoup is not available
        import re
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def _generate_excerpt(content: str, query: str, max_length: int = 250) -> str:
    """
    Generate an excerpt from the content focused around the query terms.
    
    Args:
        content: The content to extract from
        query: The search query to focus on
        max_length: Maximum length of the excerpt
        
    Returns:
        An excerpt centered around the query keywords
    """
    if not content or not query:
        return ""
        
    # Tokenize query into keywords
    keywords = set(re.findall(r'\w+', query.lower()))
    
    # Find the best paragraph that contains the most keywords
    paragraphs = content.split('\n')
    best_paragraph = ""
    best_score = 0
    
    for para in paragraphs:
        if len(para) < 20:  # Skip very short paragraphs
            continue
            
        para_lower = para.lower()
        score = sum(1 for keyword in keywords if keyword in para_lower)
        
        if score > best_score:
            best_score = score
            best_paragraph = para
            
    # If no good paragraph found, just use the beginning
    if not best_paragraph and paragraphs:
        best_paragraph = paragraphs[0]
        
    # If the paragraph is too long, extract a window around the keywords
    if len(best_paragraph) > max_length:
        # Find the position of the first keyword
        first_pos = float('inf')
        for keyword in keywords:
            pos = best_paragraph.lower().find(keyword)
            if pos != -1 and pos < first_pos:
                first_pos = pos
                
        if first_pos == float('inf'):
            # No keyword found, use the beginning
            excerpt = best_paragraph[:max_length] + "..."
        else:
            # Create a window around the first keyword
            start = max(0, first_pos - max_length // 2)
            end = min(len(best_paragraph), start + max_length)
            excerpt = ("..." if start > 0 else "") + best_paragraph[start:end] + ("..." if end < len(best_paragraph) else "")
    else:
        excerpt = best_paragraph
        
    return excerpt

# The actual 'Tool' object that you pass to the agent:
confluence_tool = Tool(
    name="confluence_search",
    func=search_confluence,
    description="""Use this tool to search internal Confluence-based documentation (SolutionBook).
    Useful for finding technical documentation, MTV references, project details, and attached files.
    Input should be a search query string.
    
    This tool can retrieve and extract content from various attachment types including:
    - PDF documents
    - Microsoft Office files (Word, Excel, PowerPoint)
    - Images (with OCR text extraction)
    - CSV/JSON data files
    
    For comprehensive results, you can use these parameters:
    - include_attachments=True (default): Retrieves and processes file attachments
    - include_attachments=False: Retrieves only page content without attachments
    - max_results=N: Controls the maximum number of results returned (default is 5)
    - space_key="SPACE": Limits search to a specific Confluence space
    - content_type="page"|"blogpost"|"comment"|"all": Filters by content type
    - extract_tables=True: Extracts tables from HTML content
    
    Example usage: "Technical implementation guide", space_key="DEV", include_attachments=True, extract_tables=True
    """
)
