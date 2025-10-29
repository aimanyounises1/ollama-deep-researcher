import os
from typing import Any, Dict, List

import requests
from duckduckgo_search import DDGS
from langsmith import traceable
from tavily import TavilyClient


def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results):
    """Format search results into a bullet-point list of sources.
    
    Args:
        search_results (dict): Tavily search response containing results
        
    Returns:
        str: Formatted string with sources and their URLs
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path


@traceable
def duckduckgo_search(query: str, max_results: int = 5, fetch_full_page: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Search the web using DuckDuckGo with caching, improved error handling, and enhanced content retrieval.
    
    Args:
        query (str): The search query to execute
        max_results (int): Maximum number of results to return
        fetch_full_page (bool): Whether to fetch the full page content
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full page content if fetch_full_page is True, otherwise same as content
                - retrieved_at (str): ISO format timestamp of when the result was retrieved
    """
    # Setup caching
    cache_dir = Path.home() / ".ollama_researcher_cache"
    cache_dir.mkdir(exist_ok=True)
    
    cache_key = f"ddg_{hashlib.md5(query.encode()).hexdigest()}"
    cache_file = cache_dir / f"{cache_key}.json"
    
    # Try to get results from cache first (valid for 24 hours)
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text())
            cache_time = cached_data.get("cache_time", 0)
            if time.time() - cache_time < 86400:  # 24 hours
                print(f"Using cached results for query: {query}")
                return cached_data["result"]
        except Exception as e:
            print(f"Cache read error (continuing with live search): {str(e)}")
    
    # Function to save results to cache
    def cache_results(result_data):
        try:
            cache_file.write_text(json.dumps({
                "cache_time": time.time(),
                "result": result_data
            }))
        except Exception as e:
            print(f"Cache write error: {str(e)}")
    
    # Implement retry logic with exponential backoff
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with DDGS() as ddgs:
                results = []
                search_results = list(ddgs.text(query, max_results=max_results))
                
                # Break early if we got results
                if search_results:
                    break
                    
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"No results returned, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error in DuckDuckGo search after {max_retries} attempts: {str(e)}")
                print(f"Full error details: {type(e).__name__}")
                return {"results": []}
                
            wait_time = 2 ** retry_count
            print(f"Error in DuckDuckGo search, retrying in {wait_time} seconds: {str(e)}")
            time.sleep(wait_time)
    
    # Process results
    results = []
    unique_urls = set()  # For deduplication
    
    for r in search_results:
        url = r.get('href')
        title = r.get('title')
        content = r.get('body')
        
        # Skip incomplete or duplicate results
        if not all([url, title, content]) or url in unique_urls:
            continue
            
        unique_urls.add(url)
        raw_content = content
        
        if fetch_full_page:
            try:
                # Improved full page content retrieval with timeout and headers
                import urllib.request

                from bs4 import BeautifulSoup
                
                # Set up request with user agent and timeout
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                response = urllib.request.urlopen(req, timeout=10)
                html = response.read()
                
                # Use BeautifulSoup for better text extraction
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text and normalize whitespace
                raw_content = ' '.join(soup.get_text().split())
                
                # Limit content length to avoid excessively large responses
                if len(raw_content) > 15000:
                    raw_content = raw_content[:15000] + "... [content truncated]"
                    
            except Exception as e:
                print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
        
        # Add result to list with timestamp
        result = {
            "title": title,
            "url": url,
            "content": content,
            "raw_content": raw_content,
            "retrieved_at": datetime.now().isoformat()
        }
        results.append(result)
    
    # Cache and return results
    result_data = {"results": results}
    cache_results(result_data)
    return result_data

@traceable
def tavily_search(query, include_raw_content=True, max_results=5):
    """Search the web using the Tavily API with caching and enhanced error handling.
    
    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
                - retrieved_at (str): ISO format timestamp of when the result was retrieved
    """
    # Setup caching
    cache_dir = Path.home() / ".ollama_researcher_cache"
    cache_dir.mkdir(exist_ok=True)
    
    cache_key = f"tavily_{hashlib.md5(query.encode()).hexdigest()}"
    cache_file = cache_dir / f"{cache_key}.json"
    
    # Try to get results from cache first (valid for 24 hours)
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text())
            cache_time = cached_data.get("cache_time", 0)
            if time.time() - cache_time < 86400:  # 24 hours
                print(f"Using cached Tavily results for query: {query}")
                return cached_data["result"]
        except Exception as e:
            print(f"Tavily cache read error (continuing with live search): {str(e)}")
    
    # Function to save results to cache
    def cache_results(result_data):
        try:
            cache_file.write_text(json.dumps({
                "cache_time": time.time(),
                "result": result_data
            }))
        except Exception as e:
            print(f"Tavily cache write error: {str(e)}")
    
    # Implement retry logic with exponential backoff
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create client and execute search
            tavily_client = TavilyClient()
            result = tavily_client.search(query, 
                                max_results=max_results, 
                                include_raw_content=include_raw_content)
            
            # Add timestamps and enhance the results
            for item in result.get('results', []):
                item['retrieved_at'] = datetime.now().isoformat()
                
                # Truncate overly large raw_content to prevent token explosion
                if include_raw_content and 'raw_content' in item and item['raw_content']:
                    if len(item['raw_content']) > 15000:
                        item['raw_content'] = item['raw_content'][:15000] + "... [content truncated]"
            
            # Cache and return results
            cache_results(result)
            return result
                
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error in Tavily search after {max_retries} attempts: {str(e)}")
                return {"results": []}
                
            wait_time = 2 ** retry_count
            print(f"Error in Tavily search, retrying in {wait_time} seconds: {str(e)}")
            time.sleep(wait_time)

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using the Perplexity API with caching, error handling, and enhanced content processing.
    
    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int): The loop step for perplexity search (starts at 0)
  
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
                - retrieved_at (str): ISO format timestamp of when the result was retrieved
    """
    # Setup caching
    cache_dir = Path.home() / ".ollama_researcher_cache"
    cache_dir.mkdir(exist_ok=True)
    
    cache_key = f"perplexity_{hashlib.md5(query.encode()).hexdigest()}_{perplexity_search_loop_count}"
    cache_file = cache_dir / f"{cache_key}.json"
    
    # Try to get results from cache first (valid for 24 hours)
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text())
            cache_time = cached_data.get("cache_time", 0)
            if time.time() - cache_time < 86400:  # 24 hours
                print(f"Using cached Perplexity results for query: {query}")
                return cached_data["result"]
        except Exception as e:
            print(f"Perplexity cache read error (continuing with live search): {str(e)}")
    
    # Function to save results to cache
    def cache_results(result_data):
        try:
            cache_file.write_text(json.dumps({
                "cache_time": time.time(),
                "result": result_data
            }))
        except Exception as e:
            print(f"Perplexity cache write error: {str(e)}")

    # Get API key from environment
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        print("Error: PERPLEXITY_API_KEY not set in environment")
        return {"results": []}

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Enhanced system prompt for better results
    system_prompt = """
    Search the web and provide comprehensive, factual information with sources. 
    Focus on recent, high-quality information from authoritative sources.
    Structure your response clearly with relevant details and examples.
    Always cite your sources for every piece of information.
    """
    
    payload = {
        "model": "sonar-pro",  # Default to sonar-pro, could be configurable
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Perform a comprehensive search for: {query}"
            }
        ]
    }
    
    # Implement retry logic with exponential backoff
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # Add timeout to prevent hanging
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"Rate limited by Perplexity API. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            # Handle other errors
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Try to extract more detailed citations if available
            try:
                # Some versions of the API provide detailed metadata about citations
                detailed_citations = data.get("choices", [{}])[0].get("message", {}).get("metadata", {}).get("citations", [])
                if detailed_citations:
                    citations = [citation.get("url", "https://perplexity.ai") for citation in detailed_citations]
                else:
                    # Fall back to simple citations list
                    citations = data.get("citations", ["https://perplexity.ai"])
            except Exception as e:
                print(f"Error extracting detailed citations: {str(e)}")
                citations = data.get("citations", ["https://perplexity.ai"])
            
            # Add timestamp
            timestamp = datetime.now().isoformat()
            
            # Return first citation with full content, others just as references
            results = [{
                "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
                "url": citations[0],
                "content": content,
                "raw_content": content,
                "retrieved_at": timestamp
            }]
            
            # Add additional citations with their own titles and snippets if possible
            for i, citation in enumerate(citations[1:], start=2):
                # Try to extract title from URL
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(citation)
                    domain = parsed_url.netloc
                    path = parsed_url.path
                    title = f"{domain}{path}"
                except:
                    title = f"Source {i}"
                    
                results.append({
                    "title": f"Perplexity Search {perplexity_search_loop_count + 1}, {title}",
                    "url": citation,
                    "content": "Referenced in Perplexity search results",
                    "raw_content": None,
                    "retrieved_at": timestamp
                })
            
            result_data = {"results": results}
            # Cache the successful results
            cache_results(result_data)
            return result_data
                
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error in Perplexity search after {max_retries} attempts: {str(e)}")
                return {"results": []}
                
            wait_time = 2 ** retry_count
            print(f"Error in Perplexity search, retrying in {wait_time} seconds: {str(e)}")
            time.sleep(wait_time)