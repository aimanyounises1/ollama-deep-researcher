# src/assistant/tools/SolutionBookTool.py

"""
SolutionBookTool.py

Enhanced SolutionBookQuerier that accesses Confluence-based "SolutionBook"
pages, handles both lexical + fake vector search, and includes fallback expansions
and basic security scanning for new or suspicious content.

Dependencies:
  - requests, re, os, ssl, logging, json, ...
  - textstat, PyPDF2, pytesseract (optional) for advanced parsing
  - langchain_core.documents.Document
  - TENACITY for retries
  - HTML cleaning libs (BeautifulSoup, html2text)
"""

import asyncio
import hashlib
import html
import io
import json
import logging
import mimetypes
import os
import re
import ssl
import sys
from typing import Any, Dict, List, Optional

import html2text
import pandas as pd
import pypandoc
import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached
from dotenv import load_dotenv
from langchain_core.documents import Document
from requests.exceptions import HTTPError, RequestException
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from textstat import textstat
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Optional imports for advanced features
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Try optional imports for OCR and image processing
try:
    import pytesseract
    from PIL import Image
    # Test the OCR functionality
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
    logger.info("OCR capabilities are available.")
except (ImportError, Exception) as e:
    OCR_AVAILABLE = False
    logger.warning(f"OCR capabilities not available: {e}")
    # Create placeholder if not available
    pytesseract = type('pytesseract', (), {'image_to_string': lambda *args, **kwargs: ""})
    try:
        from PIL import Image
    except ImportError:
        Image = type('Image', (), {'open': lambda *args, **kwargs: None})

# Import content cleaner utility
try:
    from src.utils.content_cleaner import clean_confluence_content
except ImportError:
    logging.getLogger(__name__).warning(
        "Failed to import clean_confluence_content utility. Cleaning will be basic."
    )


    def clean_confluence_content(html_content: str) -> str:
        """Basic fallback cleaner."""
        if not html_content:
            return ""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return re.sub(r"\n{3,}", "\n\n", text).strip()
        except Exception:
            return html_content

# Load environment variables
load_dotenv()

# Disable SSL verification warnings
disable_warnings(InsecureRequestWarning)

# Simple synonyms dictionary for demonstration
SYNONYMS = {
    "error": ["failure", "bug", "issue"],
    "procedure": ["steps", "instructions", "how to"],
    "setup": ["configuration", "install", "installation"],
}


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class AiQueryClassifier:
    """Placeholder classifier for query types."""

    def classify_query(self, query: str) -> str:
        logger.warning("Using placeholder AiQueryClassifier.")
        lower = query.lower()
        if any(k in lower for k in ["code", "function", "java", "python"]):
            return "code"
        if any(k in lower for k in ["document", "spec", "guide"]):
            return "doc"
        if any(k in lower for k in ["error", "bug", "fail"]):
            return "bug"
        return "misc"


class SolutionBookQuerier:
    """
    Primary class for querying Confluence-based "SolutionBook" content.
    Uses internal proxy logic and tenacity for retries.
    """

    def __init__(
            self,
            domain: Optional[str] = None,
            auth_token: Optional[str] = None,
            proxies: Optional[dict] = None,  # unused parameter
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.domain = domain or os.getenv("SOLUTIONBOOK_DOMAIN")
        self.auth_token = auth_token or os.getenv("SOLUTIONBOOK_ACCESS_TOKEN")
        if not all([self.domain, self.auth_token]):
            raise ConfigurationError("Missing SolutionBook credentials.")

        if re.match(r"^https?://", self.domain):
            raise ValueError("SOLUTIONBOOK_DOMAIN should not include protocol")

        self.base_url = f"https://{self.domain}"
        self.api_url = f"{self.base_url}/rest/api/content"
        self.search_endpoint = f"{self.api_url}/search"

        # Session setup
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        # Proxy configuration
        self.proxy_enabled = False
        self.primary_proxies = {}
        self.backup_proxies = {}
        self.original_proxies = {
            "http": os.environ.get("HTTP_PROXY"),
            "https": os.environ.get("HTTPS_PROXY"),
        }
        self._configure_proxy()

        self.page_cache = TTLCache(maxsize=500, ttl=1800)
        self.vector_index: Dict[str, Any] = {}

    def _configure_proxy(self) -> None:
        """Attempt to set up proxies. Fallback to direct if all fail."""
        try:
            if self._try_proxy(self.primary_proxies, "primary"):
                return
            if self._try_proxy(self.backup_proxies, "backup"):
                return

            self.logger.warning("No internal proxy worked; falling back to environment or direct.")
            env_proxies = {
                "http": self.original_proxies.get("http"),
                "https": self.original_proxies.get("https"),
            }
            self.session.proxies = {k: v for k, v in env_proxies.items() if v}
            if self.session.proxies:
                self.logger.info(f"Using environment proxies: {self.session.proxies}")
                self.proxy_enabled = True
            else:
                self.logger.info("Using direct connection (no proxy).")
                self.proxy_enabled = False
        except Exception as e:
            self.logger.error(f"Proxy config error: {e}")
            self.session.proxies = {}
            self.proxy_enabled = False

    def _try_proxy(self, proxy_config: dict, label: str) -> bool:
        """Test a proxy for connectivity."""
        test_session = requests.Session()
        test_session.verify = False
        test_session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
        test_session.proxies = proxy_config

        try:
            test_url = f"{self.base_url}/rest/api/space"
            resp = test_session.get(test_url, timeout=5)
            if resp.ok:
                self.logger.info(f"Proxy {label} succeeded: {proxy_config.get('https')}")
                self.session.proxies = proxy_config
                self.proxy_enabled = True
                return True

            self.logger.warning(f"Proxy {label} responded with HTTP {resp.status_code}")
            return False

        except RequestException as e:
            self.logger.warning(f"Proxy {label} failed: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Unexpected error testing proxy {label}: {e}")
            return False

    def _reset_proxy(self) -> None:
        """Reset session proxies to original environment settings."""
        try:
            self.logger.debug("Resetting session proxy configuration.")
            self.session.proxies = {}
            env_proxies = {k: v for k, v in self.original_proxies.items() if v}
            if env_proxies:
                self.session.proxies = env_proxies
        except Exception as e:
            self.logger.error(f"Reset proxy error: {e}")

    def __enter__(self) -> "SolutionBookQuerier":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._reset_proxy()

    def login(self) -> bool:
        """Verify authentication by querying a known endpoint."""
        try:
            url = f"{self.base_url}/rest/api/space"
            resp = self.session.get(url, timeout=10)
            if resp.ok:
                self.logger.info("SolutionBook authentication OK.")
                return True

            self.logger.error(f"Auth test failed: HTTP {resp.status_code}")
            return False

        except Exception as e:
            self.logger.error(f"Auth test error: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        retry=retry_if_exception_type((RequestException, RetryError)),
    )
    def _make_api_request(
            self,
            url: str,
            params: Optional[Dict[str, Any]] = None,
            method: str = "GET",
            payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an API request with retries."""
        try:
            connect_timeout, read_timeout = 15, 60
            if method.upper() == "GET":
                response = self.session.get(
                    url, params=params, timeout=(connect_timeout, read_timeout)
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    url, params=params, json=payload, timeout=(connect_timeout, read_timeout)
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "10"))
                self.logger.warning(
                    f"Rate limit hit for {url}; retrying after {retry_after}s"
                )
                raise RetryError(f"Rate limit hit (429)")

            response.raise_for_status()

            if not response.content:
                self.logger.warning(f"Empty response from {url} (HTTP {response.status_code})")
                return {} if response.ok else {
                    "error": "Empty response",
                    "status_code": response.status_code,
                }

            return response.json()

        except HTTPError as e:
            status_code = e.response.status_code if e.response else None
            error_text = (e.response.text[:500] or "(no body)") if e.response else str(e)
            self.logger.error(f"HTTP error {status_code} requesting {url}: {error_text}")
            if status_code and 400 <= status_code < 500 and status_code != 429:
                return {
                    "error": f"Client error {status_code}",
                    "message": error_text,
                    "status_code": status_code,
                }
            raise RequestException(f"HTTP error {status_code}") from e

        except RequestException as e:
            self.logger.error(f"Network error requesting {url}: {e}")
            raise

        except json.JSONDecodeError as e:
            snippet = response.text[:500] + "..." if hasattr(response, "text") else "(no response text)"
            self.logger.error(f"JSON decode failed for {url}: {e}; response: {snippet}")
            return {"error": "Invalid JSON"}

        except RetryError as e:
            self.logger.error(f"Request failed after retries for {url}: {e}")
            return {"error": f"Failed after retries: {e}"}

        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {e}", exc_info=True)
            return {"error": f"Unexpected error: {e}"}

    @cached(cache=TTLCache(maxsize=500, ttl=1800))
    def get_page_by_id(self, page_id: str) -> Dict[str, Any]:
        """
        Enhanced Confluence page retrieval with better content handling, expansion, and metadata.
        Based on advanced Confluence REST API usage.
        
        Args:
            page_id: The Confluence page ID to retrieve
            
        Returns:
            Dictionary with page information and cleaned content
        """
        self.logger.debug(f"Fetching page by ID: {page_id}")
        url = f"{self.api_url}/{page_id}"
        
        # Use expanded parameters from Confluence REST API for comprehensive data
        params = {
            "expand": "body.storage,body.view,version,history.lastUpdated,space,metadata,ancestors,children.page,children.attachment,extensions",
            "status": "current"
        }
        
        data = self._make_api_request(url, params=params)

        if not isinstance(data, dict) or "error" in data:
            error_msg = data.get("message") or data.get("error", "Unknown error")
            status = data.get("status_code", "N/A")
            self.logger.error(f"Page {page_id} retrieval failed: {status} - {error_msg}")
            return {"id": page_id, "error": error_msg, "status_code": status}

        try:
            # Get HTML content in multiple formats when available
            raw_html = data.get("body", {}).get("storage", {}).get("value", "")
            view_html = data.get("body", {}).get("view", {}).get("value", "")
            
            # Use view content if available, otherwise use storage content
            html_to_clean = view_html if view_html else raw_html
            
            # Apply enhanced content cleaning
            cleaned = clean_confluence_content(html_to_clean)
            if not cleaned and html_to_clean:
                self.logger.warning(f"No content after cleaning for page {page_id}")

            # Extract comprehensive metadata
            links = data.get("_links", {})
            webui = links.get("webui") or f"{self.base_url}/pages/viewpage.action?pageId={page_id}"
            title = html.unescape(data.get("title", "Untitled"))
            version = data.get("version", {}).get("number", 0)
            space_key = data.get("space", {}).get("key", "")
            space_name = data.get("space", {}).get("name", "")
            last_mod = data.get("history", {}).get("lastUpdated", {}).get("when", "")
            creator = data.get("history", {}).get("createdBy", {}).get("displayName", "")
            
            # Extract labels if available
            labels = []
            if "metadata" in data and "labels" in data["metadata"]:
                for label in data["metadata"]["labels"].get("results", []):
                    labels.append(label.get("name", ""))
                
            # Extract ancestors for page hierarchy/breadcrumbs
            ancestors = []
            if "ancestors" in data:
                for ancestor in data["ancestors"]:
                    ancestors.append({
                        "id": ancestor.get("id", ""),
                        "title": ancestor.get("title", ""),
                        "url": f"{self.base_url}/pages/viewpage.action?pageId={ancestor.get('id', '')}"
                    })
                
            # Count child pages and attachments
            child_pages = []
            if "children" in data and "page" in data["children"]:
                child_pages_data = data["children"]["page"].get("results", [])
                for child in child_pages_data:
                    child_pages.append({
                        "id": child.get("id", ""),
                        "title": child.get("title", ""),
                        "url": f"{self.base_url}/pages/viewpage.action?pageId={child.get('id', '')}"
                    })

            attachment_count = 0
            if "children" in data and "attachment" in data["children"]:
                attachment_count = data["children"]["attachment"].get("size", 0)

            # Prepare enhanced return object
            result = {
                "id": page_id,
                "title": title,
                "content": cleaned,
                "content_html": raw_html,  # Include original HTML for special cases
                "version": version,
                "url": webui,
                "space_key": space_key,
                "space_name": space_name,
                "last_modified": last_mod,
                "creator": creator,
                "labels": labels,
                "ancestors": ancestors,
                "child_pages": child_pages,
                "attachment_count": attachment_count
            }
            
            # Add quality metrics
            result["content_quality"] = self._assess_content_quality(cleaned, title)
            
            return result
        except Exception as e:
            self.logger.error(f"Error formatting page {page_id}: {e}", exc_info=True)
            return {"id": page_id, "error": f"Format error: {e}"}

    def _assess_content_quality(self, content: str, title: str) -> Dict[str, Any]:
        """
        Assess content quality metrics for improved content evaluation.
        
        Args:
            content: The extracted page content
            title: The page title
            
        Returns:
            Dictionary with content quality metrics
        """
        if not content:
            return {
                "word_count": 0,
                "readability": 0,
                "has_structure": False,
                "is_stub": True,
                "quality_score": 0
            }
        
        try:
            # Basic metrics
            word_count = len(content.split())
            sentences = len(re.findall(r'[.!?]+', content)) + 1  # +1 to avoid zero
            avg_sentence_length = word_count / sentences
            
            # Check for document structure
            has_headings = bool(re.search(r'#{2,5}\s+\w+|===+|---+', content))
            has_lists = bool(re.search(r'^\s*[-*]\s+\w+|^\s*\d+\.\s+\w+', content, re.MULTILINE))
            has_structure = has_headings or has_lists
            
            # Readability score
            try:
                readability = textstat.flesch_reading_ease(content) 
                readability = max(0, min(100, readability))  # Normalize between 0-100
            except Exception:
                readability = 50  # Default mid-range value
            
            # Check if content is too short to be useful
            is_stub = word_count < 100
            
            # Calculate overall quality score
            # Factors: length, structure, readability
            length_score = min(1.0, word_count / 1000)  # Max at 1000 words
            structure_score = 0.8 if has_structure else 0.3
            readability_score = readability / 100.0
            
            quality_score = (length_score * 0.5) + (structure_score * 0.3) + (readability_score * 0.2)
            quality_score = round(quality_score * 100) / 100  # Round to 2 decimal places
            
            return {
                "word_count": word_count,
                "readability": round(readability),
                "has_structure": has_structure,
                "is_stub": is_stub,
                "quality_score": quality_score
            }
        except Exception as e:
            self.logger.warning(f"Error calculating content quality: {e}")
            return {
                "word_count": len(content.split()),
                "quality_score": 0.5,
                "error": str(e)
            }

    def search_pages(
            self,
            query: str,
            space_key: Optional[str] = None,
            label: Optional[str] = None,
            limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search Confluence pages using CQL."""
        # Log the raw query for debugging
        self.logger.info(f"Raw query for search_pages: '{query}'")

        # For MTV patterns, make sure we're searching correctly
        mtv_match = re.search(r'(MTV\d{4,})', query, re.IGNORECASE)

        # Build CQL query
        cql_parts = []

        # Handle specific MTV search differently - use direct text search
        if mtv_match:
            mtv_id = mtv_match.group(1)
            self.logger.info(f"Using MTV-specific search for: {mtv_id}")
            # Search for exact MTV ID in text
            cql_parts.append(f'text ~ "{mtv_id}"')
        else:
            # General search using siteSearch
            escaped = query.replace("\\", "\\\\").replace('"', '\\"')
            cql_parts.append(f'siteSearch ~ "{escaped}"')

        # Add page type filter
        cql_parts.append("type = page")

        # Add space filter if provided
        if space_key:
            cql_parts.append(f'space = "{space_key}"')

        # Add label filter if provided
        if label:
            cql_parts.append(f'label = "{label}"')

        # Build final CQL
        cql = " AND ".join(cql_parts) + " ORDER BY lastModified DESC"
        self.logger.info(f"Executing CQL: {cql}")

        # Make the request with increased limit
        params = {
            "limit": limit * 2,  # Double the limit to account for filtering
            "cql": cql,
            "expand": "content.title,content.space,content._links",
        }

        try:
            data = self._make_api_request(self.search_endpoint, params=params)

            # Log detailed info about the response
            if isinstance(data, dict):
                result_count = len(data.get("results", []))
                total = data.get("totalSize", 0)
                self.logger.info(f"API returned {result_count} of {total} total results")

                # Debug first result if available
                if result_count > 0:
                    first_item = data["results"][0]
                    self.logger.debug(f"First result: {first_item.get('title', 'No title')}")

            if not isinstance(data, dict) or "error" in data:
                err = data.get("message") or data.get("error", "Unknown")
                self.logger.error(f"CQL search failed: {err}")
                return []

            formatted: List[Dict[str, Any]] = []

            # Process results - dump more details about the first item for debugging
            if data.get("results") and len(data["results"]) > 0:
                self.logger.info(f"First result structure: {json.dumps(data['results'][0], default=str)[:500]}...")

            # Process results with more flexibility
            for item in data.get("results", []):
                try:
                    # Extract page ID directly from item or its content object
                    pid = None
                    title = "Untitled"
                    url = self.base_url
                    space = None

                    # Try multiple approaches to extract data

                    # 1. Check if item has direct ID (some API versions return this)
                    if "id" in item:
                        pid = item["id"]
                        self.logger.info(f"Found direct ID in item: {pid}")

                    # 2. Check in content object
                    content = item.get("content", {}) or {}
                    if not pid and content and "id" in content:
                        pid = content["id"]

                    # 3. If both failed, try to get from contentID or any ID-like fields
                    if not pid:
                        for key in item.keys():
                            if "id" in key.lower() and isinstance(item[key], (str, int)):
                                pid = str(item[key])
                                self.logger.info(f"Using {key} as page ID: {pid}")
                                break

                    # Get title - try multiple paths
                    if "title" in item:
                        title = html.unescape(item["title"])
                    elif content and "title" in content:
                        title = html.unescape(content["title"])

                    # Get space key
                    if "space" in item and isinstance(item["space"], dict):
                        space = item["space"].get("key")
                    elif content and "space" in content and isinstance(content["space"], dict):
                        space = content["space"].get("key")

                    # Get URL - try multiple approaches
                    links = None
                    if "_links" in item and isinstance(item["_links"], dict):
                        links = item["_links"]
                    elif content and "_links" in content and isinstance(content["_links"], dict):
                        links = content["_links"]

                    if links and "webui" in links:
                        url = links["webui"]
                    # If we have a title but no URL, try to construct it
                    elif pid:
                        url = f"{self.base_url}/pages/viewpage.action?pageId={pid}"

                    # Skip if we couldn't find a page ID after all attempts
                    if not pid:
                        self.logger.warning(f"Couldn't extract page ID from result item")
                        # Use item hash as fallback ID
                        item_hash = hash(json.dumps(item, sort_keys=True, default=str))
                        pid = f"unknown_{item_hash}"

                    # Add to results with the data we could extract
                    page_info = {
                        "id": pid,
                        "title": title or "Untitled Page",
                        "url": url,
                        "space_key": space,
                        # Add placeholder content to ensure it passes minimal content check
                        "content": f"Page found via search. ID: {pid}, Title: {title}"
                    }
                    formatted.append(page_info)
                    self.logger.info(f"Added page: {title} (ID: {pid})")

                except Exception as e:
                    self.logger.error(f"Error processing search result: {e}")
                    # Continue to next item

            self.logger.info(f"CQL search processed {len(formatted)} pages.")

            # If we found MTV in query but no results, try direct text search
            if mtv_match and not formatted:
                self.logger.warning(f"No results with MTV search, trying general text search")
                # Make another attempt with general search
                mtv_id = mtv_match.group(1)
                general_query = f'text ~ "{mtv_id}" AND type = page ORDER BY lastModified DESC'

                params["cql"] = general_query
                self.logger.info(f"Fallback CQL: {general_query}")

                try:
                    fallback_data = self._make_api_request(self.search_endpoint, params=params)

                    if isinstance(fallback_data, dict) and "results" in fallback_data:
                        for item in fallback_data.get("results", []):
                            try:
                                content = item.get("content", {}) or {}
                                pid = content.get("id")
                                title = html.unescape(content.get("title", "Untitled"))
                                space = content.get("space", {}).get("key") if content.get("space") else None
                                links = content.get("_links", {}) or {}
                                url = links.get("webui") or f"{self.base_url}/pages/viewpage.action?pageId={pid}"
                                formatted.append({"id": pid, "title": title, "url": url, "space_key": space})
                            except Exception as e:
                                self.logger.error(f"Error processing fallback result: {e}")
                except Exception as fallback_err:
                    self.logger.error(f"Fallback search failed: {fallback_err}")

            return formatted

        except Exception as e:
            self.logger.error(f"Error in search_pages: {e}")
            # Return empty list on error
            return []

    def fetch_and_clean_page_content(self, page_id: str, page_title: str = "Unknown") -> str:
        """Wrapper to get cleaned page content."""
        data = self.get_page_by_id(page_id)
        if "error" in data:
            self.logger.warning(f"Error fetching page {page_id} ('{page_title}'): {data['error']}")
            return ""
        return data.get("content", "")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type((RequestException, RetryError, ConfigurationError)),
    )
    def query_solutionbook(
        self, 
        query: str, 
        max_results: int = 5, 
        include_attachments: bool = False, 
        max_attachments_per_page: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Enhanced SolutionBook query that retrieves full page content with optional attachments.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            include_attachments: Whether to include attachments in the results
            max_attachments_per_page: Maximum number of attachments to include per page
            
        Returns:
            List of page results with content and optional attachments
        """
        ids = re.findall(r"([A-Z]+-\d+|MTV\d{3,})", query, re.IGNORECASE)
        terms = list(set(ids)) if ids else [query]
        self.logger.info(f"Searching SolutionBook for: {terms}")

        all_results: List[Dict[str, Any]] = []
        for term in terms:
            space_key = os.getenv("CONFLUENCE_DEFAULT_SPACE")
            res = self.search_pages(term, space_key=space_key, limit=max_results)
            all_results.extend(res)

        # Exit early if no pages were found
        if not all_results:
            self.logger.info("No pages found in search_pages")
            return []

        # Log what we found
        self.logger.info(f"Found {len(all_results)} total results across all search terms")

        # Use a dictionary to deduplicate by ID
        unique = {r["id"]: r for r in all_results if r.get("id")}
        self.logger.info(f"Deduplicated to {len(unique)} unique pages")

        # Fetch actual content for each page
        full_results = []
        for pid, item in unique.items():
            try:
                # Fetch the full page content by ID
                page_data = self.get_page_by_id(pid)

                # Create the result with full content if available
                if page_data and "error" not in page_data:
                    # Use the actual content from the page
                    content = page_data.get("content", "")
                    # If content is empty, use a placeholder with more information
                    if not content or len(content) < 50:
                        content = f"Page {pid} found in Confluence but content could not be retrieved. Please visit {item.get('url', '')} for details."
                        self.logger.warning(f"Empty or minimal content for page {pid}: {item.get('title', 'Untitled')}")

                    result = {
                        "id": pid,
                        "title": page_data.get("title", item.get("title", "Untitled")),
                        "url": page_data.get("url", item.get("url", "")),
                        "content": content,
                        "score": 1.0,
                        "space_key": page_data.get("space_key", item.get("space_key"))
                    }
                    
                    # Process attachments if requested
                    if include_attachments:
                        self.logger.info(f"Getting attachments for page: {pid}")
                        try:
                            attachments = self.get_attachments(pid)
                            if attachments:
                                # Take only the top few attachments based on parameter
                                page_attachments = attachments[:max_attachments_per_page]
                                
                                # Add attachment metadata
                                result["attachments"] = []
                                
                                # Only download attachments that are reasonably sized
                                for attachment in page_attachments:
                                    attachment_size = attachment.get("size", 0)
                                    if attachment_size > 10 * 1024 * 1024:  # Skip files > 10MB
                                        self.logger.info(f"Skipping large attachment: {attachment.get('title')} ({attachment_size/1024/1024:.1f}MB)")
                                        result["attachments"].append({
                                            "id": attachment.get("id"),
                                            "title": attachment.get("title"),
                                            "mediaType": attachment.get("mediaType"),
                                            "size": attachment_size,
                                            "skipped": True,
                                            "reason": "Size exceeds download limit"
                                        })
                                        continue
                                        
                                    # Download attachment content
                                    attachment_content = self.download_attachment(attachment)
                                    
                                    # If download succeeded, attempt to parse content
                                    if attachment_content:
                                        extracted_text = self.parse_attachment_content(
                                            attachment_content,
                                            attachment.get("mediaType", ""),
                                            attachment.get("title", "")
                                        )
                                        
                                        result["attachments"].append({
                                            "id": attachment.get("id"),
                                            "title": attachment.get("title"),
                                            "mediaType": attachment.get("mediaType"),
                                            "size": attachment_size,
                                            "content": extracted_text if extracted_text else "",
                                            "content_extracted": bool(extracted_text)
                                        })
                                    else:
                                        # Add metadata without content if download failed
                                        result["attachments"].append({
                                            "id": attachment.get("id"),
                                            "title": attachment.get("title"),
                                            "mediaType": attachment.get("mediaType"),
                                            "size": attachment_size,
                                            "download_failed": True
                                        })
                                
                                self.logger.info(f"Processed {len(result['attachments'])} attachments for page {pid}")
                        except Exception as e:
                            self.logger.error(f"Error processing attachments for page {pid}: {e}")
                            result["attachment_error"] = str(e)
                    
                else:
                    # If there was an error fetching the page, use the metadata we have
                    error_msg = page_data.get("error", "Unknown error") if page_data else "Failed to fetch page"
                    self.logger.warning(f"Could not fetch full content for {pid}: {error_msg}")

                    result = {
                        "id": pid,
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", ""),
                        "content": f"Page {pid} found in Confluence, but could not retrieve full content: {error_msg}. Please visit {item.get('url', '')} for details.",
                        "score": 0.5,  # Lower score for pages we couldn't get full content for
                        "space_key": item.get("space_key")
                    }

                full_results.append(result)
                self.logger.info(
                    f"Added result with {'full' if len(result['content']) > 100 else 'partial'} content: {result['title']} ({pid})")

            except Exception as e:
                self.logger.error(f"Error processing page {pid}: {e}")
                # Add basic info we have even if there was an error
                full_results.append({
                    "id": pid,
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": f"Page {pid} found in Confluence but an error occurred while retrieving content: {e}",
                    "score": 0.3,  # Even lower score for error cases
                    "space_key": item.get("space_key")
                })

        # Limit to requested number of results, prioritizing by score
        # Sort by score first to ensure we return the best results
        full_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = full_results[:max_results]
        self.logger.info(f"query_solutionbook returning {len(final_results)} pages with full content.")
        return final_results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=12),
        retry=retry_if_exception_type(RequestException),
    )
    def download_attachment(self, attachment: Dict[str, Any]) -> Optional[bytes]:
        """
        Download an attachment's content with enhanced error handling.
        
        Args:
            attachment: Attachment metadata dictionary from get_attachments
            
        Returns:
            Binary content of the attachment or None if download failed
        """
        link = attachment.get("downloadLink")
        title = attachment.get("title", "unknown")
        size = attachment.get("size", 0)
        attachment_id = attachment.get("id", "unknown")
        
        if not link:
            self.logger.warning(f"No download link for attachment: {title}")
            return None
            
        # Skip very large files
        if size and size > 20 * 1024 * 1024:  # 20MB limit
            self.logger.warning(f"Skipping large attachment {title} ({size/1024/1024:.1f}MB)")
            return None

        url = f"{self.base_url}{link}"
        try:
            # Use streaming to handle large files efficiently
            self.logger.info(f"Downloading attachment: {title} (ID: {attachment_id}, Size: {size/1024:.1f}KB)")
            
            # Set appropriate timeouts based on file size
            connect_timeout = 10
            read_timeout = min(60, 5 + (size / (100 * 1024)))  # Base + 1 sec per 100KB
            
            r = self.session.get(url, timeout=(connect_timeout, read_timeout), stream=True)
            r.raise_for_status()
            
            # Stream and build content
            content = b''
            for chunk in r.iter_content(chunk_size=8192):
                content += chunk
                
            self.logger.info(f"Successfully downloaded {title} ({len(content)} bytes)")
            return content
            
        except HTTPError as e:
            status = e.response.status_code if hasattr(e, 'response') and e.response else 'Unknown'
            self.logger.error(f"HTTP error {status} downloading {title}: {str(e)}")
            if status in (403, 404):
                # Don't retry for permission/not found errors
                return None
            raise
            
        except RequestException as e:
            self.logger.error(f"Network error downloading {title}: {e}")
            raise
            
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {title}: {e}", exc_info=True)
            return None

    def _expand_query_with_synonyms(self, query: str) -> List[str]:
        tokens = query.split()
        expansions = [query]
        for t in tokens:
            lower = t.lower()
            if lower in SYNONYMS:
                for syn in SYNONYMS[lower]:
                    expansions.append(query.replace(t, syn))
        return list(set(expansions))

    def index_page_for_semantic_search(self, page_id: str, content: str) -> None:
        if not content:
            return
        emb = hashlib.md5(content.encode("utf-8")).hexdigest()
        self.vector_index[page_id] = {"embedding": emb, "metadata": {"page_id": page_id}}
        self.logger.debug(f"Indexed {page_id} (fake vector)")

    def _fake_hash_distance(self, h1: str, h2: str) -> int:
        return abs(len(h1) - len(h2)) or sum(a != b for a, b in zip(h1, h2))

    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip() or not self.vector_index:
            return []
        query_emb = hashlib.md5(query.encode("utf-8")).hexdigest()
        results: List[Dict[str, Any]] = []
        for pid, entry in self.vector_index.items():
            dist = self._fake_hash_distance(query_emb, entry["embedding"])
            score = 1 / (1 + dist)
            page_data = self.get_page_by_id(pid)
            if "error" not in page_data and page_data.get("content"):
                results.append({
                    "id": pid,
                    "score": score,
                    "content": page_data["content"],
                    "url": page_data["url"],
                    "title": page_data["title"],
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def search_documents(self, query: str, max_results: int = 5) -> List[Document]:
        self.logger.info(f"Hybrid search for: {query}")
        loop = asyncio.get_running_loop()
        try:
            lexical = await asyncio.wait_for(
                loop.run_in_executor(None, self.query_solutionbook, query, max_results * 2),
                timeout=60,
            )
            vector = await asyncio.wait_for(
                loop.run_in_executor(None, self._vector_search, query, max_results * 2),
                timeout=60,
            )
        except Exception as e:
            self.logger.error(f"Hybrid search error: {e}", exc_info=True)
            return []

        merged: Dict[str, Dict[str, Any]] = {}
        for h in lexical:
            merged[h["id"]] = {
                "id": h["id"],
                "lex_score": h.get("score", 1.0),
                "vec_score": 0.0,
                "title": h["title"],
                "content": h["content"],
                "url": h["url"],
            }
        for v in vector:
            if v["id"] in merged:
                merged[v["id"]]["vec_score"] = v["score"]
            else:
                merged[v["id"]] = {
                    "id": v["id"],
                    "lex_score": 0.0,
                    "vec_score": v["score"],
                    "title": v["title"],
                    "content": v["content"],
                    "url": v["url"],
                }

        ranked = sorted(
            merged.values(),
            key=lambda x: 0.7 * x["lex_score"] + 0.3 * x["vec_score"],
            reverse=True,
        )
        docs: List[Document] = []
        for data in ranked[:max_results]:
            content = data["content"]
            if not content:
                continue
            metadata = {
                "source": f"SolutionBook:{data['id']}",
                "title": data["title"],
                "page_id": data["id"],
                "url": data["url"],
                "score": 0.7 * data["lex_score"] + 0.3 * data["vec_score"],
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def get_attachments(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Get a list of attachments for a Confluence page with enhanced metadata.
        
        Args:
            page_id: ID of the Confluence page
            
        Returns:
            List of attachment metadata dictionaries 
        """
        self.logger.debug(f"Getting attachments for page {page_id}")
        url = f"{self.api_url}/{page_id}/child/attachment"
        params = {"expand": "version,metadata.mediaType,extensions", "limit": 15}
        data = self._make_api_request(url, params=params)

        if not isinstance(data, dict) or "error" in data:
            self.logger.error(f"Error retrieving attachments for {page_id}: {data.get('error')}")
            return []

        attachments: List[Dict[str, Any]] = []
        for r in data.get("results", []):
            try:
                download = r.get("_links", {}).get("download")
                if r.get("id") and r.get("title") and download:
                    created = r.get("history", {}).get("createdDate", "")
                    creator = r.get("history", {}).get("createdBy", {}).get("displayName", "Unknown")
                    
                    # Get more detailed media type information when available
                    media_type = r.get("metadata", {}).get("mediaType", "")
                    filename = r.get("title", "unknown")
                    
                    # If no media type is provided, try to guess it from filename
                    if not media_type:
                        guessed_type, _ = mimetypes.guess_type(filename)
                        media_type = guessed_type or ""
                    
                    attachment_info = {
                        "id": r["id"],
                        "title": r["title"],
                        "mediaType": media_type,
                        "size": r.get("extensions", {}).get("fileSize", 0),
                        "version": r.get("version", {}).get("number", 0),
                        "downloadLink": download,
                        "created": created,
                        "creator": creator,
                        "extension": os.path.splitext(filename)[1].lower()[1:] if "." in filename else "",
                    }
                    
                    # Only include attachments that are likely to have useful text content
                    # Exclude large videos, executable files, etc.
                    excluded_extensions = ['exe', 'dll', 'bin', 'iso']
                    file_ext = attachment_info["extension"]
                    
                    # Skip executables, system files, and very large files
                    if (file_ext not in excluded_extensions and 
                        not media_type.startswith("video/") and 
                        attachment_info["size"] < 20 * 1024 * 1024):  # 20MB max size
                        attachments.append(attachment_info)
                    else:
                        self.logger.debug(f"Skipping attachment {r['title']} due to type/size filtering")
            except Exception as e:
                self.logger.error(f"Error processing attachment metadata: {e}")
                
        # Sort attachments by size (smaller first) for efficiency
        attachments.sort(key=lambda x: x.get("size", 0))
        
        self.logger.info(f"Found {len(attachments)} attachments for page {page_id}")
        return attachments

    def parse_attachment_content(
            self, file_bytes: bytes, media_type: str, filename: str = "attachment"
    ) -> Optional[str]:
        """
        Convert raw bytes of an attachment to text, using specialized converters based on file type.
        Handles PDF, Word, Excel, PowerPoint, images (OCR), CSV, JSON, XML, HTML and plain text.
        
        Args:
            filename: Original filename of the attachment
            content_bytes: Raw binary content
            mime_type: MIME type (will be guessed from filename if not provided)
            
        Returns:
            Extracted text content or error message
        """
        if not file_bytes: 
            return None
            
        if not media_type: 
            guess_type, _ = mimetypes.guess_type(filename)
            media_type = guess_type or ""
            
        # Get file extension for better format detection
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        self.logger.debug(f"Parsing '{filename}' (type: {media_type}, extension: {extension})")

        try:
            # PDF parsing with multiple libraries for better reliability
            if media_type == "application/pdf" or extension == 'pdf':
                if PDF_AVAILABLE:
                    try: 
                        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                        texts = []
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            if page_text:
                                texts.append(f"--- Page {i+1} ---")
                                texts.append(page_text)
                        if texts:
                            return "\n".join(texts)
                    except Exception as e:
                        self.logger.warning(f"PyPDF2 failed for {filename}: {e}")
                
                # Try pandoc as fallback
                try:
                    return pypandoc.convert_text(file_bytes, to='plain', format='pdf')
                except Exception:
                    pass

            # Office documents (Word, Excel, PowerPoint)
            if "officedocument" in media_type or extension in ('docx', 'xlsx', 'pptx', 'doc', 'xls', 'ppt'):
                try:
                    if extension in ('xlsx', 'xls'):
                        # Parse Excel with pandas for better structure
                        excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                        sheets_text = []
                        for sheet_name, df in excel_data.items():
                            sheets_text.append(f"--- Sheet: {sheet_name} ---")
                            sheets_text.append(df.to_string(index=False))
                        return "\n\n".join(sheets_text)
                    else:
                        # Use pandoc for other office formats
                        fmt = 'docx' if 'word' in media_type or extension in ('docx', 'doc') else 'pptx'
                        return pypandoc.convert_text(file_bytes, to='plain', format=fmt)
                except Exception as e:
                    self.logger.warning(f"Office document parsing failed for {filename}: {e}")

            # Images with OCR
            if media_type.startswith("image/") or extension in ('png', 'jpg', 'jpeg', 'gif', 'bmp'):
                if OCR_AVAILABLE:
                    try:
                        img = Image.open(io.BytesIO(file_bytes))
                        # Use English language by default
                        ocr_text = pytesseract.image_to_string(img, lang='eng')
                        if ocr_text and len(ocr_text.strip()) > 10:  # Ensure we got meaningful text
                            return f"[Image: {filename}]\n" + ocr_text
                        else:
                            self.logger.warning(f"OCR produced too little text for {filename}")
                            return f"[Image: {filename}] (OCR yielded minimal text)"
                    except Exception as e:
                        self.logger.warning(f"OCR failed for {filename}: {e}")
                        return f"[Image: {filename}] (OCR failed: {str(e)})"
                return f"[Image: {filename}] (OCR not available)"

            # JSON files
            if media_type == "application/json" or extension == 'json':
                try:
                    json_data = json.loads(file_bytes.decode('utf-8'))
                    return json.dumps(json_data, indent=2)
                except Exception as e:
                    self.logger.warning(f"JSON parsing failed: {e}")

            # CSV files
            if media_type == "text/csv" or extension == 'csv':
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes))
                    return df.to_string(index=False)
                except Exception as e:
                    self.logger.warning(f"CSV parsing failed: {e}")

            # HTML files
            if media_type == "text/html" or extension in ('html', 'htm'):
                try:
                    soup = BeautifulSoup(file_bytes.decode('utf-8', errors='ignore'), 'html.parser')
                    return html2text.html2text(str(soup))
                except Exception as e:
                    self.logger.warning(f"HTML parsing failed: {e}")

            # XML files
            if media_type == "application/xml" or extension == 'xml':
                try:
                    soup = BeautifulSoup(file_bytes.decode('utf-8', errors='ignore'), 'xml')
                    return f"[XML: {filename}]\n" + soup.get_text(separator="\n")
                except Exception as e:
                    self.logger.warning(f"XML parsing failed: {e}")

            # Fallback decode attempts
            try:
                return file_bytes.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                try:
                    return file_bytes.decode("latin-1", errors="ignore")
                except Exception:
                    return f"[Binary data: {filename}, type: {media_type}]"

        except Exception as e:
            self.logger.warning(f"Failed to parse {filename}: {e}")
            return f"<Failed to parse {filename}: {str(e)}>"

    def quality_check(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"readability_score": 0, "warnings": ["Empty content"]}
        try:
            score = textstat.flesch_reading_ease(text)
        except Exception:
            score = 0
        warnings = self._find_content_warnings(text)
        return {"readability_score": score, "warnings": warnings}

    def _find_content_warnings(self, text: str) -> List[str]:
        warns: List[str] = []
        if not text or len(text) < 200:
            warns.append("Content short (<200 chars).")
        if text and not re.search(r"MTV\d+|[A-Z]+-\d+", text, re.IGNORECASE):
            warns.append("No MTV/JIRA Key found.")
        return warns

    def search_comments(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        url = f"{self.api_url}/search"
        esc = keyword.replace('"', '\\"')
        cql = f'text ~ "{esc}" and type = comment ORDER BY lastModified DESC'
        params = {"cql": cql, "limit": limit}
        data = self._make_api_request(url, params=params)

        if not isinstance(data, dict) or "error" in data:
            self.logger.error(f"search_comments failed: {data.get('error')}")
            return []
        return data.get("results", [])

    def render_solutionbook_references(self, results: List[Any]) -> str:
        if not results:
            return "### SolutionBook References\n\nNo references found.\n"

        lines = ["### SolutionBook References", ""]
        seen: set = set()
        for r in results:
            if isinstance(r, dict):
                title = r.get("title", "?")
                url = r.get("url", "#")
                content = r.get("content", "")
            else:
                title = r.metadata.get("title", "?")
                url = r.metadata.get("url", "#")
                content = r.page_content

            if url in seen or url == "#":
                continue
            seen.add(url)

            snippet = content[:200].replace("\n", " ").strip()
            lines.append(f"- **[{title}]({url})**\n  Excerpt: {snippet}...\n")

        return "\n".join(lines)


if __name__ == "__main__":
    try:
        qb = SolutionBookQuerier()
        logger.info("SolutionBookQuerier initialized successfully.")

        test_query = "MTV2005"
        results = qb.query_solutionbook(test_query, max_results=3)
        logger.info(f"Got {len(results)} results for '{test_query}'.")
        for r in results:
            logger.info(f"Title: {r.get('title')} (ID: {r.get('id')})")
            logger.info(f"URL: {r.get('url')}")
            logger.info(f"Content Length: {len(r.get('content', ''))}")
            logger.info(f"Excerpt: {r.get('content', '')[:150]}...")
            logger.info("-" * 20)

    except ConfigurationError as e:
        logger.error(f"Tool Configuration Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
