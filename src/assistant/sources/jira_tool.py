# src/assistant/tools/jira_tool.py
import io
import json
import logging
import mimetypes
import os
import re
import sys
import asyncio # Added asyncio
import time # Added time for retry sleep
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import pandas as pd
import pypandoc
import requests
import urllib3
from cachetools import TTLCache, cached
from dotenv import load_dotenv
from jira import JIRA, Issue # Keep original JIRA import alongside patch
from jira.exceptions import JIRAError
from langchain.docstore.document import Document
from openpyxl import load_workbook
# --- START Enhancement: Import Tenacity ---
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
# --- END Enhancement ---

# Import JIRA patch to fix session issue
from src.assistant.sources.jira_patch import JIRA

# Optional imports for advanced features
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure basic logging
# (Using configuration from graph_2.py or main entry point is preferred)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Set level explicitly if needed

# Load environment variables
load_dotenv()


class ConfigurationError(Exception):
    """Raised when JIRA configuration or authentication fails."""
    pass


class JiraProject:
    """
    Enhanced class to handle JIRA operations with improved accuracy and robustness.
    """

    # --- START Enhancement: Default Fields ---
    # Define default fields to request for faster initial searches
    # Added more fields that might be useful for context/summarization
    DEFAULT_JIRA_FIELDS = "key,summary,status,issuetype,description,updated,priority,assignee,reporter,created,labels,fixVersions,components,project,resolution"
    # --- END Enhancement ---

    def __init__(
        self,
        server: str,
        token: str = None,
        username: str = None,
        password: str = None,
        verify_ssl: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        connection_retries: int = 2,
        jira_read_timeout: int = 60,
    ):
        """
        Initialize the JIRA connection with enhanced error handling.
        
        Args:
            server: JIRA server URL
            token: JIRA API token (preferred auth method)
            username: JIRA username (only used if token not provided)
            password: JIRA password (only used if token not provided)
            verify_ssl: Whether to verify SSL certificates
            proxies: Optional proxy configuration
            timeout: Connection timeout in seconds
            connection_retries: Number of retries for failed connections
            jira_read_timeout: Timeout for read operations (for large issues)
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Store configuration
        self.server = server
        self.verify_ssl = verify_ssl
        self.token = token
        self.username = username
        self.password = password
        self.timeout = timeout
        self.connection_retries = connection_retries
        self.jira_read_timeout = jira_read_timeout

        # Synonym map (can be loaded from config)
        self.synonyms_map = {} # Example: {'deploy': ['release', 'rollout']}

        try:
            # Initialize JIRA client with proper options
            # Ensure token_auth is used correctly
            self.jira = JIRA(
                options={
                    'server': server,
                    'verify': False, # Keep verify=False as per original code and ssl_fix.py usage
                    # Set separate connect and read timeouts
                    'timeout': (self.timeout, self.jira_read_timeout)
                },
                token_auth=token # Use token_auth for API token
            )

            # Test connection during init
            if not self.validate_connection():
                 # Log the error within validate_connection
                 raise ConfigurationError("JIRA connection validation failed during initialization.")

            logger.info("JIRA connection established and validated successfully.")

        except JIRAError as e:
            # Log specific details from JIRAError if possible
            status = getattr(e, 'status_code', 'N/A')
            text = getattr(e, 'text', str(e))
            logger.error(f"JIRA authentication/configuration failed: Status Code: {status}, Text: {text}", exc_info=True)
            raise ConfigurationError(f"Invalid JIRA credentials, permissions, or server issue: {text}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"JIRA connection error: {e}", exc_info=True)
            raise ConfigurationError(f"Cannot connect to JIRA server at {server}: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during JIRA initialization: {e}", exc_info=True)
             raise ConfigurationError(f"Unexpected error initializing JIRA: {e}") from e

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3),
           retry=retry_if_exception_type((JIRAError, requests.exceptions.RequestException, asyncio.TimeoutError, ConnectionError, RetryError))) # Added RetryError
    # --- END Enhancement ---
    def validate_connection(self) -> bool:
        """Comprehensive check to confirm JIRA is reachable and properly configured."""
        try:
            # Check server info
            logger.debug("Validating JIRA: Fetching server info...")
            server_info = self.jira.server_info()
            if not server_info:
                logger.error("Could not retrieve JIRA server info")
                return False
            logger.debug(f"JIRA Server Info: OK (Version: {server_info.get('version')})")

            # Check authentication
            logger.debug("Validating JIRA: Fetching user info...")
            user = self.jira.myself()
            if not user:
                logger.error("Could not retrieve user information (authentication check)")
                return False
            logger.debug(f"JIRA User Info: OK (User: {user.get('emailAddress') or user.get('name')})")

            # Test basic search capability (simple, fast query)
            logger.debug("Validating JIRA: Performing test search...")
            # --- START FIX: Use valid test JQL ---
            test_jql = 'created >= -1m ORDER BY created DESC' # Find issues created recently
            # --- END FIX ---
            test_fields = 'key' # Request only the key field
            try:
                # Run the synchronous search directly, retry handles transient issues
                _ = self.jira.search_issues(test_jql, fields=test_fields, maxResults=1)
                logger.debug("JIRA Search Test: OK")
            except JIRAError as e:
                 logger.error(f"Search test failed with JIRAError: {e.status_code} - {e.text}")
                 # Re-raise specific errors that indicate fundamental issues not helped by retry
                 if e.status_code in [401, 403]: raise e
                 # Otherwise, let retry handle it if applicable by its type
                 # Re-raise other JIRAErrors for retry mechanism
                 raise e
            except Exception as e:
                logger.error(f"Search test failed with unexpected error: {e}")
                raise # Re-raise for retry decorator if applicable

            logger.info("JIRA connection validated successfully")
            return True

        except JIRAError as e:
            # Log specific JIRA errors during validation after retries
            status = getattr(e, 'status_code', 'N/A')
            text = getattr(e, 'text', str(e))
            if status == 401: logger.error("JIRA validation failed: Authentication error (401). Check API Token.")
            elif status == 403: logger.error("JIRA validation failed: Permission error (403). Check user permissions.")
            elif status == 400: logger.error(f"JIRA validation failed: Bad Request (400) - check JQL/request: {text}")
            else: logger.error(f"JIRA validation failed: {status} - {text}")
            return False
        except requests.exceptions.RequestException as e:
             logger.error(f"JIRA validation failed: Network error - {e}")
             return False
        except RetryError as e:
             logger.error(f"JIRA validation failed after multiple retries: {e}")
             return False
        except Exception as e:
            # Catch errors from retry or unexpected issues
            logger.error(f"Unexpected error during validation after retries: {e}", exc_info=True)
            return False

    # --------------------------------------------------------------------------
    # 1) Basic Issue Searching
    # --------------------------------------------------------------------------
    def _expand_query_with_synonyms(self, query: str) -> List[str]:
        """Expand a query with synonyms (currently placeholder)."""
        # Placeholder: Implement actual synonym logic if needed
        return [query]

    def _build_fallback_jql(self, base_query: str) -> List[str]:
        """Generate fallback JQL queries."""
        expansions = self._expand_query_with_synonyms(base_query)
        jqls = []
        for exp in expansions:
            # Example simple fallback approach
            # Consider adding project context if relevant and available
            # project_key = os.getenv("JIRA_DEFAULT_PROJECT")
            # project_clause = f"PROJECT = {project_key} AND " if project_key else ""
            jqls.append(f'text ~ "{exp}" ORDER BY updated DESC') # Order by updated often more relevant
        return jqls

    # --- START Enhancement: Add Retry ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=12), # Slightly longer wait
           retry=retry_if_exception_type((JIRAError, requests.exceptions.RequestException, RetryError))) # Add RetryError
    # --- END Enhancement ---
    def search_issues(self, jql: str, fields: Optional[str] = None, max_results: int = 50, startAt: int = 0) -> List[Issue]:
        """Search for JIRA issues by JQL (synchronous version) with field limiting and pagination."""
        try:
            # --- START Enhancement: Use Default Fields ---
            fields_to_request = fields or self.DEFAULT_JIRA_FIELDS
            logger.debug(f"Executing JQL: {jql} | Fields: {fields_to_request} | MaxResults: {max_results} | StartAt: {startAt}")
            # --- END Enhancement ---
            # --- START Enhancement: Add startAt for pagination ---
            # The jira-python library handles the REST call here
            return self.jira.search_issues(jql, fields=fields_to_request, maxResults=max_results, startAt=startAt)
            # --- END Enhancement ---
        # --- START Enhancement: Specific Error Handling ---
        except JIRAError as e:
             status = getattr(e, 'status_code', 'N/A')
             text = getattr(e, 'text', str(e))
             # Handle common errors more specifically
             if status == 400: logger.error(f"JIRA Bad Request (likely JQL syntax error) for '{jql}': {text}")
             elif status == 401: logger.error(f"JIRA Authentication Error for '{jql}': {text}")
             elif status == 403: logger.error(f"JIRA Permission Error for '{jql}': {text}")
             elif status == 429: logger.warning(f"JIRA Rate Limit hit for '{jql}'. Retrying..."); raise RetryError(f"Rate limit hit (429) for JQL: {jql}") from e # Trigger retry
             else: logger.error(f"JIRA error ({status}) for JQL '{jql}': {text}")
             # Re-raise JIRAError so @retry can potentially catch it if applicable (e.g., 5xx)
             raise e
        # --- END Enhancement ---
        except requests.exceptions.RequestException as e:
             logger.error(f"Network error searching JIRA with JQL '{jql}': {e}", exc_info=True)
             raise e # Re-raise for @retry
        except Exception as e:
            # Log unexpected errors but wrap them to signal failure clearly
            logger.error(f"Unexpected error searching JIRA with JQL '{jql}': {e}", exc_info=True)
            # Wrap in JIRAError for consistent handling by caller/retry logic
            raise JIRAError(f"Unexpected JIRA search error: {e}", status_code=500) from e

    async def search_issues_async(self, jql: str, fields: Optional[str] = None, max_results: int = 50, startAt: int = 0) -> List[Issue]:
        """Search for JIRA issues by JQL (async version) with field limiting, pagination and retry."""
        try:
            # Use asyncio.to_thread to run the enhanced synchronous search_issues method
            loop = asyncio.get_running_loop()
            # --- START Enhancement: Use configured timeout ---
            # Use combined timeout for the async wrapper call
            timeout = self.timeout + self.jira_read_timeout
            return await asyncio.wait_for(
                loop.run_in_executor(None, self.search_issues, jql, fields, max_results, startAt),
                timeout=timeout
            )
            # --- END Enhancement ---
        except asyncio.TimeoutError:
             logger.error(f"Async JIRA search timed out after {timeout}s for JQL: {jql}")
             return []
        except Exception as e:
            # Errors inside search_issues are already logged by it, just log the async wrapper context
            logger.error(f"Error in async wrapper for JIRA search '{jql}': {type(e).__name__}", exc_info=False) # Less verbose exc_info here
            return []

    def search_all_issues(self, jql: str, fields: Optional[str] = None, page_size: int = 50) -> List[Issue]:
        """Retrieve ALL issues matching a JQL query using pagination."""
        all_issues = []
        start_at = 0
        retries = 0
        max_pagination_retries = 3 # Limit retries for pagination errors specifically
        while True:
             try:
                  logger.debug(f"Fetching issues page for JQL: {jql} | StartAt: {start_at} | PageSize: {page_size}")
                  # Call the synchronous search_issues which has its own retry logic for transient errors
                  issues_page = self.search_issues(jql, fields=fields, max_results=page_size, startAt=start_at)

                  # If search_issues returns empty list due to error handled internally, stop pagination
                  if issues_page is None: # Should not happen if errors raise exceptions
                       logger.error(f"Search_issues returned None unexpectedly at startAt={start_at}. Stopping pagination.")
                       break

                  if not issues_page:
                       logger.debug(f"No more issues found at startAt={start_at}.")
                       break # Stop if no issues returned (end of results)

                  all_issues.extend(issues_page)
                  start_at += len(issues_page)
                  retries = 0 # Reset pagination retries on success

                  # Check if the number of results returned is less than page_size, indicating the end
                  if len(issues_page) < page_size:
                       logger.debug("Last page reached.")
                       break
             except (JIRAError, requests.exceptions.RequestException, RetryError) as e:
                  retries += 1
                  logger.error(f"Error fetching paginated JIRA results (startAt={start_at}, attempt={retries}): {e}")
                  if retries >= max_pagination_retries:
                       logger.error(f"Max retries ({max_pagination_retries}) reached for pagination at startAt={start_at}. Stopping.")
                       break # Stop pagination after max retries for a page
                  # Wait briefly before retrying pagination error (exponential backoff)
                  wait_time = 2**retries
                  logger.info(f"Retrying pagination after {wait_time} seconds...")
                  time.sleep(wait_time)
             except Exception as e:
                  logger.error(f"Unexpected error during pagination (startAt={start_at}): {e}", exc_info=True)
                  break # Stop on unexpected errors
        logger.info(f"Retrieved a total of {len(all_issues)} issues for JQL: {jql}")
        return all_issues


    def robust_search_issues(self, query_text: str, fields: Optional[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search function trying ID patterns first, then falling back to text, using specified fields."""
        try:
            mtv_match = re.search(r'(MTV\d{4,})', query_text, re.IGNORECASE)
            jira_match = re.search(r'([A-Z]+-\d+)', query_text, re.IGNORECASE)
            general_id_match = re.search(r'([A-Z0-9]+-\d+|MTV\d{3,})', query_text, re.IGNORECASE)

            logger.info(f"Robust searching JIRA with query: {query_text}")

            issues = []
            # --- START Enhancement: Use Specified/Default Fields ---
            fields_to_request = fields or self.DEFAULT_JIRA_FIELDS
            # --- END Enhancement ---

            # --- START Enhancement: Optimize JQL & Project Context ---
            # Example Project Key (should come from config or context for better accuracy)
            project_key = os.getenv("JIRA_DEFAULT_PROJECT")
            project_clause = f"PROJECT = {project_key} AND " if project_key else ""
            # --- END Enhancement ---

            # Prioritize specific ID searches
            if jira_match:
                jira_id = jira_match.group(1).upper() # Normalize key
                logger.info(f"Found JIRA key pattern: {jira_id}. Searching by key/text.")
                # Escape quotes for text search part
                jql = f'{project_clause}(key = "{jira_id}" OR text ~ \'"{jira_id}"\') ORDER BY updated DESC'
                issues = self.search_issues(jql, fields=fields_to_request, max_results=max_results)
            elif mtv_match:
                mtv_id = mtv_match.group(1).upper() # Normalize ID
                logger.info(f"Found MTV pattern: {mtv_id}. Searching JIRA text.")
                jql = f'{project_clause}text ~ \'"{mtv_id}"\' ORDER BY updated DESC' # Escape quotes
                issues = self.search_issues(jql, fields=fields_to_request, max_results=max_results)
            elif general_id_match:
                 general_id = general_id_match.group(1).upper() # Normalize ID
                 logger.info(f"Found general ID pattern: {general_id}. Trying key/text search.")
                 jql = f'{project_clause}(key = "{general_id}" OR text ~ \'"{general_id}"\') ORDER BY updated DESC'
                 issues = self.search_issues(jql, fields=fields_to_request, max_results=max_results)

            # If ID search yielded no results OR no ID was found, fallback to broader text search
            if not issues:
                logger.info(f"No results from ID search (or no ID found). Falling back to text search for: '{query_text}'")
                # Escape potential special chars for JQL text search
                escaped_query = query_text.replace('\\', '\\\\').replace('"', '\\"')
                # Search summary OR description OR comment for better relevance
                jql = f'{project_clause}(summary ~ \'"{escaped_query}"\' OR description ~ \'"{escaped_query}"\' OR comment ~ \'"{escaped_query}"\') ORDER BY updated DESC'
                issues = self.search_issues(jql, fields=fields_to_request, max_results=max_results)

            return self._format_issues(issues)

        except Exception as e:
            # Catch errors from search_issues if retry fails
            logger.error(f"Error in robust_search_issues for query '{query_text}': {e}", exc_info=True)
            return []

    # Keep search_by_id and search_by_text for potential direct use, but ensure they use field limiting
    def search_by_id(self, id_str: str, fields: Optional[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for issues by ID (either JIRA key or ID in text)"""
        try:
            fields_to_request = fields or self.DEFAULT_JIRA_FIELDS
            id_str_upper = id_str.upper() # Normalize
            # Add project context if possible
            project_key = os.getenv("JIRA_DEFAULT_PROJECT")
            project_clause = f"PROJECT = {project_key} AND " if project_key else ""
            # project_clause = "" # Keep project agnostic unless specified

            if re.match(r'[A-Z]+-\d+', id_str_upper):
                logger.info(f"Searching for JIRA issue with key: {id_str_upper}")
                jql = f'{project_clause}key = "{id_str_upper}"'
            else:
                logger.info(f"Searching for issues containing ID in text: {id_str}")
                jql = f'{project_clause}text ~ \'"{id_str}"\'' # Escape quotes
            jql += ' ORDER BY updated DESC'
            issues = self.search_issues(jql, fields=fields_to_request, max_results=max_results)
            return self._format_issues(issues)
        except Exception as e:
            logger.error(f"Error searching by ID {id_str}: {e}", exc_info=True)
            return []

    def search_by_text(self, jql: str, fields: Optional[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using custom JQL with default field limiting"""
        try:
            fields_to_request = fields or self.DEFAULT_JIRA_FIELDS
            logger.info(f"Executing JQL: {jql}")
            issues = self.search_issues(jql, fields=fields_to_request, max_results=max_results)
            return self._format_issues(issues)
        except Exception as e:
            logger.error(f"Error searching with JQL {jql}: {e}", exc_info=True)
            return []

    def _format_issues(self, issues: List[Issue]) -> List[Dict[str, Any]]:
        """Format JIRA issues into consistent dictionary structure with enhanced critical field extraction."""
        formatted_issues = []
        if not issues: return formatted_issues

        for issue in issues:
            # Basic check if it looks like an Issue object
            if not hasattr(issue, 'key') or not hasattr(issue, 'fields'):
                 logger.warning(f"Skipping invalid issue object: {issue}")
                 continue

            issue_data = {"key": issue.key, "url": f"{self.server}/browse/{issue.key}"}

            # Safely access fields using getattr with defaults
            issue_data["summary"] = getattr(issue.fields, 'summary', 'N/A') or 'N/A' # Ensure not None

            status_obj = getattr(issue.fields, 'status', None)
            issue_data["status"] = getattr(status_obj, 'name', 'Unknown') if status_obj else "Unknown"

            issuetype_obj = getattr(issue.fields, 'issuetype', None)
            issue_data["type"] = getattr(issuetype_obj, 'name', 'Unknown') if issuetype_obj else "Unknown"

            priority_obj = getattr(issue.fields, 'priority', None)
            issue_data["priority"] = getattr(priority_obj, 'name', 'N/A') if priority_obj else "N/A"

            assignee_obj = getattr(issue.fields, 'assignee', None)
            issue_data["assignee"] = getattr(assignee_obj, 'displayName', 'Unassigned') if assignee_obj else "Unassigned"

            reporter_obj = getattr(issue.fields, 'reporter', None)
            issue_data["reporter"] = getattr(reporter_obj, 'displayName', 'N/A') if reporter_obj else "N/A"

            issue_data["created"] = getattr(issue.fields, 'created', 'N/A')
            issue_data["updated"] = getattr(issue.fields, 'updated', 'N/A')

            # --- ADDED: Extract Acceptance Criteria ---
            # Look for acceptance criteria in both dedicated field and description
            acceptance_criteria = ""
            
            # Try common custom field names for acceptance criteria
            for field_name in dir(issue.fields):
                if 'acceptancecriteria' in field_name.lower() or 'acceptance_criteria' in field_name.lower():
                    criteria = getattr(issue.fields, field_name, None)
                    if criteria:
                        acceptance_criteria = criteria
                        break
                        
            # If not found in custom fields, look for section in description
            if not acceptance_criteria:
                description = getattr(issue.fields, 'description', '')
                if description:
                    # Common patterns for acceptance criteria in description
                    ac_patterns = [
                        r'(?i)acceptance criteria:?(.*?)(?:^#|\Z)', 
                        r'(?i)acceptance criteria:?(.*?)(?:\n\n|\Z)',
                        r'(?i)acceptance criteria:?(.*)',
                    ]
                    for pattern in ac_patterns:
                        match = re.search(pattern, description, re.DOTALL | re.MULTILINE)
                        if match:
                            acceptance_criteria = match.group(1).strip()
                            break
            
            issue_data["acceptance_criteria"] = acceptance_criteria
            
            # --- ADDED: Extract Business Value/US Feature Type ---
            business_value = ""
            us_feature_type = ""
            
            # Look for business value and US/Feature Type in custom fields
            for field_name in dir(issue.fields):
                if 'businessvalue' in field_name.lower() or 'business_value' in field_name.lower():
                    business_value = getattr(issue.fields, field_name, '') or ''
                if 'featuretype' in field_name.lower() or 'feature_type' in field_name.lower():
                    us_feature_type = getattr(issue.fields, field_name, '') or ''
                    
            issue_data["business_value"] = business_value
            issue_data["feature_type"] = us_feature_type
            
            # --- ADDED: Extract Technical Integration Points ---
            # Look for integration points in description and other fields
            description = getattr(issue.fields, 'description', '')
            integration_points = []
            
            # Common systems seen in screenshots
            systems = ["ESB", "DLS", "OMS", "Digital BE", "QPC", "RPC", "UPC", "OMS"]
            integration_pattern = r'\b(' + '|'.join(systems) + r')\b'
            
            if description:
                matches = re.findall(integration_pattern, description)
                integration_points.extend(matches)
                
            # Check summary for integration points too
            summary = getattr(issue.fields, 'summary', '')
            if summary:
                matches = re.findall(integration_pattern, summary)
                integration_points.extend(matches)
                
            # Deduplicate
            issue_data["integration_points"] = list(set(integration_points))
            
            # --- ADDED: Get Sub-Tasks ---
            # Use the API to get subtasks if available
            subtasks = getattr(issue.fields, 'subtasks', [])
            subtask_data = []
            
            for subtask in subtasks:
                if hasattr(subtask, 'key') and hasattr(subtask, 'fields'):
                    subtask_info = {
                        "key": subtask.key,
                        "summary": getattr(subtask.fields, 'summary', ''),
                        "status": getattr(getattr(subtask.fields, 'status', None), 'name', 'Unknown') if hasattr(subtask.fields, 'status') else "Unknown"
                    }
                    subtask_data.append(subtask_info)
                    
            issue_data["subtasks"] = subtask_data
            
            # --- ADDED: Extract Technical Field Mappings ---
            # Attempt to find mapped fields (as seen in "Mapping of attributes from OMS")
            mapping_data = []
            
            # Look for attribute mapping patterns in description
            if description:
                # Look for patterns like "X will be set by Y as returned by Z"
                # or "This attribute will be populated by X during Y"
                mapping_patterns = [
                    r'(\w+[\s\w]*)\s+will\s+be\s+(?:set|populated|returned|mapped)\s+by\s+(\w+[\s\w]*)',
                    r'(\w+[\s\w]*)\s+attribute\s+will\s+be\s+(?:populated|returned|set)\s+by\s+(\w+[\s\w]*)',
                    r'(\w+[\s\w]*)\s+-\s+will\s+be\s+set\s+by\s+(\w+[\s\w]*)',
                ]
                
                for pattern in mapping_patterns:
                    for match in re.finditer(pattern, description, re.IGNORECASE):
                        attr_name = match.group(1).strip()
                        source_system = match.group(2).strip()
                        mapping_data.append({
                            "attribute": attr_name,
                            "source_system": source_system
                        })
                        
            issue_data["field_mappings"] = mapping_data

            # Truncate description if present and requested
            if description:
                # Limit description length for performance/clarity in summaries
                desc_limit = 1000
                issue_data["description"] = description[:desc_limit] + ("..." if len(description) > desc_limit else "")
            else:
                issue_data["description"] = "No description." # Provide default

            issue_data["labels"] = getattr(issue.fields, 'labels', [])
            # Ensure fixVersions/components are lists and have 'name' attribute
            issue_data["fixVersions"] = [v.name for v in getattr(issue.fields, 'fixVersions', []) if hasattr(v, 'name')]
            issue_data["components"] = [c.name for c in getattr(issue.fields, 'components', []) if hasattr(c, 'name')]

            resolution_obj = getattr(issue.fields, 'resolution', None)
            issue_data["resolution"] = getattr(resolution_obj, 'name', None) if resolution_obj else None # Keep as None if unresolved

            project_obj = getattr(issue.fields, 'project', None)
            issue_data["project_key"] = getattr(project_obj, 'key', None) if project_obj else None
            issue_data["project_name"] = getattr(project_obj, 'name', None) if project_obj else None

            formatted_issues.append(issue_data)

        logger.info(f"Formatted {len(formatted_issues)} JIRA issues from {len(issues)} raw results with enhanced critical fields.")
        return formatted_issues

    # --------------------------------------------------------------------------
    # 2) Retrieving & Processing Single Issues
    # --------------------------------------------------------------------------
    # --- START Enhancement: Robust get_issue ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((JIRAError, requests.exceptions.RequestException, asyncio.TimeoutError, RetryError))) # Added RetryError
    # --- END Enhancement ---
    def get_issue(self, issue_key: str, fields: str = "*all") -> Optional[Issue]:
        """Get a JIRA issue with enhanced error handling and retries."""
        try:
            self.logger.info(f"Fetching issue {issue_key} with fields: {fields}")
            
            # Use synchronous method for synchronous context
            # FIXED: Replace direct call with proper synchronous method to avoid coroutine warnings
            if fields == "*all":
                issue = self.jira.issue(issue_key)  # Use default fields
            else:
                issue = self.jira.issue(issue_key, fields=fields, expand='renderedFields')  # Direct call for sync context
                
            return issue
        except JIRAError as e:
            self.logger.error(f"JIRA error getting issue {issue_key}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting issue {issue_key}: {e}")
            return None
    # --- END Enhancement ---

    def get_issue_details(self, issue_key: str, download_attachments: bool = True) -> Dict[str, Any]:
        """
        Return a dictionary with relevant fields from a JIRA issue.
        
        Args:
            issue_key: The JIRA issue key
            download_attachments: If True, download and process attachments content
            
        Returns:
            Dictionary with issue details including processed attachments
        """
        try:
            issue = self.get_issue(issue_key, fields="*all,renderedFields") # Uses enhanced get_issue with retries
            if not issue:
                 return {"key": issue_key, "error": f"Could not retrieve issue {issue_key} after retries."}

            # Use _format_issues logic for consistency, passing the single issue in a list
            formatted = self._format_issues([issue])
            details = formatted[0] if formatted else {"key": issue_key, "error": f"Could not format issue {issue_key}"}

            # Add comments and attachments which are not default fields if formatting didn't error
            if not details.get("error"):
                details["comments"] = []
                details["attachments"] = []
                # Safely get comments (check attribute existence)
                if hasattr(issue, 'fields') and hasattr(issue.fields, 'comment') and issue.fields.comment and hasattr(issue.fields.comment, 'comments'):
                     details["comments"] = [
                          {
                              "author": getattr(getattr(c, 'author', None), 'displayName', 'Unknown Author'), # Safe access
                              "created": getattr(c, 'created', 'N/A'),
                              "body": getattr(c, 'body', '')[:1000] + ("..." if len(getattr(c, 'body', '')) > 1000 else "") # Truncate
                          }
                          for c in issue.fields.comment.comments
                     ]
                
                # --- ENHANCED: Get and process attachments with content ---
                attachments_raw = []
                if hasattr(issue, 'fields') and hasattr(issue.fields, 'attachment') and issue.fields.attachment:
                    attachments_raw = issue.fields.attachment
                
                # Process attachments and download content if requested
                details["attachments"] = []
                if attachments_raw:
                    logger.info(f"Processing {len(attachments_raw)} attachments for issue {issue_key}")
                    
                    for att in attachments_raw:
                        # Skip if attachment object is invalid
                        if not hasattr(att, 'filename'):
                            continue
                            
                        attachment_info = {
                            "id": getattr(att, 'id', 'unknown'),
                            "filename": getattr(att, 'filename', 'unknown'),
                            "mime_type": getattr(att, 'mimeType', 'unknown'),
                            "size": getattr(att, 'size', 0),
                            "created": getattr(att, 'created', 'N/A'),
                            "url": getattr(att, 'content', None),
                            "content_processed": False
                        }
                        
                        # Download and process content if requested and URL exists
                        if download_attachments and attachment_info["url"]:
                            try:
                                # Only attempt to download if size is reasonable (max 10MB)
                                if attachment_info["size"] > 10 * 1024 * 1024:
                                    logger.warning(f"Skipping large attachment: {attachment_info['filename']} ({attachment_info['size']} bytes)")
                                    attachment_info["content_text"] = "[Attachment too large to download]"
                                    attachment_info["content_preview"] = "[Attachment too large]"
                                    details["attachments"].append(attachment_info)
                                    continue
                                    
                                # Download the attachment content
                                content_bytes = self.download_attachment_content(att)
                                if content_bytes:
                                    attachment_info["content_processed"] = True
                                    
                                    # Attempt to convert to text based on mime type
                                    extracted_text = self.convert_attachment_to_text(
                                        attachment_info["filename"], 
                                        content_bytes, 
                                        attachment_info["mime_type"]
                                    )
                                    
                                    # Add the extracted text with reasonable size limits
                                    if extracted_text:
                                        # Cap text length to avoid excessive payload size
                                        max_text_len = 10000  # 10K chars max per attachment
                                        attachment_info["content_text"] = extracted_text[:max_text_len]
                                        if len(extracted_text) > max_text_len:
                                            attachment_info["content_text"] += "\n... [TRUNCATED - CONTENT TOO LARGE] ..."
                                            
                                        # Also store a short preview
                                        preview_len = 200
                                        attachment_info["content_preview"] = extracted_text[:preview_len].replace('\n', ' ')
                                        if len(extracted_text) > preview_len:
                                            attachment_info["content_preview"] += "..."
                                    else:
                                        attachment_info["content_text"] = "[No text content could be extracted]"
                                        attachment_info["content_preview"] = "[Binary content]"
                                    
                                    # Special handling for different file types
                                    if attachment_info["mime_type"].startswith("image/") and hasattr(att, 'thumbnail'):
                                        attachment_info["thumbnail"] = att.thumbnail
                                
                            except Exception as att_err:
                                logger.error(f"Error processing attachment {attachment_info['filename']}: {att_err}")
                                attachment_info["content_text"] = f"[Error processing attachment: {str(att_err)}]"
                                attachment_info["content_preview"] = "[Processing error]"
                        
                        details["attachments"].append(attachment_info)
                    
                    logger.info(f"Processed {len(details['attachments'])} attachments for issue {issue_key}")

            return details
        except Exception as e:
             # Catch errors from get_issue if all retries fail
             logger.error(f"Failed to get details for {issue_key} after retries: {e}", exc_info=True)
             return {"key": issue_key, "error": f"Failed to get details for issue {issue_key}: {e}"}

    # --- ADDED: New Method to Get Parent-Child Relationships ---
    def get_issue_hierarchy(self, issue_key: str, download_attachments: bool = True) -> Dict[str, Any]:
        """
        Get the full hierarchy information for an issue (epic/parent/subtasks).
        
        Args:
            issue_key: The JIRA issue key
            download_attachments: If True, download and process attachments content
            
        Returns:
            Dictionary with parent, epic, and child relationships
        """
        hierarchy = {
            "key": issue_key,
            "parent": None,
            "epic": None,
            "children": [],
            "subtasks": []
        }
        
        try:
            # Get the issue with additional fields
            issue = self.get_issue(issue_key, fields="*all")
            if not issue:
                return hierarchy
                
            # Check for parent
            parent_field = None
            for field_name in dir(issue.fields):
                if 'parent' in field_name.lower():
                    parent = getattr(issue.fields, field_name, None)
                    if parent and hasattr(parent, 'key'):
                        hierarchy["parent"] = {
                            "key": parent.key,
                            "summary": getattr(parent.fields, 'summary', '') if hasattr(parent, 'fields') else '',
                            "type": getattr(parent.fields.issuetype, 'name', '') if (hasattr(parent, 'fields') and hasattr(parent.fields, 'issuetype')) else ''
                        }
                        break
            
            # Check for epic link
            for field_name in dir(issue.fields):
                if 'epic' in field_name.lower():
                    epic = getattr(issue.fields, field_name, None)
                    if epic and isinstance(epic, str):  # Epic link is usually stored as a string (epic key)
                        try:
                            epic_issue = self.get_issue(epic)
                            if epic_issue:
                                hierarchy["epic"] = {
                                    "key": epic_issue.key,
                                    "summary": getattr(epic_issue.fields, 'summary', ''),
                                    "status": getattr(getattr(epic_issue.fields, 'status', None), 'name', '') if hasattr(epic_issue.fields, 'status') else ''
                                }
                        except Exception as e:
                            logger.warning(f"Error getting epic details for {epic}: {e}")
            
            # Get subtasks
            subtasks = getattr(issue.fields, 'subtasks', [])
            for subtask in subtasks:
                if hasattr(subtask, 'key'):
                    try:
                        subtask_details = self.get_issue_details(subtask.key, download_attachments=download_attachments)
                        hierarchy["subtasks"].append(subtask_details)
                    except Exception as e:
                        logger.warning(f"Error getting subtask details for {subtask.key}: {e}")
                        # Add minimal info if details can't be fetched
                        hierarchy["subtasks"].append({
                            "key": subtask.key,
                            "summary": getattr(subtask.fields, 'summary', '') if hasattr(subtask, 'fields') else '',
                            "status": getattr(getattr(subtask.fields, 'status', None), 'name', '') if (hasattr(subtask, 'fields') and hasattr(subtask.fields, 'status')) else ''
                        })
            
            # Look for issues that have this issue as their parent
            try:
                children_jql = f'parent = {issue_key}'
                children = self.search_issues(children_jql, max_results=20)
                for child in children:
                    # Skip if it's already in subtasks
                    if any(st.get('key') == child.key for st in hierarchy["subtasks"]):
                        continue
                    hierarchy["children"].append({
                        "key": child.key,
                        "summary": getattr(child.fields, 'summary', ''),
                        "status": getattr(getattr(child.fields, 'status', None), 'name', '') if hasattr(child.fields, 'status') else '',
                        "type": getattr(getattr(child.fields, 'issuetype', None), 'name', '') if hasattr(child.fields, 'issuetype') else ''
                    })
            except Exception as e:
                logger.warning(f"Error searching for child issues of {issue_key}: {e}")
                
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error getting issue hierarchy for {issue_key}: {e}", exc_info=True)
            return hierarchy


    def _get_attachments_info(self, issue: Issue) -> List[Dict[str, Any]]:
        """Return minimal attachment info for a JIRA Issue."""
        # (Keep existing implementation, ensure safe attribute access)
        if not hasattr(issue, 'fields') or not hasattr(issue.fields, 'attachment') or not issue.fields.attachment:
            return []
        attachments = []
        for att in issue.fields.attachment:
             # Basic check if 'att' looks like an attachment object
             if not hasattr(att, 'filename'): continue
             attachments.append({
                "filename": getattr(att, 'filename', 'unknown'),
                "mime_type": getattr(att, 'mimeType', 'unknown'),
                "size": getattr(att, 'size', 0),
                "created": getattr(att, 'created', 'N/A'),
                "url": getattr(att, 'content', None) # URL to the attachment content
             })
        return attachments

    # --------------------------------------------------------------------------
    # 3) Attachments: Downloading & Converting to Text
    # --------------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=12), # Increased max wait
           retry=retry_if_exception_type(requests.exceptions.RequestException)) # Retry on general request exceptions
    def download_attachment_content(self, attachment) -> Optional[bytes]:
        """Download raw bytes of a JIRA attachment with retries."""
        # (Keep existing implementation from previous response)
        url, filename = None, 'unknown'
        if isinstance(attachment, dict): url, filename = attachment.get('url'), attachment.get('filename', 'unknown')
        elif hasattr(attachment, 'content'): url, filename = attachment.content, getattr(attachment, 'filename', 'unknown')
        else: logger.error(f"Invalid attachment object type: {type(attachment)}"); return None
        if not url: logger.warning(f"No download URL for attachment: {filename}"); return None

        headers = {"Authorization": f"Bearer {self.token}"}
        logger.debug(f"Downloading attachment: {filename} from {url}")
        try:
            # Use requests session for consistency (verify=False needed for internal JIRA often)
            # Use configured timeouts
            response = requests.get(url, headers=headers, verify=False,
                                    timeout=(self.timeout, self.jira_read_timeout * 2), # Longer read timeout for download
                                    stream=True) # Use stream for potentially large files
            response.raise_for_status() # Check for HTTP errors

            # Read content carefully with streaming
            content_buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                 content_buffer.write(chunk)
            content = content_buffer.getvalue()

            logger.info(f"Downloaded attachment '{filename}' ({len(content)} bytes)")
            return content
        except requests.exceptions.HTTPError as e:
             logger.error(f"HTTP error downloading {filename}: {e.response.status_code} - {e.response.text[:200]}") # Log partial text
             # Dont retry client errors unless 429 (handled by session retry potentially)
             if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                  return None
             raise requests.exceptions.RequestException from e # Re-raise 5xx/429 for retry
        except requests.exceptions.RequestException as e:
             logger.error(f"Network error downloading {filename}: {e}")
             raise e # Re-raise for retry
        except Exception as e:
             logger.error(f"Unexpected error downloading {filename}: {e}", exc_info=True)
             return None

    def convert_attachment_to_text(self, filename: str, content_bytes: bytes, mime_type: str = "") -> str:
        """Convert raw bytes of an attachment to text, using docx/pdf converters, OCR, etc."""
        # (Keep existing implementation from previous response)
        if not content_bytes: return ""
        if not mime_type: guess_type, _ = mimetypes.guess_type(filename); mime_type = guess_type or ""
        logger.debug(f"Converting attachment '{filename}' (type: {mime_type})")
        
        # Get file extension
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # PDF
        if mime_type == "application/pdf" or extension == 'pdf':
            if PDF_AVAILABLE:
                try: 
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
                    text = []
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text.append(f"--- Page {i+1} ---")
                            text.append(page_text)
                    if text:
                        return "\n".join(text)
                except Exception as e: 
                    logger.warning(f"PyPDF2 PDF parse failed for {filename}: {e}")
            try: return pypandoc.convert_text(content_bytes, to='plain', format='pdf')
            except Exception as e: logger.warning(f"pypandoc PDF conversion failed for {filename}: {e}"); return "<PDF parsing failed>"
        
        # Word
        elif mime_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"] or extension in ['doc', 'docx']:
            try: return pypandoc.convert_text(content_bytes, to='plain', format='docx' if 'openxmlformats' in mime_type or extension == 'docx' else 'doc')
            except Exception as e: logger.warning(f"pypandoc Word conversion failed for {filename}: {e}")
        
        # Excel
        elif mime_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or extension in ['xls', 'xlsx', 'csv']:
            try:
                # For CSV, use pandas directly
                if extension == 'csv' or mime_type == 'text/csv':
                    try:
                        df = pd.read_csv(io.BytesIO(content_bytes))
                        return df.to_string(index=False)
                    except Exception as csv_err:
                        logger.warning(f"CSV parse error for {filename}: {csv_err}")
                        
                # For Excel files
                wb = load_workbook(filename=io.BytesIO(content_bytes), read_only=True, data_only=True)
                text = []
                for sheet_name in wb.sheetnames:
                     text.append(f"--- Sheet: {sheet_name} ---")
                     for row in wb[sheet_name].iter_rows(values_only=True): 
                        text.append(" | ".join([str(c) if c is not None else "" for c in row]).strip())
                return "\n".join([line for line in text if line.strip()]) # Filter empty lines
            except Exception as e: logger.warning(f"Excel parse failed for {filename}: {e}"); return "<Excel parse error>"
        
        # JSON data
        elif mime_type == "application/json" or extension == 'json':
            try:
                json_data = json.loads(content_bytes.decode('utf-8', errors='ignore'))
                return json.dumps(json_data, indent=2)
            except Exception as e:
                logger.warning(f"JSON parse error for {filename}: {e}")
                return "<JSON parsing failed>"
        
        # HTML content
        elif mime_type in ["text/html"] or extension in ['html', 'htm']:
            try:
                # Try to use BeautifulSoup if available
                from bs4 import BeautifulSoup
                html_content = content_bytes.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text(separator='\n', strip=True)
            except ImportError:
                # Direct decode if BeautifulSoup not available
                return content_bytes.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"HTML parse error for {filename}: {e}")
                return "<HTML parsing failed>"
        
        # Image / OCR
        elif mime_type.startswith("image/") and OCR_AVAILABLE:
            try: 
                image = Image.open(io.BytesIO(content_bytes))
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')
            except Exception as e: logger.warning(f"OCR parse error for image {filename}: {e}"); return "<OCR parse error>"
        
        # Plain text variants
        elif mime_type.startswith("text/") or extension in ['txt', 'md', 'py', 'java', 'js', 'c', 'cpp', 'h', 'sh', 'bat', 'sql']:
             try: return content_bytes.decode("utf-8", errors="ignore")
             except Exception: pass # Fall through
             
        # Fallback decode
        try: return content_bytes.decode("utf-8", errors="ignore")
        except Exception: pass
        try: return content_bytes.decode("latin-1", errors="ignore")
        except Exception as e: logger.warning(f"Cannot decode attachment {filename} as text: {e}"); return "<Binary or undecodable content>"
        return "<Binary or unsupported file>" # Final fallback


    # --------------------------------------------------------------------------
    # 4) MTV-Focused Methods
    # --------------------------------------------------------------------------
    def _remove_prefix(self, mtv_number: str) -> str:
        """Remove 'MTV' if present, leaving numeric part."""
        return re.sub(r'^MTV', '', mtv_number, flags=re.IGNORECASE).strip()

    async def process_mtv(self, mtv_number: str) -> List[Document]:
        """Fetch JIRA issues referencing an MTV, convert to Document objects."""
        # Use robust search for potentially better matching
        loop = asyncio.get_running_loop()
        try:
             # Run synchronous robust_search_issues in executor
             issues_data = await asyncio.wait_for(
                 loop.run_in_executor(None, self.robust_search_issues, mtv_number, None, 20), # Increased max results
                 timeout=self.jira_read_timeout * 1.5 # Allow more time for MTV search
             )
        except asyncio.TimeoutError:
             logger.error(f"Timeout processing MTV {mtv_number}")
             return []
        except Exception as e:
             logger.error(f"Error running robust search for MTV {mtv_number}: {e}")
             return []

        docs = []
        # Process the dictionary results from robust_search_issues
        for issue_data in issues_data:
             # Ensure data is dict and has key
             if not isinstance(issue_data, dict) or not issue_data.get("key"): continue
             # Create Langchain Document
             doc = Document(
                  page_content=issue_data.get("description", ""), # Use formatted description
                  metadata={
                       "key": issue_data.get("key"), "summary": issue_data.get("summary"),
                       "status": issue_data.get("status"), "type": issue_data.get("type"),
                       "priority": issue_data.get("priority"), "assignee": issue_data.get("assignee"),
                       "reporter": issue_data.get("reporter"), "created": issue_data.get("created"),
                       "updated": issue_data.get("updated"), "labels": issue_data.get("labels", []),
                       "fixVersions": issue_data.get("fixVersions", []), "components": issue_data.get("components", []),
                       "project_key": issue_data.get("project_key"), "url": issue_data.get("url"),
                       "source": f"JIRA:{issue_data.get('key')}"
                  }
             )
             docs.append(doc)
        logger.info("Found %d JIRA documents referencing %s", len(docs), mtv_number)
        return docs

    async def get_mtv_documents(self, mtv_number: str) -> List[Document]:
        """Alias for process_mtv."""
        return await self.process_mtv(mtv_number)

    # --------------------------------------------------------------------------
    # 5) Example Security Analysis (Keep placeholder or implement fully)
    # --------------------------------------------------------------------------
    def perform_security_analysis(self, issue_key: str) -> Dict[str, Any]:
        """Example method for basic security checks on an issue."""
        # (Keep existing implementation from previous response)
        info = {"issue_key": issue_key, "security_label": False, "risk_comments": [], "security_approved": None}
        try:
            issue = self.get_issue(issue_key) # Uses enhanced get_issue
            if not issue: return info
            labels = getattr(issue.fields, 'labels', [])
            if "Security" in labels or "security" in labels: info["security_label"] = True
            if hasattr(issue.fields, 'comment') and issue.fields.comment and hasattr(issue.fields.comment, 'comments'):
                for comment in issue.fields.comment.comments:
                    text_lower = getattr(comment, 'body', '').lower()
                    if any(term in text_lower for term in ["risk", "vulnerability", "exploit", "security hole", "threat"]):
                        info["risk_comments"].append({ "comment_author": getattr(getattr(comment, 'author', None), 'displayName', 'Unknown'), "snippet": getattr(comment, 'body', '')[:150] + "...", "date": getattr(comment, 'created', 'N/A') })
            # Example custom field check (adapt ID)
            # security_field_id = 'customfield_12345'
            # if hasattr(issue.fields, security_field_id): info["security_approved"] = bool(getattr(issue.fields, security_field_id))
        except Exception as e: logger.error(f"Error during security analysis for {issue_key}: {e}", exc_info=True)
        return info

    # --------------------------------------------------------------------------
    # 6) Epics & Linking (Keep as is, uses enhanced get_issue)
    # --------------------------------------------------------------------------
    def get_epic_details(self, epic_key: str) -> dict:
        """Retrieve summary/status/description of an Epic by key."""
        # (Keep existing implementation from previous response)
        try:
            epic = self.get_issue(epic_key) # Uses enhanced get_issue
            if not epic: return {"key": epic_key, "error": f"Could not retrieve epic {epic_key}"}
            return { 'key': epic.key, 'summary': getattr(epic.fields, 'summary', 'N/A'), 'status': getattr(getattr(epic.fields, 'status', None), 'name', 'Unknown'), 'url': f"{self.server}/browse/{epic_key}", 'description': getattr(epic.fields, 'description', '') or "", }
        except Exception as e: logger.error(f"Error getting epic details {epic_key}: {e}"); return {"key": epic_key, "error": str(e)}


    def get_issues_in_epic(self, epic_key: str, fields: Optional[str] = None) -> List[Issue]:
        """Return all issues in a given Epic."""
        # (Keep existing implementation from previous response)
        epic_link_field_name = '"Epic Link"' # Default legacy name - change if needed or use env var
        # Consider adding project clause if applicable: f'PROJECT = {project_key} AND ...'
        query = f'{epic_link_field_name} = "{epic_key}" ORDER BY updated DESC'
        fields_to_request = fields or self.DEFAULT_JIRA_FIELDS
        # Use paginated search for potentially many issues in an epic
        return self.search_all_issues(query, fields=fields_to_request)


    # --------------------------------------------------------------------------
    # 7) Creating & Updating Issues (Keep as is, ensure robust error handling)
    # --------------------------------------------------------------------------
    def create_issue(self, project_key: str, summary: str, description: str, issue_type: str = "Task") -> Optional[Issue]:
        """Create a new JIRA issue with minimal fields."""
        # (Keep existing implementation from previous response)
        try:
            logger.info(f"Creating issue in project {project_key} with summary: {summary}")
            new_issue = self.jira.create_issue(fields={'project': {'key': project_key}, 'summary': summary, 'description': description, 'issuetype': {'name': issue_type}})
            logger.info(f"Created issue {new_issue.key}")
            return new_issue
        except JIRAError as e: logger.error(f"Issue creation failed: {e.status_code} - {e.text}"); return None
        except Exception as e: logger.error(f"Unexpected error creating issue: {e}", exc_info=True); return None

    def transition_issue(self, issue_key: str, transition_name: str) -> bool:
        """Transition a JIRA issue to a new status by name."""
        # (Keep existing implementation from previous response)
        try:
            issue = self.get_issue(issue_key) # Uses enhanced get_issue
            if not issue: return False
            transitions = self.jira.transitions(issue)
            target_transition = next((t for t in transitions if t['name'].lower() == transition_name.lower()), None)
            if target_transition:
                self.jira.transition_issue(issue, target_transition['id'])
                logger.info(f"Transitioned issue {issue_key} to '{transition_name}' (ID: {target_transition['id']})")
                return True
            else: logger.warning(f"Transition '{transition_name}' not found for {issue_key}. Available: {[t['name'] for t in transitions]}"); return False
        except JIRAError as e: logger.error(f"Failed to transition {issue_key} to '{transition_name}': {e.status_code} - {e.text}"); return False
        except Exception as e: logger.error(f"Unexpected error transitioning {issue_key}: {e}", exc_info=True); return False

    # --------------------------------------------------------------------------
    # 8) CSV/Excel Handling (Keep as is)
    # --------------------------------------------------------------------------
    def read_mtv_excel(self, file_path: str) -> List[Dict[str, str]]:
        """Example method for reading MTV items from an Excel file."""
        # (Keep existing implementation)
        try: df = pd.read_excel(file_path)
        except Exception as e: logger.error(f"Failed to read Excel {file_path}: {e}"); return []
        mtv_list = []
        for _, row in df.iterrows():
             mtv_id = str(row.get('MTV ID', '')).strip(); mtv_name = str(row.get('Title', ''))
             if mtv_id and not mtv_id.startswith("MTV"): mtv_id = "MTV" + "".join(filter(str.isdigit, mtv_id)) # Normalize ID format
             mtv_list.append({"MTV ID": mtv_id, "Title": mtv_name, "Description": str(row.get('Description', ''))})
        return mtv_list

    def fetch_all_mtv_content(self, file_path: str) -> Dict[str, str]:
        """For each MTV in Excel, retrieve JIRA issues and store their text."""
        # (Keep existing implementation - uses async process_mtv)
        mtv_list = self.read_mtv_excel(file_path)
        results = {}
        async def fetch_one(mtv_id):
            if not mtv_id: return ""
            try: docs = await self.process_mtv(mtv_id); return "\n\n".join([d.page_content for d in docs])
            except Exception as e: logger.error(f"Error processing MTV {mtv_id} in fetch_all: {e}"); return "<Error fetching content>"
        async def fetch_all():
            tasks = {entry.get("MTV ID", f"invalid_{i}"): fetch_one(entry.get("MTV ID")) for i, entry in enumerate(mtv_list) if entry.get("MTV ID")} # Ensure key exists
            if not tasks: return {} # Handle empty MTV list
            results_list = await asyncio.gather(*tasks.values())
            return dict(zip(tasks.keys(), results_list))
        try:
             # Get or create event loop
             try: loop = asyncio.get_running_loop()
             except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
             # Ensure loop is running if we started a new one
             if not loop.is_running(): results = loop.run_until_complete(fetch_all())
             else: results = asyncio.run_coroutine_threadsafe(fetch_all(), loop).result() # If called from thread

        except Exception as e:
             logger.error(f"Error fetching all MTV content: {e}", exc_info=True)
             results = {entry.get("MTV ID", f"error_{i}"):"<Error fetching content>" for i, entry in enumerate(mtv_list)}
        return results


    # --------------------------------------------------------------------------
    # 9) Rendering references for final RAG (Keep as is)
    # --------------------------------------------------------------------------
    def render_jira_references(self, issues_list: List[Union[Document, dict, Issue]]) -> str:
        """Convert a list of Issue objects, Document objects or raw dict issues into a Markdown reference block."""
        # (Keep existing implementation from previous response)
        if not issues_list: return "### JIRA References\n\nNo JIRA issues found.\n"
        markdown = "### JIRA References\n\n"; processed_keys = set()
        for issue in issues_list:
            key, summary, status, link = 'Unknown', '(No summary)', '(No status)', '#'
            if isinstance(issue, Document): key = issue.metadata.get('key', 'Unknown'); summary = issue.metadata.get('summary', '(No summary)'); status = issue.metadata.get('status', '(No status)'); link = issue.metadata.get('url', f"{self.server}/browse/{key}")
            elif isinstance(issue, Issue): key = issue.key; summary = getattr(issue.fields, 'summary', '(No summary)'); status = getattr(getattr(issue.fields, 'status', None), 'name', '(No status)'); link = f"{self.server}/browse/{key}"
            elif isinstance(issue, dict): key = issue.get('key', '???'); summary = issue.get('summary', '(No summary)'); status = issue.get('status', '(No status)'); link = issue.get('url', f"{self.server}/browse/{key}")
            if key != 'Unknown' and key != '???' and key not in processed_keys: markdown += f"- **[{key}]({link})** - *{status}*: {summary}\n"; processed_keys.add(key)
        return markdown + "\n"

# ----- Example Main usage (async-friendly) -----
# Keep main guard and example usage as is for testing

if __name__ == "__main__":
    import asyncio
    async def main():
        try:
            jira_server = os.getenv("JIRA_SERVER")
            jira_token = os.getenv("JIRA_API_TOKEN")
            if not jira_server or not jira_token:
                logger.error("Missing required environment variables JIRA_SERVER and/or JIRA_API_TOKEN")
                return

            project = JiraProject(jira_server, jira_token)
            # Connection validation is now done in __init__

            # Example: robust search
            query = "MTV2005 rollback"
            issues_dict_list = project.robust_search_issues(query, max_results=3) # This is synchronous call
            logger.info(f"\nRobust Search Results for '{query}':")
            if issues_dict_list:
                for issue_dict in issues_dict_list:
                    logger.info(f"- {issue_dict.get('key')} ({issue_dict.get('status')}): {issue_dict.get('summary')}")
            else:
                logger.info("No issues found.")

            # Example: Get details for a specific issue
            issue_key_to_get = "VIT-61928" # Replace with a valid key
            logger.info(f"\nDetails for {issue_key_to_get}:")
            details = project.get_issue_details(issue_key_to_get) # Synchronous call
            if details and not details.get("error"):
                # Use default=str for datetime objects if not handled otherwise
                logger.info(json.dumps(details, indent=2, default=str))
            else:
                logger.error(f"Could not get details for {issue_key_to_get}: {details.get('error')}")


        except ConfigurationError as e:
             logger.error(f"Configuration Error: {e}")
        except Exception as e:
            logger.exception("Main encountered an error: %s", e)
        finally:
            logger.info("Enhanced JIRA script completed.")

    # Run the async main function
    try:
        # Setup asyncio loop if running directly
        if sys.version_info >= (3, 8) and sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script execution interrupted by user")
    except Exception as e:
        logger.exception("Unexpected error occurred at top level: %s", str(e))
