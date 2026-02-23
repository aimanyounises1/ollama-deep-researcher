import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union

import aiohttp

logger = logging.getLogger(__name__)

class JiraClient:
    """
    Client for interacting with Jira API
    """
    def __init__(
        self, 
        server: str = "", 
        username: str = "", 
        password: str = "",
        timeout: int = 60
    ):
        """
        Initialize Jira client with connection parameters
        
        Args:
            server: Jira server URL
            username: Jira username
            password: Jira API token
            timeout: Request timeout in seconds
        """
        self.server = server or os.environ.get("JIRA_SERVER", "")
        self.username = username or os.environ.get("JIRA_USERNAME", "")
        self.password = password or os.environ.get("JIRA_API_TOKEN", "")
        self.timeout = timeout
        
        # Validate configuration
        if not self.server:
            logger.warning("No Jira server URL provided")
        if not self.username or not self.password:
            logger.warning("Jira credentials not fully provided")
    
    async def search_issues(self, query: str, max_results: int = 25, fields: Optional[str] = None) -> Union[List[Dict[str, Any]], str]:
        """
        Search for Jira issues using JQL
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            fields: Comma-separated list of fields to include
            
        Returns:
            List of Jira issues or error message
        """
        # Check if credentials are available
        if not self.server or not self.username or not self.password:
            logger.error("Jira API credentials or server URL not configured")
            return "Error: Jira API credentials or server URL not configured"
        
        # Convert query to JQL if it's not already JQL
        if not any(op in query for op in ["=", "~", ">", "<", "AND", "OR", "NOT"]):
            # Simple text search
            jql = f'text ~ "{query}" ORDER BY created DESC'
        else:
            # Assume it's already JQL
            jql = query
            
        fields_param = fields or "summary,description,status,priority,assignee,reporter,created,updated,issuetype,labels,comment"
            
        api_url = f"{self.server.rstrip('/')}/rest/api/2/search"
        params = {
            "jql": jql,
            "maxResults": max_results,
            "fields": fields_param
        }
        
        try:
            auth = aiohttp.BasicAuth(self.username, self.password)
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.get(api_url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "issues" in data and isinstance(data["issues"], list):
                            logger.info(f"Found {len(data['issues'])} Jira issues")
                            return data["issues"]
                        else:
                            logger.warning("Unexpected Jira API response structure")
                            return []
                    else:
                        error_msg = f"Jira API error: HTTP {response.status}"
                        logger.error(error_msg)
                        try:
                            error_data = await response.text()
                            logger.error(f"Jira API error details: {error_data}")
                        except:
                            pass
                        return error_msg
        except Exception as e:
            logger.error(f"Error connecting to Jira API: {e}")
            return f"Error connecting to Jira API: {str(e)}"
            
    async def search_issues_by_ids(self, ids: List[str], fields: Optional[str] = None) -> Union[List[Dict[str, Any]], str]:
        """
        Search for specific Jira issues by their IDs
        
        Args:
            ids: List of issue IDs or keys
            fields: Comma-separated list of fields to include
            
        Returns:
            List of Jira issues or error message
        """
        if not ids:
            return []
            
        # Create JQL for direct ID/key search
        id_list = ",".join([f'"{id}"' for id in ids])
        jql = f"id in ({id_list}) OR key in ({id_list})"
        
        # Some IDs might be in different formats (e.g., "MTV-1234" or "VIT-5678")
        # Try to extract them if they match common patterns
        mtv_pattern = [id for id in ids if "mtv" in id.lower()]
        vit_pattern = [id for id in ids if "vit" in id.lower()]
        
        # Add any specific ID patterns found
        if mtv_pattern:
            mtv_list = ",".join([f'"{id}"' for id in mtv_pattern])
            jql += f" OR key in ({mtv_list})"
        if vit_pattern:
            vit_list = ",".join([f'"{id}"' for id in vit_pattern])
            jql += f" OR key in ({vit_list})"
            
        # Add a text search component for the IDs as well, to catch references
        text_search = " OR ".join([f'text ~ "{id}"' for id in ids])
        jql = f"({jql}) OR ({text_search})"
        
        logger.info(f"Searching Jira with JQL: {jql}")
        return await self.search_issues(jql, max_results=50, fields=fields)
        
    async def get_issue(self, issue_key: str, expand: Optional[str] = None) -> Union[Dict[str, Any], str]:
        """
        Get detailed information about a specific Jira issue
        
        Args:
            issue_key: Jira issue key (e.g., "PROJ-123")
            expand: Comma-separated list of fields to expand
            
        Returns:
            Jira issue details or error message
        """
        # Check if credentials are available
        if not self.server or not self.username or not self.password:
            logger.error("Jira API credentials or server URL not configured")
            return "Error: Jira API credentials or server URL not configured"
            
        api_url = f"{self.server.rstrip('/')}/rest/api/2/issue/{issue_key}"
        params = {}
        if expand:
            params["expand"] = expand
            
        try:
            auth = aiohttp.BasicAuth(self.username, self.password)
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.get(api_url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully retrieved details for issue {issue_key}")
                        return data
                    else:
                        error_msg = f"Jira API error: HTTP {response.status}"
                        logger.error(error_msg)
                        try:
                            error_data = await response.text()
                            logger.error(f"Jira API error details: {error_data}")
                        except:
                            pass
                        return error_msg
        except Exception as e:
            logger.error(f"Error connecting to Jira API: {e}")
            return f"Error connecting to Jira API: {str(e)}"
    
    async def get_issue_attachments(self, issue_key: str) -> Union[List[Dict[str, Any]], str]:
        """
        Get attachments for a specific Jira issue
        
        Args:
            issue_key: Jira issue key (e.g., "PROJ-123")
            
        Returns:
            List of attachments or error message
        """
        issue_data = await self.get_issue(issue_key, expand="attachment")
        
        if isinstance(issue_data, dict) and "fields" in issue_data:
            if "attachment" in issue_data["fields"]:
                return issue_data["fields"]["attachment"]
            else:
                return []
        elif isinstance(issue_data, str) and issue_data.startswith("Error"):
            return issue_data
        else:
            return [] 
