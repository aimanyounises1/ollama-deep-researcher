"""
Integration module for various research tools.
"""

import logging
from typing import Optional, Dict, Any, List

# Create dummy classes for those that might not exist
class DummySearcher:
    """A fallback class when the actual class is not available."""
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Dummy search method."""
        return []
        
    def analyze_file_content(self, content: str) -> Dict[str, Any]:
        """Dummy analysis method."""
        return {}
        
    async def get_changes(self, path: str) -> List[Dict[str, Any]]:
        """Dummy get changes method."""
        return []
        
    async def get_issue(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """Dummy get issue method."""
        return None

# Import with fallbacks
try:
    from src.assistant.sources.solution_book import SolutionBookQuerier
except ImportError:
    SolutionBookQuerier = DummySearcher
    
try:
    from src.assistant.sources.perforce import search_perforce
except ImportError:
    search_perforce = lambda query: []
    
try:
    from src.assistant.sources.confluence import search_confluence
except ImportError:
    search_confluence = lambda query: []
    
try:
    from src.assistant.sources.perforce_client import PerforceHelper
except ImportError:
    PerforceHelper = DummySearcher
    
try:
    from src.assistant.sources.jira_tool import JiraProject
except ImportError:
    JiraProject = DummySearcher
    
try:
    from src.assistant.analyzers.security_analyzer import SecurityAnalyzer
except ImportError:
    SecurityAnalyzer = DummySearcher
    
try:
    from src.assistant.sources.jira_runner import search_jira
except ImportError:
    search_jira = lambda query: []

logger = logging.getLogger(__name__)

class ToolManager:
    """Manager class for handling various research tools."""
    
    def __init__(self):
        self._solution_book = None
        self._perforce = None
        self._jira = None
        self._security = None
        
    @property
    def solution_book(self):
        if not self._solution_book:
            self._solution_book = SolutionBookQuerier()
        return self._solution_book
        
    @property
    def perforce(self):
        if not self._perforce:
            self._perforce = PerforceHelper()
        return self._perforce
        
    @property
    def jira(self):
        if not self._jira:
            self._jira = JiraProject()
        return self._jira
        
    @property
    def security(self):
        if not self._security:
            self._security = SecurityAnalyzer()
        return self._security

    async def search_solution_book(self, query: str) -> List[Dict[str, Any]]:
        """Search the solution book."""
        try:
            return await self.solution_book.search(query)
        except Exception as e:
            logger.error(f"Error searching solution book: {e}")
            return []

    async def search_perforce(self, query: str) -> List[Dict[str, Any]]:
        """Search Perforce."""
        try:
            if callable(search_perforce):
                return await search_perforce(query)
            return []
        except Exception as e:
            logger.error(f"Error searching Perforce: {e}")
            return []

    async def search_confluence(self, query: str) -> List[Dict[str, Any]]:
        """Search Confluence."""
        try:
            if callable(search_confluence):
                return await search_confluence(query)
            return []
        except Exception as e:
            logger.error(f"Error searching Confluence: {e}")
            return []

    async def search_jira(self, query: str) -> List[Dict[str, Any]]:
        """Search JIRA."""
        try:
            if callable(search_jira):
                return await search_jira(query)
            return []
        except Exception as e:
            logger.error(f"Error searching JIRA: {e}")
            return []

    async def analyze_security(self, content: str) -> Dict[str, Any]:
        """Analyze security aspects of content."""
        try:
            return self.security.analyze_file_content(content)
        except Exception as e:
            logger.error(f"Error analyzing security: {e}")
            return {}

    async def get_perforce_changes(self, path: str) -> List[Dict[str, Any]]:
        """Get Perforce changes for a path."""
        try:
            return await self.perforce.get_changes(path)
        except Exception as e:
            logger.error(f"Error getting Perforce changes: {e}")
            return []

    async def get_jira_details(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a JIRA issue."""
        try:
            return await self.jira.get_issue(issue_key)
        except Exception as e:
            logger.error(f"Error getting JIRA details: {e}")
            return None

# Create a global instance
tool_manager = ToolManager() 
