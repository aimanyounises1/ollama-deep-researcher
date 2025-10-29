"""
Type definitions for the research package.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, TypedDict, Annotated
from uuid import uuid4
import operator
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict, total=False):
    """State for the research graph."""
    # Core fields
    research_topic: str
    search_query: str
    enterprise_research_results: Annotated[List[str], operator.add]  # Raw results from search_all_sources
    sources_gathered: Annotated[List[str], operator.add]  # Combined list for potential summarization
    running_summary: str  # The iteratively refined summary
    research_loop_count: int
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Analysis results
    security_findings: List[Dict[str, Any]]
    technical_validation: Dict[str, Any]
    reflection_data: Union[Dict[str, Any], str]
    perforce_diff_analysis: Dict[str, Any]
    filtered_jira_results: List[Dict[str, Any]]
    
    # Confluence chunking fields
    full_confluence_text: str  # Full text from Confluence to be processed in chunks
    confluence_chunk_idx: int  # Current chunk index being processed
    confluence_total_chunks: int  # Total number of chunks of Confluence content
    
    # Enhanced features
    knowledge_graph: Dict[str, Any]
    citations: List[Dict[str, Any]]  # Citations extracted during summarization/analysis
    web_search_results: List[Dict[str, Any]]  # Raw web results
    alternative_terms: List[str]
    _relevance_check: str  # Added for corrective RAG
    
    # User interaction
    user_feedback: Dict[str, Any]
    export_format: Optional[str]
    exported_data: Dict[str, Any]
    
    # Error handling and metadata
    _errors: List[Dict[str, Any]]
    _recovery_actions: List[str]
    _recovery_metadata: Dict[str, Any]
    metadata: Dict[str, Any]

class SearchAPI(Enum):
    """Supported search APIs."""
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

class ExportFormat(Enum):
    """Supported export formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"

class Citation:
    """Citation tracking for research sources."""
    
    def __init__(self, source_id: str, source_type: str, content: str, url: Optional[str] = None):
        self.id = str(uuid4())[:8]
        self.source_id = source_id
        self.source_type = source_type
        self.content = content
        self.url = url
        self.timestamp = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "content": self.content,
            "url": self.url,
            "timestamp": self.timestamp
        } 