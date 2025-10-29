# src/assistant/state.py

from typing import List, Dict, Any, Union, Annotated, Set, Optional
import operator
from langchain_core.messages import BaseMessage

class ResearchState(Dict[str, Any]):
    """
    TypedDict for storing research state. This replaces the large inline definition
    from graph_2.py with a more maintainable class.
    """
    
    # Note: TypedDict is used for static type checking but this implementation
    # uses Dict[str, Any] for runtime flexibility with proper annotations
    
    # Expected fields with proper typing:
    # --- Core state ---
    # research_topic: str
    # search_query: str
    # running_summary: str
    # research_loop_count: int
    # messages: Annotated[List[BaseMessage], operator.add]

    # --- Raw Tool Results ---
    # raw_jira_results: List[Dict[str, Any]]
    # raw_confluence_results: List[Union[Dict[str, Any], str]]
    # raw_perforce_results: List[Dict[str, Any]]
    # raw_web_search_results: List[Dict[str, Any]]
    
    # --- Binary Content Processing ---
    # raw_binary_attachments: List[Dict[str, Any]]
    # processed_binary_content: Dict[str, List[Dict[str, Any]]]
    # binary_content_summaries: Dict[str, str]

    # --- Chunking State ---
    # jira_chunks: List[List[Dict[str, Any]]]
    # jira_chunk_idx: int
    # jira_total_chunks: int
    # confluence_chunks: List[List[Union[Dict[str, Any], str]]]
    # confluence_chunk_idx: int
    # confluence_total_chunks: int
    # perforce_chunks: List[List[Dict[str, Any]]]
    # perforce_chunk_idx: int
    # perforce_total_chunks: int

    # --- Processed Results ---
    # enterprise_research_results: Annotated[List[str], operator.add]
    # sources_gathered: Annotated[List[str], operator.add]
    # pre_processed_summaries: Dict[str, List[str]]
    # jira_summary: str
    # confluence_summary: str
    # perforce_summary: str
    # web_summary: str

    # --- Analysis Results ---
    # security_findings: List[Dict[str, Any]]
    # technical_validation: Dict[str, Any]
    # reflection_data: Union[Dict[str, Any], str]
    # perforce_diff_analysis: Dict[str, Any]]
    # verification_results: Dict[str, Any]
    # citations: List[Dict[str, Any]]
    # knowledge_graph: Dict[str, Any]
    # cross_source_analysis_text: str
    # refined_summary_context: str
    # quiz_questions: List[str]
    # quiz_answers: List[Dict[str, str]]
    # identified_corrections: List[str]

    # --- Deduplication & Tracking ---
    # _content_hashes: Set[int]
    # _jira_keys_seen: Set[str]
    # _perforce_cl_numbers: Set[str]
    # _confluence_urls_seen: Set[str]
    # _confluence_ids_seen: Set[str]

    # --- Control Flow & Meta ---
    # alternative_terms: List[str]
    # _relevance_check: str
    # __hop_counter: int
    # _last_confluence_chunk_idx: int
    # _last_jira_chunk_idx: int
    # _last_perforce_chunk_idx: int
    # _stuck_confluence_count: int
    # _stuck_jira_count: int
    # _stuck_perforce_count: int
    # _errors: List[Dict[str, Any]]
    # _recovery_actions: List[str]
    # _recovery_metadata: Dict[str, Any]
    # export_format: Optional[str]
    # exported_data: Dict[str, Any]
    # config: Dict[str, Any]
    # topic_classification: str

    def __init__(self, *args, **kwargs):
        """Initialize the ResearchState with default values for required fields."""
        super().__init__(*args, **kwargs)
        
        # Set default values for essential fields if not present
        if "research_loop_count" not in self:
            self["research_loop_count"] = 0
        if "enterprise_research_results" not in self:
            self["enterprise_research_results"] = []
        if "sources_gathered" not in self:
            self["sources_gathered"] = []
        if "_errors" not in self:
            self["_errors"] = []
        if "_jira_keys_seen" not in self:
            self["_jira_keys_seen"] = set()
        if "_perforce_cl_numbers" not in self:
            self["_perforce_cl_numbers"] = set()
        if "_confluence_urls_seen" not in self:
            self["_confluence_urls_seen"] = set()
        if "_confluence_ids_seen" not in self:
            self["_confluence_ids_seen"] = set()
        if "_content_hashes" not in self:
            self["_content_hashes"] = set()
        if "messages" not in self:
            self["messages"] = []
            
    # Helper methods for easier access to state
    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the state with a default if not present."""
        return self.get(key, default)
        
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values at once."""
        self.update(updates)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchState':
        """Create a ResearchState from a dictionary."""
        return cls(**data)