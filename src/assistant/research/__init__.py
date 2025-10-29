"""
Research package for deep learning-based research and analysis.
"""

from .types import ResearchState, Citation, ExportFormat, SearchAPI
from .tools import tool_manager
from .graph import build_research_graph
from .engine import OpenDeepResearchEngine
from .search import (
    search_all_sources,
    search_jira,
    search_perforce,
    search_confluence,
    search_vectorstore,
    search_web,
    hybrid_retrieval,
    add_documents_to_retrieval,
    ensure_vectorstore_has_documents,
    initialize_hybrid_retriever,
    create_ensemble_retriever
)
from .error_handling import (
    error_recovery,
    try_route_enterprise,
    try_get_recovery_action,
    try_check_summarization,
    try_check_technical_needs,
    try_determine_continuation
)
from .knowledge_graph import (
    generate_knowledge_graph,
    extract_citations,
    visualize_knowledge_graph,
    save_knowledge_graph
)

__all__ = [
    'ResearchState',
    'Citation',
    'ExportFormat',
    'SearchAPI',
    'OpenDeepResearchEngine',
    'build_research_graph',
    'search_all_sources',
    'search_jira',
    'search_perforce',
    'search_confluence',
    'search_vectorstore',
    'search_web',
    'hybrid_retrieval',
    'add_documents_to_retrieval',
    'ensure_vectorstore_has_documents',
    'initialize_hybrid_retriever',
    'create_ensemble_retriever',
    'error_recovery',
    'try_route_enterprise',
    'try_get_recovery_action',
    'try_check_summarization',
    'try_check_technical_needs',
    'try_determine_continuation',
    'generate_knowledge_graph',
    'extract_citations',
    'visualize_knowledge_graph',
    'save_knowledge_graph',
    'tool_manager'
] 