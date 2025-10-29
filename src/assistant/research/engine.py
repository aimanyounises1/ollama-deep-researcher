"""
Research engine implementation.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from .types import ResearchState
from .graph import build_research_graph
from .tools import tool_manager

logger = logging.getLogger(__name__)

# Optional visualization support
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd
    HAS_VISUALIZATION = True
except ImportError:
    logging.warning(
        "Visualization libraries (networkx, matplotlib) not available. Graph visualization will be disabled.")
    HAS_VISUALIZATION = False

    # Define placeholder objects to avoid NameError
    class NXPlaceholder:
        DiGraph = object
        spring_layout = lambda *args, **kwargs: {}
        draw_networkx_nodes = lambda *args, **kwargs: None
        draw_networkx_edges = lambda *args, **kwargs: None
        draw_networkx_labels = lambda *args, **kwargs: None

    class PLTPlaceholder:
        figure = lambda *args, **kwargs: None
        Line2D = lambda *args, **kwargs: object()
        cm = type('cm', (), {'tab10': lambda x: x})()
        legend = lambda *args, **kwargs: None
        title = lambda *args, **kwargs: None
        axis = lambda *args, **kwargs: None
        tight_layout = lambda *args, **kwargs: None
        show = lambda *args, **kwargs: None
        close = lambda *args, **kwargs: None
        savefig = lambda *args, **kwargs: None

    class PDPlaceholder:
        DataFrame = lambda *args, **kwargs: None
        json_normalize = lambda *args, **kwargs: None
        ExcelWriter = lambda *args, **kwargs: None

    nx = NXPlaceholder()
    plt = PLTPlaceholder()
    pd = PDPlaceholder()

class OpenDeepResearchEngine:
    """Enhanced research engine that properly handles Confluence chunking."""
    
    def __init__(self):
        self.app = build_research_graph()
        self.citations = []
        self.knowledge_graphs = {}
        logger.info("OpenDeepResearchEngine initialized")

    async def run_research(self, topic: str, include_web: bool = True,
                           export_format: Optional[str] = None) -> Dict[str, Any]:
        """Run research with parallel processing and enhanced features."""
        initial_state = {
            "research_topic": topic,
            "search_query": "",
            "enterprise_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 0,
            "running_summary": "",
            "security_findings": [],
            "technical_validation": {},
            "knowledge_graph": {},
            "citations": [],
            "user_feedback": {},
            "export_format": export_format,
            "web_search_results": []
        }

        config = RunnableConfig(
            recursion_limit=10,
            configurable={
                "include_web_search": include_web,
                "export_format": export_format,
            }
        )

        logger.info(f"Starting research on topic: {topic}")
        final_state = await self.app.ainvoke(initial_state, config=config)
        logger.info(f"Research completed on topic: {topic}")

        if final_state.get('citations'):
            self.citations.extend(final_state['citations'])

        if final_state.get('knowledge_graph'):
            self.knowledge_graphs[topic] = final_state['knowledge_graph']

        return {
            "summary": final_state['running_summary'],
            "citations": final_state.get('citations', []),
            "knowledge_graph": final_state.get('knowledge_graph', {}),
            "security_findings": final_state.get('security_findings', []),
            "technical_validation": final_state.get('technical_validation', {}),
            "exported_data": final_state.get('exported_data', None)
        }

    def get_citations(self) -> List[Dict[str, Any]]:
        """Get all tracked citations."""
        return self.citations

    def visualize_knowledge_graph(self, topic: Optional[str] = None) -> None:
        """Visualize the knowledge graph for a specific topic or the most recent one."""
        if not HAS_VISUALIZATION:
            logger.warning("Visualization libraries not available")
            return

        if not self.knowledge_graphs:
            logger.warning("No knowledge graphs available")
            return

        graph_data = self.knowledge_graphs.get(topic) if topic else list(self.knowledge_graphs.values())[-1]

        if not graph_data:
            logger.warning(f"No knowledge graph found for topic: {topic}")
            return

        G = nx.DiGraph()

        for node in graph_data.get('nodes', []):
            G.add_node(node['id'],
                       label=node.get('label', node['id']),
                       type=node.get('type', 'concept'),
                       weight=node.get('importance', 1))

        for edge in graph_data.get('edges', []):
            G.add_edge(edge['source'], edge['target'],
                       label=edge.get('relation', 'related_to'),
                       weight=edge.get('strength', 1))

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)

        node_types = {d['type'] for _, d in G.nodes(data=True)}
        colors = plt.cm.tab10(range(len(node_types)))
        color_map = dict(zip(node_types, colors))

        for node_type, color in color_map.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[color], alpha=0.8)

        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        nx.draw_networkx_labels(G, pos)

        legend_items = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type)
            for node_type, color in color_map.items()]

        plt.legend(handles=legend_items, loc='upper right')
        plt.title(f"Knowledge Graph: {topic or 'Most Recent Research'}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_knowledge_graph(self, topic: Optional[str] = None,
                             file_format: str = "png",
                             filename: Optional[str] = None) -> str:
        """Save the knowledge graph visualization to a file."""
        if not HAS_VISUALIZATION:
            logger.warning("Visualization libraries not available")
            return "Visualization libraries not installed"

        if not self.knowledge_graphs:
            logger.warning("No knowledge graphs available")
            return "No knowledge graphs available"

        graph_data = self.knowledge_graphs.get(topic) if topic else list(self.knowledge_graphs.values())[-1]

        if not graph_data:
            logger.warning(f"No knowledge graph found for topic: {topic}")
            return f"No knowledge graph found for topic: {topic}"

        if not filename:
            topic_str = topic or "recent_research"
            safe_name = re.sub(r'[^\w\-_]', '_', topic_str)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"knowledge_graph_{safe_name}_{timestamp}.{file_format}"

        G = nx.DiGraph()

        for node in graph_data.get('nodes', []):
            G.add_node(
                node['id'],
                label=node.get('label', node['id']),
                type=node.get('type', 'concept'),
                weight=node.get('importance', 1)
            )

        for edge in graph_data.get('edges', []):
            G.add_edge(
                edge['source'],
                edge['target'],
                label=edge.get('relation', 'related_to'),
                weight=edge.get('strength', 1)
            )

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)

        node_types = {d['type'] for _, d in G.nodes(data=True)}
        colors = plt.cm.tab10(range(len(node_types)))
        color_map = dict(zip(node_types, colors))

        for node_type, color in color_map.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[color], alpha=0.8)

        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        nx.draw_networkx_labels(G, pos)

        legend_items = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type)
            for node_type, color in color_map.items()]

        plt.legend(handles=legend_items, loc='upper right')
        plt.title(f"Knowledge Graph: {topic or 'Research'}")
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(filename, format=file_format, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Knowledge graph saved to {filename}")
        return filename

    async def analyze_security(self, state: ResearchState, config: RunnableConfig) -> ResearchState:
        """Analyze security aspects of the research content."""
        try:
            # Get content to analyze
            content = state.get("running_summary", "")
            if not content:
                return state
                
            # Use security analyzer tool
            security_findings = await tool_manager.analyze_security(content)
            state["security_findings"] = security_findings
            
            return state
        except Exception as e:
            logger.error(f"Error in security analysis: {e}")
            state.setdefault("_errors", []).append({
                "stage": "security_analysis",
                "message": str(e)
            })
            return state
            
    async def get_perforce_changes(self, state: ResearchState) -> ResearchState:
        """Get Perforce changes related to the research topic."""
        try:
            # Get relevant paths from state
            paths = state.get("perforce_paths", [])
            if not paths:
                return state
                
            changes = []
            for path in paths:
                path_changes = await tool_manager.get_perforce_changes(path)
                changes.extend(path_changes)
                
            state["perforce_diff_analysis"] = {
                "changes": changes,
                "total_changes": len(changes)
            }
            
            return state
        except Exception as e:
            logger.error(f"Error getting Perforce changes: {e}")
            state.setdefault("_errors", []).append({
                "stage": "perforce_analysis",
                "message": str(e)
            })
            return state
            
    async def get_jira_details(self, state: ResearchState) -> ResearchState:
        """Get detailed JIRA information for relevant issues."""
        try:
            # Get JIRA keys from state
            jira_keys = state.get("jira_keys", [])
            if not jira_keys:
                return state
                
            jira_details = []
            for key in jira_keys:
                details = await tool_manager.get_jira_details(key)
                if details:
                    jira_details.append(details)
                    
            state["filtered_jira_results"] = jira_details
            
            return state
        except Exception as e:
            logger.error(f"Error getting JIRA details: {e}")
            state.setdefault("_errors", []).append({
                "stage": "jira_analysis",
                "message": str(e)
            })
            return state 