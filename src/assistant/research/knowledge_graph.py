"""
Knowledge graph functionality for the research package.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langsmith import traceable
from src.utils.config import Configuration
from .types import ResearchState
from .utils import clean_llm_output

logger = logging.getLogger(__name__)

@traceable
async def generate_knowledge_graph(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Generate a knowledge graph from the research results."""
    try:
        # Get the summary and sources
        summary = state.get("running_summary", "")
        sources = state.get("sources_gathered", [])
        
        if not summary and not sources:
            return state
            
        # Use Ollama to generate knowledge graph
        cfg = Configuration.from_runnable_config(config)
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.local_llm,
            temperature=0
        )
        
        prompt = f"""Generate a knowledge graph from the following information:

Summary:
{summary}

Sources:
{json.dumps(sources, indent=2)}

Knowledge Graph (in JSON format with nodes and edges):"""
        
        response = await llm.ainvoke(prompt)
        try:
            graph_data = json.loads(response.content)
            state["knowledge_graph"] = graph_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse knowledge graph JSON")
            state["knowledge_graph"] = {
                "nodes": [],
                "edges": []
            }
            
        return state
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {e}")
        state.setdefault("_errors", []).append({
            "stage": "knowledge_graph",
            "message": str(e)
        })
        return state

@traceable
async def extract_citations(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Extract citations from the research results."""
    try:
        # Get all sources
        sources = state.get("sources_gathered", [])
        if not sources:
            return state
            
        citations = []
        for source in sources:
            # Extract source type
            source_type = "unknown"
            if "=== Jira Results ===" in source:
                source_type = "jira"
            elif "=== Confluence Results ===" in source:
                source_type = "confluence"
            elif "=== Perforce Results ===" in source:
                source_type = "perforce"
            elif "=== Vectorstore Results ===" in source:
                source_type = "vectorstore"
                
            # Create citation
            citation = {
                "source_id": str(hash(source)),
                "source_type": source_type,
                "content": source[:500] + "..." if len(source) > 500 else source,
                "timestamp": datetime.now().isoformat()
            }
            citations.append(citation)
            
        state["citations"] = citations
        return state
    except Exception as e:
        logger.error(f"Error extracting citations: {e}")
        state.setdefault("_errors", []).append({
            "stage": "citation_extraction",
            "message": str(e)
        })
        return state

def visualize_knowledge_graph(graph_data: Dict[str, Any], topic: Optional[str] = None) -> None:
    """Visualize the knowledge graph."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.get('nodes', []):
            G.add_node(
                node['id'],
                label=node.get('label', node['id']),
                type=node.get('type', 'concept'),
                weight=node.get('importance', 1)
            )
            
        # Add edges
        for edge in graph_data.get('edges', []):
            G.add_edge(
                edge['source'],
                edge['target'],
                label=edge.get('relation', 'related_to'),
                weight=edge.get('strength', 1)
            )
            
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors by type
        node_types = {d['type'] for _, d in G.nodes(data=True)}
        colors = plt.cm.tab10(range(len(node_types)))
        color_map = dict(zip(node_types, colors))
        
        for node_type, color in color_map.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[color], alpha=0.8)
            
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        # Create legend
        legend_items = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type)
            for node_type, color in color_map.items()
        ]
        
        plt.legend(handles=legend_items, loc='upper right')
        plt.title(f"Knowledge Graph: {topic or 'Research'}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Visualization libraries not available")
    except Exception as e:
        logger.error(f"Error visualizing knowledge graph: {e}")

def save_knowledge_graph(graph_data: Dict[str, Any], topic: Optional[str] = None,
                        file_format: str = "png", filename: Optional[str] = None) -> str:
    """Save the knowledge graph visualization to a file."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        if not filename:
            topic_str = topic or "recent_research"
            safe_name = re.sub(r'[^\w\-_]', '_', topic_str)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"knowledge_graph_{safe_name}_{timestamp}.{file_format}"
            
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.get('nodes', []):
            G.add_node(
                node['id'],
                label=node.get('label', node['id']),
                type=node.get('type', 'concept'),
                weight=node.get('importance', 1)
            )
            
        # Add edges
        for edge in graph_data.get('edges', []):
            G.add_edge(
                edge['source'],
                edge['target'],
                label=edge.get('relation', 'related_to'),
                weight=edge.get('strength', 1)
            )
            
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors by type
        node_types = {d['type'] for _, d in G.nodes(data=True)}
        colors = plt.cm.tab10(range(len(node_types)))
        color_map = dict(zip(node_types, colors))
        
        for node_type, color in color_map.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[color], alpha=0.8)
            
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        # Create legend
        legend_items = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type)
            for node_type, color in color_map.items()
        ]
        
        plt.legend(handles=legend_items, loc='upper right')
        plt.title(f"Knowledge Graph: {topic or 'Research'}")
        plt.axis('off')
        plt.tight_layout()
        
        # Save to file
        plt.savefig(filename, format=file_format, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Knowledge graph saved to {filename}")
        return filename
        
    except ImportError:
        logger.warning("Visualization libraries not available")
        return "Visualization libraries not installed"
    except Exception as e:
        logger.error(f"Error saving knowledge graph: {e}")
        return f"Error saving knowledge graph: {str(e)}" 