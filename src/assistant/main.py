#!/usr/bin/env python3
"""
Main entry point for the Ollama Deep Researcher.
Provides programmatic access to the research engine.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, Optional

import nest_asyncio

from src.assistant.graph_2 import OpenDeepResearchEngine, check_ollama_server

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def run_research(
    topic: str,
    include_web: bool = True,
    export_format: Optional[str] = None,
    visualize_graph: bool = False,
    save_graph: bool = False,
    graph_format: str = "png",
    graph_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run research on a given topic.
    
    Args:
        topic: The research topic
        include_web: Whether to include web search results
        export_format: Format to export the research results (markdown, json, html, csv, excel)
        visualize_graph: Whether to visualize the knowledge graph
        save_graph: Whether to save the knowledge graph
        graph_format: Format to save the graph (png, pdf, svg)
        graph_filename: Custom filename for the saved graph
        
    Returns:
        Dictionary containing research results
    """
    # Check if Ollama server is running
    if not check_ollama_server():
        logger.error("Ollama server is not running. Please start it before continuing.")
        sys.exit(1)
    
    # Initialize the research engine
    engine = OpenDeepResearchEngine()
    
    # Run the research
    logger.info(f"Starting research on topic: {topic}")
    result = await engine.run_research(
        topic=topic,
        include_web=include_web,
        export_format=export_format
    )
    logger.info("Research completed successfully")
    
    # Handle knowledge graph visualization if requested
    if visualize_graph and result.get("knowledge_graph"):
        logger.info("Visualizing knowledge graph")
        engine.visualize_knowledge_graph()
    
    # Save the knowledge graph if requested
    if save_graph and result.get("knowledge_graph"):
        logger.info(f"Saving knowledge graph as {graph_format}")
        saved_file = engine.save_knowledge_graph(
            format=graph_format,
            filename=graph_filename
        )
        logger.info(f"Graph saved to: {saved_file}")
    
    return result


def command_line():
    """Command line interface for direct script execution."""
    parser = argparse.ArgumentParser(description="Ollama Deep Researcher - Advanced research engine")
    parser.add_argument("topic", help="Research topic to investigate")
    parser.add_argument("--no-web", dest="include_web", action="store_false", 
                        help="Disable web search integration")
    parser.add_argument("--export", dest="export_format", choices=["markdown", "json", "html", "csv", "excel"],
                        help="Export format for research results")
    parser.add_argument("--visualize", dest="visualize_graph", action="store_true",
                        help="Visualize the knowledge graph")
    parser.add_argument("--save-graph", dest="save_graph", action="store_true",
                        help="Save the knowledge graph visualization")
    parser.add_argument("--graph-format", default="png", choices=["png", "pdf", "svg"],
                        help="Format for the saved graph")
    parser.add_argument("--graph-filename", help="Custom filename for the saved graph")
    parser.add_argument("--output", "-o", help="Output file for the research summary")
    
    args = parser.parse_args()
    
    # Run the research
    result = asyncio.run(run_research(
        topic=args.topic,
        include_web=args.include_web,
        export_format=args.export_format,
        visualize_graph=args.visualize_graph,
        save_graph=args.save_graph,
        graph_format=args.graph_format,
        graph_filename=args.graph_filename
    ))
    
    # Output the summary
    summary = result["summary"]
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
    else:
        print(summary)
    
    return 0


if __name__ == "__main__":
    sys.exit(command_line()) 