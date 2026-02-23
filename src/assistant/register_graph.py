#!/usr/bin/env python3
"""
Script to register graph_3 with LangGraph server.

Usage:
    python -m src.assistant.register_graph
"""

import sys
import os
import logging
import subprocess
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure parent directory is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from langgraph.graph import StateGraph
    from langgraph.graph.state import CompiledStateGraph
    logger.info("Successfully imported langgraph modules")
except ImportError as e:
    logger.error(f"Failed to import langgraph: {e}")
    logger.error("Please install langgraph: pip install langgraph")
    sys.exit(1)

# Import the research graph
try:
    from src.assistant.graph_3 import build_graph
    logger.info("Successfully imported build_graph from graph_3")

    # Compile the graph
    graph = build_graph()
    logger.info("Successfully built research graph")

    # Validate graph type
    if not isinstance(graph, (StateGraph, CompiledStateGraph)):
        logger.error(f"Expected StateGraph or CompiledStateGraph, got {type(graph)}")
        sys.exit(1)

    logger.info(f"Graph type: {type(graph).__name__}")

    # Export graph to file
    output_dir = os.path.join(os.path.dirname(__file__), "exported_graph")
    os.makedirs(output_dir, exist_ok=True)
    export_path = os.path.join(output_dir, "research_graph.json")

    if isinstance(graph, StateGraph):
        with open(export_path, "w") as f:
            f.write(graph.to_json())
    else:
        # CompiledStateGraph
        with open(export_path, "w") as f:
            json.dump(graph.get_graph().to_json(), f, indent=2)

    logger.info(f"Graph exported to: {export_path}")

    # Attempt automatic registration with LangGraph server
    logger.info("\nTo register this graph with LangGraph server:")
    logger.info("1. Go to: https://smith.langchain.com/studio/?baseUrl=http://localhost:8123")
    logger.info("2. Click on 'Create Assistant'")
    logger.info("3. Upload the exported graph file")
    logger.info(f"   Path: {export_path}")

    logger.info("\nAttempting automatic registration...")
    try:
        result = subprocess.run(
            ["langgraph", "up", "--config", export_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info(f"Command output: {result.stdout}")
            logger.info("Successfully registered graph with LangGraph server")
        else:
            logger.warning(f"Automatic registration failed: {result.stderr}")
            logger.info("Please follow the manual registration steps above")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Automatic registration failed: {e}")
        logger.info("Please follow the manual registration steps above")

except ImportError as e:
    logger.error(f"Failed to import graph: {e}")
    logger.error("Make sure graph_3.py exists in src/assistant/")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    sys.exit(1)
