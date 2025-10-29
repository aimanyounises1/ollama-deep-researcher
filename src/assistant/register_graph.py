#!/usr/bin/env python3
"""
Script to register the graph_2 with LangGraph server.
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

# Make sure we can import from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from langgraph.graph import StateGraph
    from langgraph.graph.state import CompiledStateGraph
    logger.info("Successfully imported langgraph modules")
except ImportError as e:
    logger.error(f"Failed to import langgraph: {e}")
    print("Please install langgraph: pip install langgraph")
    sys.exit(1)

# Try to import our graph
try:
    # First, try importing the compiled graph
    from src.assistant.graph_2 import build_research_graph
    logger.info("Successfully imported build_research_graph from graph_2")

    # Compile the graph
    graph = build_research_graph()
    logger.info("Successfully built research graph")
    
    # Check if it's a valid graph type
    if not isinstance(graph, (StateGraph, CompiledStateGraph)):
        logger.error(f"Expected StateGraph or CompiledStateGraph, got {type(graph)}")
        sys.exit(1)
    
    logger.info(f"Graph is of type: {type(graph)}")
    
    # Export graph
    output_dir = os.path.join(os.path.dirname(__file__), "exported_graph")
    os.makedirs(output_dir, exist_ok=True)
    
    # Export the graph to a file - handle both graph types
    export_path = os.path.join(output_dir, "research_graph.json")
    
    if isinstance(graph, StateGraph):
        with open(export_path, "w") as f:
            f.write(graph.to_json())
    else:  # CompiledStateGraph
        # CompiledStateGraph doesn't have to_json directly, use its config
        with open(export_path, "w") as f:
            # Get the serializable config representation
            config = graph.get_graph_config()
            json.dump(config, f, indent=2)
    
    logger.info(f"Successfully exported graph to {export_path}")
    
    # Instructions for manual registration
    print("\nTo register this graph with LangGraph server:")
    print(f"1. Go to: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024")
    print("2. Click on 'Create Assistant'")
    print("3. Upload the exported graph file")
    print(f"   Path: {export_path}")
    
    # Try to automatically register using langgraph-cli if available
    try:
        print("\nAttempting automatic registration...")
        result = subprocess.run(
            ["langgraph", "assistants", "upload", 
             "--file", export_path, 
             "--name", "DeepResearchAssistant", 
             "--description", "AI Assistant for deep enterprise research",
             "--api-url", "http://127.0.0.1:2024"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Command output: {result.stdout}")
        print("Successfully registered graph with LangGraph server")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Automatic registration failed: {e}")
        print("Please follow the manual registration steps above")

except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    sys.exit(1) 