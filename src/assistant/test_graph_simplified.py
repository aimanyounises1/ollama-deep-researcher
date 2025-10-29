#!/usr/bin/env python3
"""
Simple test script for graph_2.py
"""

import sys
import os
from typing import Dict, Any

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# First, try to import the build_research_graph function
try:
    print("Testing import from src.assistant.graph_2...")
    from src.assistant.graph_2 import build_research_graph
    print("✅ Successfully imported build_research_graph")
    
    # Try to compile the graph
    print("Testing graph compilation...")
    graph = build_research_graph()
    print(f"✅ Successfully compiled graph: {type(graph)}")
    
    # Try to invoke the graph with a basic input
    print("Testing graph execution...")
    inputs: Dict[str, Any] = {
        "research_topic": "LangGraph architecture",
        "search_query": "How does LangGraph work?",
        "research_loop_count": 0,
        "sources_gathered": [],
        "enterprise_research_results": [],
        "running_summary": "",
    }
    
    # Try to run the graph with the inputs
    result = graph.invoke(inputs)
    print("✅ Successfully invoked graph")
    print(f"Result type: {type(result)}")
    
    print("\nGraph is working correctly! ✅")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 