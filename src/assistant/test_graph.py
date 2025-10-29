#!/usr/bin/env python

import importlib.util
import sys
from pathlib import Path

# Add the src directory to the Python path if not already there
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Try to import the graph module
try:
    from assistant.graph_2 import graph
    print("Successfully imported graph from graph_2.py")
    print(f"Graph type: {type(graph)}")
    print(f"Graph config: {graph.config}")
    print("Graph nodes:")
    for node_name in graph.nodes:
        print(f"  - {node_name}")
except Exception as e:
    print(f"Error importing graph: {e}")
    import traceback
    traceback.print_exc()

print("\nTrying alternative import path:")
try:
    # Alternative import approach
    spec = importlib.util.spec_from_file_location("graph_2", current_dir / "graph_2.py")
    graph_2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_2_module)
    
    if hasattr(graph_2_module, "graph"):
        print("Found graph attribute in graph_2.py")
        print(f"Graph type: {type(graph_2_module.graph)}")
    else:
        print("No graph attribute found in graph_2.py")
        print("Available attributes:", dir(graph_2_module))
except Exception as e:
    print(f"Error with alternative import: {e}")
    import traceback
    traceback.print_exc() 