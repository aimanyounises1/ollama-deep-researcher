#!/usr/bin/env python3
"""
Command-line interface for the Ollama Deep Researcher.
"""

import asyncio
import os
import sys
from enum import Enum
from typing import Optional, List

import nest_asyncio
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.assistant.graph_2 import (
    OpenDeepResearchEngine, 
    ExportFormat,
    check_ollama_server
)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize Rich console for pretty output
console = Console()
app = typer.Typer(
    help="Ollama Deep Researcher: Advanced enterprise research engine",
    add_completion=False
)


class OutputFormat(str, Enum):
    """Output formats for the research results."""
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    EXCEL = "excel"


@app.command("research")
def research_command(
    topic: str = typer.Argument(..., help="The research topic to investigate"),
    include_web: bool = typer.Option(True, "--web/--no-web", help="Include web search results"),
    export_format: Optional[OutputFormat] = typer.Option(
        None, "--export", "-e", help="Export format for the research results"
    ),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Visualize the knowledge graph"),
    save_graph: bool = typer.Option(False, "--save-graph", "-s", help="Save the knowledge graph visualization"),
    graph_format: str = typer.Option("png", "--graph-format", "-g", help="Format for the knowledge graph image"),
    graph_filename: Optional[str] = typer.Option(None, "--filename", "-f", help="Filename for the saved graph"),
    open_results: bool = typer.Option(False, "--open", "-o", help="Open results after completion"),
):
    """Perform in-depth research on a topic with knowledge graph generation."""
    
    # Check if Ollama server is running
    if not check_ollama_server():
        console.print("[bold red]Error:[/] Ollama server is not running. Please start it before continuing.")
        sys.exit(1)
    
    # Initialize engine
    engine = OpenDeepResearchEngine()
    
    # Create a progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        console=console
    ) as progress:
        task = progress.add_task("[bold]Researching...[/]", total=None)
        
        # Run the research
        try:
            result = asyncio.run(engine.run_research(
                topic=topic,
                include_web=include_web,
                export_format=export_format.value if export_format else None
            ))
            progress.update(task, completed=True, description="[bold green]Research completed![/]")
        except Exception as e:
            progress.update(task, completed=True, description="[bold red]Research failed![/]")
            console.print(f"[bold red]Error:[/] {str(e)}")
            sys.exit(1)
    
    # Display the summary
    console.print("\n")
    console.print(Panel(Markdown(result["summary"]), title="Research Summary", expand=False))
    
    # Handle visualization if requested
    if visualize and result.get("knowledge_graph"):
        console.print("[bold blue]Visualizing knowledge graph...[/]")
        engine.visualize_knowledge_graph()
    
    # Save the graph if requested
    if save_graph and result.get("knowledge_graph"):
        console.print("[bold blue]Saving knowledge graph...[/]")
        saved_file = engine.save_knowledge_graph(
            format=graph_format,
            filename=graph_filename
        )
        console.print(f"[bold green]Graph saved:[/] {saved_file}")
    
    # Show citation information
    if result.get("citations"):
        console.print("\n[bold]Citation Information:[/]")
        table = Table(title="Citations")
        table.add_column("ID", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Content", style="white")
        
        for citation in result["citations"][:5]:  # Show first 5 citations
            table.add_row(
                citation.get("id", ""),
                citation.get("source_type", "Unknown"),
                citation.get("content", "")[:100] + "..." if len(citation.get("content", "")) > 100 else citation.get("content", "")
            )
        
        console.print(table)
        
        if len(result["citations"]) > 5:
            console.print(f"[italic]...and {len(result['citations']) - 5} more citations[/italic]")
    
    # Show export information
    if result.get("exported_data"):
        export_info = result["exported_data"]
        console.print(f"\n[bold green]Research exported:[/] {export_info.get('filename')}")
        
        # Open the exported file if requested
        if open_results and os.path.exists(export_info.get('filename', '')):
            console.print("[bold blue]Opening results...[/]")
            try:
                if sys.platform == 'darwin':  # macOS
                    os.system(f"open {export_info.get('filename')}")
                elif sys.platform == 'win32':  # Windows
                    os.system(f"start {export_info.get('filename')}")
                else:  # Linux
                    os.system(f"xdg-open {export_info.get('filename')}")
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/] Could not open file: {str(e)}")


@app.command("visualize")
def visualize_command(
    topic: Optional[str] = typer.Argument(None, help="The specific research topic to visualize"),
    save: bool = typer.Option(False, "--save", "-s", help="Save the visualization to a file"),
    format: str = typer.Option("png", "--format", "-f", help="Format for the saved visualization"),
    filename: Optional[str] = typer.Option(None, "--name", "-n", help="Filename for the saved visualization"),
):
    """Visualize a knowledge graph from previous research."""
    
    # Initialize engine
    engine = OpenDeepResearchEngine()
    
    # Check if we have knowledge graphs
    if not engine.knowledge_graphs:
        console.print("[bold yellow]Warning:[/] No knowledge graphs available to visualize.")
        sys.exit(1)
    
    if topic and topic not in engine.knowledge_graphs:
        console.print(f"[bold yellow]Warning:[/] No knowledge graph found for topic: {topic}")
        available_topics = list(engine.knowledge_graphs.keys())
        console.print(f"[bold]Available topics:[/] {', '.join(available_topics)}")
        sys.exit(1)
    
    if save:
        # Save the visualization
        saved_file = engine.save_knowledge_graph(
            topic=topic,
            format=format,
            filename=filename
        )
        console.print(f"[bold green]Visualization saved:[/] {saved_file}")
    else:
        # Show the visualization
        console.print("[bold blue]Visualizing knowledge graph...[/]")
        engine.visualize_knowledge_graph(topic=topic)


@app.command("list-citations")
def list_citations_command(
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of citations to display"),
    export: bool = typer.Option(False, "--export", "-e", help="Export citations to a file"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv, json)"),
):
    """List all citations from previous research sessions."""
    
    # Initialize engine
    engine = OpenDeepResearchEngine()
    
    # Get citations
    citations = engine.get_citations()
    
    if not citations:
        console.print("[bold yellow]Warning:[/] No citations available.")
        sys.exit(1)
    
    # Display citations in a table
    table = Table(title=f"Citations ({len(citations)} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Type", style="blue")
    table.add_column("Content", style="white")
    
    for citation in citations[:limit]:
        table.add_row(
            citation.get("id", ""),
            citation.get("source_id", "Unknown"),
            citation.get("source_type", "Unknown"),
            citation.get("content", "")[:100] + "..." if len(citation.get("content", "")) > 100 else citation.get("content", "")
        )
    
    console.print(table)
    
    # Export if requested
    if export:
        import json
        import csv
        from datetime import datetime
        
        filename = f"citations_{datetime.now().strftime('%Y%m%d%H%M%S')}.{format}"
        
        if format.lower() == "json":
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(citations, f, indent=2)
        elif format.lower() == "csv":
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=citations[0].keys())
                writer.writeheader()
                writer.writerows(citations)
        else:
            console.print(f"[bold red]Error:[/] Unsupported export format: {format}")
            sys.exit(1)
        
        console.print(f"[bold green]Citations exported to:[/] {filename}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main() 