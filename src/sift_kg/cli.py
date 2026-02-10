"""CLI interface for sift-kg using Typer.

This module defines the main command-line interface for the sift-kg tool.
Commands are registered with the Typer app and provide placeholder
implementations until respective phases are completed.
"""

import typer
from rich.console import Console

# Initialize Typer app with configuration
app = typer.Typer(
    name="sift",
    help="Document-to-knowledge-graph pipeline",
    add_completion=True,
    rich_markup_mode="rich",
)

# Rich console for colored output
console = Console()


@app.command()
def extract(
    directory: str = typer.Argument(..., help="Directory containing documents to process"),
    model: str = typer.Option("openai/gpt-4o-mini", help="LLM model to use"),
) -> None:
    """Extract entities and relations from documents using LLM.

    Process all documents in the specified directory and extract
    structured knowledge (entities, relations, dates) into a graph.
    """
    console.print("[yellow]⚠️  Not implemented yet (Phase 3)[/yellow]")
    console.print(f"Would process directory: {directory}")
    console.print(f"Would use model: {model}")
    raise typer.Exit(0)


@app.command()
def review() -> None:
    """Review and merge duplicate entities interactively.

    Launch an interactive TUI for reviewing entity merge candidates
    and confirming or rejecting proposed merges.
    """
    console.print("[yellow]⚠️  Not implemented yet (Phase 5)[/yellow]")
    raise typer.Exit(0)


@app.command()
def narrate() -> None:
    """Generate narrative summaries from the knowledge graph.

    Create human-readable summaries and timelines from extracted
    entities and relations.
    """
    console.print("[yellow]⚠️  Not implemented yet (Phase 6)[/yellow]")
    raise typer.Exit(0)


@app.command()
def init() -> None:
    """Initialize a new sift-kg project in the current directory.

    Create configuration files and directory structure for a new
    document processing project.
    """
    console.print("[yellow]⚠️  Not implemented yet (Plan 01-02)[/yellow]")
    raise typer.Exit(0)


@app.command()
def info() -> None:
    """Display project information and configuration.

    Show the current project's configuration, domain schema,
    and processing status.
    """
    console.print("[yellow]⚠️  Not implemented yet (Plan 01-02)[/yellow]")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
