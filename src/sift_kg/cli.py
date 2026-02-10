"""CLI interface for sift-kg using Typer.

This module defines the main command-line interface for the sift-kg tool.
Commands are registered with the Typer app and provide placeholder
implementations until respective phases are completed.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sift_kg.config import SiftConfig

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
    # Load and validate configuration
    config = SiftConfig()
    try:
        config.validate_api_keys(model)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

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
    env_example_path = Path(".env.example")

    # Check if .env.example already exists
    if env_example_path.exists():
        overwrite = typer.confirm("Overwrite existing .env.example?")
        if not overwrite:
            raise typer.Exit(0)

    # .env.example template content
    env_template = """# sift-kg Configuration
# Copy this file to .env and fill in your API keys

# === LLM API Keys ===
# At least one provider required. Get keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - Anthropic: https://console.anthropic.com/settings/keys
# - Ollama: No key needed (local models)

SIFT_OPENAI_API_KEY=sk-...
SIFT_ANTHROPIC_API_KEY=sk-ant-...

# === Model Configuration ===
# Format: provider/model-name
# Examples: openai/gpt-4o-mini, anthropic/claude-haiku, ollama/llama3.3

SIFT_DEFAULT_MODEL=openai/gpt-4o-mini

# === Output Configuration ===
# Directory for all output files (extractions, graph, narrative)

SIFT_OUTPUT_DIR=output

# === Domain Configuration (Optional) ===
# Path to custom domain YAML file (default domain used if not set)

# SIFT_DOMAIN_PATH=path/to/custom/domain.yaml
"""

    # Write .env.example
    env_example_path.write_text(env_template)
    console.print("[green]✓[/green] Created .env.example")

    # Print next steps
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Copy .env.example to .env")
    console.print("  2. Add your API keys to .env")
    console.print("  3. Run: [bold]sift extract ./docs/[/bold]")

    raise typer.Exit(0)


@app.command()
def info() -> None:
    """Display project information and configuration.

    Show the current project's configuration, domain schema,
    and processing status.
    """
    # Load configuration
    config = SiftConfig()

    # Check if output directory exists
    if not config.output_dir.exists():
        console.print("[yellow]No output directory found. Run 'sift extract' first.[/yellow]")
        raise typer.Exit(0)

    # Create stats table
    table = Table(title="Project Information", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    # Configuration stats
    table.add_row("Output Directory", str(config.output_dir))
    table.add_row("Default Model", config.default_model)

    # Processing stats
    extractions_dir = config.output_dir / "extractions"
    if extractions_dir.exists():
        doc_count = len(list(extractions_dir.glob("*.json")))
        table.add_row("Documents Processed", str(doc_count))
    else:
        table.add_row("Documents Processed", "0")

    # Output file stats
    graph_exists = "Yes" if (config.output_dir / "graph_data.json").exists() else "No"
    table.add_row("Graph File Exists", graph_exists)

    narrative_exists = "Yes" if (config.output_dir / "narrative.md").exists() else "No"
    table.add_row("Narrative File Exists", narrative_exists)

    # Placeholder stats for future phases
    table.add_row("Entities Extracted", "Not available (run after Phase 4)")
    table.add_row("Relations Extracted", "Not available (run after Phase 4)")
    table.add_row("Total Cost Spent", "Not tracked yet (Phase 3+)")

    # Display table
    console.print(table)
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
