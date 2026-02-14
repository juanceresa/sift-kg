"""CLI interface for sift-kg."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sift_kg.config import SiftConfig

app = typer.Typer(
    name="sift",
    help="Document-to-knowledge-graph pipeline",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()


def _load_domain(config: SiftConfig, domain_name: str = "default"):
    """Load domain config from user path or bundled name.

    Priority: --domain CLI flag > SIFT_DOMAIN_PATH env > sift.yaml > bundled default

    The domain value from sift.yaml can be a file path or a bundled name
    (e.g. "academic", "osint"). If the path doesn't exist as a file, it's
    tried as a bundled domain name.
    """
    from sift_kg.domains.loader import DomainLoader

    loader = DomainLoader()
    if config.domain_path:
        if config.domain_path.exists():
            return loader.load_from_path(config.domain_path)
        # Try as a bundled domain name (e.g. domain: academic in sift.yaml)
        name = str(config.domain_path)
        if name in loader.list_bundled():
            return loader.load_bundled(name)
        return loader.load_from_path(config.domain_path)  # let it raise
    return loader.load_bundled(domain_name)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    # Suppress noisy libraries
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# Pipeline Commands
# ============================================================================


@app.command()
def extract(
    directory: str = typer.Argument(..., help="Directory containing documents to process"),
    model: str = typer.Option(None, help="LLM model (e.g. openai/gpt-4o-mini)"),
    domain: str | None = typer.Option(None, help="Path to custom domain YAML"),
    domain_name: str = typer.Option("default", "--domain-name", "-d", help="Bundled domain name (e.g. osint)"),
    max_cost: float | None = typer.Option(None, help="Maximum cost budget in USD"),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Characters per chunk (larger = fewer API calls, lower cost)"),
    concurrency: int = typer.Option(4, "-c", "--concurrency", help="Concurrent LLM calls per document"),
    rpm: int = typer.Option(40, "--rpm", help="Max requests per minute (prevents rate limit waste)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-extract all documents, ignoring cached results"),
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Extract entities and relations from documents."""
    _setup_logging(verbose)
    config = SiftConfig()
    effective_model = model or config.default_model

    try:
        config.validate_api_keys(effective_model)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    # Load domain
    if domain:
        config.domain_path = Path(domain)
    domain_config = _load_domain(config, domain_name)

    # Output dir
    output_dir = Path(output) if output else config.output_dir

    # Discover documents
    from sift_kg.ingest.reader import discover_documents

    doc_dir = Path(directory)
    if not doc_dir.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {directory}")
        raise typer.Exit(1)

    docs = discover_documents(doc_dir)
    if not docs:
        console.print(f"[yellow]No supported documents found in {directory}[/yellow]")
        raise typer.Exit(0)

    console.print(f"[cyan]Domain:[/cyan] {domain_config.name}")
    console.print(f"[cyan]Model:[/cyan] {effective_model}")
    console.print(f"[cyan]Documents:[/cyan] {len(docs)}")
    if max_cost:
        console.print(f"[cyan]Budget:[/cyan] ${max_cost:.2f}")
    console.print()

    # Set up LLM client and run extraction
    from sift_kg.extract.extractor import extract_all
    from sift_kg.extract.llm_client import LLMClient

    llm = LLMClient(model=effective_model, rpm=rpm)
    results = extract_all(docs, llm, domain_config, output_dir, max_cost=max_cost, concurrency=concurrency, chunk_size=chunk_size, force=force)

    # Summary
    successful = [r for r in results if not r.error]
    total_entities = sum(len(r.entities) for r in successful)
    total_relations = sum(len(r.relations) for r in successful)

    console.print()
    console.print("[green]Extraction complete![/green]")
    console.print(f"  Documents processed: {len(successful)}/{len(docs)}")
    console.print(f"  Entities extracted: {total_entities}")
    console.print(f"  Relations extracted: {total_relations}")
    console.print(f"  Total cost: ${llm.total_cost_usd:.4f}")
    console.print(f"  Output: {output_dir / 'extractions'}")
    console.print()
    console.print("Next: [cyan]sift build[/cyan] to construct the knowledge graph")


@app.command()
def build(
    domain: str | None = typer.Option(None, help="Path to custom domain YAML"),
    domain_name: str = typer.Option("default", "--domain-name", "-d", help="Bundled domain name (e.g. osint)"),
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    review_threshold: float = typer.Option(0.7, help="Flag relations below this confidence"),
    no_postprocess: bool = typer.Option(False, help="Skip redundancy removal"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Build knowledge graph from extraction results."""
    _setup_logging(verbose)
    config = SiftConfig()
    output_dir = Path(output) if output else config.output_dir

    # Load domain config to check review_required relation types
    if domain:
        config.domain_path = Path(domain)
    domain_config = _load_domain(config, domain_name)

    from sift_kg.graph.builder import build_graph, flag_relations_for_review, load_extractions

    extractions = load_extractions(output_dir)
    if not extractions:
        console.print(f"[yellow]No extractions found in {output_dir / 'extractions'}[/yellow]")
        console.print("Run [cyan]sift extract[/cyan] first.")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading:[/cyan] {len(extractions)} extraction files")

    domain_rel_types = set(domain_config.relation_types.keys()) if domain_config.relation_types else None
    kg = build_graph(extractions, postprocess=not no_postprocess, domain_relation_types=domain_rel_types)

    # Save graph
    graph_path = output_dir / "graph_data.json"
    kg.save(graph_path)

    # Flag relations for review
    review_types = {
        name for name, cfg in domain_config.relation_types.items()
        if cfg.review_required
    }
    flagged = flag_relations_for_review(kg, review_threshold, review_types)

    if flagged:
        from sift_kg.resolve.io import write_relation_review
        from sift_kg.resolve.models import RelationReviewEntry, RelationReviewFile

        entries = [RelationReviewEntry(**f) for f in flagged]
        review_file = RelationReviewFile(
            review_threshold=review_threshold, relations=entries
        )
        review_path = output_dir / "relation_review.yaml"
        write_relation_review(review_file, review_path)

    console.print()
    console.print("[green]Graph built![/green]")
    console.print(f"  Entities: {kg.entity_count}")
    console.print(f"  Relations: {kg.relation_count}")
    if flagged:
        console.print(f"  Flagged for review: {len(flagged)} relations")
    console.print(f"  Output: {graph_path}")
    console.print()
    console.print("Next: [cyan]sift resolve[/cyan] to find duplicate entities")


@app.command()
def resolve(
    model: str = typer.Option(None, help="LLM model for entity resolution"),
    domain: str | None = typer.Option(None, help="Path to custom domain YAML"),
    domain_name: str = typer.Option("default", "--domain-name", "-d", help="Bundled domain name (e.g. osint)"),
    concurrency: int = typer.Option(4, "-c", "--concurrency", help="Concurrent LLM calls"),
    rpm: int = typer.Option(40, "--rpm", help="Max requests per minute"),
    use_embeddings: bool = typer.Option(
        False, "--embeddings", help="Use semantic clustering (requires: pip install sift-kg[embeddings])"
    ),
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Find duplicate entities using LLM-based resolution."""
    _setup_logging(verbose)
    config = SiftConfig()
    effective_model = model or config.default_model
    output_dir = Path(output) if output else config.output_dir

    try:
        config.validate_api_keys(effective_model)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    # Load domain for system context
    if domain:
        config.domain_path = Path(domain)
    domain_config = _load_domain(config, domain_name)
    system_context = domain_config.system_context or ""

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        console.print("[yellow]No graph found.[/yellow] Run [cyan]sift build[/cyan] first.")
        raise typer.Exit(1)

    from sift_kg.extract.llm_client import LLMClient
    from sift_kg.graph.knowledge_graph import KnowledgeGraph
    from sift_kg.resolve.io import write_proposals
    from sift_kg.resolve.resolver import find_merge_candidates

    kg = KnowledgeGraph.load(graph_path)
    console.print(f"[cyan]Domain:[/cyan] {domain_config.name}")
    console.print(f"[cyan]Graph:[/cyan] {kg.entity_count} entities, {kg.relation_count} relations")

    llm = LLMClient(model=effective_model, rpm=rpm)
    merge_file = find_merge_candidates(kg, llm, concurrency=concurrency, use_embeddings=use_embeddings, system_context=system_context)

    if not merge_file.proposals:
        console.print("[green]No duplicates found![/green]")
        return

    proposals_path = output_dir / "merge_proposals.yaml"
    write_proposals(merge_file, proposals_path)

    console.print()
    console.print(f"[green]Found {len(merge_file.proposals)} merge proposals[/green]")
    console.print(f"  Cost: ${llm.total_cost_usd:.4f}")
    console.print(f"  Output: {proposals_path}")
    console.print()
    console.print("Next: [cyan]sift review[/cyan] to approve/reject merges interactively")
    console.print("  Or edit [cyan]{proposals_path}[/cyan] manually (DRAFT → CONFIRMED/REJECTED)")
    console.print("  Then: [cyan]sift apply-merges[/cyan]")


@app.command(name="apply-merges")
def apply_merges_cmd(
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Apply confirmed entity merges and relation rejections."""
    _setup_logging(verbose)
    config = SiftConfig()
    output_dir = Path(output) if output else config.output_dir

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        console.print("[yellow]No graph found.[/yellow] Run [cyan]sift build[/cyan] first.")
        raise typer.Exit(1)

    from sift_kg.graph.knowledge_graph import KnowledgeGraph
    from sift_kg.resolve.engine import apply_merges, apply_relation_rejections
    from sift_kg.resolve.io import read_proposals, read_relation_review

    kg = KnowledgeGraph.load(graph_path)
    console.print(f"[cyan]Graph:[/cyan] {kg.entity_count} entities, {kg.relation_count} relations")

    # Apply entity merges
    proposals_path = output_dir / "merge_proposals.yaml"
    merge_stats = {"merges_applied": 0}
    if proposals_path.exists():
        merge_file = read_proposals(proposals_path)
        confirmed_count = len(merge_file.confirmed)
        if confirmed_count:
            merge_stats = apply_merges(kg, merge_file)
            console.print(f"  Entity merges applied: {merge_stats['merges_applied']}")
        else:
            console.print("  No confirmed entity merges found")
    else:
        console.print("  No merge proposals file found")

    # Apply relation rejections
    review_path = output_dir / "relation_review.yaml"
    rejected_count = 0
    if review_path.exists():
        review_file = read_relation_review(review_path)
        rejected_count = apply_relation_rejections(kg, review_file)
        if rejected_count:
            console.print(f"  Relations rejected: {rejected_count}")

    if merge_stats.get("merges_applied", 0) or rejected_count:
        kg.save(graph_path)
        console.print()
        console.print("[green]Graph updated![/green]")
        console.print(f"  Entities: {kg.entity_count}")
        console.print(f"  Relations: {kg.relation_count}")
    else:
        console.print()
        console.print("[yellow]No changes to apply.[/yellow]")
        console.print("Edit merge_proposals.yaml or relation_review.yaml first.")

    console.print()
    console.print("Next: [cyan]sift narrate[/cyan] to generate narrative summary")


# ============================================================================
# Review Commands
# ============================================================================


@app.command()
def review(
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    auto_approve: float = typer.Option(
        0.85, "--auto-approve",
        help="Auto-confirm proposals where all members meet this confidence (0-1). Set to 1.0 to disable.",
    ),
    auto_reject: float = typer.Option(
        0.5, "--auto-reject",
        help="Auto-reject relations below this confidence (0-1). Set to 0.0 to disable.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Interactively review merge proposals and flagged relations."""
    _setup_logging(verbose)
    config = SiftConfig()
    output_dir = Path(output) if output else config.output_dir

    from sift_kg.resolve.io import (
        read_proposals,
        read_relation_review,
        write_proposals,
        write_relation_review,
    )
    from sift_kg.resolve.reviewer import review_merges, review_relations

    proposals_path = output_dir / "merge_proposals.yaml"
    review_path = output_dir / "relation_review.yaml"

    has_merges = proposals_path.exists()
    has_relations = review_path.exists()

    if not has_merges and not has_relations:
        console.print("[yellow]Nothing to review.[/yellow]")
        console.print("Run [cyan]sift resolve[/cyan] (entity merges) or [cyan]sift build[/cyan] (relation flags) first.")
        raise typer.Exit(0)

    # Review merge proposals
    if has_merges:
        merge_file = read_proposals(proposals_path)
        if merge_file.draft:
            review_merges(merge_file, auto_approve_threshold=auto_approve)
            write_proposals(merge_file, proposals_path)
            console.print()
        else:
            console.print("[dim]No DRAFT merge proposals to review.[/dim]")

    # Review flagged relations
    if has_relations:
        relation_file = read_relation_review(review_path)
        if relation_file.draft:
            review_relations(
                relation_file,
                auto_approve_threshold=auto_approve,
                auto_reject_threshold=auto_reject,
            )
            write_relation_review(relation_file, review_path)
            console.print()
        else:
            console.print("[dim]No DRAFT flagged relations to review.[/dim]")

    console.print()
    console.print("Next: [cyan]sift apply-merges[/cyan] to apply your decisions")


# ============================================================================
# Utility Commands
# ============================================================================


@app.command()
def search(
    query: str = typer.Argument(..., help="Search term (matches entity names and aliases)"),
    relations: bool = typer.Option(False, "-r", "--relations", help="Show connected entities"),
    description: bool = typer.Option(False, "-d", "--description", help="Show entity description (requires sift narrate)"),
    entity_type: str | None = typer.Option(None, "-t", "--type", help="Filter by entity type (e.g. PERSON)"),
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Search entities in the knowledge graph by name or alias."""
    _setup_logging(verbose)
    config = SiftConfig()
    output_dir = Path(output) if output else config.output_dir

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        console.print("[yellow]No graph found.[/yellow] Run [cyan]sift build[/cyan] first.")
        raise typer.Exit(1)

    import json as json_mod

    from sift_kg.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph.load(graph_path)

    # Load descriptions if requested
    descriptions: dict[str, str] = {}
    if description:
        desc_path = output_dir / "entity_descriptions.json"
        if desc_path.exists():
            descriptions = json_mod.loads(desc_path.read_text())
        else:
            console.print("[dim]No descriptions found. Run [cyan]sift narrate[/cyan] first.[/dim]")

    query_lower = query.lower()
    matches: list[tuple[str, dict]] = []

    for node_id, data in kg.graph.nodes(data=True):
        name = data.get("name", "")
        if entity_type and data.get("entity_type", "").upper() != entity_type.upper():
            continue

        # Search name
        if query_lower in name.lower():
            matches.append((node_id, data))
            continue

        # Search aliases
        attrs = data.get("attributes", {})
        aliases = attrs.get("aliases", []) or attrs.get("also_known_as", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        if any(query_lower in str(a).lower() for a in aliases):
            matches.append((node_id, data))

    if not matches:
        console.print(f"[yellow]No entities matching \"{query}\"[/yellow]")
        raise typer.Exit(0)

    console.print(f"[cyan]{len(matches)} result{'s' if len(matches) != 1 else ''}[/cyan]\n")

    for node_id, data in matches:
        name = data.get("name", "")
        etype = data.get("entity_type", "UNKNOWN")
        degree = kg.graph.degree(node_id)
        sources = data.get("source_documents", [])

        # Aliases
        attrs = data.get("attributes", {})
        aliases = attrs.get("aliases", []) or attrs.get("also_known_as", [])
        if isinstance(aliases, str):
            aliases = [aliases] if aliases else []

        console.print(f"  [bold]{etype}:[/bold] {name}")
        if aliases:
            console.print(f"    [dim]aka:[/dim] {', '.join(str(a) for a in aliases)}")
        console.print(f"    [dim]Connections:[/dim] {degree}")
        if sources:
            console.print(f"    [dim]Sources:[/dim] {', '.join(sources)}")

        # Description
        if description and node_id in descriptions:
            desc_text = descriptions[node_id]
            if len(desc_text) > 300:
                desc_text = desc_text[:300] + "..."
            console.print(f"    [dim]Description:[/dim] {desc_text}")

        # Relations
        if relations:
            limit = 1000 if verbose else 10
            all_rels: list[str] = []

            for _, target, edata in kg.graph.edges(node_id, data=True):
                rel = edata.get("relation_type", "RELATED_TO")
                target_name = kg.graph.nodes[target].get("name", target)
                all_rels.append(f"    [green]→[/green] {rel} → {target_name}")

            for source, _, edata in kg.graph.in_edges(node_id, data=True):
                if source == node_id:
                    continue
                rel = edata.get("relation_type", "RELATED_TO")
                source_name = kg.graph.nodes[source].get("name", source)
                all_rels.append(f"    [green]←[/green] {source_name} → {rel}")

            for line in all_rels[:limit]:
                console.print(line)
            if len(all_rels) > limit:
                console.print(f"    [dim]... {len(all_rels) - limit} more (use --verbose to show all)[/dim]")

        console.print()


@app.command()
def export(
    fmt: str = typer.Argument("graphml", help="Export format: json, graphml, gexf, csv"),
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    export_path: str | None = typer.Option(None, "--to", help="Export file/directory path"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Export the knowledge graph to GraphML, GEXF, CSV, or JSON."""
    _setup_logging(verbose)
    config = SiftConfig()
    output_dir = Path(output) if output else config.output_dir

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        console.print("[yellow]No graph found.[/yellow] Run [cyan]sift build[/cyan] first.")
        raise typer.Exit(1)

    from sift_kg.export import SUPPORTED_FORMATS, export_graph
    from sift_kg.graph.knowledge_graph import KnowledgeGraph

    if fmt.lower() not in SUPPORTED_FORMATS:
        console.print(f"[red]Unsupported format:[/red] {fmt}")
        console.print(f"Supported: {', '.join(SUPPORTED_FORMATS)}")
        raise typer.Exit(1)

    # Guard: catch `sift export --to json` (user meant `sift export json`)
    if export_path and export_path.lower() in SUPPORTED_FORMATS and fmt == "graphml":
        fmt = export_path.lower()
        export_path = None

    kg = KnowledgeGraph.load(graph_path)

    # Load entity descriptions if available (from sift narrate)
    descriptions: dict[str, str] | None = None
    desc_path = output_dir / "entity_descriptions.json"
    if desc_path.exists():
        import json
        descriptions = json.loads(desc_path.read_text())
        console.print(f"  Including {len(descriptions)} entity descriptions")

    if export_path:
        dest = Path(export_path)
    elif fmt == "csv":
        dest = output_dir / "csv"
    else:
        ext = {"json": "json", "graphml": "graphml", "gexf": "gexf"}[fmt.lower()]
        dest = output_dir / f"graph.{ext}"

    result = export_graph(kg, dest, fmt, descriptions=descriptions)

    console.print(f"[green]Exported![/green] ({fmt.upper()})")
    console.print(f"  Entities: {kg.entity_count}")
    console.print(f"  Relations: {kg.relation_count}")
    if fmt.lower() == "csv":
        console.print(f"  Output: {result}/entities.csv, {result}/relations.csv")
    else:
        console.print(f"  Output: {result}")


@app.command()
def view(
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    to: str | None = typer.Option(None, "--to", help="Output HTML path"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open in browser"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Open an interactive graph visualization in your browser."""
    _setup_logging(verbose)
    config = SiftConfig()
    output_dir = Path(output) if output else config.output_dir

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        console.print("[yellow]No graph found.[/yellow] Run [cyan]sift build[/cyan] first.")
        raise typer.Exit(1)

    from sift_kg.graph.knowledge_graph import KnowledgeGraph
    from sift_kg.visualize import generate_view

    kg = KnowledgeGraph.load(graph_path)
    dest = Path(to) if to else output_dir / "graph.html"

    # Load entity descriptions if narrate has been run
    desc_path = output_dir / "entity_descriptions.json"
    if desc_path.exists():
        console.print(f"[cyan]Descriptions:[/cyan] loaded from {desc_path.name}")

    result = generate_view(
        kg, dest, open_browser=not no_open,
        descriptions_path=desc_path if desc_path.exists() else None,
    )

    console.print("[green]View generated![/green]")
    console.print(f"  Entities: {kg.entity_count}")
    console.print(f"  Relations: {kg.relation_count}")
    console.print(f"  Output: {result}")
    if no_open:
        console.print(f"  Open in browser: [cyan]file://{result.resolve()}[/cyan]")


@app.command()
def domains() -> None:
    """List available bundled domains."""
    from sift_kg.domains.loader import DomainLoader

    loader = DomainLoader()
    available = loader.list_bundled()

    if not available:
        console.print("[yellow]No bundled domains found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Available Domains", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Entities", justify="right")
    table.add_column("Relations", justify="right")

    for name in available:
        domain_config = loader.load_bundled(name)
        desc = domain_config.description.strip().split("\n")[0]  # First line
        table.add_row(
            name,
            desc,
            str(len(domain_config.entity_types)),
            str(len(domain_config.relation_types)),
        )

    console.print(table)
    console.print()
    console.print("Usage: [cyan]sift extract ./docs --domain-name osint[/cyan]")
    console.print("Custom: [cyan]sift extract ./docs --domain path/to/domain.yaml[/cyan]")


@app.command()
def narrate(
    model: str = typer.Option(None, help="LLM model for narrative generation"),
    domain: str | None = typer.Option(None, help="Path to custom domain YAML"),
    domain_name: str = typer.Option("default", "--domain-name", "-d", help="Bundled domain name (e.g. osint)"),
    output: str | None = typer.Option(None, "-o", help="Output directory"),
    no_descriptions: bool = typer.Option(False, help="Skip per-entity descriptions"),
    max_cost: float | None = typer.Option(None, help="Maximum cost budget in USD"),
    communities_only: bool = typer.Option(False, "--communities-only", help="Only regenerate community labels (~$0.01)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
) -> None:
    """Generate narrative summary from the knowledge graph."""
    _setup_logging(verbose)
    config = SiftConfig()
    effective_model = model or config.default_model
    output_dir = Path(output) if output else config.output_dir

    try:
        config.validate_api_keys(effective_model)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        console.print("[yellow]No graph found.[/yellow] Run [cyan]sift build[/cyan] first.")
        raise typer.Exit(1)

    from sift_kg.extract.llm_client import LLMClient
    from sift_kg.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph.load(graph_path)

    if communities_only:
        from sift_kg.narrate.generator import regenerate_communities

        llm = LLMClient(model=effective_model)
        comm_path = regenerate_communities(kg=kg, llm=llm, output_dir=output_dir)
        console.print(f"[green]Communities regenerated:[/green] {comm_path}")
        console.print(f"  Cost: ${llm.total_cost_usd:.4f}")
        return

    # Load domain for system context
    if domain:
        config.domain_path = Path(domain)
    domain_config = _load_domain(config, domain_name)
    system_context = domain_config.system_context or ""

    from sift_kg.narrate.generator import generate_narrative

    console.print(f"[cyan]Graph:[/cyan] {kg.entity_count} entities, {kg.relation_count} relations")
    console.print(f"[cyan]Model:[/cyan] {effective_model}")
    if max_cost:
        console.print(f"[cyan]Budget:[/cyan] ${max_cost:.2f}")
    console.print()

    llm = LLMClient(model=effective_model)
    narrative_path = generate_narrative(
        kg=kg,
        llm=llm,
        output_dir=output_dir,
        system_context=system_context,
        include_entity_descriptions=not no_descriptions,
        max_cost=max_cost,
    )

    console.print()
    console.print("[green]Narrative generated![/green]")
    console.print(f"  Output: {narrative_path}")
    console.print(f"  Cost: ${llm.total_cost_usd:.4f}")
    console.print()
    console.print("Pipeline complete! Review the narrative at:")
    console.print(f"  [cyan]{narrative_path}[/cyan]")


@app.command()
def init(
    domain: str | None = typer.Option(None, help="Path to custom domain YAML to set in project config"),
) -> None:
    """Initialize a new sift-kg project in the current directory."""
    env_example_path = Path(".env.example")
    sift_yaml_path = Path("sift.yaml")

    # Create .env.example
    if not env_example_path.exists() or typer.confirm("Overwrite existing .env.example?", default=False):
        env_template = """# sift-kg Configuration
# Copy this file to .env and fill in your API keys

# === LLM API Keys ===
# At least one required. Ollama needs no key (local models).
SIFT_OPENAI_API_KEY=
SIFT_ANTHROPIC_API_KEY=

# === Model Configuration ===
# Format: provider/model-name
SIFT_DEFAULT_MODEL=openai/gpt-4o-mini
"""
        env_example_path.write_text(env_template)
        console.print("[green]Created .env.example[/green]")

    # Create sift.yaml project config
    if not sift_yaml_path.exists() or typer.confirm("Overwrite existing sift.yaml?", default=False):
        project_config = "# sift-kg project config\n# All commands pick up these settings automatically.\n\n"
        if domain:
            project_config += f"domain: {domain}\n"
        else:
            project_config += "# domain: path/to/domain.yaml\n"
        project_config += "# model: openai/gpt-4o-mini\n"
        project_config += "# output: output\n"
        sift_yaml_path.write_text(project_config)
        console.print("[green]Created sift.yaml[/green]")

    console.print("\nNext steps:")
    console.print("  1. cp .env.example .env")
    console.print("  2. Add your API key to .env")
    if not domain:
        console.print("  3. Edit sift.yaml to set your domain (or use --domain flag)")
        console.print("  4. sift extract ./docs/")
    else:
        console.print("  3. sift extract ./docs/")
    console.print()
    console.print("Available domains: [cyan]sift domains[/cyan]")
    raise typer.Exit(0)


@app.command()
def info() -> None:
    """Display project configuration and processing stats."""
    config = SiftConfig()
    domain_config = _load_domain(config)

    table = Table(title="sift-kg Project Info", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Domain", domain_config.name)
    table.add_row("Entity Types", ", ".join(domain_config.get_entity_type_names()))
    table.add_row("Relation Types", str(len(domain_config.get_relation_type_names())))
    table.add_row("Default Model", config.default_model)
    table.add_row("Output Directory", str(config.output_dir))

    extractions_dir = config.output_dir / "extractions"
    if extractions_dir.exists():
        doc_count = len(list(extractions_dir.glob("*.json")))
        table.add_row("Documents Processed", str(doc_count))
    else:
        table.add_row("Documents Processed", "0")

    graph_path = config.output_dir / "graph_data.json"
    if graph_path.exists():
        from sift_kg.graph.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph.load(graph_path)
        table.add_row("Graph", f"{kg.entity_count} entities, {kg.relation_count} relations")
    else:
        table.add_row("Graph", "Not built")

    # Check merge/review status
    proposals_path = config.output_dir / "merge_proposals.yaml"
    if proposals_path.exists():
        from sift_kg.resolve.io import read_proposals

        mf = read_proposals(proposals_path)
        table.add_row(
            "Merge Proposals",
            f"{len(mf.confirmed)} confirmed, {len(mf.draft)} draft, {len(mf.rejected)} rejected"
        )

    review_path = config.output_dir / "relation_review.yaml"
    if review_path.exists():
        from sift_kg.resolve.io import read_relation_review

        rf = read_relation_review(review_path)
        table.add_row(
            "Relation Review",
            f"{len(rf.confirmed)} confirmed, {len(rf.draft)} draft, {len(rf.rejected)} rejected"
        )

    narrative_exists = (config.output_dir / "narrative.md").exists()
    table.add_row("Narrative Generated", "Yes" if narrative_exists else "No")

    console.print(table)
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
