"""Library-usable pipeline functions.

Each function corresponds to a CLI command but takes explicit parameters
instead of reading from config/CLI args. Use these from Jupyter notebooks,
web apps, or anywhere you want sift-kg as a library.
"""

import logging
from pathlib import Path

from sift_kg.domains.models import DomainConfig
from sift_kg.extract.llm_client import LLMClient
from sift_kg.extract.models import DocumentExtraction
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.resolve.models import MergeFile, RelationReviewFile

logger = logging.getLogger(__name__)


def run_extract(
    doc_dir: Path,
    model: str,
    domain: DomainConfig,
    output_dir: Path,
    max_cost: float | None = None,
) -> list[DocumentExtraction]:
    """Extract entities and relations from all documents in a directory.

    Args:
        doc_dir: Directory containing documents (PDF, text, HTML)
        model: LLM model string (e.g. "openai/gpt-4o-mini")
        domain: Domain configuration
        output_dir: Where to save extraction JSON files
        max_cost: Budget cap in USD

    Returns:
        List of DocumentExtraction results
    """
    from sift_kg.extract.extractor import extract_all
    from sift_kg.ingest.reader import discover_documents

    docs = discover_documents(doc_dir)
    if not docs:
        logger.warning(f"No supported documents found in {doc_dir}")
        return []

    llm = LLMClient(model=model)
    return extract_all(docs, llm, domain, output_dir, max_cost=max_cost)


def run_build(
    output_dir: Path,
    domain: DomainConfig,
    review_threshold: float = 0.7,
    postprocess: bool = True,
) -> KnowledgeGraph:
    """Build knowledge graph from extraction results.

    Also flags relations for review and saves the graph + review file.

    Args:
        output_dir: Directory with extraction JSON files
        domain: Domain configuration (for review_required types)
        review_threshold: Flag relations below this confidence
        postprocess: Whether to remove redundant edges

    Returns:
        Populated KnowledgeGraph
    """
    from sift_kg.graph.builder import build_graph, flag_relations_for_review, load_extractions
    from sift_kg.resolve.io import write_relation_review
    from sift_kg.resolve.models import RelationReviewEntry

    extractions = load_extractions(output_dir)
    if not extractions:
        raise FileNotFoundError(f"No extractions found in {output_dir / 'extractions'}")

    domain_rel_types = set(domain.relation_types.keys()) if domain.relation_types else None
    domain_rel_configs = {
        name: (cfg.source_types, cfg.target_types, cfg.symmetric)
        for name, cfg in domain.relation_types.items()
    } if domain.relation_types else None
    domain_canonical = {
        name: (cfg.canonical_names, cfg.canonical_fallback_type)
        for name, cfg in domain.entity_types.items()
        if cfg.canonical_names
    } or None
    kg = build_graph(
        extractions,
        postprocess=postprocess,
        domain_relation_types=domain_rel_types,
        domain_relation_configs=domain_rel_configs,
        domain_canonical_entities=domain_canonical,
    )

    # Save graph
    graph_path = output_dir / "graph_data.json"
    kg.save(graph_path)

    # Flag relations for review
    review_types = {
        name for name, cfg in domain.relation_types.items()
        if cfg.review_required
    }
    flagged = flag_relations_for_review(kg, review_threshold, review_types)

    if flagged:
        entries = [RelationReviewEntry(**f) for f in flagged]
        review_file = RelationReviewFile(
            review_threshold=review_threshold, relations=entries
        )
        write_relation_review(review_file, output_dir / "relation_review.yaml")

    return kg


def run_resolve(
    output_dir: Path,
    model: str,
) -> MergeFile:
    """Find duplicate entities using LLM-based resolution.

    Args:
        output_dir: Directory with graph_data.json
        model: LLM model string

    Returns:
        MergeFile with DRAFT proposals
    """
    from sift_kg.resolve.io import write_proposals
    from sift_kg.resolve.resolver import find_merge_candidates

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph found at {graph_path}")

    kg = KnowledgeGraph.load(graph_path)
    llm = LLMClient(model=model)
    merge_file, variant_relations = find_merge_candidates(kg, llm)

    if merge_file.proposals:
        write_proposals(merge_file, output_dir / "merge_proposals.yaml")

    if variant_relations:
        from sift_kg.resolve.io import read_relation_review, write_relation_review
        from sift_kg.resolve.models import RelationReviewFile

        review_path = output_dir / "relation_review.yaml"
        if review_path.exists():
            review_file = read_relation_review(review_path)
        else:
            review_file = RelationReviewFile()
        review_file.relations.extend(variant_relations)
        write_relation_review(review_file, review_path)

    return merge_file


def run_apply_merges(output_dir: Path) -> dict:
    """Apply confirmed entity merges and relation rejections.

    Args:
        output_dir: Directory with graph_data.json and review files

    Returns:
        Stats dict with merges_applied, rejected_count
    """
    from sift_kg.resolve.engine import apply_merges, apply_relation_rejections
    from sift_kg.resolve.io import read_proposals, read_relation_review

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph found at {graph_path}")

    kg = KnowledgeGraph.load(graph_path)

    merge_stats = {"merges_applied": 0}
    proposals_path = output_dir / "merge_proposals.yaml"
    if proposals_path.exists():
        merge_file = read_proposals(proposals_path)
        if merge_file.confirmed:
            merge_stats = apply_merges(kg, merge_file)

    rejected_count = 0
    review_path = output_dir / "relation_review.yaml"
    if review_path.exists():
        review_file = read_relation_review(review_path)
        rejected_count = apply_relation_rejections(kg, review_file)

    if merge_stats.get("merges_applied", 0) or rejected_count:
        kg.save(graph_path)

    return {"merges_applied": merge_stats.get("merges_applied", 0), "rejected_count": rejected_count}


def run_narrate(
    output_dir: Path,
    model: str,
    system_context: str = "",
    include_entity_descriptions: bool = True,
    max_cost: float | None = None,
) -> Path:
    """Generate narrative summary from the knowledge graph.

    Args:
        output_dir: Directory with graph_data.json
        model: LLM model string
        system_context: Optional domain context for LLM
        include_entity_descriptions: Generate per-entity descriptions
        max_cost: Budget cap in USD

    Returns:
        Path to generated narrative.md
    """
    from sift_kg.narrate.generator import generate_narrative

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph found at {graph_path}")

    kg = KnowledgeGraph.load(graph_path)
    llm = LLMClient(model=model)

    return generate_narrative(
        kg=kg,
        llm=llm,
        output_dir=output_dir,
        system_context=system_context,
        include_entity_descriptions=include_entity_descriptions,
        max_cost=max_cost,
    )


def run_export(
    output_dir: Path,
    fmt: str = "json",
    export_path: Path | None = None,
) -> Path:
    """Export the knowledge graph to a specified format.

    Args:
        output_dir: Directory with graph_data.json
        fmt: Export format — "json", "graphml", "gexf", or "csv"
        export_path: Where to write output (default: output_dir/graph.{fmt})

    Returns:
        Path to the exported file or directory
    """
    from sift_kg.export import export_graph

    graph_path = output_dir / "graph_data.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph found at {graph_path}")

    kg = KnowledgeGraph.load(graph_path)

    if export_path is None:
        if fmt == "csv":
            export_path = output_dir / "csv"
        else:
            ext = {"json": "json", "graphml": "graphml", "gexf": "gexf"}[fmt]
            export_path = output_dir / f"graph.{ext}"

    return export_graph(kg, export_path, fmt)


def run_pipeline(
    doc_dir: Path,
    model: str,
    domain: DomainConfig,
    output_dir: Path,
    max_cost: float | None = None,
    include_narrative: bool = True,
) -> Path:
    """Run the full pipeline: extract → build → narrate.

    Skips resolve/apply-merges (those require human review).

    Args:
        doc_dir: Directory containing documents
        model: LLM model string
        domain: Domain configuration
        output_dir: Output directory for all artifacts
        max_cost: Budget cap in USD
        include_narrative: Whether to generate narrative at the end

    Returns:
        Path to output directory
    """
    run_extract(doc_dir, model, domain, output_dir, max_cost=max_cost)
    run_build(output_dir, domain)

    if include_narrative:
        system_context = domain.system_context or ""
        run_narrate(output_dir, model, system_context=system_context, max_cost=max_cost)

    return output_dir
