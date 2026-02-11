"""Entity and relation extraction from document text.

Orchestrates the full extraction pipeline: chunk text → extract entities →
extract relations → merge chunk results → persist to disk.
"""

import json
import logging
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from sift_kg.domains.models import DomainConfig
from sift_kg.extract.llm_client import LLMClient
from sift_kg.extract.models import (
    DocumentExtraction,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from sift_kg.extract.prompts import build_entity_prompt, build_relation_prompt
from sift_kg.ingest.chunker import TextChunk, chunk_text
from sift_kg.ingest.reader import read_document

logger = logging.getLogger(__name__)


def extract_from_text(
    text: str,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
    chunk_size: int = 5000,
) -> DocumentExtraction:
    """Extract entities and relations from document text.

    Pure logic — no file I/O. Testable without file fixtures.

    Args:
        text: Document text content
        doc_id: Document identifier
        llm: LLM client for API calls
        domain: Domain configuration
        chunk_size: Characters per chunk

    Returns:
        DocumentExtraction with all entities and relations
    """
    chunks = chunk_text(text, chunk_size=chunk_size)
    all_entities: list[ExtractedEntity] = []
    all_relations: list[ExtractedRelation] = []

    cost_before = llm.total_cost_usd

    for chunk in chunks:
        result = _extract_chunk(chunk, doc_id, llm, domain)
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)

    unique_entities = _dedupe_entities(all_entities)

    return DocumentExtraction(
        document_id=doc_id,
        document_path="",
        chunks_processed=len(chunks),
        entities=unique_entities,
        relations=all_relations,
        cost_usd=llm.total_cost_usd - cost_before,
        model_used=llm.model,
    )


def extract_document(
    doc_path: Path,
    llm: LLMClient,
    domain: DomainConfig,
    output_dir: Path,
    chunk_size: int = 5000,
) -> DocumentExtraction:
    """Extract entities and relations from a single document file.

    Reads the file, calls extract_from_text, and saves results to disk.
    Idempotent — skips documents that already have extraction JSON.

    Args:
        doc_path: Path to document file
        llm: LLM client for API calls
        domain: Domain configuration
        output_dir: Where to save extraction JSON
        chunk_size: Characters per chunk

    Returns:
        DocumentExtraction with all entities and relations
    """
    doc_id = doc_path.stem
    extraction_path = output_dir / "extractions" / f"{doc_id}.json"

    # Skip if already extracted (idempotent)
    if extraction_path.exists():
        logger.info(f"Skipping {doc_id} (already extracted)")
        raw = json.loads(extraction_path.read_text())
        return DocumentExtraction(**raw)

    logger.info(f"Extracting: {doc_path.name}")

    # Read document
    try:
        text = read_document(doc_path)
    except Exception as e:
        logger.error(f"Failed to read {doc_path.name}: {e}")
        return DocumentExtraction(
            document_id=doc_id,
            document_path=str(doc_path),
            error=str(e),
            model_used=llm.model,
        )

    if not text.strip():
        logger.warning(f"Empty text from {doc_path.name}")
        return DocumentExtraction(
            document_id=doc_id,
            document_path=str(doc_path),
            error="Empty document",
            model_used=llm.model,
        )

    # Extract
    extraction = extract_from_text(text, doc_id, llm, domain, chunk_size)
    extraction.document_path = str(doc_path)

    # Persist
    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_path.write_text(extraction.model_dump_json(indent=2))

    logger.info(
        f"  {doc_id}: {len(extraction.entities)} entities, "
        f"{len(extraction.relations)} relations ({extraction.chunks_processed} chunks)"
    )
    return extraction


def _extract_chunk(
    chunk: TextChunk,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
) -> ExtractionResult:
    """Extract entities and relations from a single chunk."""
    # Entity extraction
    entity_prompt = build_entity_prompt(chunk.text, doc_id, domain)
    try:
        entity_data = llm.call_json(entity_prompt)
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Entity extraction failed for {doc_id} chunk {chunk.chunk_index}: {e}")
        return ExtractionResult(source_document=doc_id, chunk_index=chunk.chunk_index)

    entities = []
    for raw in entity_data.get("entities", []):
        try:
            entities.append(ExtractedEntity(
                name=raw.get("name", ""),
                entity_type=raw.get("entity_type", "UNKNOWN"),
                attributes=raw.get("attributes", {}),
                confidence=float(raw.get("confidence", 0.5)),
                context=raw.get("context", ""),
            ))
        except Exception as e:
            logger.debug(f"Skipping malformed entity: {e}")

    # Relation extraction (only if we got entities)
    relations = []
    if entities:
        entity_list = [{"name": e.name, "entity_type": e.entity_type} for e in entities]
        relation_prompt = build_relation_prompt(chunk.text, entity_list, doc_id, domain)

        try:
            relation_data = llm.call_json(relation_prompt)
            for raw in relation_data.get("relations", []):
                try:
                    relations.append(ExtractedRelation(
                        relation_type=raw.get("relation_type", "ASSOCIATED_WITH"),
                        source_entity=raw.get("source_entity", ""),
                        target_entity=raw.get("target_entity", ""),
                        confidence=float(raw.get("confidence", 0.5)),
                        evidence=raw.get("evidence", ""),
                    ))
                except Exception as e:
                    logger.debug(f"Skipping malformed relation: {e}")
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Relation extraction failed for {doc_id} chunk {chunk.chunk_index}: {e}")

    return ExtractionResult(
        entities=entities,
        relations=relations,
        source_document=doc_id,
        chunk_index=chunk.chunk_index,
    )


def _dedupe_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Remove duplicate entities within a document by name+type."""
    seen: dict[str, ExtractedEntity] = {}
    for e in entities:
        key = f"{e.entity_type}:{e.name.lower().strip()}"
        if key in seen:
            # Keep higher confidence version
            if e.confidence > seen[key].confidence:
                seen[key] = e
        else:
            seen[key] = e
    return list(seen.values())


def extract_all(
    doc_paths: list[Path],
    llm: LLMClient,
    domain: DomainConfig,
    output_dir: Path,
    max_cost: float | None = None,
) -> list[DocumentExtraction]:
    """Extract entities and relations from multiple documents.

    Args:
        doc_paths: List of document paths
        llm: LLM client
        domain: Domain configuration
        output_dir: Output directory
        max_cost: Budget cap in USD (None = no limit)

    Returns:
        List of DocumentExtraction results
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("${task.fields[cost]:.3f}"),
    ) as progress:
        task = progress.add_task("Extracting...", total=len(doc_paths), cost=0.0)

        for doc_path in doc_paths:
            # Check budget
            if max_cost and llm.total_cost_usd >= max_cost:
                logger.warning(
                    f"Budget limit reached: ${llm.total_cost_usd:.2f} / ${max_cost:.2f}. "
                    f"Processed {len(results)}/{len(doc_paths)} documents."
                )
                break

            result = extract_document(doc_path, llm, domain, output_dir)
            results.append(result)
            progress.update(task, advance=1, cost=llm.total_cost_usd)

    return results
