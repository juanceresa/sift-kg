"""Entity and relation extraction from document text.

Orchestrates the full extraction pipeline: chunk text → extract entities →
extract relations → merge chunk results → persist to disk.

Uses async concurrency to process multiple chunks in parallel.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
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
from sift_kg.extract.prompts import build_combined_prompt
from sift_kg.ingest.chunker import TextChunk, chunk_text
from sift_kg.ingest.reader import read_document

logger = logging.getLogger(__name__)

# Default concurrent LLM calls. Balances speed vs rate limits.
DEFAULT_CONCURRENCY = 4


def _check_stale(
    existing: DocumentExtraction,
    model: str,
    domain_name: str,
    chunk_size: int,
) -> str | None:
    """Check if an existing extraction is stale. Returns reason string or None."""
    # No metadata = old extraction before incremental support, re-extract
    if not existing.domain_name and not existing.chunk_size:
        return "missing metadata (pre-incremental extraction)"
    if existing.model_used != model:
        return f"model changed ({existing.model_used} → {model})"
    if existing.domain_name != domain_name:
        return f"domain changed ({existing.domain_name} → {domain_name})"
    if existing.chunk_size != chunk_size:
        return f"chunk size changed ({existing.chunk_size} → {chunk_size})"
    return None


def extract_from_text(
    text: str,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
    chunk_size: int = 10000,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> DocumentExtraction:
    """Extract entities and relations from document text.

    Pure logic — no file I/O. Testable without file fixtures.
    Runs async extraction internally for concurrency.
    """
    return asyncio.run(
        _aextract_from_text(text, doc_id, llm, domain, chunk_size, concurrency)
    )


async def _aextract_from_text(
    text: str,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
    chunk_size: int,
    concurrency: int,
) -> DocumentExtraction:
    """Async extraction — processes chunks concurrently."""
    chunks = chunk_text(text, chunk_size=chunk_size)
    cost_before = llm.total_cost_usd
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(chunk: TextChunk) -> ExtractionResult:
        async with sem:
            return await _aextract_chunk(chunk, doc_id, llm, domain)

    results = await asyncio.gather(*[_bounded(c) for c in chunks])

    all_entities: list[ExtractedEntity] = []
    all_relations: list[ExtractedRelation] = []
    for result in results:
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
    chunk_size: int = 10000,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
) -> DocumentExtraction:
    """Extract entities and relations from a single document file.

    Reads the file, calls extract_from_text, and saves results to disk.
    Incremental — skips documents whose extraction config matches.
    Use force=True to re-extract regardless.
    """
    doc_id = doc_path.stem
    extraction_path = output_dir / "extractions" / f"{doc_id}.json"

    if extraction_path.exists() and not force:
        existing = DocumentExtraction(**json.loads(extraction_path.read_text()))
        reason = _check_stale(existing, llm.model, domain.name, chunk_size)
        if reason is None:
            logger.info(f"Skipping {doc_id} (already extracted)")
            return existing
        logger.info(f"Re-extracting {doc_id}: {reason}")

    logger.info(f"Extracting: {doc_path.name}")

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

    extraction = extract_from_text(text, doc_id, llm, domain, chunk_size, concurrency)
    extraction.document_path = str(doc_path)
    extraction.domain_name = domain.name
    extraction.chunk_size = chunk_size
    extraction.extracted_at = datetime.now(UTC).isoformat()

    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_path.write_text(extraction.model_dump_json(indent=2))

    logger.info(
        f"  {doc_id}: {len(extraction.entities)} entities, "
        f"{len(extraction.relations)} relations ({extraction.chunks_processed} chunks)"
    )
    return extraction


async def _aextract_chunk(
    chunk: TextChunk,
    doc_id: str,
    llm: LLMClient,
    domain: DomainConfig,
) -> ExtractionResult:
    """Extract entities and relations from a single chunk (async).

    Uses a combined prompt (1 LLM call) instead of separate entity + relation
    calls (2 LLM calls). Falls back to entity-only on parse failure.
    """
    prompt = build_combined_prompt(chunk.text, doc_id, domain)
    try:
        data = await llm.acall_json(prompt)
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Extraction failed for {doc_id} chunk {chunk.chunk_index}: {e}")
        return ExtractionResult(source_document=doc_id, chunk_index=chunk.chunk_index)

    entities = []
    for raw in data.get("entities", []):
        try:
            entities.append(ExtractedEntity(
                name=raw.get("name", ""),
                entity_type=raw.get("entity_type", "UNKNOWN"),
                attributes=raw.get("attributes", {}),
                confidence=float(raw.get("confidence", 0.5)),
                context=raw.get("context", ""),
            ))
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Skipping malformed entity: {e}")

    relations = []
    for raw in data.get("relations", []):
        try:
            relations.append(ExtractedRelation(
                relation_type=raw.get("relation_type", "ASSOCIATED_WITH"),
                source_entity=raw.get("source_entity", ""),
                target_entity=raw.get("target_entity", ""),
                confidence=float(raw.get("confidence", 0.5)),
                evidence=raw.get("evidence", ""),
            ))
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Skipping malformed relation: {e}")

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
    concurrency: int = DEFAULT_CONCURRENCY,
    chunk_size: int = 10000,
    force: bool = False,
) -> list[DocumentExtraction]:
    """Extract entities and relations from multiple documents.

    All chunks across all documents share a single semaphore and rate limiter,
    so slots aren't wasted waiting between documents.
    """
    return asyncio.run(
        _aextract_all(doc_paths, llm, domain, output_dir, max_cost, concurrency, chunk_size, force)
    )


async def _aextract_all(
    doc_paths: list[Path],
    llm: LLMClient,
    domain: DomainConfig,
    output_dir: Path,
    max_cost: float | None,
    concurrency: int,
    chunk_size: int,
    force: bool = False,
) -> list[DocumentExtraction]:
    """Async extraction across all documents with shared concurrency."""
    sem = asyncio.Semaphore(concurrency)
    extraction_dir = output_dir / "extractions"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # Read all docs and prepare chunks upfront (cheap, no LLM calls)
    doc_work: list[tuple[Path, str, str, list[TextChunk]]] = []
    cached: list[DocumentExtraction] = []

    for doc_path in doc_paths:
        doc_id = doc_path.stem
        extraction_path = extraction_dir / f"{doc_id}.json"

        if extraction_path.exists() and not force:
            existing = DocumentExtraction(**json.loads(extraction_path.read_text()))
            reason = _check_stale(existing, llm.model, domain.name, chunk_size)
            if reason is None:
                logger.info(f"Skipping {doc_id} (already extracted)")
                cached.append(existing)
                continue
            logger.info(f"Re-extracting {doc_id}: {reason}")

        try:
            text = read_document(doc_path)
        except Exception as e:
            logger.error(f"Failed to read {doc_path.name}: {e}")
            cached.append(DocumentExtraction(
                document_id=doc_id, document_path=str(doc_path),
                error=str(e), model_used=llm.model,
            ))
            continue

        if not text.strip():
            logger.warning(f"Empty text from {doc_path.name}")
            cached.append(DocumentExtraction(
                document_id=doc_id, document_path=str(doc_path),
                error="Empty document", model_used=llm.model,
            ))
            continue

        chunks = chunk_text(text, chunk_size=chunk_size)
        doc_work.append((doc_path, doc_id, text, chunks))

    if not doc_work:
        return cached

    # Flatten all chunks across all docs, tagged with their doc info
    all_tasks: list[tuple[str, TextChunk]] = []
    for _, doc_id, _, chunks in doc_work:
        for chunk in chunks:
            all_tasks.append((doc_id, chunk))

    total_chunks = len(all_tasks)
    completed_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("${task.fields[cost]:.3f}"),
    ) as progress:
        ptask = progress.add_task("Extracting...", total=total_chunks, cost=0.0)

        async def _bounded(doc_id: str, chunk: TextChunk) -> tuple[str, ExtractionResult]:
            nonlocal completed_chunks
            if max_cost and llm.total_cost_usd >= max_cost:
                return doc_id, ExtractionResult(source_document=doc_id, chunk_index=chunk.chunk_index)
            async with sem:
                result = await _aextract_chunk(chunk, doc_id, llm, domain)
                completed_chunks += 1
                progress.update(ptask, completed=completed_chunks, cost=llm.total_cost_usd)
                return doc_id, result

        chunk_results = await asyncio.gather(
            *[_bounded(doc_id, chunk) for doc_id, chunk in all_tasks]
        )

    # Group results by document
    doc_results: dict[str, list[ExtractionResult]] = {}
    for doc_id, result in chunk_results:
        doc_results.setdefault(doc_id, []).append(result)

    # Build DocumentExtraction per doc and save
    extractions = list(cached)
    for doc_path, doc_id, _, chunks in doc_work:
        results = doc_results.get(doc_id, [])
        cost_for_doc = sum(
            getattr(r, '_cost', 0.0) for r in results
        )

        all_entities: list[ExtractedEntity] = []
        all_relations: list[ExtractedRelation] = []
        for r in results:
            all_entities.extend(r.entities)
            all_relations.extend(r.relations)

        extraction = DocumentExtraction(
            document_id=doc_id,
            document_path=str(doc_path),
            chunks_processed=len(chunks),
            entities=_dedupe_entities(all_entities),
            relations=all_relations,
            cost_usd=cost_for_doc,
            model_used=llm.model,
            domain_name=domain.name,
            chunk_size=chunk_size,
            extracted_at=datetime.now(UTC).isoformat(),
        )

        extraction_path = extraction_dir / f"{doc_id}.json"
        extraction_path.write_text(extraction.model_dump_json(indent=2))

        logger.info(
            f"  {doc_id}: {len(extraction.entities)} entities, "
            f"{len(extraction.relations)} relations ({extraction.chunks_processed} chunks)"
        )
        extractions.append(extraction)

    if max_cost and llm.total_cost_usd >= max_cost:
        logger.warning(
            f"Budget limit reached: ${llm.total_cost_usd:.2f} / ${max_cost:.2f}"
        )

    return extractions
