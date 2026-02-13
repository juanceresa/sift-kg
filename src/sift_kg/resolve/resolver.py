"""LLM-based entity resolution.

Asks the LLM to identify which entities in the graph refer to the same
real-world entity. Groups candidates by type and sends batches to the LLM.
No training step needed — works out of the box.

Uses async concurrency to process type batches in parallel.
"""

import asyncio
import json
import logging

from sift_kg.extract.llm_client import LLMClient
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.resolve.models import MergeFile, MergeMember, MergeProposal

logger = logging.getLogger(__name__)

# Only resolve entity types that commonly have duplicates
RESOLVABLE_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "EVENT"}

# Don't send more than this many entities to the LLM at once.
# Slim payloads (name/aliases only) make 100 feasible within token limits.
MAX_BATCH_SIZE = 100

# Overlap between consecutive batches so entities near boundaries
# appear in both, eliminating cross-batch blind spots.
BATCH_OVERLAP = 20


def find_merge_candidates(
    kg: KnowledgeGraph,
    llm: LLMClient,
    entity_types: list[str] | None = None,
    concurrency: int = 4,
    use_embeddings: bool = False,
    system_context: str = "",
) -> MergeFile:
    """Find entities that likely refer to the same real-world thing.

    Args:
        kg: Knowledge graph with entities
        llm: LLM client for similarity judgments
        entity_types: Types to resolve (default: PERSON, ORG, LOCATION, EVENT)
        concurrency: Max concurrent LLM calls
        use_embeddings: Use semantic clustering instead of alphabetical batching
        system_context: Domain context to help LLM understand entity names

    Returns:
        MergeFile with DRAFT proposals
    """
    return asyncio.run(
        _afind_merge_candidates(kg, llm, entity_types, concurrency, use_embeddings, system_context)
    )


async def _afind_merge_candidates(
    kg: KnowledgeGraph,
    llm: LLMClient,
    entity_types: list[str] | None,
    concurrency: int,
    use_embeddings: bool = False,
    system_context: str = "",
) -> MergeFile:
    """Async implementation — resolves all type batches concurrently."""
    types_to_check = entity_types or [
        t for t in RESOLVABLE_TYPES
        if any(
            data.get("entity_type") == t
            for _, data in kg.graph.nodes(data=True)
        )
    ]

    # Build all (batch, entity_type) pairs
    tasks = []
    sem = asyncio.Semaphore(concurrency)

    for entity_type in types_to_check:
        entities = []
        for nid, data in kg.graph.nodes(data=True):
            if data.get("entity_type") != entity_type:
                continue
            attrs = data.get("attributes", {})
            aliases = attrs.get("aliases", []) or attrs.get("also_known_as", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            entities.append({
                "id": nid,
                "name": data.get("name", ""),
                "aliases": aliases,
                "attributes": attrs,
            })

        if len(entities) < 2:
            continue

        logger.info(f"Resolving {len(entities)} {entity_type} entities")

        # Build batches: semantic clustering or alphabetical windows
        if use_embeddings:
            try:
                from sift_kg.resolve.clustering import cluster_entities_by_embedding

                batches = cluster_entities_by_embedding(entities)
            except ImportError:
                logger.warning(
                    "Embedding clustering unavailable, falling back to alphabetical"
                )
                entities.sort(key=lambda e: e["name"].lower())
                batches = _build_overlapping_batches(entities)
            except Exception as e:
                logger.warning(f"Clustering failed ({e}), falling back to alphabetical")
                entities.sort(key=lambda e: e["name"].lower())
                batches = _build_overlapping_batches(entities)
        else:
            entities.sort(key=lambda e: e["name"].lower())
            batches = _build_overlapping_batches(entities)

        for batch_idx, batch in enumerate(batches):
            if len(batch) < 2:
                continue
            if len(batches) > 1:
                logger.info(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch)} entities")

            async def _bounded(b: list[dict], et: str) -> list[MergeProposal]:
                async with sem:
                    return await _aresolve_type_batch(b, et, llm, system_context)

            tasks.append(_bounded(batch, entity_type))

    if not tasks:
        return MergeFile(proposals=[])

    batch_results = await asyncio.gather(*tasks)
    all_proposals = [p for batch in batch_results for p in batch]

    # Overlapping windows can produce duplicate proposals — deduplicate.
    all_proposals = _deduplicate_proposals(all_proposals)

    logger.info(f"Found {len(all_proposals)} merge proposals across {len(types_to_check)} types")
    return MergeFile(proposals=all_proposals)


def _build_overlapping_batches(entities: list[dict]) -> list[list[dict]]:
    """Split entities into batches with overlap at boundaries.

    For lists that fit in one batch, returns a single batch.
    For larger lists, consecutive batches share BATCH_OVERLAP entities
    so duplicates near boundaries are seen together.
    """
    n = len(entities)
    if n <= MAX_BATCH_SIZE:
        return [entities]

    batches = []
    step = MAX_BATCH_SIZE - BATCH_OVERLAP
    for start in range(0, n, step):
        batch = entities[start:start + MAX_BATCH_SIZE]
        if len(batch) < 2:
            break
        batches.append(batch)
        # Stop if this batch already reaches the end
        if start + MAX_BATCH_SIZE >= n:
            break
    return batches


def _deduplicate_proposals(proposals: list[MergeProposal]) -> list[MergeProposal]:
    """Remove duplicate proposals from overlapping batches.

    When the same canonical_id appears in multiple proposals, keep the one
    with the most members (and highest average confidence as tiebreaker).
    Merge member lists from duplicates to catch cross-batch matches.
    """
    by_canonical: dict[str, list[MergeProposal]] = {}
    for p in proposals:
        by_canonical.setdefault(p.canonical_id, []).append(p)

    deduped = []
    for canonical_id, group in by_canonical.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue

        # Merge members from all proposals for this canonical entity.
        # Keep highest confidence per member.
        seen_members: dict[str, MergeMember] = {}
        best_reason = ""
        best_confidence = 0.0
        entity_type = group[0].entity_type
        canonical_name = group[0].canonical_name

        for p in group:
            avg_conf = sum(m.confidence for m in p.members) / len(p.members) if p.members else 0
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_reason = p.reason
                canonical_name = p.canonical_name
            for m in p.members:
                if m.id not in seen_members or m.confidence > seen_members[m.id].confidence:
                    seen_members[m.id] = m

        deduped.append(MergeProposal(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            status="DRAFT",
            members=list(seen_members.values()),
            reason=best_reason,
        ))

    return deduped


async def _aresolve_type_batch(
    entities: list[dict],
    entity_type: str,
    llm: LLMClient,
    system_context: str = "",
) -> list[MergeProposal]:
    """Ask LLM to identify duplicate entities within a type (async)."""
    # Only send identity-relevant fields — full attribute dicts drown out
    # name/alias signals and waste tokens.
    identity_keys = {"role", "title", "occupation", "position", "aka"}
    entity_dicts = []
    for e in entities:
        entry: dict = {"id": e["id"], "name": e["name"]}
        if e.get("aliases"):
            entry["aliases"] = e["aliases"]
        attrs = e.get("attributes", {})
        identity_attrs = {k: v for k, v in attrs.items() if k in identity_keys and v}
        if identity_attrs:
            entry["attributes"] = identity_attrs
        entity_dicts.append(entry)

    entity_list = json.dumps(entity_dicts, indent=2, ensure_ascii=False)

    context_section = ""
    if system_context:
        context_section = f"DOMAIN CONTEXT:\n{system_context}\n\n"

    prompt = f"""{context_section}Analyze these {entity_type} entities and identify groups that refer to the same real-world entity.

ENTITIES:
{entity_list}

Look for:
- Name variations (abbreviations, nicknames, full legal names vs common names, misspellings, transliterations)
- Aliases — if an entity's aliases list contains a name matching another entity, they are very likely the same
- Same entity referenced differently across documents
- DO NOT merge entities that are genuinely different (e.g., father and son with similar names)

Return valid JSON only:
{{
  "groups": [
    {{
      "canonical_id": "id of the best/most complete entity",
      "canonical_name": "the preferred name",
      "member_ids": ["id1", "id2"],
      "confidence": 0.0-1.0,
      "reason": "brief explanation"
    }}
  ]
}}

If no duplicates found, return {{"groups": []}}.
OUTPUT JSON:"""

    try:
        data = await llm.acall_json(prompt)
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Entity resolution failed for {entity_type}: {e}")
        return []

    # Parse response into MergeProposals
    proposals = []
    entity_lookup = {e["id"]: e["name"] for e in entities}

    for group in data.get("groups", []):
        canonical_id = group.get("canonical_id", "")
        member_ids = group.get("member_ids", [])
        confidence = float(group.get("confidence", 0.5))

        if not canonical_id or len(member_ids) < 2:
            continue

        if canonical_id not in member_ids:
            member_ids.insert(0, canonical_id)

        members = []
        for mid in member_ids:
            if mid != canonical_id and mid in entity_lookup:
                members.append(MergeMember(
                    id=mid,
                    name=entity_lookup.get(mid, mid),
                    confidence=confidence,
                ))

        if not members:
            continue

        proposals.append(MergeProposal(
            canonical_id=canonical_id,
            canonical_name=entity_lookup.get(canonical_id, canonical_id),
            entity_type=entity_type,
            status="DRAFT",
            members=members,
            reason=group.get("reason", ""),
        ))

    return proposals
