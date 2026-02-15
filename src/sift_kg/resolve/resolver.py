"""LLM-based entity resolution.

Asks the LLM to identify which entities in the graph refer to the same
real-world entity. Groups candidates by type and sends batches to the LLM.
No training step needed — works out of the box.

Uses async concurrency to process type batches in parallel.
"""

import asyncio
import json
import logging

from unidecode import unidecode

from sift_kg.extract.llm_client import LLMClient
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.graph.prededup import _TITLE_PREFIXES
from sift_kg.resolve.models import (
    MergeFile,
    MergeMember,
    MergeProposal,
    RelationReviewEntry,
)

logger = logging.getLogger(__name__)

# Entity types to skip during resolution (source documents, not real entities)
SKIP_TYPES = {"DOCUMENT"}

# Don't send more than this many entities to the LLM at once.
# Slim payloads (name/aliases only) make 100 feasible within token limits.
MAX_BATCH_SIZE = 100

# Overlap between consecutive batches so entities near boundaries
# appear in both, eliminating cross-batch blind spots.
BATCH_OVERLAP = 20


def _person_sort_key(name: str) -> str:
    """Sort PERSON entities by surname so title/first-name variants cluster.

    "Mr. Edwards", "Bradley Edwards", "Edwards" all sort under "edwards".
    "Detective Joe Recarey", "Joseph Recarey" sort under "recarey".
    """
    normalized = unidecode(name).lower().strip()
    # Strip title prefixes
    changed = True
    while changed:
        changed = False
        for prefix in _TITLE_PREFIXES:
            if normalized.startswith(prefix + " "):
                normalized = normalized[len(prefix) + 1:].strip()
                changed = True
                break
    # Sort by last word (surname), then full name as tiebreaker
    parts = normalized.split()
    surname = parts[-1] if parts else normalized
    return f"{surname} {normalized}"


def find_merge_candidates(
    kg: KnowledgeGraph,
    llm: LLMClient,
    entity_types: list[str] | None = None,
    concurrency: int = 4,
    use_embeddings: bool = False,
    system_context: str = "",
) -> tuple[MergeFile, list[RelationReviewEntry]]:
    """Find entities that likely refer to the same real-world thing.

    Args:
        kg: Knowledge graph with entities
        llm: LLM client for similarity judgments
        entity_types: Types to resolve (default: all types except DOCUMENT)
        concurrency: Max concurrent LLM calls
        use_embeddings: Use semantic clustering instead of alphabetical batching
        system_context: Domain context to help LLM understand entity names

    Returns:
        Tuple of (MergeFile with DRAFT proposals, list of variant relation proposals)
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
) -> tuple[MergeFile, list[RelationReviewEntry]]:
    """Async implementation — resolves all type batches concurrently."""
    if entity_types:
        types_to_check = entity_types
    else:
        # Resolve all entity types present in the graph, except DOCUMENT
        types_to_check = sorted({
            data.get("entity_type")
            for _, data in kg.graph.nodes(data=True)
            if data.get("entity_type") and data.get("entity_type") not in SKIP_TYPES
        })

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

        # Build batches: semantic clustering or surname/alphabetical windows
        sort_key = (
            (lambda e: _person_sort_key(e["name"]))
            if entity_type == "PERSON"
            else (lambda e: e["name"].lower())
        )

        if use_embeddings:
            try:
                from sift_kg.resolve.clustering import cluster_entities_by_embedding

                batches = cluster_entities_by_embedding(entities)
            except ImportError:
                logger.warning(
                    "Embedding clustering unavailable, falling back"
                )
                entities.sort(key=sort_key)
                batches = _build_overlapping_batches(entities)
            except Exception as e:
                logger.warning(f"Clustering failed ({e}), falling back")
                entities.sort(key=sort_key)
                batches = _build_overlapping_batches(entities)
        else:
            entities.sort(key=sort_key)
            batches = _build_overlapping_batches(entities)

        for batch_idx, batch in enumerate(batches):
            if len(batch) < 2:
                continue
            if len(batches) > 1:
                logger.info(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch)} entities")

            async def _bounded(b: list[dict], et: str) -> tuple[list[MergeProposal], list[RelationReviewEntry]]:
                async with sem:
                    return await _aresolve_type_batch(b, et, llm, system_context)

            tasks.append(_bounded(batch, entity_type))

    if not tasks:
        return MergeFile(proposals=[]), []

    batch_results = await asyncio.gather(*tasks)
    all_proposals = [p for proposals, _ in batch_results for p in proposals]
    all_variants = [v for _, variants in batch_results for v in variants]

    # Overlapping windows can produce duplicate proposals — deduplicate.
    all_proposals = _deduplicate_proposals(all_proposals)

    # Cross-type dedup: find entities with same name but different types.
    # No LLM needed — if the name is identical, it's almost certainly the
    # same entity with an inconsistent type assignment.
    cross_type_proposals = _find_cross_type_duplicates(kg)
    all_proposals.extend(cross_type_proposals)

    logger.info(f"Found {len(all_proposals)} merge proposals and {len(all_variants)} variant relations across {len(types_to_check)} types")
    return MergeFile(proposals=all_proposals), all_variants


def _find_cross_type_duplicates(kg: KnowledgeGraph) -> list[MergeProposal]:
    """Find entities with the same name but different types.

    When the LLM extracts "reading comprehension" as both CONCEPT and
    PHENOMENON, these are the same entity with an inconsistent type.
    The canonical is the one with more connections (more context for
    the type assignment). All relations get combined on merge.
    """
    from collections import defaultdict

    # Group by normalized name
    name_groups: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for nid, data in kg.graph.nodes(data=True):
        entity_type = data.get("entity_type", "")
        if entity_type in SKIP_TYPES:
            continue
        name = data.get("name", "").strip().lower()
        if not name:
            continue
        degree = kg.graph.degree(nid)
        name_groups[name].append((nid, entity_type, degree))

    proposals = []
    for _name, group in name_groups.items():
        # Only care about names that appear under multiple types
        types = {t for _, t, _ in group}
        if len(types) < 2:
            continue

        # Canonical = highest degree
        group.sort(key=lambda x: x[2], reverse=True)
        canonical_id, canonical_type, _ = group[0]
        canonical_name = kg.graph.nodes[canonical_id].get("name", canonical_id)

        members = []
        for mid, _mtype, _ in group[1:]:
            members.append(MergeMember(
                id=mid,
                name=kg.graph.nodes[mid].get("name", mid),
                confidence=0.95,
            ))

        member_types = ", ".join(t for _, t, _ in group[1:])
        proposals.append(MergeProposal(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            entity_type=canonical_type,
            status="DRAFT",
            members=members,
            reason=f"Same name across types ({canonical_type} vs {member_types}). Relations will be combined.",
        ))

    if proposals:
        logger.info(f"Cross-type dedup: found {len(proposals)} entities with same name across different types")

    return proposals


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


def _strip_person_titles(name: str) -> str:
    """Strip title prefixes from a person name for resolve payloads.

    Returns the stripped name, or the original if nothing changed.
    """
    normalized = unidecode(name).strip()
    lower = normalized.lower()
    changed = True
    while changed:
        changed = False
        for prefix in _TITLE_PREFIXES:
            if lower.startswith(prefix + " "):
                normalized = normalized[len(prefix) + 1:].strip()
                lower = normalized.lower()
                changed = True
                break
    return normalized


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
        aliases = list(e.get("aliases") or [])

        # For PERSON entities, strip titles and add the bare name as an alias
        # so the LLM can see "Joe Recarey" next to "Joseph Recarey".
        if entity_type == "PERSON":
            stripped = _strip_person_titles(e["name"])
            if stripped.lower() != e["name"].lower() and stripped not in aliases:
                aliases.insert(0, stripped)

        if aliases:
            entry["aliases"] = aliases
        attrs = e.get("attributes", {})
        identity_attrs = {k: v for k, v in attrs.items() if k in identity_keys and v}
        if identity_attrs:
            entry["attributes"] = identity_attrs
        entity_dicts.append(entry)

    entity_list = json.dumps(entity_dicts, indent=2, ensure_ascii=False)

    context_section = ""
    if system_context:
        context_section = f"DOMAIN CONTEXT:\n{system_context}\n\n"

    # Build type-appropriate dedup hints
    person_types = {"PERSON", "RESEARCHER"}
    if entity_type in person_types:
        type_hints = """Look for:
- Name variations (abbreviations, nicknames, full legal names vs common names, misspellings, transliterations)
- Title/honorific prefixes that don't change identity (Dr., Mr., Detective, Judge, etc.)
- First name vs nickname variants
- Aliases — if an entity's aliases list contains a name matching another entity, they are very likely the same
- Same person referenced differently across documents
- DO NOT merge genuinely different people (e.g., father and son, or unrelated people sharing a surname)"""
    else:
        type_hints = """Look for:
- Acronyms vs spelled-out forms of the same thing
- Spacing, punctuation, and capitalization variants of the same name
- A name with and without a redundant qualifier (e.g. "X method" vs just "X")
- Same entity referenced with slightly different wording across documents
- Aliases — if an entity's aliases list contains a name matching another entity, they are very likely the same
- DO NOT merge entities that are genuinely distinct variants, versions, or subtypes of each other"""

    prompt = f"""{context_section}Analyze these {entity_type} entities and identify:
1. **Duplicates** — entities that refer to the exact same thing (merge them)
2. **Variants** — entities that are a subtype, version, or specific implementation of a parent entity (link them with EXTENDS)

ENTITIES:
{entity_list}

{type_hints}

IMPORTANT: If entity B is a variant/subtype/version of entity A (not the same thing, but derived from it), put it in "variants" NOT "groups". Only true duplicates go in "groups".

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
  ],
  "variants": [
    {{
      "parent_id": "id of the parent/base entity",
      "child_id": "id of the variant/subtype",
      "confidence": 0.0-1.0,
      "reason": "brief explanation"
    }}
  ]
}}

If nothing found, return {{"groups": [], "variants": []}}.
OUTPUT JSON:"""

    try:
        data = await llm.acall_json(prompt)
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Entity resolution failed for {entity_type}: {e}")
        return [], []

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

    # Parse variant relations (EXTENDS)
    variant_relations = []
    for variant in data.get("variants", []):
        parent_id = variant.get("parent_id", "")
        child_id = variant.get("child_id", "")
        if not parent_id or not child_id:
            continue
        if parent_id not in entity_lookup or child_id not in entity_lookup:
            continue

        variant_relations.append(RelationReviewEntry(
            source_id=child_id,
            source_name=entity_lookup[child_id],
            target_id=parent_id,
            target_name=entity_lookup[parent_id],
            relation_type="EXTENDS",
            confidence=float(variant.get("confidence", 0.7)),
            evidence=variant.get("reason", ""),
            status="DRAFT",
            flag_reason="Variant relationship discovered during entity resolution",
        ))

    return proposals, variant_relations
