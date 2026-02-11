"""LLM-based entity resolution.

Asks the LLM to identify which entities in the graph refer to the same
real-world entity. Groups candidates by type and sends batches to the LLM.
No training step needed — works out of the box.
"""

import json
import logging

from sift_kg.extract.llm_client import LLMClient
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.resolve.models import MergeFile, MergeMember, MergeProposal

logger = logging.getLogger(__name__)

# Only resolve entity types that commonly have duplicates
RESOLVABLE_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "EVENT"}

# Don't send more than this many entities to the LLM at once
MAX_BATCH_SIZE = 50


def find_merge_candidates(
    kg: KnowledgeGraph,
    llm: LLMClient,
    entity_types: list[str] | None = None,
) -> MergeFile:
    """Find entities that likely refer to the same real-world thing.

    Args:
        kg: Knowledge graph with entities
        llm: LLM client for similarity judgments
        entity_types: Types to resolve (default: PERSON, ORG, LOCATION, EVENT)

    Returns:
        MergeFile with DRAFT proposals
    """
    types_to_check = entity_types or [
        t for t in RESOLVABLE_TYPES
        if any(
            data.get("entity_type") == t
            for _, data in kg.graph.nodes(data=True)
        )
    ]

    all_proposals: list[MergeProposal] = []

    for entity_type in types_to_check:
        # Gather entities of this type
        entities = [
            {"id": nid, "name": data.get("name", ""), "attributes": data.get("attributes", {})}
            for nid, data in kg.graph.nodes(data=True)
            if data.get("entity_type") == entity_type
        ]

        if len(entities) < 2:
            continue

        logger.info(f"Resolving {len(entities)} {entity_type} entities")

        # Split into batches if too many
        for batch_start in range(0, len(entities), MAX_BATCH_SIZE):
            batch = entities[batch_start:batch_start + MAX_BATCH_SIZE]
            if len(batch) < 2:
                continue
            if len(entities) > MAX_BATCH_SIZE:
                batch_num = batch_start // MAX_BATCH_SIZE + 1
                total_batches = (len(entities) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
                logger.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} entities")
            proposals = _resolve_type_batch(batch, entity_type, llm)
            all_proposals.extend(proposals)

    logger.info(f"Found {len(all_proposals)} merge proposals across {len(types_to_check)} types")
    return MergeFile(proposals=all_proposals)


def _resolve_type_batch(
    entities: list[dict],
    entity_type: str,
    llm: LLMClient,
) -> list[MergeProposal]:
    """Ask LLM to identify duplicate entities within a type."""
    entity_list = json.dumps(
        [{"id": e["id"], "name": e["name"], "attributes": e["attributes"]} for e in entities],
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""Analyze these {entity_type} entities and identify groups that refer to the same real-world entity.

ENTITIES:
{entity_list}

Look for:
- Name variations (abbreviations, nicknames, misspellings, transliterations)
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
        data = llm.call_json(prompt)
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

        # Canonical must be in the member list
        if canonical_id not in member_ids:
            member_ids.insert(0, canonical_id)

        # Build members (excluding canonical since it's the merge target)
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
