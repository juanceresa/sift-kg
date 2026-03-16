---
name: sift-kg
description: Use sift-kg as an AI second brain — a persistent knowledge graph your agent operates from across sessions. Use proactively at session start to orient, when answering questions about the user's projects or domain, when generating ideas or suggestions, when user asks "what do I know about X", "how does X connect to Y", "find connections", "what should I work on", "give me ideas", or when you need to understand the structure of the user's knowledge. Not needed for one-off document analysis — sift's CLI commands work fine on their own for that.
---

# sift-kg: AI Second Brain

The knowledge graph is your persistent, structured memory of the user's world — entities, relationships, communities, and how they all connect. It is built from the user's documents and persists across sessions.

**Treat the graph as your persistent understanding of the user's world.** When the user asks about their projects, domain, or knowledge — check the graph first. It grounds your responses in real entities and relationships instead of guessing. You don't need to query it for every interaction, but for anything related to the user's work or knowledge, the graph should inform your answer.

## Session Start: Orient Yourself

At the start of every session, orient from the graph:

```bash
sift info --json -o output/
```

This returns: `domain`, `entity_types`, `entities` (count), `relations` (count), `documents_processed`, `merge_proposals` and `relation_review` status (if dedup has been run), and `narrative_generated`. If no graph exists yet, go to Build the Graph below.

Then load the structural map:

```bash
sift topology -o output/
```

This returns:
- **Communities** — clusters of related entities with member counts and top entity IDs
- **Bridges** — entities connecting multiple communities, sorted by cross-community edge count
- **Community connections** — which clusters are linked and how strongly (shared edges, bridge count)
- **Isolated** — entities with no substantive connections

**Keep this topology in mind for the session.** Community labels and bridge entities help you navigate the user's knowledge. When relevant questions come up, you already have the structural context.

**Note on community labels:** Labels are auto-generated as "Community 1", "Community 2" unless `sift narrate` has been run (which generates descriptive LLM labels like "Palm Beach Elite Network"). To understand what a generically-labeled community represents, look at its `top_entities` list — these are entity IDs (e.g., `person:harry_boyte`, `program:civic_architecture`), not display names. Query interesting IDs with `sift query` to get full context.

**Note on `top_entities`:** These are entity IDs, not display names. Read them by convention: `person:harry_boyte` = a person named Harry Boyte, `company:palantir_technologies` = a company named Palantir Technologies.

## Querying the Graph

### Explore an Entity

```bash
sift query "topic or name" -o output/             # fuzzy name search
sift query "person:exact_entity_id" -o output/     # exact ID lookup
sift query "topic" --depth 2 -o output/            # 2-hop neighborhood
sift query "topic" -t PERSON -o output/            # filter by entity type
```

Returns: matched entity with community membership and bridge status, plus the full subgraph (nodes + edges) around it. If multiple entities match, the top result (by connection count) is returned with an `other_matches` list — re-query with the exact entity ID for a specific one.

### Lightweight Lookup

```bash
sift search "name" --json -o output/                    # find entities
sift search "name" --json --relations -o output/         # include direct relations
sift search "name" --json --description -o output/       # include descriptions
```

Use search when you just need to find an entity or check if something exists. Use query when you need the full neighborhood subgraph.

## Reasoning Patterns

### Pattern: Ground Responses in the Graph

When the user asks about their work — "what do I know about X?", "tell me about my project Y" — **always query the graph first** before responding:

```bash
sift query "X" -o output/
```

Read the match info (community, bridge status, connections) and the subgraph. Base your answer on actual entities and relationships from the graph. If the entity isn't in the graph, say so — don't invent connections.

### Pattern: Link Knowledge Islands

This is the highest-value reasoning pattern. It identifies opportunities to connect knowledge areas that are structurally separated.

**Step 1:** Get topology and identify disconnected or weakly connected community pairs.

```bash
sift topology -o output/
```

Look at `community_connections`. Community pairs with 0 `shared_edges` are completely disconnected — knowledge islands. Pairs with low `shared_edges` (1-3) are weakly connected.

**Step 2:** For each disconnected pair, examine what each community contains.

Read the `top_entities` (entity IDs) from each community in the topology output. Look at entity types — if both communities contain overlapping types (e.g., both have CONCEPT entities, or both have ORGANIZATION entities), there may be a semantic connection that isn't structurally represented yet.

**Step 3:** Query top entities from each side to find shared concepts.

```bash
sift query "top_entity_from_community_A" -o output/
sift query "top_entity_from_community_B" -o output/
```

Compare the two subgraphs. Look for:
- **Shared relation types** — if both subgraphs have FUNDED_BY or LOCATED_IN relations, the domains share a structural pattern
- **Common neighbor entities** — entities that appear in both neighborhoods are natural bridge candidates
- **Conceptual overlap** — entities in one subgraph that could logically relate to entities in the other

**Step 4:** Articulate the connection for the user.

Tell them: "Your [Community A topic] and [Community B topic] are currently disconnected in your knowledge graph. But [entity X] in the first area and [entity Y] in the second share [specific relationship or concept]. Connecting these could [specific value]."

The insight is not that the connection exists — the user might already sense it vaguely. The value is identifying **which specific entities** to connect and **why now**, grounded in the actual graph structure.

### Pattern: Generate Grounded Suggestions

When the user asks "what should I work on?" or "give me ideas":

```bash
sift topology -o output/
```

Base suggestions on graph structure:
- **Knowledge islands** (disconnected communities) — suggest exploring the connection between them
- **High-degree bridge entities** — these are the user's most structurally important concepts; suggest deepening them
- **Weakly connected community pairs** — suggest adding documents that would strengthen the bridge
- **Entity types concentrated in one community** — if all PERSON entities are in one cluster, the other clusters may lack the human dimension

Prefer suggestions grounded in specific entities, communities, or structural features from the topology over generic advice.

### Pattern: Track Changes After Rebuild

When the user adds new documents and rebuilds the graph:

```bash
# Before rebuild — capture current state
sift topology -o output/ > /tmp/topology_before.json

# Rebuild
sift extract ./new-docs/ -o output/   # confirm with user first (costs money)
sift build -o output/

# After rebuild — compare
sift topology -o output/
```

Compare the new topology against the previous snapshot:
- New communities that appeared
- Communities that merged or split
- New bridge entities
- Community pairs that gained or lost connections
- Changes in entity/relation counts

Report these changes to the user — "Adding those documents created a new cluster around [topic] and connected it to your existing [community] via [bridge entity]."

## Building the Graph

When the user provides new documents or wants to create/update the graph:

```bash
sift extract ./documents/ -o output/    # extract entities and relations (LLM, costs money)
sift build -o output/                    # construct graph + detect communities
```

**Always confirm with the user before running extract or resolve** — these make LLM API calls that cost money.

Extraction is cached — only new/changed documents are processed on re-run. The graph grows incrementally; entity IDs are deterministic (`{type}:{normalized_name}`) so the same entity from different documents auto-merges.

For better entity deduplication:

```bash
sift resolve -o output/        # LLM proposes entity merges (costs money)
sift review                    # user approves/rejects interactively
sift apply-merges -o output/   # apply confirmed merges
```

## Key Concepts

**Entity IDs** follow the format `{type}:{normalized_name}` — all lowercase, underscores for spaces. Examples: `person:jeffrey_epstein`, `company:palantir_technologies`, `location:new_york`.

**Communities** are clusters of densely connected entities detected by Louvain algorithm. They represent natural groupings — topics, domains, networks of people.

**Bridge entities** connect multiple communities. High cross-community edge count = structurally important. These are the nodes that link otherwise separate knowledge areas.

**Substantive connections** exclude DOCUMENT nodes and MENTIONED_IN edges (provenance metadata). All agent-facing counts and subgraphs use this filtered view.

**Graph scale:** For graphs under ~500 entities, `graph_data.json` can be loaded directly into context. For larger graphs, always use `sift topology` and `sift query` — never attempt to load the full JSON.

## Output Files

```
output/
├── graph_data.json          # full knowledge graph (nodes + edges)
├── communities.json         # community assignments (entity_id -> label)
├── extractions/             # per-document extraction results (cached)
├── narrative.md             # prose narrative (optional, from sift narrate)
└── entity_descriptions.json # entity descriptions (optional, from sift narrate)
```

## Troubleshooting

**"No graph found"** — Run `sift extract` then `sift build` first. The graph must be built before querying.

**"No communities found"** — Run `sift build` (communities are detected during build). If the graph is very small (<16 entities), community detection may not produce meaningful results.

**Empty query results** — Try a broader search term, or check `sift search "term" --json` to see what entities exist. Entity names may differ from what you expect — check aliases.

**Large graph, slow queries** — Use `sift topology` for the overview instead of loading `graph_data.json`. Use `sift query` with `--depth 1` (default) to keep subgraphs manageable.
