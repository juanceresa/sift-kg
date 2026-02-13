# sift-kg

**Turn any collection of documents into a knowledge graph.**

No code, no database, no infrastructure — just a CLI and your documents. Define what to extract in YAML (or use the built-in defaults), and get a browsable, exportable knowledge graph. sift-kg handles the rest: entity extraction, duplicate resolution with your approval, and narrative generation that traces connections across your entire collection.

```bash
pip install sift-kg

sift init                           # create sift.yaml + .env.example
sift extract ./documents/           # extract entities & relations
sift build                          # build knowledge graph
sift resolve                        # find duplicate entities
sift review                         # approve/reject merges interactively
sift apply-merges                   # apply your decisions
sift narrate                        # generate narrative summary
sift view                           # interactive graph in your browser
sift export graphml                 # export to Gephi, yEd, Cytoscape, etc.
```

## How It Works

```
Documents (PDF, text, HTML)
       ↓
  Text Extraction (pdfplumber, local)
       ↓
  Entity & Relation Extraction (LLM)
       ↓
  Knowledge Graph (NetworkX, JSON)
       ↓
  Entity Resolution (LLM proposes → you review)
       ↓
  Narrative Generation (LLM)
       ↓
  Interactive Viewer (browser) / Export (GraphML, GEXF, CSV)
```

Every entity and relation links back to the source document and passage. You control what gets merged. The graph is yours.

## Features

- **Zero-config start** — point at a folder, get a knowledge graph. Or drop a `sift.yaml` in your project for persistent settings
- **Any LLM provider** — OpenAI, Anthropic, Ollama (local/private), or any LiteLLM-compatible provider
- **Domain-configurable** — define custom entity types and relation types in YAML
- **Human-in-the-loop** — sift proposes entity merges, you approve or reject in an interactive terminal UI
- **CLI search** — `sift search "SBF"` finds entities by name or alias, with optional relation and description output
- **Interactive viewer** — explore your graph in-browser with search, type toggles, and entity descriptions
- **Export anywhere** — GraphML (yEd, Cytoscape), GEXF (Gephi), CSV, or native JSON for advanced analysis
- **Narrative generation** — structured summaries tracing connections across your documents
- **Source provenance** — every extraction links to the document and passage it came from
- **Multilingual** — extracts from documents in any language, outputs a unified English knowledge graph. Proper names stay as-is, non-Latin scripts are romanized automatically
- **Budget controls** — set `--max-cost` to cap LLM spending
- **Runs locally** — your documents stay on your machine

## Use Cases

- **Investigative journalism** — analyze FOIA releases, court filings, and document leaks
- **OSINT research** — map entity networks from public records
- **Academic research** — build structured datasets from historical archives
- **Legal review** — extract and connect entities across document collections
- **Genealogy** — trace family relationships across vital records

## For OSINT & Investigations

sift-kg ships with a bundled `osint` domain that adds entity types for shell companies, financial instruments, and government agencies, plus relation types like `BENEFICIAL_OWNER_OF` and `SANCTIONS_LISTED`:

```bash
sift extract ./docs/ --domain-name osint
```

The human-in-the-loop merge review is designed for this — the LLM proposes, you verify. Nothing gets merged without your approval, and every extraction links back to the source document and passage.

See [`examples/ftx/`](examples/ftx/) for a complete pipeline run on 9 articles about the FTX collapse (373 entities, 1184 relations). [**Explore the graph live**](https://juanceresa.github.io/sift-kg/graph.html) — no install, no API key.

## Civic Table

Looking for a hosted platform with OCR, forensic legal analysis, and analyst verification?

[**Civic Table**](https://github.com/juanceresa/forensic_analysis_platform) is a forensic intelligence platform built on the sift-kg pipeline. It adds OCR for scanned/degraded documents (Google Cloud Vision), a 4-tier verification system where analysts and JDs validate AI-extracted facts before they're treated as evidence, LaTeX dossier generation for legal submissions, and a web interface for sharing results with clients and families. Built for property restitution, investigative journalism, and any context where documentary provenance matters.

sift-kg is the open-source CLI. Civic Table is the full platform — and where the output gets vetted by analysts and JDs before it carries evidentiary weight.

## Installation

Requires Python 3.11+.

```bash
pip install sift-kg
```

For semantic clustering during entity resolution (optional, ~2GB for PyTorch):

```bash
pip install sift-kg[embeddings]
```

For development:

```bash
git clone https://github.com/juanceresa/sift-kg.git
cd sift-kg
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize and configure

```bash
sift init                     # creates sift.yaml + .env.example
cp .env.example .env          # copy and add your API key
```

`sift init` generates a `sift.yaml` project config so you don't need flags on every command:

```yaml
# sift.yaml
domain: domain.yaml           # or a bundled name like "osint"
model: openai/gpt-4o-mini
```

Set your API key in `.env`:
```
SIFT_OPENAI_API_KEY=sk-...
```

Or use Anthropic, Ollama, or any LiteLLM provider:
```
SIFT_ANTHROPIC_API_KEY=sk-ant-...
```

Settings priority: CLI flags > env vars > `.env` > `sift.yaml` > defaults. You can override anything from `sift.yaml` with a flag on any command.

### 2. Extract entities and relations

```bash
sift extract ./my-documents/
```

Reads PDFs, text files, and HTML. Extracts entities and relations using your configured LLM. Results saved as JSON in `output/extractions/`.

### 3. Build the knowledge graph

```bash
sift build
```

Constructs a NetworkX graph from all extractions. Automatically deduplicates near-identical entity names (plurals, Unicode variants, case differences) before they become graph nodes. Flags low-confidence relations for review. Saves to `output/graph_data.json`.

### 4. Resolve duplicate entities

See [Entity Resolution Workflow](#entity-resolution-workflow) below for the full guide — especially important for genealogy, legal, and investigative use cases where accuracy matters.

### 5. Explore and export

**Interactive viewer** — for exploration and investigation:

```bash
sift view                     # → opens output/graph.html in your browser
```

Opens a force-directed graph in your browser with entity descriptions, color-coded types, search, type toggles, and a detail sidebar. This is the intended way to explore your graph — click on entities, trace connections, read the evidence.

**CLI search** — query entities directly from the terminal:

```bash
sift search "Sam Bankman"          # search by name
sift search "SBF"                  # search by alias
sift search "Caroline" -r          # show relations
sift search "FTX" -d -t ORGANIZATION  # descriptions + type filter
```

**Static exports** — for analysis tools where you want custom layout, filtering, or styling:

```bash
sift export graphml           # → output/graph.graphml (Gephi, yEd, Cytoscape)
sift export gexf              # → output/graph.gexf (Gephi native)
sift export csv               # → output/csv/entities.csv + relations.csv
sift export json              # → output/graph.json
```

Use GraphML/GEXF when you want to control node sizing, edge weighting, custom color schemes, or apply graph algorithms (centrality, community detection) in dedicated tools.

### 6. Generate narrative

```bash
sift narrate
```

Produces `output/narrative.md` — a structured summary with entity profiles tracing connections across your documents.

## Domain Configuration

sift-kg ships with two bundled domains:

```bash
sift domains                  # list available domains
```

| Domain | Entity Types | Relation Types | Use Case |
|--------|-------------|----------------|----------|
| `default` | PERSON, ORGANIZATION, LOCATION, EVENT, DOCUMENT | 9 general relations | Any document corpus |
| `osint` | Adds SHELL_COMPANY, FINANCIAL_INSTRUMENT, GOVERNMENT_AGENCY | Adds BENEFICIAL_OWNER_OF, SANCTIONS_LISTED, etc. | Investigations, FOIA |

Use a bundled domain:
```bash
sift extract ./docs/ --domain-name osint
```

Or create your own `domain.yaml`:
```yaml
name: My Domain
entity_types:
  PERSON:
    description: People and individuals
    extraction_hints:
      - Look for full names with titles
  COMPANY:
    description: Business entities
relation_types:
  EMPLOYED_BY:
    description: Employment relationship
    source_types: [PERSON]
    target_types: [COMPANY]
  OWNS:
    description: Ownership relationship
    symmetric: false
    review_required: true
```

```bash
sift extract ./docs/ --domain path/to/domain.yaml
```

## Library API

Use sift-kg from Python — Jupyter notebooks, scripts, web apps:

```python
from sift_kg import load_domain, run_extract, run_build, run_narrate, export_graph
from sift_kg import KnowledgeGraph
from pathlib import Path

# Load domain and run extraction
domain = load_domain()  # or load_domain(bundled_name="osint")
results = run_extract(Path("./docs"), "openai/gpt-4o-mini", domain, Path("./output"))

# Build graph
kg = run_build(Path("./output"), domain)
print(f"{kg.entity_count} entities, {kg.relation_count} relations")

# Export
export_graph(kg, Path("./output/graph.graphml"), "graphml")

# Or run the full pipeline
from sift_kg import run_pipeline
run_pipeline(Path("./docs"), "openai/gpt-4o-mini", domain, Path("./output"))
```

## Project Structure

After running the pipeline, your output directory contains:

```
output/
├── extractions/               # Per-document extraction JSON
│   ├── document1.json
│   └── document2.json
├── graph_data.json            # Knowledge graph (native format)
├── merge_proposals.yaml       # Entity merge proposals (DRAFT/CONFIRMED/REJECTED)
├── relation_review.yaml       # Flagged relations for review
├── narrative.md               # Generated narrative summary
├── entity_descriptions.json   # Entity descriptions (loaded by viewer)
├── graph.html                 # Interactive graph visualization
├── graph.graphml              # GraphML export (if exported)
├── graph.gexf                 # GEXF export (if exported)
└── csv/                       # CSV export (if exported)
    ├── entities.csv
    └── relations.csv
```

## Entity Resolution Workflow

When you're building a knowledge graph from family records, legal filings, or any documents where accuracy matters, you want full control over which entities get merged. sift-kg never merges anything without your approval.

The workflow has three layers, each catching different kinds of duplicates:

### Layer 1: Automatic Pre-Dedup (during `sift build`)

Before entities become graph nodes, sift deterministically collapses names that are obviously the same. No LLM involved, no cost, no review needed:

- **Unicode normalization** — "Jose Garcia" and "Jose Garcia" become one node
- **Title stripping** — "Detective Joe Recarey" and "Joe Recarey" merge (strips ~35 common prefixes: Dr., Mr., Judge, Senator, etc.)
- **Singularization** — "Companies" and "Company" merge
- **Fuzzy string matching** — [SemHash](https://github.com/MinishLab/semhash) at 0.95 threshold catches near-identical strings like "MacAulay" vs "Mac Aulay"

This happens automatically every time you run `sift build`. These are the trivial cases — spelling variants that would clutter your graph without adding information.

### Layer 2: LLM Proposes Merges (during `sift resolve`)

The LLM sees batches of entities and identifies ones that likely refer to the same real-world thing. It produces a `merge_proposals.yaml` file where every proposal starts as `DRAFT`:

```bash
sift resolve                  # uses domain from sift.yaml
sift resolve --domain osint   # or specify explicitly
```

If you have a domain configured, the LLM uses that context to make better judgments about entity names specific to your field.

This generates proposals like:

```yaml
proposals:
- canonical_id: person:samuel_benjamin_bankman_fried
  canonical_name: Samuel Benjamin Bankman-Fried
  entity_type: PERSON
  status: DRAFT                    # ← you decide
  members:
  - id: person:bankman_fried
    name: Bankman-Fried
    confidence: 0.99
  reason: Same person referenced with full name vs. surname only.

- canonical_id: person:stephen_curry
  canonical_name: Stephen Curry
  entity_type: PERSON
  status: DRAFT                    # ← you decide
  members:
  - id: person:steph_curry
    name: Steph Curry
    confidence: 0.99
  reason: Same basketball player referenced with nickname 'Steph' and full name 'Stephen'.
```

**Nothing is merged yet.** The LLM is proposing, not deciding.

### Layer 3: You Review and Decide

You have two options for reviewing proposals:

**Option A: Interactive terminal review**

```bash
sift review
```

Walks through each `DRAFT` proposal one by one. For each, you see the canonical entity, the proposed merge members, the LLM's confidence and reasoning. You approve, reject, or skip.

High-confidence proposals (>0.85 by default) are auto-approved, and low-confidence relations (<=0.5 by default) are auto-rejected:
```bash
sift review                        # uses defaults: --auto-approve 0.85, --auto-reject 0.5
sift review --auto-approve 0.90    # raise the auto-approve threshold
sift review --auto-reject 0.3      # lower the auto-reject threshold
sift review --auto-approve 1.0     # disable auto-approve, review everything manually
```

**Option B: Edit the YAML directly**

Open `output/merge_proposals.yaml` in any text editor. Change `status: DRAFT` to `CONFIRMED` or `REJECTED`:

```yaml
- canonical_id: person:stephen_curry
  canonical_name: Stephen Curry
  entity_type: PERSON
  status: CONFIRMED                # ← approve this merge
  members:
  - id: person:steph_curry
    name: Steph Curry
    confidence: 0.99
  reason: Same basketball player...

- canonical_id: person:winklevoss_twins
  canonical_name: Winklevoss twins
  entity_type: PERSON
  status: REJECTED                 # ← these are distinct people, don't merge
  members:
  - id: person:cameron_winklevoss
    name: Cameron Winklevoss
    confidence: 0.95
  reason: ...
```

**For high-accuracy use cases** (genealogy, legal review), we recommend editing the YAML directly so you can study each proposal carefully. The file is designed to be human-readable.

### Layer 3b: Relation Review

During `sift build`, relations below the confidence threshold (default 0.7) or of types marked `review_required` in your domain config get flagged in `output/relation_review.yaml`:

```yaml
review_threshold: 0.7
relations:
- source_name: Alice Smith
  target_name: Acme Corp
  relation_type: WORKS_FOR
  confidence: 0.45
  evidence: "Alice mentioned she used to work near the Acme building."
  status: DRAFT                    # ← you decide: CONFIRMED or REJECTED
  flag_reason: Low confidence (0.45 < 0.7)
```

Same workflow: review with `sift review` or edit the YAML, then apply.

### Layer 4: Apply Your Decisions

Once you've reviewed everything:

```bash
sift apply-merges
```

This does three things:
1. **Confirmed entity merges** — member entities are absorbed into the canonical entity. All their relations are rewired. Source documents are combined. The member nodes are removed.
2. **Rejected relations** — removed from the graph entirely.
3. **DRAFT proposals** — left untouched. You can come back to them later.

The graph is saved back to `output/graph_data.json`. You can re-export, narrate, or visualize the cleaned graph.

### Iterating

Entity resolution isn't always one-pass. After merging, new duplicates may become apparent. You can re-run:

```bash
sift resolve                  # find new duplicates in the cleaned graph
sift review                   # review the new proposals
sift apply-merges             # apply again
```

Each run is additive — previous `CONFIRMED`/`REJECTED` decisions in `merge_proposals.yaml` are preserved.

### Recommended Workflow by Use Case

| Use Case | Suggested Approach |
|---|---|
| **Quick exploration** | `sift review --auto-approve 0.85` — approve high-confidence, review the rest |
| **Genealogy / family records** | Edit YAML manually, `--auto-approve 1.0` — review every single merge |
| **Legal / investigative** | Edit YAML manually, use `sift view` to inspect the graph between rounds |
| **Large corpus (1000+ entities)** | `sift resolve --embeddings` for better batching, then interactive review |

## Deduplication Internals

The pre-dedup and LLM batching techniques are inspired by [KGGen](https://github.com/stochastic-sisyphus/KGGen) (NeurIPS 2025) by [@stochastic-sisyphus](https://github.com/stochastic-sisyphus). KGGen uses SemHash for deterministic entity deduplication and embedding-based clustering for grouping entities before LLM comparison. sift-kg adapts these into its human-in-the-loop review workflow.

### Embedding-Based Clustering (optional)

By default, `sift resolve` sorts entities alphabetically and splits them into overlapping batches for LLM comparison. This works well when duplicates have similar spelling — but "Robert Smith" (R) and "Bob Smith" (B) end up in different batches and never get compared.

```bash
pip install sift-kg[embeddings]    # sentence-transformers + scikit-learn (~2GB, pulls PyTorch)
sift resolve --embeddings
```

This replaces alphabetical batching with KMeans clustering on sentence embeddings (all-MiniLM-L6-v2). Semantically similar names cluster together regardless of spelling.

| | Default (alphabetical) | `--embeddings` |
|---|---|---|
| Install size | Included | ~2GB (PyTorch) |
| First-run overhead | None | ~90MB model download |
| Per-run overhead | Sorting only | Encoding (<1s for hundreds of entities) |
| Cross-alphabet duplicates | Missed if in different batches | Caught |
| Small graphs (<100/type) | Same result | Same result |

Falls back to alphabetical batching if dependencies aren't installed or clustering fails.

## License

MIT
