# sift-kg

**Turn any pile of documents into a knowledge graph.**

Domain-configurable entity extraction, human-in-the-loop entity resolution, and narrative generation — from the command line.

```bash
pip install sift-kg

sift init                           # create .env.example with config
sift extract ./documents/           # extract entities & relations
sift build                          # build knowledge graph
sift resolve                        # find duplicate entities
sift review                         # approve/reject merges interactively
sift apply-merges                   # apply your decisions
sift narrate                        # generate narrative summary
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
  Export (JSON, GraphML, GEXF, CSV)
       ↓
  Narrative Generation (LLM)
```

Every entity and relation links back to the source document and passage. You control what gets merged. The graph is yours.

## Features

- **Zero-config start** — point at a folder, get a knowledge graph
- **Any LLM provider** — OpenAI, Anthropic, Ollama (local/private), or any LiteLLM-compatible provider
- **Domain-configurable** — define custom entity types and relation types in YAML
- **Human-in-the-loop** — sift proposes entity merges, you approve or reject in an interactive terminal UI
- **Export anywhere** — GraphML (yEd, Cytoscape), GEXF (Gephi), CSV, or native JSON
- **Narrative generation** — structured summaries tracing connections across your documents
- **Source provenance** — every extraction links to the document and passage it came from
- **Budget controls** — set `--max-cost` to cap LLM spending
- **Runs locally** — your documents stay on your machine

## Use Cases

- **Investigative journalism** — analyze FOIA releases, court filings, and document leaks
- **OSINT research** — map entity networks from public records
- **Academic research** — build structured datasets from historical archives
- **Legal review** — extract and connect entities across document collections
- **Genealogy** — trace family relationships across vital records

## Installation

Requires Python 3.11+.

```bash
pip install sift-kg
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
sift init                     # creates .env.example
cp .env.example .env          # copy and add your API key
```

Set your API key in `.env`:
```
SIFT_OPENAI_API_KEY=sk-...
SIFT_DEFAULT_MODEL=openai/gpt-4o-mini
```

Or use Anthropic, Ollama, or any LiteLLM provider:
```
SIFT_ANTHROPIC_API_KEY=sk-ant-...
SIFT_DEFAULT_MODEL=anthropic/claude-haiku-4-5-20251001
```

### 2. Extract entities and relations

```bash
sift extract ./my-documents/
```

Reads PDFs, text files, and HTML. Extracts entities and relations using your configured LLM. Results saved as JSON in `output/extractions/`.

### 3. Build the knowledge graph

```bash
sift build
```

Constructs a NetworkX graph from all extractions. Flags low-confidence relations for review. Saves to `output/graph_data.json`.

### 4. Find and resolve duplicates

```bash
sift resolve                  # LLM proposes entity merges
sift review                   # interactive terminal review
sift apply-merges             # apply confirmed merges
```

Or edit `output/merge_proposals.yaml` directly — change `status: DRAFT` to `CONFIRMED` or `REJECTED`.

### 5. Export

```bash
sift export graphml           # → output/graph.graphml (Gephi, yEd, Cytoscape)
sift export gexf              # → output/graph.gexf (Gephi native)
sift export csv               # → output/csv/entities.csv + relations.csv
sift export json              # → output/graph.json
```

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
├── extractions/          # Per-document extraction JSON
│   ├── document1.json
│   └── document2.json
├── graph_data.json       # Knowledge graph (native format)
├── merge_proposals.yaml  # Entity merge proposals (DRAFT/CONFIRMED/REJECTED)
├── relation_review.yaml  # Flagged relations for review
├── narrative.md          # Generated narrative summary
├── graph.graphml         # GraphML export (if exported)
├── graph.gexf            # GEXF export (if exported)
└── csv/                  # CSV export (if exported)
    ├── entities.csv
    └── relations.csv
```

## License

MIT
