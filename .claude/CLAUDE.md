# CLAUDE.md — sift-kg

> Standalone open-source CLI: documents → knowledge graph → entity dedup → narrative

## Quick Reference

```bash
# Dev setup
cd /Users/juanceresa/Desktop/cs/sift-kg
source venv/bin/activate
pip install -e ".[dev]"

# Run CLI
sift --help
sift extract ./docs/              # model/domain from sift.yaml
sift build
sift resolve
sift apply-merges
sift narrate

# Tests
pytest
ruff check src/
```

## Architecture

```
sift extract   sift build     sift resolve     sift apply-merges   sift narrate   sift view
     │               │                │                  │                 │              │
     ▼               ▼                ▼                  ▼                 ▼              ▼
┌─────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐
│ ingest/  │→│ extract/   │→│  graph/   │→│ resolve/  │→│ narrate/  │→│ visualize  │
│ reader   │  │ llm_client │  │  builder  │  │ resolver  │  │ generator │  │ (pyvis)    │
│ chunker  │  │ extractor  │  │  postproc │  │ engine    │  │ prompts   │  │            │
│          │  │ prompts    │  │  kg       │  │ io        │  │           │  │            │
└─────────┘  └───────────┘  └──────────┘  └──────────┘  └──────────┘  └───────────┘
```

**Data flow:** All stages persist to `output/` directory. Each stage is idempotent.

## Pipeline Commands

1. `sift extract ./docs/` — Read documents, extract entities and relations via LLM
2. `sift build` — Construct knowledge graph from extractions, pre-dedup near-identical names, flag relations for review
3. `sift resolve` — Find duplicate entities using LLM-based comparison (`--embeddings` for semantic batching)
4. Edit `merge_proposals.yaml` — DRAFT → CONFIRMED/REJECTED
5. Edit `relation_review.yaml` — DRAFT → CONFIRMED/REJECTED
6. `sift apply-merges` — Apply confirmed merges + remove rejected relations
7. `sift narrate` — Generate markdown narrative with entity descriptions
8. `sift view` — Interactive pyvis graph visualization in browser

Utility commands: `sift init` (create .env.example), `sift info` (show project stats)

## Key Design Decisions

1. **LiteLLM** — supports OpenAI, Anthropic, Ollama with one interface
2. **Typer** — type-hint CLI, less boilerplate
3. **LLM-based entity resolution** — no training step, works out of the box
4. **Generic Entity model** — single model with `entity_type: str`, driven by domain config
5. **pdfplumber + plain text** (no OCR) — 90% of target docs are text-searchable
6. **YAML-based review workflow** — merge proposals and relation reviews as human-editable YAML
7. **Combined extraction prompt** — entities + relations in one LLM call per chunk (halves API calls)
8. **Degree-ranked narration** — overview uses top 50 entities by connectivity, descriptions capped at 100
9. **Async concurrency** — all LLM-heavy steps (extract, narrate) use shared semaphore + rate limiter
10. **SemHash pre-dedup** — deterministic merge of near-identical names during build (KGGen-inspired)
11. **Embedding clustering** — optional semantic batching for resolve step (KGGen-inspired, heavy deps)
12. **sift.yaml project config** — per-project settings (domain, model) without flags; priority: CLI > env > .env > sift.yaml > defaults
13. **Title prefix stripping** — pre-dedup strips ~35 common title prefixes (Detective, Dr., Judge, etc.) before comparing names

## Module Map

| Module | Purpose |
|---|---|
| `config.py` | Pydantic-settings (API keys, model, output dir) + sift.yaml project config |
| `domains/` | Domain config models, YAML loader, registry |
| `ingest/` | Document reader (PDF, text, HTML), text chunker |
| `extract/` | LLM client (LiteLLM), extractor, prompts, models |
| `graph/` | KnowledgeGraph (NetworkX), builder, postprocessor, prededup |
| `resolve/` | LLM resolver, merge models, I/O, graph surgery engine, clustering |
| `narrate/` | Narrative generator, prompts |
| `visualize.py` | Interactive pyvis graph viewer with description integration |

## Known Gotchas

- Response parsing: LLM sometimes returns JSON wrapped in markdown code fences. Strip with regex.
- Postprocessor: transitive redundancy removal works on same-type relations only (LOCATED_IN). Also prunes isolated entities (no substantive connections).
- Entity IDs: `{type}:{normalized_name}` using unidecode — same entity from different docs auto-merges
- Linter strips "unused" module-level imports between edits — use inline imports as workaround

## Output Structure

```
output/
├── extractions/              # Per-document extraction JSON
├── graph_data.json           # Knowledge graph (nodes + edges + metadata)
├── merge_proposals.yaml      # Entity merge proposals (DRAFT/CONFIRMED/REJECTED)
├── relation_review.yaml      # Flagged relations (DRAFT/CONFIRMED/REJECTED)
├── narrative.md              # Prose narrative with entity descriptions
├── entity_descriptions.json  # Description sidecar (loaded by viewer)
└── graph.html                # Interactive pyvis visualization
```

## Code Standards

- Python 3.11+, type hints on all functions
- Pydantic for all data models
- `pathlib` not `os.path`
- `ruff` for linting
- No bare `except:` — catch specific exceptions
- Fail fast on missing config (validate early in CLI commands)

## What NOT to Build (v0.2 scope)

- No OCR pipeline
- No translation
- No ML-based dedup training (LLM-based only for now)
- ~~No frontend/visualization~~ (pyvis viewer added)
- No dossier/PDF generation
- No document grouping (multi-part docs)
- No verification tiers (use confidence scores)
- No database server (files only)
