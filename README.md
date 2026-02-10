# sift-kg

**Turn any pile of documents into a knowledge graph.**

Domain-configurable entity extraction, human-in-the-loop entity resolution, and narrative generation — from the command line.

---

```bash
pip install sift-kg

sift extract ./documents/              # extract entities & relations
sift review                            # approve/reject entity merges
sift narrate                           # generate a narrative from the graph
```

## Overview

sift-kg is an end-to-end CLI tool for building knowledge graphs from unstructured document collections. It handles text extraction, LLM-powered entity and relation extraction, entity deduplication with human review, and narrative generation — in a single pipeline.

### Pipeline

```
Documents (PDF, text, HTML, email)
        |
        v
   Text Extraction (local, free)
        |
        v
   Entity & Relation Extraction (LLM)
        |
        v
   Knowledge Graph (persistent, with provenance)
        |
        v
   Entity Resolution (ML clustering + human review)
        |
        v
   Narrative Generation (LLM)
```

## Features

- **Zero-config start** — point at a folder, get a knowledge graph
- **Domain-configurable** — define custom entity types and relation types in YAML
- **Human-in-the-loop entity resolution** — sift proposes entity merges, you approve or reject
- **Narrative generation** — structured summaries tracing connections across your documents
- **Source provenance** — every entity and relation links back to the source document and passage
- **Multi-model** — OpenAI, Anthropic, Ollama (local/private), or any LiteLLM-compatible provider
- **Runs locally** — your documents stay on your machine

## Use Cases

- Investigative journalism — analyze FOIA releases, court filings, and document leaks
- OSINT research — map entity networks from public records
- Academic research — build structured datasets from historical archives and primary sources
- Legal review — extract and connect entities across document collections

## Status

Under active development.

<!--
## Quickstart

Coming soon.

## Domain Configuration

Coming soon.

## Examples

Coming soon.
-->

## License

MIT
