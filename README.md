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

## Installation

**Requirements:** Python 3.11 or higher

```bash
pip install sift-kg
```

For development:

```bash
git clone https://github.com/civictable/sift-kg.git
cd sift-kg
pip install -e ".[dev]"
```

### Shell Completion

Install shell completion for your shell:

```bash
sift --install-completion
```

Then restart your terminal. Supports bash, zsh, fish, and PowerShell.

## Quick Start

```bash
# Show available commands
sift --help

# Initialize a project (coming in Phase 1)
sift init

# Extract entities and relations (coming in Phase 3)
sift extract ./documents/

# Review entity merges (coming in Phase 5)
sift review

# Generate narrative summaries (coming in Phase 6)
sift narrate
```

**Note:** All commands currently show placeholder warnings until respective phases are completed. The CLI structure and installation are functional as of v0.2.0.

## Status

Under active development. Phase 1 (scaffolding) in progress.

Current functionality:
- ✅ Package installation via pip
- ✅ CLI command structure with `sift` entry point
- ✅ Shell completion support
- ⏳ Project initialization (Plan 01-02)
- ⏳ Document extraction (Phase 3)
- ⏳ Entity resolution (Phase 5)
- ⏳ Narrative generation (Phase 6)

<!--
## Domain Configuration

Coming soon.

## Examples

Coming soon.
-->

## License

MIT
