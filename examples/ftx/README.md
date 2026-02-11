# FTX Collapse Example

Complete sift-kg pipeline output from 9 Wikipedia articles about the FTX cryptocurrency exchange collapse.

## Quick Look

Open the interactive graph viewer — no install needed:

```bash
open output/graph.html     # macOS
xdg-open output/graph.html # Linux
```

Or with sift installed: `sift view --output examples/ftx/output`

## What's Here

```
docs/                          # 9 source documents (~148K total)
output/
  extractions/                 # Per-document entity+relation JSON from LLM
  graph_data.json              # Knowledge graph (373 entities, 1184 relations)
  merge_proposals.yaml         # Entity merge decisions (CONFIRMED/REJECTED)
  entity_descriptions.json     # AI-generated entity descriptions
  narrative.md                 # Prose narrative with entity profiles
  graph.html                   # Interactive pyvis graph viewer
```

## Pipeline Stats

| Step | Result |
|------|--------|
| Documents | 9 text files (FTX, Alameda, Binance, key people) |
| Extraction | ~777 raw entities from LLM |
| Pre-dedup (semhash) | 777 → 750 (27 deterministic merges) |
| Build + postprocess | 432 entities, 1201 relations |
| Resolve (3 passes) | 59 entities merged via LLM + human review |
| Final graph | 373 entities, 1184 relations |
| Narrative | Overview + 100 entity descriptions |
| Model | claude-haiku-4-5-20251001 |
| Total cost | ~$0.28 (extraction was separate) |

## Re-run It Yourself

```bash
pip install sift-kg

# Start from the existing extractions (free — no LLM calls)
sift build --output examples/ftx/output

# Or re-run the full pipeline from scratch
sift extract examples/ftx/docs --model openai/gpt-4o-mini --output my-output
sift build --output my-output
sift resolve --output my-output --model openai/gpt-4o-mini
sift review --output my-output
sift apply-merges --output my-output
sift narrate --output my-output --model openai/gpt-4o-mini
sift view --output my-output
```
