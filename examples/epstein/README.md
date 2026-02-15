# Giuffre v. Maxwell Example

Complete sift-kg pipeline output from the unsealed Giuffre v. Maxwell court deposition (36 document sections, OCR-scanned PDF).

## Quick Look

Open the interactive graph viewer — no install needed:

```bash
open output/graph.html     # macOS
xdg-open output/graph.html # Linux
```

Or with sift installed: `sift view -o examples/epstein/output`

## What's Here

```
docs/                          # Source PDF (unsealed court deposition)
output/
  extractions/                 # Per-document entity+relation JSON from LLM
  graph_data.json              # Knowledge graph (226 entities, 708 relations)
  merge_proposals.yaml         # Entity merge decisions (16 confirmed merges)
  relation_review.yaml         # Flagged relations reviewed
  communities.json             # Detected graph communities
  entity_descriptions.json     # AI-generated entity descriptions
  narrative.md                 # Prose narrative with entity profiles
  graph.html                   # Interactive pyvis graph viewer
```

## Pipeline Stats

| Step | Result |
|------|--------|
| Documents | 1 PDF, 36 sections (unsealed court deposition) |
| Build + postprocess | 226 entities, 708 relations |
| Entity types | 93 persons, 56 locations, 24 organizations, 15 events, 2 vehicles |
| Resolve | 16 entity merges confirmed via LLM + human review |
| Final graph | 190 entities, 387 relations |
| Narrative | Overview + entity descriptions |
| Model | claude-haiku-4-5-20251001 |

## Re-run It Yourself

```bash
pip install sift-kg

# Start from the existing extractions (free — no LLM calls)
sift build -o examples/epstein/output

# Or re-run the full pipeline from scratch
sift extract examples/epstein/docs --model openai/gpt-4o-mini -o my-output --ocr
sift build -o my-output
sift resolve -o my-output --model openai/gpt-4o-mini
sift review -o my-output
sift apply-merges -o my-output
sift narrate -o my-output --model openai/gpt-4o-mini
sift view -o my-output
```
