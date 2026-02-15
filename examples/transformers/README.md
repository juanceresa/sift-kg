# Transformers Research Example

Complete sift-kg pipeline output from 12 foundational transformer architecture papers.

## Quick Look

Open the interactive graph viewer — no install needed:

```bash
open output/graph.html     # macOS
xdg-open output/graph.html # Linux
```

Or with sift installed: `sift view -o examples/transformers/output`

## What's Here

```
docs/                          # 12 source papers (PDF)
output/
  extractions/                 # Per-document entity+relation JSON from LLM
  graph_data.json              # Knowledge graph (425 entities, 1122 relations)
  communities.json             # Detected graph communities
  entity_descriptions.json     # AI-generated entity descriptions
  narrative.md                 # Prose narrative with entity profiles
  graph.html                   # Interactive pyvis graph viewer
```

## Pipeline Stats

| Step | Result |
|------|--------|
| Documents | 12 PDFs (Attention Is All You Need, BERT, GPT-2, GPT-3, ViT, DALL-E, etc.) |
| Extraction | Entities and relations via LLM |
| Build + postprocess | 425 entities, 1122 relations |
| Entity types | 118 systems, 73 concepts, 71 researchers, 70 methods, 34 phenomena, 25 findings |
| Narrative | Overview + entity descriptions |
| Domain | `academic` (bundled) |
| Model | claude-haiku-4-5-20251001 |
| Total cost | ~$0.72 |

## Re-run It Yourself

```bash
pip install sift-kg

# Start from the existing extractions (free — no LLM calls)
sift build -o examples/transformers/output

# Or re-run the full pipeline from scratch with your own papers
mkdir my-papers && cp your-pdfs/*.pdf my-papers/
sift extract my-papers --model openai/gpt-4o-mini -o my-output --domain academic
sift build -o my-output
sift resolve -o my-output --model openai/gpt-4o-mini
sift review -o my-output
sift apply-merges -o my-output
sift narrate -o my-output --model openai/gpt-4o-mini
sift view -o my-output
```
