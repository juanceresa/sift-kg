# Ontoverse Computational Pathology Example

Complete sift-kg pipeline output from 16 papers in AstraZeneca's open-source [Ontoverse Zotero sandbox][src] — a corpus on computational pathology AI: foundation models, generative pathology, and spatial biomarkers.

[src]: https://github.com/AstraZeneca/ontoverse-kg-choreographer/blob/main/zotero_library/OntoverseSandbox.rdf

## Quick Look

Open the interactive graph viewer — no install needed:

```bash
open output/graph.html     # macOS
xdg-open output/graph.html # Linux
```

Or with sift installed: `sift view -o examples/ontoverse_az/output`

## What's Here

```
output/
  communities.json             # 13 themed graph communities
  entity_descriptions.json     # AI-generated entity descriptions
  narrative.md                 # Prose narrative with entity profiles
  graph.html                   # Interactive pyvis graph viewer
```

## Pipeline Stats

| Step | Result |
|------|--------|
| Documents | 16 PDFs (UNI, CONCH, TITAN, PixCell, PLUTO-4, ZoomLDM, ∞-Brush, MIPHEI-ViT, etc.) |
| Extraction | 1074 entities, 1140 relations (155 chunks) |
| Build + postprocess | 518 entities, 1365 relations, 13 communities |
| Resolve + merge | 32 merges applied (manually curated from 35 proposals) |
| Narrate | 216 entity descriptions + overview + timeline |
| Final | 486 entities, 1363 relations, 13 themed communities |
| Domain | `academic` (bundled) |
| Model | openai/gpt-4o-mini |
| Total cost | ~$0.41 |

## Re-run It Yourself

```bash
pip install sift-kg

# Run the full pipeline on your own pathology / AI papers
mkdir my-papers && cp your-pdfs/*.pdf my-papers/
sift extract my-papers --model openai/gpt-4o-mini -o my-output --domain academic
sift build -o my-output
sift resolve -o my-output --model openai/gpt-4o-mini
sift review -o my-output
sift apply-merges -o my-output
sift narrate -o my-output --model openai/gpt-4o-mini
sift view -o my-output
```
