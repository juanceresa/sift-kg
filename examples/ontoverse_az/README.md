# AstraZeneca Ontoverse — sift-kg demo

Built for the 2026-05-12 AstraZeneca computational pathology lab presentation.

## What this is

A 16-paper slice of [AstraZeneca's Ontoverse Zotero sandbox][ontoverse-zotero] run through the full sift-kg pipeline. The slice is designed to produce a graph that AZ's comp-path lab will recognize as **their world**: AZ's own image-generation papers anchored among the pathology foundation models they build on (UNI, CONCH, TITAN, PixCell, PLUTO-4) plus the diffusion peers they share an architectural lineage with.

[ontoverse-zotero]: https://github.com/AstraZeneca/ontoverse-kg-choreographer/blob/main/zotero_library/OntoverseSandbox.rdf

## Files

```
parse_rdf.py             Parses the full Ontoverse RDF into manifest.{json,csv} + collections.json
manifest.{json,csv}      All 157 papers in AZ's sandbox with DOI / authors / PDF URL / themes
collections.json         220 AZ-curated thematic groupings (Foundation Models, Diffusion, etc.)
demo_subset.yaml         The 16 papers chosen for the demo + rationale per paper
download_pdfs.sh         Pulls those 16 PDFs into docs/
auto_confirm.py          Auto-curates merge proposals (confirm >= 0.85, otherwise leave DRAFT)
run_pipeline.sh          Full extract -> build -> resolve -> apply-merges -> narrate
sift.yaml                Project config (domain: academic)
docs/                    The 16 source PDFs
output/                  Pipeline outputs (graph_data.json, graph.html, narrative.md, ...)
OntoverseSandbox.rdf     Snapshot of the source RDF (May 2026)
```

## The 16 papers

| # | Paper | Role |
|---|---|---|
| 01 | ReStainGAN (AZ) | IHC to IF stain translation — AZ's GAN-era work |
| 02 | The Ontoverse (AZ) | The meta-paper — becomes a node in *our* graph |
| 03 | Mask-guided cross-image attention diffusion (AZ) | Zero-shot histopath image generation |
| 04 | MSDM (AZ) | Multimodal conditioned diffusion for segmentation |
| 05 | UNI | Foundational pathology self-supervised model |
| 06 | CONCH | Vision-language pathology FM |
| 07 | TITAN | Multimodal whole-slide FM |
| 08 | PixCell | Generative pathology FM (direct peer to AZ's diffusion work) |
| 09 | PLUTO-4 | Frontier pathology FMs |
| 10 | Survey on Computational Pathology FMs | Hub — cites every FM in 05-09 |
| 11 | Do Histopath FMs Eliminate Batch Effects? | Benchmarks UNI/CONCH/TITAN |
| 12 | Pathology FMs are Scanner Sensitive (ScanGen) | Critique of the FM cluster |
| 13 | ZoomLDM | Multi-scale latent diffusion for WSI |
| 14 | ∞-Brush | Diffusion in infinite dimensions for large images |
| 15 | Counterfactual Trajectories with Latent Diffusion | Counterfactual + diffusion + concept discovery |
| 16 | MIPHEI-ViT | **Killer bridge:** uses ViT FMs to predict IF from H&E (combines #05-09 with AZ's ReStainGAN topic) |

## Quick look (no install needed)

```bash
open examples/ontoverse_az/output/graph.html
```

## Pipeline stats (actual run, 2026-05-10)

| Step | Result | Cost |
|---|---|---|
| Extract | 1074 entities, 1140 relations (155 chunks) | $0.2447 |
| Build | 518 entities, 1365 relations, 14 communities | (no LLM) |
| Resolve | 35 merge proposals, 46 flagged relations | $0.0054 |
| Merge curation | Hand-reviewed: 30 confirmed, 5 rejected | (no LLM) |
| Apply-merges | 32 merges applied | (no LLM) |
| Narrate | 216 entity descriptions + overview + timeline | $0.0767 |
| **Total** | **486 entities, 1363 relations, 14 communities** | **$0.41** |

### Merge curation calls

Manually reviewed all 35 resolve proposals. Rejected 5; kept 30. Rejections:

| Proposal | Why rejected |
|---|---|
| `multi-head cross-attention` + `multi-headed-ABMIL` (1.0 conf) | LLM hallucination — different concepts (transformer mechanism vs MIL variant) |
| `PANDA` + `PANDA-Small` (0.9 conf) | Different datasets — full vs benchmarking subset |
| `counterfactuals` + `counterfactual_explanations` + `counterfactual_analysis` (0.8 conf) | `counterfactual explanations` is a distinct XAI subfield |
| `Fréchet Inception Distance` + `Fréchet H-optimus/Virchow Distance` (0.7 conf) | Different metrics with different backbones — the variants exist *because* they differ from FID |
| `CRC` + `CRC polyp classification` + `CRC Screening` + `CRC tissue classification` (0.8 conf) | Disease vs three different classification tasks |

## Reproduce

```bash
cd <repo-root>
source venv/bin/activate
bash examples/ontoverse_az/download_pdfs.sh
bash examples/ontoverse_az/run_pipeline.sh
sift view -o examples/ontoverse_az/output
```

Expected cost (gpt-4o-mini): ~$0.30-$0.50 total. Capped at $3 by the script.

## Talking points for the AZ demo

Verified from the actual graph (see `output/topology_snapshot.json` and `sift query` checks):

1. **"We ran sift on your own sandbox."** The Ontoverse paper is itself a node — `SYSTEM: The Ontoverse`, 11 connections, correctly attributed to Zimmermann + Wiktorek, identified as a KG system using NER + hierarchical topic modelling + RAG for drug discovery in computational pathology.
2. **UNI is the dominant hub** with 92 connections bridging 8 of the 14 communities — it's the substrate that everything else compares itself against (REMEDIS, ResNet-50, CTransPath all show up with `SUPPORTS`/`IMPLEMENTS` edges to UNI).
3. **Community 3 = AZ's generative-pathology peer group**: anchored on ZoomLDM, ∞-Brush, PixCell-1024, plus latent diffusion + cross-attention neural operator concepts.
4. **Community 5 = AZ's direct commercial story**: PixCell + virtual staining + data scarcity + BRCA. This is exactly the "synthetic pathology images for downstream tasks" thesis AZ has been working on.
5. **MIPHEI-ViT is the killer bridge** (37 connections, bridges 3 communities): on one side it connects to ViT/U-Net/CycleGAN/LoRA (the FM substrate); on the other to H&E, multiplex IF, and specific immune markers CD4/CD8/FOXP3/Pan-CK/CD3e. It captures exactly the "use ViT FMs to solve AZ's stain translation problem" story.
6. **No training step.** The graph is produced from LLM extraction alone — no labeled data, no fine-tuning, no entity-resolution training. Just `sift extract` and a 250-line domain YAML. Total cost: **$0.33**.

## Useful queries during the demo

```bash
sift query "The Ontoverse" --pretty -o examples/ontoverse_az/output
sift query "UNI" -t SYSTEM --pretty -o examples/ontoverse_az/output
sift query "MIPHEI" --pretty -o examples/ontoverse_az/output
sift search "PixCell" -o examples/ontoverse_az/output
sift view --community "Community 5" -o examples/ontoverse_az/output  # AZ's commercial cluster
sift view --neighborhood "UNI" --depth 2 -o examples/ontoverse_az/output  # focus on the FM hub
```
