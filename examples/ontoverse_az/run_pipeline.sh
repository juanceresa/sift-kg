#!/usr/bin/env bash
# Full sift-kg pipeline over the AstraZeneca Ontoverse computational pathology demo subset.
# Run from the repo root with the venv activated.
set -euo pipefail

EXAMPLE="examples/ontoverse_az"
OUT="$EXAMPLE/output"
DOMAIN="academic"
MODEL="openai/gpt-4o-mini"

echo "[1/6] extract"
sift extract "$EXAMPLE/docs" -o "$OUT" -d "$DOMAIN" --model "$MODEL" --max-cost 3.00

echo "[2/6] build"
sift build -o "$OUT" -d "$DOMAIN"

echo "[3/6] resolve"
sift resolve -o "$OUT" -d "$DOMAIN" --model "$MODEL"

echo "[4/6] auto-confirm high-confidence merges + relations"
python "$EXAMPLE/auto_confirm.py" "$OUT"

echo "[5/6] apply-merges"
sift apply-merges -o "$OUT"

echo "[6/6] narrate"
sift narrate -o "$OUT" -d "$DOMAIN" --model "$MODEL" --max-cost 1.00

echo
echo "Done. Open the viewer:"
echo "  sift view -o $OUT"
echo "  # or directly: open $OUT/graph.html"
