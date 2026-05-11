#!/usr/bin/env bash
# Download the 16 demo PDFs for the AstraZeneca Ontoverse computational pathology demo.
# Usage: bash download_pdfs.sh
set -uo pipefail

cd "$(dirname "$0")/docs"

# parallel arrays: name | url
NAMES=(
  "01_az_ReStainGAN.pdf"
  "02_az_Ontoverse.pdf"
  "03_az_MaskGuidedDiffusion.pdf"
  "04_az_MSDM.pdf"
  "05_fm_UNI.pdf"
  "06_fm_CONCH.pdf"
  "07_fm_TITAN.pdf"
  "08_fm_PixCell.pdf"
  "09_fm_PLUTO4.pdf"
  "10_ev_FMSurvey.pdf"
  "11_ev_BatchEffects.pdf"
  "12_ev_ScannerSensitive.pdf"
  "13_df_ZoomLDM.pdf"
  "14_df_InfBrush.pdf"
  "15_df_CounterfactualDiff.pdf"
  "16_br_MIPHEI_ViT.pdf"
)
URLS=(
  "https://arxiv.org/pdf/2403.06545.pdf"
  "https://arxiv.org/pdf/2408.03339.pdf"
  "https://arxiv.org/pdf/2407.11664.pdf"
  "https://arxiv.org/pdf/2510.09121.pdf"
  "https://arxiv.org/pdf/2308.15474.pdf"
  "https://arxiv.org/pdf/2307.12914.pdf"
  "https://arxiv.org/pdf/2411.19666.pdf"
  "https://arxiv.org/pdf/2506.05127.pdf"
  "https://arxiv.org/pdf/2511.02826.pdf"
  "https://arxiv.org/pdf/2501.15724.pdf"
  "https://arxiv.org/pdf/2411.05489.pdf"
  "https://arxiv.org/pdf/2507.22092.pdf"
  "https://arxiv.org/pdf/2411.16969.pdf"
  "https://arxiv.org/pdf/2407.14709.pdf"
  "https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1.full.pdf"
  "https://arxiv.org/pdf/2505.10294.pdf"
)

ok=0; fail=0; skipped=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  url="${URLS[$i]}"
  if [[ -s "$name" ]]; then
    echo "skip   $name (exists, $(wc -c < "$name") bytes)"
    skipped=$((skipped+1))
    continue
  fi
  if curl -fsSL -A "Mozilla/5.0" -o "$name" "$url"; then
    size=$(wc -c < "$name")
    if (( size < 10000 )); then
      echo "FAIL   $name (only $size bytes from $url)"
      rm -f "$name"
      fail=$((fail+1))
    else
      echo "ok     $name ($size bytes)"
      ok=$((ok+1))
    fi
  else
    echo "FAIL   $name from $url"
    fail=$((fail+1))
  fi
done

echo
echo "Summary: $ok ok, $fail failed, $skipped skipped"
ls -lh
