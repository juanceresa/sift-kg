"""Auto-confirm high-confidence merge proposals and relation reviews.

Strategy (intentionally conservative — keeps demo graph clean without manual review):
  - merge_proposals.yaml: CONFIRM only if every member's confidence >= MERGE_THRESHOLD.
    Leave the rest as DRAFT (apply-merges will skip them).
  - relation_review.yaml: CONFIRM only if confidence >= RELATION_THRESHOLD,
    else REJECT below RELATION_REJECT (clearly weak), else leave DRAFT.

Run after `sift build` (which produces both YAMLs) and before `sift apply-merges`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

MERGE_THRESHOLD = 0.85          # confirm merges this confident or higher
RELATION_THRESHOLD = 0.80       # confirm relations this confident or higher
RELATION_REJECT_BELOW = 0.30    # actively reject clearly-weak relations


def process_merges(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        print(f"  (skipped — {path.name} does not exist)")
        return 0, 0, 0
    data = yaml.safe_load(path.read_text()) or {}
    confirmed = drafted = total = 0
    for p in data.get("proposals", []):
        total += 1
        if p.get("status") != "DRAFT":
            continue
        members = p.get("members", [])
        confidences = [m.get("confidence", 0) for m in members]
        if confidences and min(confidences) >= MERGE_THRESHOLD:
            p["status"] = "CONFIRMED"
            confirmed += 1
        else:
            drafted += 1
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return confirmed, drafted, total


def process_relations(path: Path) -> tuple[int, int, int, int]:
    if not path.exists():
        print(f"  (skipped — {path.name} does not exist)")
        return 0, 0, 0, 0
    data = yaml.safe_load(path.read_text()) or {}
    confirmed = rejected = drafted = total = 0
    for r in data.get("relations", []):
        total += 1
        if r.get("status") != "DRAFT":
            continue
        c = r.get("confidence", 0)
        if c >= RELATION_THRESHOLD:
            r["status"] = "CONFIRMED"
            confirmed += 1
        elif c < RELATION_REJECT_BELOW:
            r["status"] = "REJECTED"
            rejected += 1
        else:
            drafted += 1
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return confirmed, rejected, drafted, total


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/ontoverse_az/output")
    print(f"Auto-curating proposals in {out}")
    print(f"  merge threshold:    >= {MERGE_THRESHOLD}")
    print(f"  relation confirm:   >= {RELATION_THRESHOLD}")
    print(f"  relation reject:    <  {RELATION_REJECT_BELOW}")

    print("\nMerges:")
    c, d, t = process_merges(out / "merge_proposals.yaml")
    print(f"  total={t}  confirmed={c}  left-draft={d}")

    print("\nRelations:")
    c, rj, d, t = process_relations(out / "relation_review.yaml")
    print(f"  total={t}  confirmed={c}  rejected={rj}  left-draft={d}")


if __name__ == "__main__":
    main()
