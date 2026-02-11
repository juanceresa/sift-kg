"""Read and write merge proposals and relation review YAML files."""

import logging
from pathlib import Path

import yaml

from sift_kg.resolve.models import MergeFile, RelationReviewFile

logger = logging.getLogger(__name__)


def write_proposals(merge_file: MergeFile, path: Path) -> None:
    """Write merge proposals to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = merge_file.model_dump()
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"Wrote {len(merge_file.proposals)} merge proposals to {path}")


def read_proposals(path: Path) -> MergeFile:
    """Read merge proposals from YAML."""
    if not path.exists():
        return MergeFile()
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return MergeFile()
    return MergeFile.model_validate(data)


def write_relation_review(review_file: RelationReviewFile, path: Path) -> None:
    """Write relation review file to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = review_file.model_dump()
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"Wrote {len(review_file.relations)} flagged relations to {path}")


def read_relation_review(path: Path) -> RelationReviewFile:
    """Read relation review from YAML."""
    if not path.exists():
        return RelationReviewFile()
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return RelationReviewFile()
    return RelationReviewFile.model_validate(data)
