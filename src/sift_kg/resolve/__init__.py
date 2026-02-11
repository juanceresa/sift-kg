"""Entity resolution — find and merge duplicate entities.

LLM-based entity similarity detection, merge proposals as YAML,
relation review for flagged relations, and graph surgery to apply.
"""

from sift_kg.resolve.engine import apply_merges, apply_relation_rejections
from sift_kg.resolve.resolver import find_merge_candidates

__all__ = ["find_merge_candidates", "apply_merges", "apply_relation_rejections"]
