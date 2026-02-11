"""Narrative generation from knowledge graphs.

Generates human-readable markdown summaries of the knowledge graph,
including an overview narrative and per-entity descriptions.
"""

from sift_kg.narrate.generator import generate_narrative

__all__ = ["generate_narrative"]
