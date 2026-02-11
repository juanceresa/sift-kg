"""Knowledge graph construction and management.

Builds a NetworkX-based knowledge graph from extraction results,
with post-processing for redundancy removal.
"""

from sift_kg.graph.builder import build_graph
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.graph.postprocessor import remove_redundant_edges

__all__ = ["KnowledgeGraph", "build_graph", "remove_redundant_edges"]
