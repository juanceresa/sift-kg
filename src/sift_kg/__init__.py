"""sift-kg: Document-to-knowledge-graph pipeline.

A zero-config CLI tool for extracting knowledge graphs from documents
using any LLM provider. Supports domain-agnostic entity and relation
extraction, interactive entity resolution, and narrative generation.
"""

__version__ = "0.3.1"

from sift_kg.domains.loader import load_domain
from sift_kg.domains.models import DomainConfig
from sift_kg.export import export_graph
from sift_kg.extract.llm_client import LLMClient
from sift_kg.graph.knowledge_graph import KnowledgeGraph
from sift_kg.pipeline import (
    run_apply_merges,
    run_build,
    run_export,
    run_extract,
    run_narrate,
    run_pipeline,
    run_resolve,
)

__all__ = [
    "__version__",
    "DomainConfig",
    "KnowledgeGraph",
    "LLMClient",
    "export_graph",
    "load_domain",
    "run_apply_merges",
    "run_build",
    "run_export",
    "run_extract",
    "run_narrate",
    "run_pipeline",
    "run_resolve",
]
