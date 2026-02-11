"""Domain configuration system for sift-kg."""

from sift_kg.domains.loader import DomainLoader, load_domain
from sift_kg.domains.models import DomainConfig, EntityTypeConfig, RelationTypeConfig

__all__ = ["DomainConfig", "DomainLoader", "EntityTypeConfig", "RelationTypeConfig", "load_domain"]
