"""Embedding-based entity clustering for semantic batching.

Replaces alphabetical sort + overlapping window batching with KMeans
clustering on sentence embeddings. Entities with semantically similar
names/aliases end up in the same cluster for LLM comparison.

Requires: pip install sift-kg[embeddings]
"""

import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def cluster_entities_by_embedding(
    entities: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    target_cluster_size: int = 100,
) -> list[list[dict]]:
    """Group entities into semantic clusters for LLM comparison.

    Entities with similar names/aliases land in the same batch,
    regardless of alphabetical order.

    Args:
        entities: List of entity dicts with 'name' and optional 'aliases'
        model_name: Sentence transformer model to use
        target_cluster_size: Approximate entities per cluster
    Returns:
        List of entity clusters (each cluster is a list of entity dicts)

    Raises:
        ImportError: If sentence-transformers or sklearn not installed
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError(
            "Embedding clustering requires sentence-transformers and scikit-learn. "
            "Install with: pip install sift-kg[embeddings]"
        )

    if len(entities) < 10:
        return [entities]

    n_clusters = max(1, len(entities) // target_cluster_size)
    if n_clusters == 1:
        return [entities]

    # Build text per entity: name + aliases for richer embedding
    texts = []
    for entity in entities:
        aliases = entity.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        text = entity["name"]
        if aliases:
            text += " " + " ".join(aliases)
        texts.append(text)

    logger.info(
        f"Clustering {len(entities)} entities into ~{n_clusters} groups "
        f"(model: {model_name})"
    )

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group entities by cluster label
    clusters: dict[int, list[dict]] = {}
    for entity, label in zip(entities, labels, strict=True):
        clusters.setdefault(int(label), []).append(entity)

    result = list(clusters.values())

    # Log cluster size distribution
    sizes = sorted(len(c) for c in result)
    logger.info(
        f"Cluster sizes: min={sizes[0]}, max={sizes[-1]}, "
        f"median={sizes[len(sizes) // 2]}, count={len(result)}"
    )

    return result
