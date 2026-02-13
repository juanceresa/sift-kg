"""Deterministic pre-deduplication of near-identical entity names.

Runs during `sift build` before entities become graph nodes.
Catches obvious duplicates (plurals, near-identical strings) so they
don't waste an LLM call in the resolve step.

Uses:
- Unicode normalization + singularization for deterministic merges
- SemHash (Model2Vec-based) at 0.95 threshold for fuzzy near-matches
"""

import logging
from collections import Counter

import inflect
from semhash import SemHash
from unidecode import unidecode

from sift_kg.extract.models import DocumentExtraction

# Suppress SemHash/model2vec download noise
for _logger_name in ("semhash", "model2vec", "minishlab"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_INFLECT_ENGINE = inflect.engine()

# Common title prefixes that don't change entity identity
_TITLE_PREFIXES = (
    "detective", "det.", "officer", "sergeant", "sgt.", "lieutenant", "lt.",
    "captain", "cpt.", "chief", "deputy", "agent", "special agent",
    "dr.", "dr", "doctor", "prof.", "professor",
    "mr.", "mr", "mrs.", "mrs", "ms.", "ms", "miss",
    "judge", "justice", "hon.", "honorable",
    "senator", "sen.", "representative", "rep.", "governor", "gov.",
    "president", "vice president",
    "attorney", "atty.", "counsel", "esquire", "esq.",
    "reverend", "rev.", "father", "sister", "brother",
    "sir", "dame", "lord", "lady",
)


def _strip_titles(name: str) -> str:
    """Strip common title prefixes from a name."""
    changed = True
    while changed:
        changed = False
        for prefix in _TITLE_PREFIXES:
            if name.startswith(prefix + " "):
                name = name[len(prefix) + 1:].strip()
                changed = True
                break
    return name


def _normalize_name(name: str) -> str:
    """Normalize entity name: lowercase, ASCII, strip titles and whitespace."""
    name = unidecode(name).lower().strip()
    name = _strip_titles(name)
    return name


def _singularize(name: str) -> str:
    """Singularize each word in name."""
    words = name.split()
    result = []
    for word in words:
        singular = _INFLECT_ENGINE.singular_noun(word)
        # singular_noun returns False if the word is already singular
        result.append(singular if singular else word)
    return " ".join(result)


def prededup_entities(
    extractions: list[DocumentExtraction],
    similarity_threshold: float = 0.95,
) -> dict[tuple[str, str], str]:
    """Map (entity_type, original_name) -> canonical_name for near-duplicates.

    Args:
        extractions: List of document extractions to scan
        similarity_threshold: SemHash threshold for fuzzy matching (0-1)

    Returns:
        Mapping from (entity_type, original_name) to canonical_name.
        Only contains entries where original_name != canonical_name.
    """
    # Collect all entity names grouped by type
    names_by_type: dict[str, list[str]] = {}
    for extraction in extractions:
        if extraction.error:
            continue
        for entity in extraction.entities:
            names_by_type.setdefault(entity.entity_type, []).append(entity.name)

    canonical_map: dict[tuple[str, str], str] = {}
    total_merged = 0

    for entity_type, names in names_by_type.items():
        if len(names) < 2:
            continue

        # Phase 1: Deterministic grouping by normalized+singularized form
        norm_groups: dict[str, list[str]] = {}
        for name in names:
            norm = _singularize(_normalize_name(name))
            norm_groups.setdefault(norm, []).append(name)

        # Pick canonical per deterministic group
        unique_canonicals: dict[str, str] = {}  # normalized -> canonical
        for norm, variants in norm_groups.items():
            canonical = _pick_canonical(variants)
            unique_canonicals[norm] = canonical
            for name in variants:
                if name != canonical:
                    canonical_map[(entity_type, name)] = canonical
                    total_merged += 1

        # Phase 2: SemHash fuzzy matching on the remaining unique normalized forms
        if len(unique_canonicals) >= 2:
            try:
                fuzzy_merges = _semhash_cluster(
                    list(unique_canonicals.keys()),
                    unique_canonicals,
                    similarity_threshold,
                )
                for norm_variant, norm_canonical in fuzzy_merges.items():
                    canonical_name = unique_canonicals[norm_canonical]
                    # Remap all original names that pointed to variant
                    for name in norm_groups[norm_variant]:
                        canonical_map[(entity_type, name)] = canonical_name
                        total_merged += 1
                    # Also remap the variant's own canonical if different
                    variant_canonical = unique_canonicals[norm_variant]
                    if variant_canonical != canonical_name:
                        canonical_map[(entity_type, variant_canonical)] = canonical_name
            except Exception as e:
                logger.warning(f"SemHash clustering failed for {entity_type}: {e}")

    if total_merged:
        unique_before = sum(len(set(names)) for names in names_by_type.values())
        unique_after = unique_before - total_merged
        logger.info(
            f"Pre-dedup: {unique_before} entities -> {unique_after} unique ({total_merged} merged)"
        )
    else:
        logger.debug("Pre-dedup: no duplicates found")

    return canonical_map


def _semhash_cluster(
    normalized_names: list[str],
    norm_to_canonical: dict[str, str],
    threshold: float,
) -> dict[str, str]:
    """Use SemHash to find fuzzy near-duplicates among normalized names.

    Returns mapping from variant normalized form -> canonical normalized form.
    """
    records = [{"text": name} for name in normalized_names]
    sh = SemHash.from_records(records=records, columns=["text"])
    result = sh.self_deduplicate(threshold=threshold)

    merges: dict[str, str] = {}
    for item in result.selected_with_duplicates:
        kept_name = item.record["text"]
        for dup_record, _score in item.duplicates:
            dup_name = dup_record["text"]
            if dup_name != kept_name:
                merges[dup_name] = kept_name

    return merges


def _pick_canonical(names: list[str]) -> str:
    """Pick the best canonical name from a group of variants.

    Priority: most frequent -> longest -> alphabetically first.
    """
    if len(names) == 1:
        return names[0]

    counts = Counter(names)
    max_count = max(counts.values())
    most_frequent = [n for n, c in counts.items() if c == max_count]

    if len(most_frequent) == 1:
        return most_frequent[0]

    # Tiebreak: longest name (likely more complete)
    max_len = max(len(n) for n in most_frequent)
    longest = [n for n in most_frequent if len(n) == max_len]

    return sorted(longest)[0]
