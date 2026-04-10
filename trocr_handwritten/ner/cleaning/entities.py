"""Owner and plantation entity resolution for civil registry acts.

Collects all unique raw names, normalises them, clusters via fuzzy matching
(rapidfuzz), and assigns canonical name + integer ID to each cluster.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Normalisation for matching (not for display)
# ---------------------------------------------------------------------------

_HONORIFICS = re.compile(
    r"\b(sieur|sr|m|mr|monsieur|dame|veuve|vve|héritiers|heritiers|"
    r"succession|épouse|epouse)\b",
    re.IGNORECASE,
)
_SUFFIXES = re.compile(
    r"\b(père|pere|fils|junior|sr|jr|aîné|aine|cadet)\b",
    re.IGNORECASE,
)
_PARTICLES = re.compile(r"\b(de|du|de la|des|le|la|les|d)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")


def _normalise_for_matching(name: str) -> str:
    """Normalise a name for fuzzy matching (not stored, only used for clustering)."""
    norm = name.lower().strip()
    # Remove accents
    norm = "".join(
        c for c in unicodedata.normalize("NFD", norm) if unicodedata.category(c) != "Mn"
    )
    norm = _HONORIFICS.sub(" ", norm)
    norm = _SUFFIXES.sub(" ", norm)
    norm = _PARTICLES.sub(" ", norm)
    norm = _PUNCT.sub(" ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


# ---------------------------------------------------------------------------
# Cluster builder
# ---------------------------------------------------------------------------


def _pick_canonical(names: list[str]) -> str:
    """Pick the canonical form: longest non-null, most frequent name."""
    from collections import Counter

    counts = Counter(names)
    # Prefer most frequent; break ties by length (longer = more complete)
    return max(counts, key=lambda n: (counts[n], len(n)))


def build_entity_table(
    raw_names: list[Optional[str]],
    threshold: int = 85,
) -> dict[str, tuple[str, int]]:
    """Cluster raw names by fuzzy similarity and assign canonical name + ID.

    Args:
        raw_names: All raw name values (including None/null).
        threshold: Minimum similarity score (0-100) to merge two names.

    Returns:
        Dict mapping each raw name -> (canonical_name, entity_id).
        None/empty values are excluded.
    """
    # Deduplicate, drop nulls
    unique = sorted(
        {
            n
            for n in raw_names
            if n and str(n).strip().lower() not in ("null", "none", "")
        }
    )

    if not unique:
        return {}

    normed = {name: _normalise_for_matching(name) for name in unique}

    # Greedy clustering: assign each name to first cluster whose centroid matches
    clusters: list[list[str]] = []
    cluster_norms: list[str] = []

    for name in unique:
        norm = normed[name]
        if not norm:
            clusters.append([name])
            cluster_norms.append(norm)
            continue

        # Find best matching existing cluster
        best_score = 0
        best_idx = -1
        for idx, cnorm in enumerate(cluster_norms):
            score = fuzz.token_sort_ratio(norm, cnorm)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score >= threshold:
            clusters[best_idx].append(name)
            # Update centroid to canonical of new cluster
            cluster_norms[best_idx] = _normalise_for_matching(
                _pick_canonical(clusters[best_idx])
            )
        else:
            clusters.append([name])
            cluster_norms.append(norm)

    # Build lookup: raw_name -> (canonical, id)
    lookup: dict[str, tuple[str, int]] = {}
    for entity_id, cluster in enumerate(clusters, start=1):
        canonical = _pick_canonical(cluster)
        for name in cluster:
            lookup[name] = (canonical, entity_id)

    return lookup


# ---------------------------------------------------------------------------
# Build both tables from a list of NERResult dicts
# ---------------------------------------------------------------------------


def load_clusters_lookup(
    clusters_json_path: Path, entity_type: str = "owner"
) -> dict[str, tuple[str, int]]:
    """Build a raw_name → (canonical, entity_id) lookup from a Claude-generated clusters JSON.

    The JSON is {"owner_clusters": [...], "plantation_clusters": [...]} where each cluster is
    {"canonical": str|null, "variants": [...]}.
    entity_type must be "owner" or "plantation".
    Singletons not listed in the file keep their raw name as canonical (caller handles this).
    """
    import json

    with open(clusters_json_path, encoding="utf-8") as f:
        data = json.load(f)

    key = "owner_clusters" if entity_type == "owner" else "plantation_clusters"
    clusters = data.get(key, [])

    lookup: dict[str, tuple[str, int]] = {}
    for entity_id, cluster in enumerate(clusters, start=1):
        canonical = cluster.get("canonical")
        for raw in cluster.get("variants", []):
            if raw and str(raw).strip():
                lookup[raw] = (canonical, entity_id)

    return lookup


def build_entity_tables(
    records: list[dict],
    owner_threshold: int = 85,
    plantation_threshold: int = 88,
) -> tuple[dict, dict]:
    """Build owner and plantation lookup tables from all NER results.

    Returns:
        (owner_lookup, plantation_lookup) — each maps raw_name -> (canonical, id).
    """
    owner_names: list[Optional[str]] = []
    plantation_names: list[Optional[str]] = []

    for r in records:
        for entity_key in ("death_act", "birth_act", "marriage_act"):
            entity = r.get(entity_key)
            if not entity:
                continue
            owner_names.append(entity.get("owner_name"))
            plantation_names.append(entity.get("habitation_name"))

    owner_lookup = build_entity_table(owner_names, threshold=owner_threshold)
    plantation_lookup = build_entity_table(
        plantation_names, threshold=plantation_threshold
    )

    return owner_lookup, plantation_lookup
