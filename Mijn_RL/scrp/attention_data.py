"""Deterministic data protocol and leakage checks for attention multiscale OCRP."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .online_crp import OnlineCRPEpisode, load_episode_manifest


PROTOCOL_ID = "online_crp_attention_multiscale_v1"
DATA_VERSION = "attention_multiscale_v1"
MASTER_SEEDS = {
    "dev_smoke": 20261001,
    "validation_seen": 20261002,
    "validation_unseen_geometry": 20261003,
    "test_seen": 20261004,
    "test_unseen_geometry": 20261005,
}
TABLE2_GROUPS = (
    (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
    (6, 6), (6, 10), (10, 6), (10, 10),
)
LARGE_SHAPES = ((20, 8), (24, 8), (20, 10), (25, 8))
UNSEEN_VALIDATION = ((12, 8, 60, 72), (16, 8, 90, 102))
UNSEEN_TEST = ((18, 10, 120, 140), (30, 8, 150, 180))
EXPECTED_SPLIT_COUNTS = {
    "dev_smoke": 50,
    "validation_seen": 620,
    "validation_unseen_geometry": 100,
    "test_seen": 1240,
    "test_unseen_geometry": 200,
}
MAX_CANONICAL_RETRIES = 1000


@dataclass(frozen=True)
class GeometryDomain:
    domain_id: str
    kind: str
    num_stacks: int
    max_tiers: int
    initial_height: int | None = None

    @property
    def parameter_group(self) -> str:
        if self.kind == "table2":
            return f"h{self.initial_height}-s{self.num_stacks}"
        return f"s{self.num_stacks}-t{self.max_tiers}"


def training_domains() -> tuple[GeometryDomain, ...]:
    table2 = tuple(
        GeometryDomain(
            domain_id=f"table2_h{height:02d}_s{stacks:02d}",
            kind="table2",
            num_stacks=stacks,
            max_tiers=height + 2,
            initial_height=height,
        )
        for height, stacks in TABLE2_GROUPS
    )
    large = tuple(
        GeometryDomain(
            domain_id=f"large_s{stacks:02d}_t{tiers:02d}",
            kind="large",
            num_stacks=stacks,
            max_tiers=tiers,
        )
        for stacks, tiers in LARGE_SHAPES
    )
    return table2 + large


def derive_seed(master_seed: int, *components: object) -> int:
    payload = json.dumps(
        [PROTOCOL_ID, int(master_seed), *components],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_invariant_fingerprint(episode: OnlineCRPEpisode) -> str:
    """Hash rank-layout identity, invariant to container labels and stack IDs."""

    rank_by_label = {
        label: rank for rank, label in enumerate(episode.retrieval_sequence, start=1)
    }
    stacks_by_rank = sorted(
        tuple(rank_by_label[label] for label in stack)
        for stack in episode.initial_stacks
    )
    payload = json.dumps(
        {
            "num_stacks": episode.num_stacks,
            "max_tiers": episode.max_tiers,
            "rank_stacks": stacks_by_rank,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def episode_to_record(episode: OnlineCRPEpisode) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "num_stacks": episode.num_stacks,
        "max_tiers": episode.max_tiers,
        "initial_stacks": [list(stack) for stack in episode.initial_stacks],
        "retrieval_sequence": list(episode.retrieval_sequence),
        "parameter_group": episode.parameter_group,
        "metadata": dict(episode.metadata),
        "canonical_id": episode.canonical_id,
        "label_invariant_fingerprint": label_invariant_fingerprint(episode),
    }


def _episode_from_rank_layout(
    *,
    episode_id: str,
    num_stacks: int,
    max_tiers: int,
    num_containers: int,
    heights: Sequence[int],
    parameter_group: str,
    metadata: Mapping[str, object],
    rng: np.random.Generator,
) -> OnlineCRPEpisode:
    if len(heights) != num_stacks or sum(heights) != num_containers:
        raise ValueError("stack heights do not match the requested instance")
    if any(height < 0 or height > max_tiers for height in heights):
        raise ValueError("a generated stack height is outside the capacity")
    labels = rng.permutation(np.arange(1, num_containers + 1, dtype=np.int64))
    stacks: list[tuple[int, ...]] = []
    cursor = 0
    for height in heights:
        stacks.append(tuple(int(value) for value in labels[cursor : cursor + height]))
        cursor += height
    return OnlineCRPEpisode(
        episode_id=episode_id,
        num_stacks=num_stacks,
        max_tiers=max_tiers,
        initial_stacks=tuple(stacks),
        retrieval_sequence=tuple(range(1, num_containers + 1)),
        parameter_group=parameter_group,
        metadata=dict(metadata),
    )


def _table2_episode(
    domain: GeometryDomain,
    *,
    split: str,
    index: int,
    master_seed: int,
    retry: int = 0,
) -> OnlineCRPEpisode:
    if domain.kind != "table2" or domain.initial_height is None:
        raise ValueError("table2 sampler requires a structural geometry domain")
    seed = derive_seed(master_seed, split, domain.domain_id, index, retry)
    rng = np.random.Generator(np.random.PCG64(seed))
    heights = (domain.initial_height,) * domain.num_stacks
    count = domain.initial_height * domain.num_stacks
    return _episode_from_rank_layout(
        episode_id=f"{split}-{domain.domain_id}-{index:04d}",
        num_stacks=domain.num_stacks,
        max_tiers=domain.max_tiers,
        num_containers=count,
        heights=heights,
        parameter_group=domain.parameter_group,
        metadata={
            "protocol": PROTOCOL_ID,
            "split": split,
            "domain_id": domain.domain_id,
            "domain_kind": domain.kind,
            "master_seed": master_seed,
            "derived_seed": seed,
            "sample_index": index,
            "retry": retry,
            "initial_height": domain.initial_height,
            "exact_original_benchmark": False,
        },
        rng=rng,
    )


def _valid_container_range(num_stacks: int, max_tiers: int) -> tuple[int, int]:
    capacity = num_stacks * max_tiers
    low = max(100, math.ceil(0.60 * capacity))
    high = min(150, math.floor(0.80 * capacity))
    if low > high:
        raise ValueError("large domain has no N in the required count/fill intersection")
    return low, high


def _random_heights(
    rng: np.random.Generator, num_stacks: int, max_tiers: int, count: int
) -> tuple[int, ...]:
    if count < 0 or count > num_stacks * max_tiers:
        raise ValueError("container count exceeds geometry capacity")
    heights = [0] * num_stacks
    for _ in range(count):
        available = [index for index, height in enumerate(heights) if height < max_tiers]
        heights[available[int(rng.integers(0, len(available)))]] += 1
    return tuple(heights)


def _variable_episode(
    *,
    episode_id: str,
    domain_id: str,
    domain_kind: str,
    num_stacks: int,
    max_tiers: int,
    count_range: tuple[int, int],
    parameter_group: str,
    split: str,
    master_seed: int,
    index: int,
    retry: int = 0,
) -> OnlineCRPEpisode:
    seed = derive_seed(master_seed, split, domain_id, index, retry)
    rng = np.random.Generator(np.random.PCG64(seed))
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    heights = _random_heights(rng, num_stacks, max_tiers, count)
    return _episode_from_rank_layout(
        episode_id=episode_id,
        num_stacks=num_stacks,
        max_tiers=max_tiers,
        num_containers=count,
        heights=heights,
        parameter_group=parameter_group,
        metadata={
            "protocol": PROTOCOL_ID,
            "split": split,
            "domain_id": domain_id,
            "domain_kind": domain_kind,
            "master_seed": master_seed,
            "derived_seed": seed,
            "sample_index": index,
            "retry": retry,
            "exact_original_benchmark": False,
        },
        rng=rng,
    )


def _large_episode(
    stacks: int,
    tiers: int,
    *,
    split: str,
    index: int,
    master_seed: int,
    retry: int = 0,
) -> OnlineCRPEpisode:
    domain_id = f"large_s{stacks:02d}_t{tiers:02d}"
    return _variable_episode(
        episode_id=f"{split}-{domain_id}-{index:04d}",
        domain_id=domain_id,
        domain_kind="large",
        num_stacks=stacks,
        max_tiers=tiers,
        count_range=_valid_container_range(stacks, tiers),
        parameter_group=f"s{stacks}-t{tiers}",
        split=split,
        master_seed=master_seed,
        index=index,
        retry=retry,
    )


def _unseen_episode(
    geometry: tuple[int, int, int, int],
    *,
    split: str,
    index: int,
    master_seed: int,
    retry: int = 0,
) -> OnlineCRPEpisode:
    stacks, tiers, count_low, count_high = geometry
    domain_id = f"unseen_s{stacks:02d}_t{tiers:02d}_n{count_low}_{count_high}"
    return _variable_episode(
        episode_id=f"{split}-{domain_id}-{index:04d}",
        domain_id=domain_id,
        domain_kind="unseen_geometry",
        num_stacks=stacks,
        max_tiers=tiers,
        count_range=(count_low, count_high),
        parameter_group=domain_id,
        split=split,
        master_seed=master_seed,
        index=index,
        retry=retry,
    )


def _load_source_manifest(path: Path, category: str) -> tuple[list[OnlineCRPEpisode], dict[str, object]]:
    episodes = list(load_episode_manifest(path))
    fingerprints = {label_invariant_fingerprint(episode) for episode in episodes}
    return episodes, {
        "category": category,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "count": len(episodes),
        "unique_label_invariant_fingerprints": len(fingerprints),
    }


def _source_paths(repo_root: Path, workspace_root: Path) -> tuple[tuple[Path, str], ...]:
    return (
        (repo_root / "data/online_crp/lora_domains/large_scale_n100_150/adapt_train.json", "allowed_large_training_pool"),
        (repo_root / "data/online_crp/lora_domains/large_scale_n100_150/validation.json", "historical_validation_exclusion"),
        (repo_root / "data/online_crp/lora_domains/large_scale_n100_150/test_sealed.json", "historical_sealed_test_exclusion"),
        (workspace_root / "Leveling/data/caserta_table2_840.json", "original_caserta_exclusion_and_external_baseline"),
        (workspace_root / "GP/data/caserta_train_840_seed20260920.json", "gp_training_corpus_exclusion"),
    )


def _source_inventory(
    repo_root: Path, workspace_root: Path
) -> tuple[dict[str, object], set[str], list[OnlineCRPEpisode]]:
    inventories = []
    excluded: set[str] = set()
    large_training: list[OnlineCRPEpisode] = []
    for path, category in _source_paths(repo_root, workspace_root):
        if not path.is_file():
            raise FileNotFoundError(f"required isolation source is missing: {path}")
        episodes, info = _load_source_manifest(path, category)
        inventories.append(info)
        if category == "allowed_large_training_pool":
            large_training = episodes
        excluded.update(label_invariant_fingerprint(episode) for episode in episodes)
    if not large_training or {
        (episode.num_stacks, episode.max_tiers) for episode in large_training
    } != set(LARGE_SHAPES):
        raise ValueError("large-scale adapt_train pool does not cover the four planned shapes")
    return {"manifests": inventories}, excluded, large_training


def _make_split(split: str) -> list[OnlineCRPEpisode]:
    seed = MASTER_SEEDS[split]
    episodes: list[OnlineCRPEpisode] = []
    if split == "dev_smoke":
        per_domain = 2
    elif split == "validation_seen":
        per_table2, per_large = 20, 50
    elif split == "test_seen":
        per_table2, per_large = 40, 100
    elif split in ("validation_unseen_geometry", "test_unseen_geometry"):
        geometries = UNSEEN_VALIDATION if split == "validation_unseen_geometry" else UNSEEN_TEST
        count_per_domain = 50 if split == "validation_unseen_geometry" else 100
        for geometry in geometries:
            for index in range(count_per_domain):
                episodes.append(_unseen_episode(geometry, split=split, index=index, master_seed=seed))
        return episodes
    else:
        raise ValueError(f"unknown fixed split {split!r}")

    for domain in training_domains():
        count = per_domain if split == "dev_smoke" else (
            per_table2 if domain.kind == "table2" else per_large
        )
        for index in range(count):
            if domain.kind == "table2":
                episodes.append(_table2_episode(domain, split=split, index=index, master_seed=seed))
            else:
                episodes.append(_large_episode(domain.num_stacks, domain.max_tiers, split=split, index=index, master_seed=seed))
    return episodes


def _regenerate_one(
    episode: OnlineCRPEpisode, *, split: str, retry: int
) -> OnlineCRPEpisode:
    domain_id = str(episode.metadata["domain_id"])
    index = int(episode.metadata["sample_index"])
    seed = MASTER_SEEDS[split]
    domains = {domain.domain_id: domain for domain in training_domains()}
    if domain_id in domains:
        domain = domains[domain_id]
        if domain.kind == "table2":
            return _table2_episode(domain, split=split, index=index, master_seed=seed, retry=retry)
        return _large_episode(domain.num_stacks, domain.max_tiers, split=split, index=index, master_seed=seed, retry=retry)
    for geometry in (*UNSEEN_VALIDATION, *UNSEEN_TEST):
        candidate_domain_id = f"unseen_s{geometry[0]:02d}_t{geometry[1]:02d}_n{geometry[2]}_{geometry[3]}"
        if candidate_domain_id == domain_id:
            return _unseen_episode(geometry, split=split, index=index, master_seed=seed, retry=retry)
    raise ValueError(f"cannot regenerate unknown geometry domain {domain_id}")


def _regenerate_without_overlap(
    split: str, episodes: Sequence[OnlineCRPEpisode], excluded: set[str]
) -> list[OnlineCRPEpisode]:
    clean: list[OnlineCRPEpisode] = []
    seen = set(excluded)
    for original in episodes:
        episode = original
        fingerprint = label_invariant_fingerprint(episode)
        if fingerprint in seen:
            for retry in range(1, MAX_CANONICAL_RETRIES + 1):
                candidate = _regenerate_one(episode, split=split, retry=retry)
                fingerprint = label_invariant_fingerprint(candidate)
                if fingerprint not in seen:
                    episode = candidate
                    break
            else:
                raise RuntimeError(f"could not isolate sample {episode.episode_id}")
        clean.append(episode)
        seen.add(label_invariant_fingerprint(episode))
    return clean


def generate_fixed_splits(
    *, repo_root: str | Path | None = None, workspace_root: str | Path | None = None
) -> Path:
    """Generate all fixed splits once and write a source/data isolation audit."""

    this_file = Path(__file__).resolve()
    resolved_repo = Path(repo_root).resolve() if repo_root else this_file.parents[1]
    resolved_workspace = Path(workspace_root).resolve() if workspace_root else resolved_repo.parent
    output = resolved_repo / "data/online_crp/attention_multiscale_v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)

    source_inventory, source_exclusions, _ = _source_inventory(resolved_repo, resolved_workspace)
    seen_all = set(source_exclusions)
    split_audits: dict[str, object] = {}
    for split in MASTER_SEEDS:
        episodes = _regenerate_without_overlap(split, _make_split(split), seen_all)
        expected = EXPECTED_SPLIT_COUNTS[split]
        if len(episodes) != expected:
            raise AssertionError(f"{split} produced {len(episodes)} rows, expected {expected}")
        canonical = [episode.canonical_id for episode in episodes]
        fingerprints = [label_invariant_fingerprint(episode) for episode in episodes]
        if len(set(episode.episode_id for episode in episodes)) != expected:
            raise ValueError(f"{split} has duplicate episode IDs")
        if len(set(canonical)) != expected or len(set(fingerprints)) != expected:
            raise ValueError(f"{split} has duplicate canonical instances")
        rows = [episode_to_record(episode) for episode in episodes]
        manifest_path = output / f"{split}.json"
        manifest_bytes = (json.dumps(rows, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        seen_all.update(fingerprints)
        group_counts = Counter(str(episode.metadata["domain_id"]) for episode in episodes)
        split_audits[split] = {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "count": len(episodes),
            "unique_episode_ids": len(set(episode.episode_id for episode in episodes)),
            "unique_canonical_ids": len(set(canonical)),
            "unique_label_invariant_fingerprints": len(set(fingerprints)),
            "domain_counts": dict(sorted(group_counts.items())),
            "n_range": [min(episode.num_containers for episode in episodes), max(episode.num_containers for episode in episodes)],
            "source_and_prior_split_overlaps": 0,
        }

    audit = {
        "protocol": PROTOCOL_ID,
        "data_version": DATA_VERSION,
        "generated_at_local_date": "2026-10-01",
        "master_seeds": MASTER_SEEDS,
        "fingerprint_definition": "SHA256 of S,T and lexicographically sorted bottom-to-top stacks encoded as true retrieval ranks",
        "source_inventory": source_inventory,
        "split_manifests": split_audits,
        "source_fingerprint_union_size": len(source_exclusions),
        "all_split_fingerprint_union_size": len(seen_all) - len(source_exclusions),
        "overlap_check": "zero intersections against every listed source manifest and all prior generated splits",
        "final_policy_evaluation_performed": False,
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def validate_fixed_splits(
    data_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, object]:
    """Independently recompute fixed-manifest hashes, counts, and isolation."""

    data_path = Path(data_root).resolve()
    audit = json.loads((data_path / "audit.json").read_text(encoding="utf-8"))
    this_file = Path(__file__).resolve()
    resolved_repo = Path(repo_root).resolve() if repo_root else this_file.parents[1]
    resolved_workspace = Path(workspace_root).resolve() if workspace_root else resolved_repo.parent
    source_inventory, source_fingerprints, _ = _source_inventory(resolved_repo, resolved_workspace)
    if audit.get("protocol") != PROTOCOL_ID or audit.get("data_version") != DATA_VERSION:
        raise ValueError("data audit protocol/version mismatch")
    if audit.get("source_inventory") != source_inventory:
        raise ValueError("source manifest hashes changed since generation")

    seen = set(source_fingerprints)
    recomputed: dict[str, object] = {}
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        manifest_path = data_path / f"{split}.json"
        episodes = load_episode_manifest(manifest_path)
        raw_records = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_hash = file_sha256(manifest_path)
        split_record = audit["split_manifests"][split]
        if file_hash != split_record["sha256"]:
            raise ValueError(f"{split} manifest hash does not match audit")
        if len(episodes) != expected or split_record["count"] != expected:
            raise ValueError(f"{split} count does not match protocol")
        canonical_ids: set[str] = set()
        fingerprints: set[str] = set()
        domain_counts: Counter[str] = Counter()
        n_values: list[int] = []
        for episode, raw_record in zip(episodes, raw_records, strict=True):
            if episode.metadata.get("protocol") != PROTOCOL_ID or episode.metadata.get("split") != split:
                raise ValueError(f"{split} episode has mismatched protocol metadata")
            canonical_ids.add(episode.canonical_id)
            fingerprint = label_invariant_fingerprint(episode)
            fingerprints.add(fingerprint)
            domain_counts[str(episode.metadata["domain_id"])] += 1
            n_values.append(episode.num_containers)
            if raw_record.get("canonical_id") != episode.canonical_id:
                raise ValueError(f"{split} stored canonical ID is inconsistent")
            if raw_record.get("label_invariant_fingerprint") != fingerprint:
                raise ValueError(f"{split} stored isolation fingerprint is inconsistent")
            if fingerprint in seen:
                raise ValueError(f"{split} overlaps a source manifest or prior split")
            seen.add(fingerprint)
            kind = episode.metadata.get("domain_kind")
            if kind == "table2":
                height = int(episode.metadata["initial_height"])
                if episode.max_tiers != height + 2 or episode.num_containers != height * episode.num_stacks:
                    raise ValueError("Table-2-style instance violates the planned geometry")
                if any(len(stack) != height for stack in episode.initial_stacks):
                    raise ValueError("Table-2-style instance does not have fixed stack heights")
            elif kind == "large":
                if (episode.num_stacks, episode.max_tiers) not in LARGE_SHAPES:
                    raise ValueError("large instance has an unplanned shape")
                if not 100 <= episode.num_containers <= 150 or not 0.60 <= episode.fill_rate <= 0.80:
                    raise ValueError("large instance violates the planned N/fill range")
            elif kind == "unseen_geometry":
                ranges = UNSEEN_VALIDATION if split == "validation_unseen_geometry" else UNSEEN_TEST
                matching = [
                    geometry for geometry in ranges
                    if (episode.num_stacks, episode.max_tiers) == geometry[:2]
                ]
                if len(matching) != 1 or not matching[0][2] <= episode.num_containers <= matching[0][3]:
                    raise ValueError("unseen instance violates its planned geometry/count range")
            else:
                raise ValueError("episode is missing a recognized domain kind")
        if len(canonical_ids) != expected or len(fingerprints) != expected:
            raise ValueError(f"{split} contains duplicate canonical instances")
        actual_domains = dict(sorted(domain_counts.items()))
        if split == "dev_smoke":
            expected_domains = {domain.domain_id: 2 for domain in training_domains()}
        elif split == "validation_seen":
            expected_domains = {
                domain.domain_id: (20 if domain.kind == "table2" else 50)
                for domain in training_domains()
            }
        elif split == "test_seen":
            expected_domains = {
                domain.domain_id: (40 if domain.kind == "table2" else 100)
                for domain in training_domains()
            }
        else:
            geometries = UNSEEN_VALIDATION if split == "validation_unseen_geometry" else UNSEEN_TEST
            per_domain = 50 if split == "validation_unseen_geometry" else 100
            expected_domains = {
                f"unseen_s{s:02d}_t{t:02d}_n{low}_{high}": per_domain
                for s, t, low, high in geometries
            }
        if actual_domains != dict(sorted(expected_domains.items())):
            raise ValueError(f"{split} per-domain counts do not match the protocol")
        if actual_domains != split_record["domain_counts"]:
            raise ValueError(f"{split} domain counts do not match audit")
        recomputed[split] = {
            "count": len(episodes),
            "manifest_sha256": file_hash,
            "unique_canonical": len(canonical_ids),
            "unique_label_invariant": len(fingerprints),
            "domains": len(domain_counts),
            "n_range": [min(n_values), max(n_values)],
        }
    return {
        "status": "VALID",
        "protocol": PROTOCOL_ID,
        "source_manifest_count": len(source_inventory["manifests"]),
        "source_fingerprint_count": len(source_fingerprints),
        "fixed_split_results": recomputed,
        "overlaps": 0,
    }


def load_training_pool(path: str | Path) -> dict[tuple[int, int], tuple[OnlineCRPEpisode, ...]]:
    episodes = load_episode_manifest(path)
    grouped: dict[tuple[int, int], list[OnlineCRPEpisode]] = defaultdict(list)
    for episode in episodes:
        shape = (episode.num_stacks, episode.max_tiers)
        if shape not in LARGE_SHAPES or not 100 <= episode.num_containers <= 150:
            raise ValueError("large-scale training pool contains an out-of-protocol instance")
        grouped[shape].append(episode)
    if set(grouped) != set(LARGE_SHAPES):
        raise ValueError("large-scale training pool does not cover every planned shape")
    return {shape: tuple(rows) for shape, rows in grouped.items()}


def make_training_domain_batch(
    domain: GeometryDomain,
    *,
    count: int,
    master_seed: int,
    epoch: int,
    slot: int,
    role: str,
    large_pool: Mapping[tuple[int, int], Sequence[OnlineCRPEpisode]],
    excluded_fingerprints: set[str] | None = None,
) -> tuple[OnlineCRPEpisode, ...]:
    """Build a deterministic paired-method batch for one domain and outer slot."""

    if count <= 0 or role not in ("source", "query", "joint"):
        raise ValueError("invalid training batch request")
    seed = derive_seed(master_seed, "train", epoch, slot, domain.domain_id, role)
    if domain.kind == "large":
        shape = (domain.num_stacks, domain.max_tiers)
        pool = tuple(large_pool.get(shape, ()))
        if len(pool) < count:
            raise ValueError(f"training pool for {shape} has fewer than {count} instances")
        rng = np.random.Generator(np.random.PCG64(seed))
        indices = rng.choice(len(pool), size=count, replace=False)
        return tuple(pool[int(index)] for index in indices)

    generated: list[OnlineCRPEpisode] = []
    seen = set(excluded_fingerprints or ())
    for index in range(count):
        accepted = False
        for retry in range(MAX_CANONICAL_RETRIES):
            episode = _table2_episode(
                domain,
                split=f"train_e{epoch:03d}_s{slot:05d}_{role}",
                index=index,
                master_seed=seed,
                retry=retry,
            )
            fingerprint = label_invariant_fingerprint(episode)
            if fingerprint not in seen:
                generated.append(episode)
                seen.add(fingerprint)
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"could not isolate fresh training sample for {domain.domain_id}")
    return tuple(generated)


__all__ = [
    "DATA_VERSION",
    "EXPECTED_SPLIT_COUNTS",
    "GeometryDomain",
    "LARGE_SHAPES",
    "MASTER_SEEDS",
    "PROTOCOL_ID",
    "TABLE2_GROUPS",
    "derive_seed",
    "episode_to_record",
    "file_sha256",
    "generate_fixed_splits",
    "label_invariant_fingerprint",
    "load_training_pool",
    "make_training_domain_batch",
    "training_domains",
    "validate_fixed_splits",
]
