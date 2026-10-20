"""PLAN-047 M1 bank contracts and implementation helpers.

This module contains the registered sampling, native actor, label, weighting,
bootstrap, release, and scientific-wall contracts. Importing it does not start
a scientific job. The bank runner refuses to execute without a separate,
hash-matching root release record.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import uuid

import numpy as np

from R.scrp.h20_r_data import HORIZONS, Q_PRIOR
from R.scrp.universal_r_utility import (
    ALGORITHMS,
    CandidateBranchLabels,
    CandidateOrigin,
    NativeCandidatePool,
    candidate_label_cache_key,
    build_native_candidate_pools,
    run_candidate_branches,
    select_strictly_below_h20,
    semantic_universal_time_window_seed,
)


PROTOCOL_ID = "universal_r_utility_v1"
RUN_ID = "r02"
M1_SMOKE_SCOPE = "M1_REAL_SMOKE_ONLY"
M1_BANK_SCOPE = "M1_DIAGNOSTIC_BANK_AND_LABELS"
M1_TRAIN_GEOMETRIES = ("G1", "G2")
M1_SOURCE_PATHS = (
    "R/scrp/universal_r_utility_m1.py",
    "R/experiments/universal_r_utility_m1.py",
    "R/tests/test_universal_r_utility_m1.py",
    "R/docs/planning/PLAN_047_M1_IMPLEMENTATION_CONTRACT.md",
    "R/tests/test_plan047_post_stop_repair.py",
    "R/output/universal_r_utility_v1/r02/RECOVERY_AUDIT/RECOVERY_SCOPE_NOTE.md",
    "R/output/universal_r_utility_v1/r02/RECOVERY_AUDIT/run_m1_child_captured.ps1",
    "R/output/universal_r_utility_v1/r02/RECOVERY_AUDIT/build_release_candidates.py",
)
M1_STAGE_TARGETS = (("first", None), ("mid", 0.50), ("late", 0.85))
M1_CONDITIONS = tuple((window, repeat) for window in (1, 2, 5, 10, 20) for repeat in (0, 1))
M1_SCIENCE_CAP_SECONDS = 8 * 60 * 60
G1_GEOMETRY = (6, 5, 24)
G2_GEOMETRY = (8, 6, 40)
_SHA256_HEX = frozenset("0123456789abcdef")


class M1ContractError(RuntimeError):
    """The M1 input, release, or completeness contract failed closed."""


class M1BudgetExceeded(M1ContractError):
    """The cumulative PLAN-047 M1 scientific wall cap was reached."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_r02_artifact_path(
    workspace: Path, value: str | Path, *, name: str
) -> Path:
    """Resolve an explicit workspace-relative artifact path inside r02."""
    if not isinstance(value, (str, Path)):
        raise M1ContractError(f"{name} must be a safe workspace-relative r02 path")
    relative = Path(str(value))
    if (
        not str(value)
        or relative.is_absolute()
        or relative.drive
        or ".." in relative.parts
    ):
        raise M1ContractError(f"{name} must be a safe workspace-relative r02 path")
    root = Path(workspace).resolve()
    r02_out = (root / "R/output/universal_r_utility_v1/r02").resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(r02_out):
        raise M1ContractError(f"{name} escapes the independent r02 namespace")
    return resolved


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    return result


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise M1ContractError(f"{name} must be lowercase SHA-256 hex")
    return text


def verify_m1_release(
    workspace: Path,
    release_path: Path | None = None,
    *,
    expected_scope: str = M1_BANK_SCOPE,
) -> dict[str, Any]:
    """Verify the independent M0 review, frozen M0 inputs, and M1 code hashes.

    The required release file is intentionally absent until root releases this
    phase. Missing files, stale hashes, incomplete M0 receipts, or scope drift
    raise ``M1ContractError`` before the runner can open any label output.
    """
    root = Path(workspace).resolve()
    m0_out = root / "R/output/universal_r_utility_v1/r01"
    r02_out = root / "R/output/universal_r_utility_v1/r02"
    default_release_name = (
        "M1_SMOKE_RELEASE.json" if expected_scope == M1_SMOKE_SCOPE else "M1_RELEASE.json"
    )
    release_file = Path(release_path).resolve() if release_path else (r02_out / default_release_name)
    if not release_file.is_file():
        raise M1ContractError(f"M1 root release record is missing: {release_file}")

    def read_json(relative: str) -> dict[str, Any]:
        path = root / relative
        if not path.is_file():
            raise M1ContractError(f"required M0 release input is missing: {relative}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise M1ContractError(f"required JSON object is invalid: {relative}")
        return value

    release = json.loads(release_file.read_text(encoding="utf-8"))
    if not isinstance(release, dict):
        raise M1ContractError("M1 release record must be a JSON object")
    if (
        release.get("protocol_id") != PROTOCOL_ID
        or release.get("run_id") != RUN_ID
        or release.get("scope") != expected_scope
        or release.get("status") != "ROOT_RELEASED"
    ):
        raise M1ContractError("M1 release identity, scope, or status is invalid")

    m0_receipt = read_json("R/output/universal_r_utility_v1/r01/M0_PREPARATION_RECEIPT.json")
    bindings = read_json("R/output/universal_r_utility_v1/r01/M0_SOURCE_BINDINGS.json")
    review = read_json("R/output/universal_r_utility_v1/r01/M0_INDEPENDENT_REVIEW.json")
    if (
        m0_receipt.get("status") != "M0_PREPARED_AWAITING_ROOT_REVIEW"
        or m0_receipt.get("process_exit_code") != 0
        or m0_receipt.get("science_wall_seconds") != 0.0
        or m0_receipt.get("diagnostic_bank_generated") is not False
        or m0_receipt.get("science_labels_generated") is not False
        or m0_receipt.get("predictor_fit") is not False
    ):
        raise M1ContractError("M0 receipt does not prove a clean preparation-only state")
    if (
        review.get("review_id") != "REVIEW-047-00"
        or review.get("decision") != "PASS_M0_PREPARATION / NO_SCIENCE_RELEASE"
        or review.get("m1_science_release") is not False
    ):
        raise M1ContractError("independent M0 review record is absent or not preparation-only")

    source_mismatches = []
    for row in bindings.get("source_files", []):
        relative = str(row.get("path", ""))
        path = root / relative
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            source_mismatches.append(relative)
    if len(bindings.get("source_files", [])) != 45 or source_mismatches:
        raise M1ContractError(
            f"frozen M0 source binding drift/missing input: {source_mismatches}"
        )

    artifact_mismatches = []
    for relative_name, expected_hash in m0_receipt.get("artifact_sha256", {}).items():
        artifact = m0_out / relative_name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            artifact_mismatches.append(relative_name)
    if len(m0_receipt.get("artifact_sha256", {})) != 9 or artifact_mismatches:
        raise M1ContractError(
            f"frozen M0 artifact drift/missing input: {artifact_mismatches}"
        )

    release_bindings = {
        "m0_source_bindings_sha256": sha256_file(m0_out / "M0_SOURCE_BINDINGS.json"),
        "m0_exclusion_manifest_sha256": sha256_file(m0_out / "M0_EXCLUSION_MANIFEST.json"),
        "fit_layout_sha256": sha256_file(m0_out / "fit.json"),
        "validation_layout_sha256": sha256_file(m0_out / "validation.json"),
        "m0_review_sha256": sha256_file(m0_out / "M0_INDEPENDENT_REVIEW.json"),
    }
    for field, actual in release_bindings.items():
        if _require_sha256(release.get(field), name=field) != actual:
            raise M1ContractError(f"M1 release is stale for {field}")

    expected_m1_sources = {
        relative: sha256_file(root / relative) for relative in M1_SOURCE_PATHS
    }
    released_m1_sources = release.get("m1_source_sha256")
    if not isinstance(released_m1_sources, dict) or released_m1_sources != expected_m1_sources:
        raise M1ContractError("M1 implementation source hashes differ from root release")
    if release.get("predictor_fit_released") is not False:
        raise M1ContractError("M1 bank release cannot silently release predictor fitting")
    verified = {
        "release_id": str(release.get("release_id", "")),
        "scope": expected_scope,
        "release_sha256": sha256_file(release_file),
        "m0_source_bindings_sha256": release_bindings["m0_source_bindings_sha256"],
        "m0_review_sha256": release_bindings["m0_review_sha256"],
        "m1_source_sha256": expected_m1_sources,
        "predictor_fit_released": False,
    }
    if expected_scope == M1_BANK_SCOPE:
        smoke_release_relative = release.get(
            "smoke_release_path",
            "R/output/universal_r_utility_v1/r02/M1_SMOKE_RELEASE.json",
        )
        smoke_release_file = _resolve_r02_artifact_path(
            root, smoke_release_relative, name="smoke_release_path"
        )
        smoke_verified = verify_m1_release(
            root, smoke_release_file, expected_scope=M1_SMOKE_SCOPE
        )
        if _require_sha256(release.get("smoke_release_sha256"), name="smoke_release_sha256") != smoke_verified["release_sha256"]:
            raise M1ContractError("bank release is not bound to the current released smoke")

        smoke_receipt_path = _resolve_r02_artifact_path(
            root, release.get("smoke_receipt_path", ""), name="smoke_receipt_path"
        )
        smoke_labels_path = _resolve_r02_artifact_path(
            root, release.get("smoke_labels_path", ""), name="smoke_labels_path"
        )
        smoke_receipt_relative = smoke_receipt_path.relative_to(root).as_posix()
        smoke_labels_relative = smoke_labels_path.relative_to(root).as_posix()
        if _require_sha256(release.get("smoke_receipt_sha256"), name="smoke_receipt_sha256") != sha256_file(smoke_receipt_path):
            raise M1ContractError("bank release smoke receipt hash is missing or stale")
        smoke_receipt = json.loads(smoke_receipt_path.read_text(encoding="utf-8"))
        if not isinstance(smoke_receipt, dict):
            raise M1ContractError("smoke receipt must be a JSON object")
        if (
            smoke_receipt.get("phase") != M1_SMOKE_SCOPE
            or smoke_receipt.get("status") != "M1_SMOKE_COMPLETE"
            or smoke_receipt.get("phase_exit_code") != 0
            or smoke_receipt.get("trainable") is not False
            or smoke_receipt.get("predictor_fit") is not False
            or smoke_receipt.get("performance_screening") is not False
        ):
            raise M1ContractError("bank release requires a passed non-trainable smoke receipt")
        if smoke_receipt.get("output_root") != smoke_labels_path.parent.parent.relative_to(root).as_posix():
            raise M1ContractError("smoke labels path does not match the smoke receipt output root")
        expected_smoke_receipt_path = (
            root / str(smoke_receipt.get("output_root", "")) / "M1_SMOKE_RECEIPT.json"
        ).resolve()
        if smoke_receipt_path.resolve() != expected_smoke_receipt_path:
            raise M1ContractError("bank release smoke receipt path does not match its output root")
        if smoke_receipt.get("verified_release", {}).get("release_sha256") != smoke_verified["release_sha256"]:
            raise M1ContractError("smoke receipt is not bound to its released smoke source")
        if smoke_receipt.get("verified_release", {}).get("m1_source_sha256") != expected_m1_sources:
            raise M1ContractError("smoke receipt source hashes differ from the current implementation")
        if (
            smoke_receipt.get("release_integrity", {}).get("end_release_sha256")
            != smoke_verified["release_sha256"]
            or smoke_receipt.get("release_integrity", {}).get("start_release_sha256")
            != smoke_verified["release_sha256"]
            or smoke_receipt.get("release_integrity", {}).get("start_m1_source_sha256")
            != expected_m1_sources
            or smoke_receipt.get("release_integrity", {}).get("end_m1_source_sha256")
            != expected_m1_sources
            or smoke_receipt.get("release_integrity", {}).get(
                "same_scope_and_source_hashes_at_end"
            ) is not True
        ):
            raise M1ContractError("smoke receipt lacks a successful end-of-run release recheck")
        label_relative_to_output = "private/branch_labels.jsonl.gz"
        expected_labels_path = (root / str(smoke_receipt["output_root"]) / label_relative_to_output).resolve()
        if smoke_labels_path.resolve() != expected_labels_path:
            raise M1ContractError("bank release smoke labels path does not identify the registered smoke labels")
        smoke_label_hash = sha256_file(smoke_labels_path)
        if (
            _require_sha256(release.get("smoke_labels_sha256"), name="smoke_labels_sha256")
            != smoke_label_hash
            or smoke_receipt.get("artifact_sha256", {}).get(label_relative_to_output) != smoke_label_hash
        ):
            raise M1ContractError("bank release smoke labels hash is missing or stale")
        first_state_validation = smoke_receipt.get("first_state_validation")
        if not isinstance(first_state_validation, dict):
            raise M1ContractError("passed smoke receipt lacks its first-state validation summary")
        if first_state_validation.get("schema") != "plan047_first_fit_state_recovery_match_v1":
            raise M1ContractError("smoke first-state validation schema is unregistered")
        first_state_sha256 = _sha256_bytes(_canonical_bytes(first_state_validation))
        if (
            _require_sha256(
                release.get("smoke_first_state_validation_sha256"),
                name="smoke_first_state_validation_sha256",
            )
            != first_state_sha256
            or smoke_receipt.get("first_state_validation_sha256") != first_state_sha256
        ):
            raise M1ContractError("bank release first-state smoke binding is missing or stale")
        candidate_sequences = first_state_validation.get("candidate_sequences")
        branch_outcomes = first_state_validation.get("branch_outcomes")
        if not isinstance(candidate_sequences, list) or not isinstance(branch_outcomes, list):
            raise M1ContractError("smoke first-state sequence/outcome lists are malformed")
        if (
            not all(
                isinstance(sequence, list)
                and len(sequence) > 0
                and all(type(value) is int for value in sequence)
                for sequence in candidate_sequences
            )
            or not all(isinstance(row, dict) for row in branch_outcomes)
        ):
            raise M1ContractError("smoke first-state sequence/outcome rows are malformed")
        outcome_sequences = [row.get("candidate_sequence") for row in branch_outcomes]
        smoke_label_count = smoke_receipt.get("counts", {}).get("unique_candidate_labels")
        if (
            not candidate_sequences
            or candidate_sequences != sorted(candidate_sequences)
            or len({json.dumps(row, separators=(",", ":")) for row in candidate_sequences})
            != len(candidate_sequences)
            or outcome_sequences != candidate_sequences
            or type(smoke_label_count) is not int
            or len(candidate_sequences) != smoke_label_count
        ):
            raise M1ContractError("smoke first-state candidates and outcomes are incomplete or duplicated")
        for outcome in branch_outcomes:
            raw_values = outcome.get("continuation_r_raw")
            rn_values = outcome.get("continuation_r_over_n")
            candidate_sequence = outcome.get("candidate_sequence")
            if (
                not isinstance(raw_values, Mapping)
                or not isinstance(rn_values, Mapping)
                or not isinstance(candidate_sequence, list)
                or type(outcome.get("candidate_relocations_included")) is not int
                or outcome.get("candidate_relocations_included") != len(candidate_sequence)
                or set(raw_values) != set(ALGORITHMS)
                or set(rn_values) != set(ALGORITHMS)
                or not all(
                    key in outcome
                    for key in (
                        "mix4_r_raw",
                        "mix4_r_over_n",
                        "em0_sensitivity_r_raw",
                        "em0_sensitivity_r_over_n",
                    )
                )
            ):
                raise M1ContractError("smoke first-state outcome lacks one of the five continuations")
            if (
                not all(type(raw_values[algorithm]) is int for algorithm in ALGORITHMS)
                or type(outcome["em0_sensitivity_r_raw"]) is not int
                or not all(math.isfinite(float(rn_values[algorithm])) for algorithm in ALGORITHMS)
                or not all(
                    math.isfinite(float(outcome[key]))
                    for key in (
                        "mix4_r_raw",
                        "mix4_r_over_n",
                        "em0_sensitivity_r_raw",
                        "em0_sensitivity_r_over_n",
                    )
                )
            ):
                raise M1ContractError("smoke first-state outcome contains nonfinite continuation values")
        assignment = smoke_receipt.get("assignment", {})
        if (
            assignment.get("split") != "fit"
            or assignment.get("layout_ordinal") != 0
            or assignment.get("window_size") != 1
            or assignment.get("repeat") != 0
            or smoke_receipt.get("first_state_id") != first_state_validation.get("state_id")
        ):
            raise M1ContractError("smoke labels do not identify the registered first fit state")
        verified["smoke_binding"] = {
            "smoke_release_path": smoke_release_file.relative_to(root).as_posix(),
            "smoke_release_sha256": smoke_verified["release_sha256"],
            "smoke_receipt_path": smoke_receipt_relative,
            "smoke_receipt_sha256": sha256_file(smoke_receipt_path),
            "smoke_labels_path": smoke_labels_relative,
            "smoke_labels_sha256": smoke_label_hash,
            "first_state_validation": first_state_validation,
            "first_state_validation_sha256": first_state_sha256,
            "assignment": assignment,
        }
    return verified


def semantic_state_fingerprint(environment: Any, provider: Any, public_frame: Any) -> str:
    """Hash the complete evaluator state without exposing raw IDs to the scorer."""
    from R.scrp.h20_r_bank import _private_signature

    signature = _private_signature(environment, provider)
    return _sha256_bytes(
        _canonical_bytes(
            {
                "private_signature": signature,
                "num_stacks": len(environment._stacks),
                "max_tiers": int(public_frame.max_tiers),
                "num_containers": len(provider._true_sequence),
            }
        )
    )


def _forecast_sha256(forecast: Sequence[int]) -> str:
    return _sha256_bytes(_canonical_bytes([int(value) for value in forecast]))


@dataclass(frozen=True)
class DecisionSnapshot:
    decision_index: int
    retrieval_count: int
    num_containers: int
    public_frame: Any
    environment: Any
    provider: Any
    private_state_fingerprint: str

    @property
    def retrieval_fraction(self) -> float:
        return self.retrieval_count / float(self.num_containers)


@dataclass(frozen=True)
class StageCapture:
    stage: str
    target_fraction: float | None
    decision_index: int | None
    retrieval_count: int | None
    actual_fraction: float | None
    snapshot: DecisionSnapshot | None
    missing_reason: str | None
    duplicate_of: str | None


class NearestStageSampler:
    """Keep first and online-nearest .50/.85 actual retrieval states.

    Ties keep the earlier decision. Only eligible legal states are considered;
    for EG the caller marks only uncached group starts eligible. Missing and
    duplicate stages are returned explicitly and are never filled by another
    sample.
    """

    def __init__(self, num_containers: int) -> None:
        self.num_containers = _strict_int(num_containers, name="num_containers")
        if self.num_containers <= 0:
            raise ValueError("num_containers must be positive")
        self._first: DecisionSnapshot | None = None
        self._nearest: dict[str, tuple[tuple[int, int], DecisionSnapshot]] = {}
        self._snapshots: dict[int, DecisionSnapshot] = {}
        self._eligible_count = 0

    def observe(
        self,
        *,
        decision_index: int,
        public_frame: Any,
        environment: Any,
        provider: Any,
        eligible: bool = True,
    ) -> None:
        index = _strict_int(decision_index, name="decision_index")
        if index < 0:
            raise ValueError("decision_index must be nonnegative")
        if not eligible:
            return
        if not tuple(getattr(public_frame, "legal_destinations", ())):
            return
        retrieval = _strict_int(public_frame.retrieval_count, name="retrieval_count")
        if not 0 <= retrieval <= self.num_containers:
            raise ValueError("retrieval count is outside the trajectory")
        if int(environment.state.retrieval_count) != retrieval:
            raise M1ContractError("sampled frame retrieval count differs from environment")
        actual_fraction = retrieval / float(self.num_containers)
        needs_first = self._first is None
        updates: list[tuple[str, tuple[int, int]]] = []
        for stage, target in (("mid", 0.50), ("late", 0.85)):
            target_percent = _strict_int(round(target * 100), name="target_percent")
            # Compare rational progress in integer units. Floating point can
            # make mathematically equal distances (e.g. 12/40 and 28/40 from
            # 0.50) differ by a few ulps and incorrectly select the later state.
            key = (abs(100 * retrieval - target_percent * self.num_containers), index)
            prior = self._nearest.get(stage)
            if prior is None or key < prior[0]:
                updates.append((stage, key))
        if not needs_first and not updates:
            self._eligible_count += 1
            return
        snapshot = self._snapshots.get(index)
        if snapshot is None:
            environment_copy, provider_copy, frame_copy = deepcopy(
                (environment, provider, public_frame)
            )
            snapshot = DecisionSnapshot(
                decision_index=index,
                retrieval_count=retrieval,
                num_containers=self.num_containers,
                public_frame=frame_copy,
                environment=environment_copy,
                provider=provider_copy,
                private_state_fingerprint=semantic_state_fingerprint(
                    environment_copy, provider_copy, frame_copy
                ),
            )
            self._snapshots[index] = snapshot
        self._eligible_count += 1
        if needs_first:
            self._first = snapshot
        for stage, key in updates:
            self._nearest[stage] = (key, snapshot)
        retained = {id(self._first)} if self._first is not None else set()
        retained.update(id(value[1]) for value in self._nearest.values())
        self._snapshots = {
            cached_index: cached_snapshot
            for cached_index, cached_snapshot in self._snapshots.items()
            if id(cached_snapshot) in retained
        }

    def finalize(self) -> tuple[StageCapture, ...]:
        selected = {
            "first": self._first,
            "mid": None if "mid" not in self._nearest else self._nearest["mid"][1],
            "late": None if "late" not in self._nearest else self._nearest["late"][1],
        }
        seen_states: dict[str, str] = {}
        result: list[StageCapture] = []
        for stage, target in M1_STAGE_TARGETS:
            snapshot = selected[stage]
            if snapshot is None:
                result.append(
                    StageCapture(
                        stage=stage,
                        target_fraction=target,
                        decision_index=None,
                        retrieval_count=None,
                        actual_fraction=None,
                        snapshot=None,
                        missing_reason="no eligible legal decision",
                        duplicate_of=None,
                    )
                )
                continue
            duplicate_of = seen_states.get(snapshot.private_state_fingerprint)
            if duplicate_of is None:
                seen_states[snapshot.private_state_fingerprint] = stage
            result.append(
                StageCapture(
                    stage=stage,
                    target_fraction=target,
                    decision_index=snapshot.decision_index,
                    retrieval_count=snapshot.retrieval_count,
                    actual_fraction=snapshot.retrieval_fraction,
                    snapshot=snapshot,
                    missing_reason=None,
                    duplicate_of=duplicate_of,
                )
            )
        return tuple(result)

    @property
    def eligible_decision_count(self) -> int:
        return self._eligible_count


@dataclass(frozen=True)
class ConditionAssignment:
    split: str
    layout_ordinal: int
    layout_fingerprint: str
    geometry_id: str
    window_size: int
    repeat: int
    condition_ordinal: int
    behavior_algorithm: str


def condition_assignments(
    *, split: str, layout_rows: Sequence[Mapping[str, Any]]
) -> tuple[ConditionAssignment, ...]:
    """Freeze balanced behavior actors before a trajectory or label exists."""
    if split not in ("fit", "validation"):
        raise ValueError("M1 split must be fit or validation")
    result: list[ConditionAssignment] = []
    for layout_ordinal, row in enumerate(layout_rows):
        fingerprint = str(row["label_invariant_fingerprint"])
        geometry = str(row["parameter_group"])
        if geometry not in ("G1", "G2"):
            raise M1ContractError(f"unknown registered geometry: {geometry}")
        for condition_ordinal, (window, repeat) in enumerate(M1_CONDITIONS):
            result.append(
                ConditionAssignment(
                    split=split,
                    layout_ordinal=layout_ordinal,
                    layout_fingerprint=fingerprint,
                    geometry_id=geometry,
                    window_size=window,
                    repeat=repeat,
                    condition_ordinal=condition_ordinal,
                    behavior_algorithm=ALGORITHMS[
                        (layout_ordinal + condition_ordinal) % len(ALGORITHMS)
                    ],
                )
            )
    return tuple(result)


def make_forecast_context(
    episode: Any,
    *,
    layout_fingerprint: str,
    window_size: int,
    repeat: int,
) -> tuple[Any, Any, tuple[int, ...], dict[str, Any]]:
    """Create one fixed public forecast and its independent provider/context."""
    from importlib import import_module
    from R.experiments.time_window_shuffle_protocol import make_time_window_forecast
    from R.scrp.h20_r_bank import _sync_retrieval_events_with_audit
    from scrp.online_crp import OnlineCRPEnvironment

    # Runtime setup puts the established EM/Mijn_RL extended package roots on
    # sys.path. Import through the owner-first ``scrp`` namespace used there;
    # importing ``EM.scrp`` would bind EM.scrp.__init__'s different local API.
    ForecastProvider = import_module("scrp.noise_aware_forecast").ForecastProvider

    seed = semantic_universal_time_window_seed(
        layout_fingerprint, window_size=window_size, repeat=repeat
    )
    forecast = tuple(
        int(value)
        for value in make_time_window_forecast(
            episode.retrieval_sequence[1:], window_size=window_size, seed=seed
        )
    )
    provider = ForecastProvider(
        episode.retrieval_sequence,
        p=0.0,
        seed=seed,
        q_prior=Q_PRIOR,
        max_horizon=20,
        forecast_future=forecast,
    )
    environment = OnlineCRPEnvironment(episode, lookahead_horizon=0)
    after_reset = int(environment.state.retrieval_count)
    reset_events = _sync_retrieval_events_with_audit(
        provider, episode.retrieval_sequence, 0, after_reset
    )
    if provider.max_horizon != 20 or abs(provider.history.q_prior - Q_PRIOR) > 1e-15:
        raise M1ContractError("M1 provider differs from fixed public h20/q prior")
    return environment, provider, forecast, {
        "protocol_id": PROTOCOL_ID,
        "layout_fingerprint": layout_fingerprint,
        "window_size": window_size,
        "repeat": repeat,
        "seed": seed,
        "forecast_sha256": _forecast_sha256(forecast),
        "reset_retrieval_count": after_reset,
        "reset_reveal_events_private": reset_events,
    }


class EMNativeActor:
    def __init__(self, runtime: Mapping[str, Any], horizon: int) -> None:
        self.runtime = runtime
        self.horizon = _strict_int(horizon, name="horizon")
        if self.horizon not in HORIZONS:
            raise ValueError("EM actor horizon is outside the registered set")
        self.invocation_count = 0

    def __call__(self, frame: Any) -> int:
        prefix = self.runtime["truncate_public_frame"](frame, self.horizon)
        action = int(
            self.runtime["em_select"](
                prefix.observation, max_tiers=frame.max_tiers, h=self.horizon
            )
        )
        self.invocation_count += 1
        return action


class GPNativeActor:
    """Exactly one GP h call per decision; never the five-horizon FIXED20 API."""

    def __init__(self, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any], horizon: int) -> None:
        from R.scrp.cross_solver_reliability_r import _run_one_horizon

        self.runtime = runtime
        self.gp_rule = gp_rule
        self.horizon = _strict_int(horizon, name="horizon")
        if self.horizon not in HORIZONS:
            raise ValueError("GP actor horizon is outside the registered set")
        self._run_one_horizon = _run_one_horizon
        self.invocation_count = 0

    def __call__(self, frame: Any) -> int:
        row = self._run_one_horizon(
            frame,
            requested_h=self.horizon,
            solver="GP",
            frozen_solver=self.gp_rule,
        )
        self.invocation_count += 1
        return int(row.destination)


class ERINativeActor:
    def __init__(self, runtime: Mapping[str, Any], horizon: int) -> None:
        self.runtime = runtime
        self.horizon = _strict_int(horizon, name="horizon")
        if self.horizon not in HORIZONS:
            raise ValueError("ERI actor horizon is outside the registered set")
        self.invocation_count = 0

    def __call__(self, frame: Any) -> int:
        prefix = self.runtime["truncate_public_frame"](frame, self.horizon)
        action = int(self.runtime["eri_select"](prefix))
        self.invocation_count += 1
        return action


class EGNativeActor:
    def __init__(self, runtime: Mapping[str, Any]) -> None:
        self.actor = runtime["eri_eg"].EG20Actor()

    @property
    def is_group_start(self) -> bool:
        return getattr(self.actor, "_plan", None) is None

    @property
    def cached_moves(self) -> int:
        return int(getattr(self.actor, "cached_moves", 0))

    def reset(self) -> None:
        self.actor.reset()

    def __call__(self, frame: Any) -> int:
        return int(self.actor.select_destination(frame))


def make_native_actor(algorithm: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any], horizon: int = 20) -> Any:
    if algorithm == "EM":
        return EMNativeActor(runtime, horizon)
    if algorithm == "GP":
        return GPNativeActor(runtime, gp_rule, horizon)
    if algorithm == "ERI":
        return ERINativeActor(runtime, horizon)
    if algorithm == "EG":
        if horizon != 20:
            raise ValueError("the registered EG cached actor is defined only at h=20")
        return EGNativeActor(runtime)
    if algorithm == "EM0":
        return EMNativeActor(runtime, 0)
    raise ValueError(f"unknown native actor: {algorithm}")


def make_continuation_actor_factories(
    runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]
) -> tuple[dict[str, Callable[[], Any]], Callable[[], Any]]:
    factories = {
        algorithm: (lambda algorithm=algorithm: make_native_actor(algorithm, runtime, gp_rule, 20))
        for algorithm in ALGORITHMS
    }
    return factories, lambda: make_native_actor("EM0", runtime, gp_rule, 0)


def propose_native_candidate_pools(
    public_frame: Any, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]
) -> tuple[NativeCandidatePool, ...]:
    """Build all five public horizon proposals from each registered proposer."""
    from R.scrp.cross_solver_reliability_r import _run_one_horizon

    proposals: dict[str, dict[int, tuple[int, ...]]] = {}
    for algorithm in ("EM", "GP", "ERI"):
        rows: dict[int, tuple[int, ...]] = {}
        for horizon in HORIZONS:
            if algorithm == "EM":
                actor = EMNativeActor(runtime, horizon)
                rows[horizon] = (actor(public_frame),)
            elif algorithm == "GP":
                raw = _run_one_horizon(
                    public_frame,
                    requested_h=horizon,
                    solver="GP",
                    frozen_solver=gp_rule,
                )
                rows[horizon] = (int(raw.destination),)
            else:
                rows[horizon] = (ERINativeActor(runtime, horizon)(public_frame),)
        proposals[algorithm] = rows
    eg_actor = runtime["r_eri_eg"].REGGroupActor()
    eg_output = eg_actor.select_destination(public_frame)
    eg_rows = {
        int(row["requested_h"]): tuple(int(value) for value in row["destinations_top_down"])
        for row in eg_output["candidate_plans"]
    }
    proposals["EG"] = eg_rows
    pools = build_native_candidate_pools(public_frame, proposals)
    if len(pools) != 4:
        raise M1ContractError("native proposal collection lost an algorithm pool")
    return pools


def collect_behavior_stages(
    episode: Any,
    environment: Any,
    provider: Any,
    *,
    behavior_algorithm: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
    check_budget: Callable[[], None] | None = None,
    private_trace_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[tuple[StageCapture, ...], dict[str, Any]]:
    """Run one preassigned native h20 behavior path and retain at most 3 states."""
    from R.scrp.h20_r_bank import (
        _private_physical_stacks,
        _public_frame,
        _sync_retrieval_events_with_audit,
    )

    if behavior_algorithm not in ALGORITHMS:
        raise ValueError("behavior algorithm is outside the registered set")
    actor = make_native_actor(behavior_algorithm, runtime, gp_rule, 20)
    sampler = NearestStageSampler(int(episode.num_containers))
    decision_index = 0
    max_steps = int(episode.num_containers) * max(1, int(episode.max_tiers) - 1)
    while not environment.state.terminated:
        if check_budget is not None:
            check_budget()
        frame = _public_frame(episode, environment, provider)
        if not frame.legal_destinations:
            raise M1ContractError("nonterminal behavior state has no legal destination")
        eligible = behavior_algorithm != "EG" or bool(actor.is_group_start)
        sampler.observe(
            decision_index=decision_index,
            public_frame=frame,
            environment=environment,
            provider=provider,
            eligible=eligible,
        )
        action = int(actor(frame))
        if action not in frame.legal_destinations:
            raise M1ContractError(f"{behavior_algorithm} behavior action is illegal")
        before_retrieval = int(environment.state.retrieval_count)
        before_relocations = int(environment.relocation_count)
        before_stacks = _private_physical_stacks(environment)
        before_q = float(provider.history.q_t)
        environment.step(action)
        after_retrieval = int(environment.state.retrieval_count)
        after_relocations = int(environment.relocation_count)
        reveals = _sync_retrieval_events_with_audit(
            provider, episode.retrieval_sequence, before_retrieval, after_retrieval
        )
        if after_relocations != before_relocations + 1:
            raise M1ContractError("behavior relocation counter did not advance once")
        if not reveals and float(provider.history.q_t) != before_q:
            raise M1ContractError("public q changed without a new reveal")
        live_count = sum(len(stack) for stack in _private_physical_stacks(environment))
        if live_count + after_retrieval != int(episode.num_containers):
            raise M1ContractError("behavior transition failed container conservation")
        if private_trace_sink is not None:
            private_trace_sink(
                {
                    "phase": "behavior",
                    "behavior_algorithm_evaluator_only": behavior_algorithm,
                    "decision_index": decision_index,
                    "public_legal_destinations": list(frame.legal_destinations),
                    "action_destination_private": action,
                    "physical_stacks_before_private": before_stacks,
                    "physical_stacks_after_private": _private_physical_stacks(environment),
                    "retrieval_count_before": before_retrieval,
                    "retrieval_count_after": after_retrieval,
                    "relocation_count_before": before_relocations,
                    "relocation_count_after": after_relocations,
                    "q_t_before_action": before_q,
                    "q_t_after_reveals": float(provider.history.q_t),
                    "reveal_events_private": reveals,
                }
            )
        decision_index += 1
        if decision_index > max_steps:
            raise M1ContractError("native behavior exceeded physical safety bound")
    if int(environment.state.retrieval_count) != int(episode.num_containers):
        raise M1ContractError("native behavior terminated before retrieving every container")
    captures = sampler.finalize()
    return captures, {
        "behavior_algorithm": behavior_algorithm,
        "behavior_decisions": decision_index,
        "eligible_decisions": sampler.eligible_decision_count,
        "actual_total_relocations": int(environment.relocation_count),
        "stage_captures": [
            {
                "stage": row.stage,
                "decision_index": row.decision_index,
                "retrieval_count": row.retrieval_count,
                "actual_fraction": row.actual_fraction,
                "missing_reason": row.missing_reason,
                "duplicate_of": row.duplicate_of,
            }
            for row in captures
        ],
    }


def mix4_training_target_r_over_n(continuations: Mapping[str, float]) -> float:
    if set(continuations) != set(ALGORITHMS):
        raise M1ContractError("mix4 target requires exactly EM20/GP20/ERI20/EG20")
    values = [float(continuations[algorithm]) for algorithm in ALGORITHMS]
    if not all(math.isfinite(value) for value in values):
        raise M1ContractError("mix4 target has nonfinite continuation values")
    return sum(values) / 4.0


def mix3_loao_target_r_over_n(
    continuations: Mapping[str, float], *, heldout_algorithm: str
) -> float:
    if set(continuations) != set(ALGORITHMS) or heldout_algorithm not in ALGORITHMS:
        raise M1ContractError("LOAO target requires four fixed continuation columns")
    values = [float(continuations[a]) for a in ALGORITHMS if a != heldout_algorithm]
    if not all(math.isfinite(value) for value in values):
        raise M1ContractError("LOAO mix3 target has nonfinite values")
    return sum(values) / 3.0


@dataclass(frozen=True)
class CandidateTrainingSample:
    layout_id: str
    geometry_id: str
    state_id: str
    behavior_algorithm: str
    pool_algorithm: str
    sequence: tuple[int, ...]
    features: tuple[float, ...]
    continuation_r_over_n: Mapping[str, float]
    origin_algorithms: tuple[str, ...]


def iter_candidate_training_samples(
    public_candidate_rows: Iterable[Mapping[str, Any]],
    private_membership_rows: Iterable[Mapping[str, Any]],
    private_label_rows: Iterable[Mapping[str, Any]],
) -> Iterator[CandidateTrainingSample]:
    """Join separated M1 streams for fit/validation without exposing metadata to score."""
    public_by_id: dict[str, Mapping[str, Any]] = {}
    for row in public_candidate_rows:
        candidate_id = str(row["candidate_id"])
        prior = public_by_id.get(candidate_id)
        if prior is not None and prior != row:
            raise M1ContractError("public candidate_id maps to conflicting feature rows")
        public_by_id[candidate_id] = row
    labels_by_id: dict[str, Mapping[str, Any]] = {}
    for row in private_label_rows:
        label_id = str(row["private_label_id"])
        prior = labels_by_id.get(label_id)
        if prior is not None and prior != row:
            raise M1ContractError("private label_id maps to conflicting physical outcomes")
        labels_by_id[label_id] = row

    for membership in private_membership_rows:
        candidate_id = str(membership["candidate_id"])
        label_id = str(membership["private_label_id"])
        public = public_by_id.get(candidate_id)
        label = labels_by_id.get(label_id)
        if public is None or label is None:
            raise M1ContractError("fit/validation join is missing a public candidate or private label")
        model_input = public.get("model_input")
        if not isinstance(model_input, Mapping) or set(model_input) != {
            "public_frame",
            "candidate_sequence",
        }:
            raise M1ContractError("candidate score input includes nonpublic or nonsequence fields")
        sequence = tuple(int(value) for value in membership["candidate_sequence"])
        if sequence != tuple(int(value) for value in model_input["candidate_sequence"]):
            raise M1ContractError("public candidate sequence differs from private pool membership")
        if sequence != tuple(int(value) for value in label["candidate_sequence"]):
            raise M1ContractError("candidate sequence differs from its bound physical label")
        if membership.get("state_id") != public.get("state_id"):
            raise M1ContractError("fit/validation public candidate state binding differs")
        if membership.get("layout_fingerprint") != public.get("layout_fingerprint"):
            raise M1ContractError("fit/validation layout binding differs across streams")
        if membership.get("behavior_algorithm") != public.get("behavior_algorithm"):
            raise M1ContractError("behavior algorithm binding differs across streams")
        features = tuple(float(value) for value in public["public_features_74"])
        if len(features) != 74 or not all(math.isfinite(value) for value in features):
            raise M1ContractError("public feature stream is not 74 finite values")
        continuations = {
            str(algorithm): float(value)
            for algorithm, value in label["continuation_r_over_n"].items()
        }
        if set(continuations) != set(ALGORITHMS):
            raise M1ContractError("private label join lacks all four continuation columns")
        origin_algorithms = tuple(
            sorted(
                {
                    str(origin["algorithm"])
                    for origin in public.get("origins_evaluator_metadata", [])
                }
            )
        )
        yield CandidateTrainingSample(
            layout_id=str(membership["layout_fingerprint"]),
            geometry_id=str(membership["geometry_id"]),
            state_id=str(membership["state_id"]),
            behavior_algorithm=str(membership["behavior_algorithm"]),
            pool_algorithm=str(membership["pool_algorithm"]),
            sequence=sequence,
            features=features,
            continuation_r_over_n=continuations,
            origin_algorithms=origin_algorithms,
        )


@dataclass(frozen=True)
class PairwiseTrainingRow:
    layout_id: str
    geometry_id: str
    state_id: str
    pool_algorithm: str
    left_sequence: tuple[int, ...]
    right_sequence: tuple[int, ...]
    x_left: tuple[float, ...]
    x_right: tuple[float, ...]
    target_difference: float
    weight: float
    target_kind: str
    heldout_algorithm: str | None
    left_origin_algorithms: tuple[str, ...]
    right_origin_algorithms: tuple[str, ...]


def build_pairwise_training_rows(
    samples: Iterable[CandidateTrainingSample],
    *,
    heldout_algorithm: str | None = None,
) -> tuple[PairwiseTrainingRow, ...]:
    """Build geometry/layout/state/pool/pair weighted rows.

    Pair construction is strictly within one native proposer pool. An LOAO
    fold drops heldout behavior states and pools, removes heldout-only proposal
    origins, and uses the remaining-three continuation target. The two
    registered geometries have equal mass; within each, eligible layouts,
    eligible states, pair-eligible pools, and pairs each divide mass equally.
    """
    if heldout_algorithm is not None and heldout_algorithm not in ALGORITHMS:
        raise ValueError("heldout algorithm is not registered")
    grouped: dict[tuple[str, str, str, str], dict[tuple[int, ...], CandidateTrainingSample]] = {}
    for sample in samples:
        if sample.pool_algorithm not in ALGORITHMS or sample.behavior_algorithm not in ALGORITHMS:
            raise M1ContractError("candidate sample has an unregistered algorithm")
        if sample.geometry_id not in M1_TRAIN_GEOMETRIES:
            raise M1ContractError("candidate sample has an unregistered geometry")
        if heldout_algorithm is not None and (
            sample.behavior_algorithm == heldout_algorithm
            or sample.pool_algorithm == heldout_algorithm
        ):
            continue
        if heldout_algorithm is not None and not any(
            algorithm != heldout_algorithm for algorithm in sample.origin_algorithms
        ):
            continue
        features = tuple(float(value) for value in sample.features)
        if len(features) != 74 or not all(math.isfinite(value) for value in features):
            raise M1ContractError("training feature row must contain 74 finite public values")
        if heldout_algorithm is None:
            target = mix4_training_target_r_over_n(sample.continuation_r_over_n)
            target_kind = "mix4_r_over_n"
        else:
            target = mix3_loao_target_r_over_n(
                sample.continuation_r_over_n, heldout_algorithm=heldout_algorithm
            )
            target_kind = "mix3_r_over_n"
        normalized = CandidateTrainingSample(
            layout_id=sample.layout_id,
            geometry_id=sample.geometry_id,
            state_id=sample.state_id,
            behavior_algorithm=sample.behavior_algorithm,
            pool_algorithm=sample.pool_algorithm,
            sequence=tuple(int(value) for value in sample.sequence),
            features=features,
            continuation_r_over_n={"target": target},
            origin_algorithms=tuple(
                sorted(
                    {
                        algorithm
                        for algorithm in sample.origin_algorithms
                        if heldout_algorithm is None or algorithm != heldout_algorithm
                    }
                )
            ),
        )
        group_key = (
            sample.layout_id,
            sample.geometry_id,
            sample.state_id,
            sample.pool_algorithm,
        )
        prior = grouped.setdefault(group_key, {}).get(normalized.sequence)
        if prior is not None and (
            prior.features != normalized.features
            or prior.continuation_r_over_n["target"] != target
        ):
            raise M1ContractError("duplicate sequence has conflicting feature or branch label")
        if prior is None:
            grouped[group_key][normalized.sequence] = normalized
        else:
            origins = tuple(sorted(set(prior.origin_algorithms + normalized.origin_algorithms)))
            grouped[group_key][normalized.sequence] = CandidateTrainingSample(
                layout_id=prior.layout_id,
                geometry_id=prior.geometry_id,
                state_id=prior.state_id,
                behavior_algorithm=prior.behavior_algorithm,
                pool_algorithm=prior.pool_algorithm,
                sequence=prior.sequence,
                features=prior.features,
                continuation_r_over_n=prior.continuation_r_over_n,
                origin_algorithms=origins,
            )

    by_state: dict[tuple[str, str, str], list[tuple[str, dict[tuple[int, ...], CandidateTrainingSample]]]] = {}
    for (layout, geometry, state, pool), candidates in grouped.items():
        if len(candidates) >= 2:
            by_state.setdefault((layout, geometry, state), []).append((pool, candidates))
    eligible_layouts_by_geometry: dict[str, set[str]] = {
        geometry: set() for geometry in M1_TRAIN_GEOMETRIES
    }
    state_ids_by_layout: dict[tuple[str, str], set[str]] = {}
    for layout, geometry, state in by_state:
        eligible_layouts_by_geometry[geometry].add(layout)
        state_ids_by_layout.setdefault((geometry, layout), set()).add(state)
    for geometry in M1_TRAIN_GEOMETRIES:
        if not eligible_layouts_by_geometry[geometry]:
            raise M1ContractError(f"pairwise training has no eligible layout for {geometry}")

    output: list[PairwiseTrainingRow] = []
    for (layout, geometry, state), eligible_pools in sorted(by_state.items()):
        state_weight = 1.0 / len(state_ids_by_layout[(geometry, layout)])
        pool_weight = 1.0 / len(eligible_pools)
        for pool, candidates in sorted(eligible_pools):
            sequences = sorted(candidates)
            pairs = tuple(itertools.combinations(sequences, 2))
            pair_weight = 1.0 / len(pairs)
            for left_sequence, right_sequence in pairs:
                left = candidates[left_sequence]
                right = candidates[right_sequence]
                left_target = float(left.continuation_r_over_n["target"])
                right_target = float(right.continuation_r_over_n["target"])
                output.append(
                    PairwiseTrainingRow(
                        layout_id=layout,
                        geometry_id=geometry,
                        state_id=state,
                        pool_algorithm=pool,
                        left_sequence=left_sequence,
                        right_sequence=right_sequence,
                        x_left=left.features,
                        x_right=right.features,
                        target_difference=left_target - right_target,
                        weight=(
                            0.5
                            / len(eligible_layouts_by_geometry[geometry])
                            * state_weight
                            * pool_weight
                            * pair_weight
                        ),
                        target_kind=("mix4_r_over_n" if heldout_algorithm is None else "mix3_r_over_n"),
                        heldout_algorithm=heldout_algorithm,
                        left_origin_algorithms=left.origin_algorithms,
                        right_origin_algorithms=right.origin_algorithms,
                    )
                )
    if not math.isclose(sum(row.weight for row in output), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise M1ContractError("pair weights do not sum to one over geometry/layout/state/pools")
    return tuple(output)


@dataclass(frozen=True)
class ValidationCandidate:
    sequence: tuple[int, ...]
    features: tuple[float, ...]
    mix4_r_raw: float
    is_h20_anchor: bool


@dataclass(frozen=True)
class ValidationPool:
    layout_id: str
    geometry_id: str
    state_id: str
    proposer_algorithm: str
    candidates: tuple[ValidationCandidate, ...]


def validation_pools_from_rows(
    public_candidate_rows: Iterable[Mapping[str, Any]],
    private_membership_rows: Iterable[Mapping[str, Any]],
    private_label_rows: Iterable[Mapping[str, Any]],
) -> tuple[ValidationPool, ...]:
    """Rebuild per-state native pools by joining the separated M1 streams."""
    public_by_id = {str(row["candidate_id"]): row for row in public_candidate_rows}
    labels_by_id = {str(row["private_label_id"]): row for row in private_label_rows}
    grouped: dict[tuple[str, str, str, str], list[ValidationCandidate]] = {}
    for membership in private_membership_rows:
        candidate_id = str(membership["candidate_id"])
        label_id = str(membership["private_label_id"])
        public = public_by_id.get(candidate_id)
        label = labels_by_id.get(label_id)
        if public is None or label is None:
            raise M1ContractError("validation join is missing a public candidate or private label")
        sequence = tuple(int(value) for value in membership["candidate_sequence"])
        if (
            sequence != tuple(int(value) for value in public["model_input"]["candidate_sequence"])
            or sequence != tuple(int(value) for value in label["candidate_sequence"])
        ):
            raise M1ContractError("validation candidate sequence differs across bound streams")
        if membership.get("state_id") != public.get("state_id"):
            raise M1ContractError("validation state binding differs across public and membership rows")
        features = tuple(float(value) for value in public["public_features_74"])
        label_cost = float(label["mix4_r_raw_primary"])
        if len(features) != 74 or not all(math.isfinite(value) for value in features):
            raise M1ContractError("validation feature row must contain 74 finite public values")
        if not math.isfinite(label_cost) or label_cost < 0:
            raise M1ContractError("validation mix4 R_raw must be finite and nonnegative")
        group_key = (
            str(membership["layout_fingerprint"]),
            str(membership["geometry_id"]),
            str(membership["state_id"]),
            str(membership["pool_algorithm"]),
        )
        grouped.setdefault(group_key, []).append(
            ValidationCandidate(
                sequence=sequence,
                features=features,
                mix4_r_raw=label_cost,
                is_h20_anchor=bool(membership["is_h20_anchor"]),
            )
        )
    output = []
    for (layout, geometry, state, algorithm), candidates in sorted(grouped.items()):
        by_sequence: dict[tuple[int, ...], ValidationCandidate] = {}
        for candidate in candidates:
            prior = by_sequence.get(candidate.sequence)
            if prior is not None and prior != candidate:
                raise M1ContractError("validation pool has duplicate sequence with conflicting data")
            by_sequence[candidate.sequence] = candidate
        output.append(
            ValidationPool(
                layout_id=layout,
                geometry_id=geometry,
                state_id=state,
                proposer_algorithm=algorithm,
                candidates=tuple(by_sequence[key] for key in sorted(by_sequence)),
            )
        )
    return tuple(output)


def evaluate_validation_pools(
    pools: Iterable[ValidationPool], scorer: Callable[[np.ndarray], float]
) -> tuple[dict[str, Any], ...]:
    """Evaluate selection, native pool oracle, and paired mix4-R_raw gain."""
    rows: list[dict[str, Any]] = []
    for pool in pools:
        if pool.proposer_algorithm not in ALGORITHMS or pool.geometry_id not in ("G1", "G2"):
            raise M1ContractError("validation pool has unregistered geometry or algorithm")
        if not pool.candidates:
            raise M1ContractError("validation pool has no candidate sequences")
        anchors = [candidate for candidate in pool.candidates if candidate.is_h20_anchor]
        if len(anchors) != 1:
            raise M1ContractError("validation pool must contain exactly one native h20 anchor")
        score_rows = []
        for candidate in pool.candidates:
            features = np.asarray(candidate.features, dtype=np.float64)
            if features.shape != (74,) or not np.isfinite(features).all():
                raise M1ContractError("validation candidate has invalid public features")
            score = float(scorer(features))
            if not math.isfinite(score):
                raise M1ContractError("validation scorer returned a nonfinite value")
            if not math.isfinite(float(candidate.mix4_r_raw)) or candidate.mix4_r_raw < 0:
                raise M1ContractError("validation label must be a finite nonnegative mix4 R_raw")
            score_rows.append((candidate, score))
        anchor = anchors[0]
        anchor_score = next(score for candidate, score in score_rows if candidate is anchor)
        better = [
            (candidate, score)
            for candidate, score in score_rows
            if candidate.sequence != anchor.sequence and score < anchor_score
        ]
        if better:
            selected, _selected_score = min(
                better, key=lambda row: (row[1], row[0].sequence)
            )
        else:
            selected = anchor
        oracle = min(pool.candidates, key=lambda candidate: (candidate.mix4_r_raw, candidate.sequence))
        native_cost = float(anchor.mix4_r_raw)
        selected_cost = float(selected.mix4_r_raw)
        oracle_cost = float(oracle.mix4_r_raw)
        if native_cost <= 0:
            raise M1ContractError("validation native h20 R_raw denominator is zero")
        rows.append(
            {
                "layout_id": pool.layout_id,
                "geometry_id": pool.geometry_id,
                "state_id": pool.state_id,
                "proposer_algorithm": pool.proposer_algorithm,
                "native_h20_mix4_r_raw": native_cost,
                "selected_mix4_r_raw": selected_cost,
                "oracle_mix4_r_raw": oracle_cost,
                "gain_r_raw": native_cost - selected_cost,
                "oracle_gap_r_raw": native_cost - oracle_cost,
                "gain_fraction_of_native_state_sensitivity": (
                    native_cost - selected_cost
                ) / native_cost,
                "oracle_gap_fraction_of_native_state_sensitivity": (
                    native_cost - oracle_cost
                ) / native_cost,
            }
        )
    return tuple(rows)


def _allowed_algorithm_tuple(allowed_algorithms: Sequence[str]) -> tuple[str, ...]:
    allowed = tuple(allowed_algorithms)
    if not allowed or len(set(allowed)) != len(allowed):
        raise M1ContractError("allowed algorithm set must be nonempty and unique")
    if any(algorithm not in ALGORITHMS for algorithm in allowed):
        raise M1ContractError("allowed algorithm set contains an unregistered algorithm")
    return allowed


def _validate_macro_rows(
    rows: Sequence[Mapping[str, Any]], allowed_algorithms: Sequence[str]
) -> tuple[str, ...]:
    allowed = _allowed_algorithm_tuple(allowed_algorithms)
    if not rows:
        raise M1ContractError("cannot macro-average an empty validation bank")
    observed = {str(row["proposer_algorithm"]) for row in rows}
    if observed != set(allowed):
        missing = sorted(set(allowed) - observed)
        unexpected = sorted(observed - set(allowed))
        raise M1ContractError(
            f"validation algorithm set mismatch; missing={missing}, unexpected={unexpected}"
        )
    geometries = {str(row["geometry_id"]) for row in rows}
    if geometries != set(M1_TRAIN_GEOMETRIES):
        raise M1ContractError(
            f"validation geometry set mismatch; observed={sorted(geometries)}"
        )
    for geometry in M1_TRAIN_GEOMETRIES:
        for algorithm in allowed:
            if not any(
                str(row["geometry_id"]) == geometry
                and str(row["proposer_algorithm"]) == algorithm
                for row in rows
            ):
                raise M1ContractError(f"validation macro missing {geometry}/{algorithm}")
    return allowed


def _macro_mean(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    allowed_algorithms: Sequence[str] = ALGORITHMS,
) -> float:
    allowed = _validate_macro_rows(rows, allowed_algorithms)
    per_layout: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["geometry_id"]), str(row["proposer_algorithm"]), str(row["layout_id"]))
        per_layout.setdefault(key, []).append(float(row[field]))
    geom_algo: list[float] = []
    for geometry in M1_TRAIN_GEOMETRIES:
        for algorithm in allowed:
            layout_values = [
                float(np.mean(values))
                for (g, a, _layout), values in per_layout.items()
                if g == geometry and a == algorithm
            ]
            if not layout_values:
                raise M1ContractError(f"validation macro missing {geometry}/{algorithm}")
            geom_algo.append(float(np.mean(layout_values)))
    return float(np.mean(geom_algo))


def validation_macro_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_algorithms: Sequence[str] = ALGORITHMS,
) -> dict[str, Any]:
    """Macro raw gain/gap and their fractions of macro native mix4 cost."""
    allowed = _validate_macro_rows(rows, allowed_algorithms)
    gain = _macro_mean(rows, "gain_r_raw", allowed_algorithms=allowed)
    gap = _macro_mean(rows, "oracle_gap_r_raw", allowed_algorithms=allowed)
    native = _macro_mean(
        rows, "native_h20_mix4_r_raw", allowed_algorithms=allowed
    )
    if native <= 0:
        raise M1ContractError("validation macro native h20 R_raw denominator is zero")
    gain_fraction = gain / native
    gap_fraction = gap / native
    if gap <= 0:
        capture_ratio = None
    else:
        capture_ratio = gain / gap
    return {
        "macro_gain_r_raw": gain,
        "macro_native_h20_mix4_r_raw": native,
        "macro_oracle_gap_r_raw": gap,
        "macro_gain_fraction_of_native": gain_fraction,
        "macro_oracle_gap_fraction_of_native": gap_fraction,
        "prediction_capture_ratio_point": capture_ratio,
        "macro_order": (
            "layout mean within geometry x proposer; equal geometry and allowed-proposer macro mean"
        ),
    }


def layout_cluster_bootstrap_gain_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = 202610124799,
    allowed_algorithms: Sequence[str] = ALGORITHMS,
) -> dict[str, Any]:
    """Two-sided percentile 90% CI, resampling layouts within geometry."""
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    allowed = _validate_macro_rows(rows, allowed_algorithms)
    per_layout: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["geometry_id"]), str(row["proposer_algorithm"]), str(row["layout_id"]))
        per_layout.setdefault(key, []).append(float(row["gain_r_raw"]))
    layout_means = {key: float(np.mean(values)) for key, values in per_layout.items()}
    layout_ids = {
        geometry: sorted({layout for g, _algorithm, layout in layout_means if g == geometry})
        for geometry in M1_TRAIN_GEOMETRIES
    }
    for geometry in layout_ids:
        if not layout_ids[geometry]:
            raise M1ContractError(f"bootstrap has no layout clusters for {geometry}")
        for algorithm in allowed:
            algorithm_layouts = {
                layout
                for g, a, layout in layout_means
                if g == geometry and a == algorithm
            }
            if algorithm_layouts != set(layout_ids[geometry]):
                raise M1ContractError(
                    f"bootstrap has incomplete layout clusters for {geometry}/{algorithm}"
                )
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        macro: list[float] = []
        for geometry in M1_TRAIN_GEOMETRIES:
            selected_layouts = rng.choice(layout_ids[geometry], size=len(layout_ids[geometry]), replace=True)
            for algorithm in allowed:
                means = [
                    layout_means[(geometry, algorithm, str(layout))]
                    for layout in selected_layouts
                ]
                macro.append(float(np.mean(means)))
        samples[draw] = float(np.mean(macro))
    return {
        "draws": draws,
        "seed": seed,
        "resample_unit": "layout within geometry; retain all state/pool/stage/repeat cells",
        "percentiles": [5, 95],
        "lower_90pct": float(np.percentile(samples, 5)),
        "upper_90pct": float(np.percentile(samples, 95)),
        "estimand": "paired macro gain in mix4 R_raw; positive favors shared selector",
    }


def stage_rows_for_trajectory(captures: Sequence[StageCapture]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "stage": row.stage,
            "target_fraction": row.target_fraction,
            "decision_index": row.decision_index,
            "retrieval_count": row.retrieval_count,
            "actual_fraction": row.actual_fraction,
            "missing_reason": row.missing_reason,
            "duplicate_of": row.duplicate_of,
            "private_state_fingerprint": (
                None if row.snapshot is None else row.snapshot.private_state_fingerprint
            ),
        }
        for row in captures
    )


class GzipJsonlWriter:
    """Append JSONL rows through a deterministic gzip stream."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.path.open("wb")
        self._gzip = gzip.GzipFile(fileobj=self._raw, mode="wb", mtime=0)
        self._text = io.TextIOWrapper(self._gzip, encoding="utf-8", newline="\n")
        self.rows_written = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._text.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        self._text.write("\n")
        self.rows_written += 1

    def close(self) -> None:
        if not self._text.closed:
            self._text.flush()
            self._text.close()
        if not self._raw.closed:
            self._raw.close()

    def __enter__(self) -> "GzipJsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ScienceBudgetLedger:
    """Crash-accounted cumulative M1 science wall budget using append-only events."""

    def __init__(
        self,
        path: Path,
        *,
        cap_seconds: float = M1_SCIENCE_CAP_SECONDS,
        clock: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.path = Path(path)
        self.cap_seconds = float(cap_seconds)
        self.clock = clock
        self.monotonic = monotonic
        self.run_token: str | None = None
        self.started_utc: datetime | None = None
        self.started_monotonic: float | None = None
        if self.cap_seconds <= 0 or not math.isfinite(self.cap_seconds):
            raise ValueError("science cap must be positive and finite")

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        output = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise M1ContractError(f"science budget ledger is malformed at line {line_number}") from exc
            if not isinstance(row, dict):
                raise M1ContractError("science budget event must be a JSON object")
            output.append(row)
        return output

    def _accounted_seconds(self, *, now: datetime | None = None) -> tuple[float, bool]:
        current = now or self.clock()
        starts: dict[str, datetime] = {}
        closed: set[str] = set()
        total = 0.0
        for row in self._events():
            token = str(row.get("run_token", ""))
            try:
                timestamp = datetime.fromisoformat(str(row["timestamp_utc"]))
            except (KeyError, ValueError) as exc:
                raise M1ContractError("science budget timestamp is missing or invalid") from exc
            if timestamp.tzinfo is None:
                raise M1ContractError("science budget timestamps must include timezone")
            event = row.get("event")
            if event == "start":
                if not token or token in starts:
                    raise M1ContractError("science budget has duplicate/empty run token")
                starts[token] = timestamp
            elif event == "stop":
                if token not in starts or token in closed:
                    raise M1ContractError("science budget stop has no unique start")
                try:
                    duration = float(
                        row.get(
                            "process_monotonic_seconds",
                            (timestamp - starts[token]).total_seconds(),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise M1ContractError("science budget stop duration is invalid") from exc
                if not math.isfinite(duration) or duration < 0:
                    raise M1ContractError("science budget stop duration is invalid")
                total += duration
                closed.add(token)
            else:
                raise M1ContractError("science budget event must be start or stop")
        open_tokens = set(starts) - closed
        for token in open_tokens:
            if token == self.run_token and self.started_monotonic is not None:
                total += max(0.0, self.monotonic() - self.started_monotonic)
            else:
                total += max(0.0, (current - starts[token]).total_seconds())
        return total, bool(open_tokens)

    def start(self, *, phase: str = "M1_DIAGNOSTIC_BANK") -> None:
        total, has_open = self._accounted_seconds()
        if has_open:
            raise M1ContractError("an earlier M1 science run has no stop event; inspect its partial output")
        if total >= self.cap_seconds:
            raise M1BudgetExceeded("cumulative M1 science wall has reached its 8-hour cap")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_token = uuid.uuid4().hex
        self.started_utc = self.clock()
        self.started_monotonic = self.monotonic()
        self._append(
            {
                "event": "start",
                "run_token": self.run_token,
                "timestamp_utc": self.started_utc.isoformat(),
                "phase": phase,
                "prior_science_wall_seconds": total,
            }
        )

    def check(self) -> None:
        if self.run_token is None or self.started_monotonic is None:
            raise M1ContractError("M1 science budget was not started")
        cumulative, _open = self._accounted_seconds()
        current_elapsed = max(0.0, self.monotonic() - self.started_monotonic)
        if cumulative >= self.cap_seconds or current_elapsed >= self.cap_seconds:
            raise M1BudgetExceeded("cumulative M1 science wall exceeded the 8-hour cap")

    def stop(self, *, status: str, exit_code: int) -> dict[str, Any]:
        if self.run_token is None or self.started_utc is None or self.started_monotonic is None:
            raise M1ContractError("M1 science budget has no active run")
        ended = self.clock()
        current_elapsed = max(0.0, self.monotonic() - self.started_monotonic)
        self._append(
            {
                "event": "stop",
                "run_token": self.run_token,
                "timestamp_utc": ended.isoformat(),
                "status": status,
                "exit_code": int(exit_code),
                "process_monotonic_seconds": current_elapsed,
            }
        )
        total, has_open = self._accounted_seconds()
        if has_open:
            raise M1ContractError("science budget stop did not close its run interval")
        if total > self.cap_seconds:
            status = "STOP_M1_SCIENCE_WALL_CAP_EXCEEDED"
        self.run_token = None
        return {
            "status": status,
            "science_wall_seconds": current_elapsed,
            "cumulative_science_wall_seconds": total,
            "cap_seconds": self.cap_seconds,
            "started_utc": self.started_utc.isoformat(),
            "finished_utc": ended.isoformat(),
        }

    def _append(self, row: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())


__all__ = (
    "ALGORITHMS",
    "CandidateTrainingSample",
    "ConditionAssignment",
    "DecisionSnapshot",
    "EGNativeActor",
    "EMNativeActor",
    "ERINativeActor",
    "GPNativeActor",
    "GzipJsonlWriter",
    "M1BudgetExceeded",
    "M1_BANK_SCOPE",
    "M1ContractError",
    "M1_CONDITIONS",
    "M1_SCIENCE_CAP_SECONDS",
    "M1_SMOKE_SCOPE",
    "M1_STAGE_TARGETS",
    "NearestStageSampler",
    "PairwiseTrainingRow",
    "PROTOCOL_ID",
    "RUN_ID",
    "ScienceBudgetLedger",
    "StageCapture",
    "ValidationCandidate",
    "ValidationPool",
    "build_pairwise_training_rows",
    "collect_behavior_stages",
    "condition_assignments",
    "evaluate_validation_pools",
    "layout_cluster_bootstrap_gain_ci",
    "make_continuation_actor_factories",
    "make_forecast_context",
    "make_native_actor",
    "mix3_loao_target_r_over_n",
    "mix4_training_target_r_over_n",
    "propose_native_candidate_pools",
    "semantic_state_fingerprint",
    "stage_rows_for_trajectory",
    "validation_macro_summary",
    "validation_pools_from_rows",
    "verify_m1_release",
)
