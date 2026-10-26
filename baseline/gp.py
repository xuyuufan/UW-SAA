"""Public-frame adapter for the vendored CrpOnlinellm GP terminals.

The source engine and formula compiler are loaded lazily so non-GP policies do
not need the vendored package or an active GP model.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable, Sequence

SOURCE_TERMINALS = ("SH", "EMP", "AVG", "CUR", "RI", "DIFF")
PUBLIC_UNKNOWN_RANK = 22


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "crp_online" / "src"


def _load_source_modules():
    root = _source_root()
    if not (root / "crp" / "gp" / "expression.py").is_file():
        raise RuntimeError(f"vendored GP source is missing under {root}")
    root_text = str(root)
    existing = sys.modules.get("crp")
    existing_path = getattr(existing, "__file__", None)
    if existing_path is not None and Path(existing_path).resolve().parent != (root / "crp").resolve():
        raise RuntimeError("a different 'crp' package was imported before the vendored GP source")
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from crp.gp.expression import compile_formula
    from crp.terminals import compute_terminals
    return compile_formula, compute_terminals


def bootstrap_source() -> None:
    """Add and load the byte-preserved source package before importing its engine."""
    _load_source_modules()


def compile_priority(formula: str) -> Callable[..., float]:
    """Compile the source prefix expression using its byte-preserved compiler."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("GP formula must be a non-empty source prefix expression")
    compile_formula, _compute_terminals = _load_source_modules()
    return compile_formula(formula)


class _PublicYard:
    """The small Yard surface consumed by source compute_terminals."""

    def __init__(self, stacks: Sequence[Sequence[int]], max_height: int, horizon: int) -> None:
        self.stacks = [list(map(int, stack)) for stack in stacks]
        self.max_height = int(max_height)
        remaining = sum(len(stack) for stack in self.stacks)
        # Relative ranks 1..h+1 are visible; Maximum is one above the last
        # public rank, capped near the end of an episode.
        self.maximum = min(remaining, int(horizon) + 1) + 1

    def value(self, container: int) -> int:
        value = int(container)
        if value == PUBLIC_UNKNOWN_RANK:
            return self.maximum
        if value < 1 or value > PUBLIC_UNKNOWN_RANK:
            raise ValueError(f"invalid occupied public rank: {value}")
        return value

    def top_value(self, stack_index: int) -> int:
        stack = self.stacks[int(stack_index)]
        if not stack:
            raise ValueError("cannot read the top of an empty stack")
        return self.value(stack[-1])


def _public_yard(frame: object, horizon: int) -> tuple[_PublicYard, int]:
    from environment.public_frame import public_stacks, require_public_frame

    require_public_frame(frame)
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) not in (0, 2, 5, 10, 20):
        raise ValueError("GP horizon must be one of 0, 2, 5, 10, or 20")
    stacks = public_stacks(frame)
    remaining = sum(len(stack) for stack in stacks)
    if remaining <= 0:
        raise ValueError("GP cannot score a terminal public frame")
    visible = {
        int(value)
        for stack in stacks
        for value in stack
        if int(value) != PUBLIC_UNKNOWN_RANK
    }
    expected_visible = set(range(1, min(remaining, int(horizon) + 1) + 1))
    if visible != expected_visible:
        raise ValueError("public known ranks do not match the explicit horizon prefix")
    source_candidates = [
        index for index, stack in enumerate(stacks) if 1 in stack
    ]
    if len(source_candidates) != 1:
        raise ValueError("public frame must contain exactly one current target rank")
    source = source_candidates[0]
    return _PublicYard(stacks, int(frame.max_tiers), int(horizon)), source


def select_destination(
    priority: Callable[..., float],
    frame: object,
    *,
    horizon: int,
) -> int:
    """Choose the lowest source-terminal priority from a censored public frame."""
    _compile_formula, compute_terminals = _load_source_modules()
    yard, source = _public_yard(frame, int(horizon))
    if not yard.stacks[source] or yard.stacks[source][-1] == 1:
        raise ValueError("GP selection requires a blocked current target")
    legal = tuple(int(value) for value in frame.legal_destinations)
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("GP selection requires unique legal destinations")
    if source in legal or any(value < 0 or value >= len(yard.stacks) for value in legal):
        raise ValueError("GP legal destinations include an invalid/source stack")

    best_destination: int | None = None
    best_priority = math.inf
    for destination in legal:
        if len(yard.stacks[destination]) >= yard.max_height:
            raise ValueError("GP legal destination is physically full")
        terminals = compute_terminals(yard, source, destination)
        score = float(priority(*(float(terminals[name]) for name in SOURCE_TERMINALS)))
        if not math.isfinite(score):
            raise ValueError("source GP priority returned a non-finite score")
        if score < best_priority or (
            score == best_priority and (best_destination is None or destination < best_destination)
        ):
            best_priority = score
            best_destination = destination
    if best_destination is None:
        raise ValueError("GP did not find a legal destination")
    return int(best_destination)

IMPLEMENTATION_FINGERPRINT_FILES = (
    "baseline/gp.py",
    "baseline/native.py",
    "uwsaa/entrypoint.py",
    "environment/public_frame.py",
    "environment/noise_aware_forecast.py",
    "environment/online_crp.py",
    "experiments/gp_h20/fitness.py",
    "experiments/gp_h20/train.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def current_fingerprints() -> dict[str, str]:
    """Verify the vendored snapshot and return inputs bound to a trained rule."""
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "third_party" / "crp_online" / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("vendored CrpOnlinellm source manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_manifest_sha = manifest.get("manifest_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    calculated_manifest_sha = _sha256_bytes(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if recorded_manifest_sha != calculated_manifest_sha:
        raise ValueError("vendored source manifest hash is invalid")
    for record in manifest.get("files", []):
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("vendored source manifest contains an unsafe path")
        path = root / "third_party" / "crp_online" / relative
        if not path.is_file():
            raise ValueError(f"vendored source file is missing: {relative.as_posix()}")
        data = path.read_bytes()
        if len(data) != int(record["size"]) or _sha256_bytes(data) != record["sha256"]:
            raise ValueError(f"vendored source file changed: {relative.as_posix()}")

    protocol_path = root / "experiments" / "gp_h20" / "protocol.json"
    bank_path = root / "experiments" / "gp_h20" / "data" / "train_bank.json"
    protocol_bytes = protocol_path.read_bytes()
    bank_sha = _sha256_bytes(bank_path.read_bytes())
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    registered_bank_sha = protocol.get("training_bank", {}).get("sha256")
    if registered_bank_sha != bank_sha:
        raise ValueError("training bank SHA-256 differs from the frozen GP protocol")
    protocol_sha = _sha256_bytes(protocol_bytes)
    implementation_records = []
    for name in IMPLEMENTATION_FINGERPRINT_FILES:
        path = root / Path(name)
        if not path.is_file():
            raise RuntimeError(f"GP implementation fingerprint input is missing: {name}")
        implementation_records.append({"path": name, "sha256": _sha256_bytes(path.read_bytes())})
    implementation_sha = _sha256_bytes(
        json.dumps(implementation_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {
        "source_manifest_sha256": calculated_manifest_sha,
        "protocol_sha256": protocol_sha,
        "train_bank_sha256": bank_sha,
        "implementation_sha256": implementation_sha,
    }
