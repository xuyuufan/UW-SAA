"""AMCTS/LSGF core from Zhang et al. (2025) with a public H20 adapter.

The tree accepts a general container RPM. The public adapter uses point-mass
groups for visible ranks and one anonymous tail group; it does not implement
the paper's operational-feature MLP/RPM prediction stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from environment.noise_aware_forecast import HORIZONS
from environment.public_frame import public_prefix, require_public_frame

Stacks = tuple[tuple[int, ...], ...]
Coordinate = tuple[int, int]
RPM = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class H20Groups:
    """Public H20 layout encoded as item IDs plus point-mass RPM rows."""

    stacks: Stacks
    max_tiers: int
    horizon: int
    tail_group: int
    rpm: RPM
    period_groups: tuple[int, ...]

    @property
    def item_count(self) -> int:
        return sum(map(len, self.stacks))


def h20_groups_from_frame(frame: object, horizon: int = 20) -> H20Groups:
    """Map visible current/future ranks to ordered groups and censor the tail."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    require_public_frame(frame)
    prefix = public_prefix(frame, horizon)
    ranks = np.asarray(prefix.observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(prefix.observation.occupied_mask, dtype=np.bool_)
    tail_group = int(horizon) + 2
    stacks: list[tuple[int, ...]] = []
    period_groups: list[int] = []
    for stack_id in range(ranks.shape[0]):
        height = int(occupied[stack_id].sum())
        values = []
        for tier in range(height):
            rank = int(ranks[stack_id, tier])
            if rank == 22:
                period = tail_group
            elif 1 <= rank <= horizon + 1:
                period = rank
            else:
                raise ValueError("public rank is outside the H20 grouping contract")
            values.append(len(period_groups))
            period_groups.append(period)
        stacks.append(tuple(values))
    current_ids = [item for item, period in enumerate(period_groups) if period == 1]
    if len(current_ids) != 1:
        raise ValueError("H20 adapter requires exactly one current target in group 1")
    return H20Groups(
        tuple(stacks), int(frame.max_tiers), int(horizon), tail_group,
        _point_mass_rpm(period_groups), tuple(period_groups),
    )


def pairwise_before(rpm: Sequence[Sequence[float]], i: int, j: int) -> float:
    """Exact two-item probability P(i precedes j), with same-period ties at .5."""
    if i == j:
        raise ValueError("pairwise retrieval probability requires distinct items")
    pi = np.asarray(rpm[i], dtype=np.float64)
    pj = np.asarray(rpm[j], dtype=np.float64)
    if pi.ndim != 1 or pj.shape != pi.shape:
        raise ValueError("RPM rows must be one-dimensional and equally sized")
    if np.any(pi < 0) or np.any(pj < 0) or not np.isclose(pi.sum(), 1.0) or not np.isclose(pj.sum(), 1.0):
        raise ValueError("each RPM row must be a probability distribution")
    less = 0.0
    ties = 0.0
    for mi, pmi in enumerate(pi):
        less += float(pmi) * float(pj[mi + 1 :].sum())
        ties += float(pmi) * float(pj[mi])
    return min(1.0, max(0.0, less + 0.5 * ties))


@dataclass(frozen=True)
class _PairwiseCache:
    """Per-search cached A/log(A) matrices (the paper's Sec. 5.1 structure)."""

    probabilities: tuple[tuple[float, ...], ...]
    log_probabilities: tuple[tuple[float, ...], ...]

    @classmethod
    def from_rpm(cls, rpm: Sequence[Sequence[float]]) -> "_PairwiseCache":
        count = len(rpm)
        probabilities = [[0.0] * count for _ in range(count)]
        log_probabilities = [[-math.inf] * count for _ in range(count)]
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                # Preserve the original ordered helper calculation exactly;
                # in particular, do not derive A[j,i] by subtraction.
                probability = pairwise_before(rpm, i, j)
                probabilities[i][j] = probability
                if probability > 0.0:
                    log_probabilities[i][j] = math.log(probability)
        return cls(
            tuple(tuple(row) for row in probabilities),
            tuple(tuple(row) for row in log_probabilities),
        )


def approximate_next_probabilities(
    rpm: Sequence[Sequence[float]], items: Sequence[int], *,
    pairwise_cache: _PairwiseCache | None = None,
) -> dict[int, float]:
    """Zhang et al. Eqs. (7)-(8), evaluated stably in log space."""
    item_ids = tuple(int(item) for item in items)
    if not item_ids:
        return {}
    log_weights: list[float] = []
    for i in item_ids:
        log_weight = 0.0
        impossible = False
        for j in item_ids:
            if i == j:
                continue
            probability = (
                pairwise_cache.probabilities[i][j]
                if pairwise_cache is not None else pairwise_before(rpm, i, j)
            )
            if probability == 0.0:
                impossible = True
                break
            log_weight += (
                pairwise_cache.log_probabilities[i][j]
                if pairwise_cache is not None else math.log(probability)
            )
        log_weights.append(-math.inf if impossible else log_weight)
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        # Eq. (8) has a zero denominator for cyclic/degenerate pairwise inputs.
        # Uniform fallback is a declared reconstruction choice for that edge.
        weight = 1.0 / len(item_ids)
        return {item: weight for item in item_ids}
    maximum = max(finite)
    weights = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in log_weights]
    denominator = math.fsum(weights)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ArithmeticError("Eq. (8) probability normalization failed")
    return {item: weight / denominator for item, weight in zip(item_ids, weights)}


def _point_mass_rpm(groups: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    periods = max(groups, default=0)
    if periods <= 0:
        return ()
    rows = []
    for group in groups:
        row = [0.0] * periods
        row[int(group) - 1] = 1.0
        rows.append(tuple(row))
    return tuple(rows)


def _locations(stacks: Stacks) -> tuple[Coordinate, ...]:
    return tuple((s, t) for s, stack in enumerate(stacks) for t in range(len(stack)))


def _items_in(stacks: Stacks) -> tuple[int, ...]:
    return tuple(item for stack in stacks for item in stack)


def _expected_period(rpm: Sequence[Sequence[float]], item: int) -> float:
    return math.fsum((period + 1) * float(probability) for period, probability in enumerate(rpm[item]))


def next_group_probabilities(
    stacks: Stacks, rpm: Sequence[Sequence[float]], *,
    pairwise_cache: _PairwiseCache | None = None,
) -> tuple[tuple[Coordinate, float], ...]:
    """Eq. (7)-(8) approximate next-target probabilities for a general RPM."""
    locations = _locations(stacks)
    if not locations:
        return ()
    item_ids = tuple(stacks[s][t] for s, t in locations)
    item_probabilities = approximate_next_probabilities(rpm, item_ids, pairwise_cache=pairwise_cache)
    return tuple(
        (location, item_probabilities[stacks[location[0]][location[1]]])
        for location in locations
    )


def _legal_destinations(stacks: Stacks, source: int, max_tiers: int) -> tuple[int, ...]:
    return tuple(index for index, stack in enumerate(stacks) if index != source and len(stack) < max_tiers)


def lsgf_score(
    stacks: Stacks,
    blocker_item: int,
    destination: int,
    rpm: Sequence[Sequence[float]],
    *,
    alpha: float = 0.05,
    pairwise_cache: _PairwiseCache | None = None,
) -> float:
    """LSGF Eq. (9): next-target probability plus alpha times squared stack size."""
    if alpha != 0.05:
        raise ValueError("the paper-fixed LSGF alpha is 0.05")
    candidate_items = tuple(stacks[destination]) + (int(blocker_item),)
    q = approximate_next_probabilities(rpm, candidate_items, pairwise_cache=pairwise_cache)
    blocker_probability = float(q[int(blocker_item)])
    return blocker_probability + 0.05 * len(stacks[destination]) ** 2


def lsgf_destination(
    stacks: Stacks,
    source: int,
    blocker_item: int,
    destinations: Sequence[int],
    rpm: Sequence[Sequence[float]],
    *,
    alpha: float = 0.05,
    pairwise_cache: _PairwiseCache | None = None,
) -> int:
    """Choose the legal destination with maximum LSGF, then leftmost on ties."""
    legal = tuple(int(value) for value in destinations)
    if not legal:
        raise ValueError("LSGF needs at least one legal destination")
    return max(
        legal,
        key=lambda dest: (
            lsgf_score(stacks, blocker_item, dest, rpm, alpha=alpha, pairwise_cache=pairwise_cache),
            -dest,
        ),
    )


def _move(stacks: Stacks, source: int, destination: int) -> Stacks:
    mutable = [list(stack) for stack in stacks]
    if source == destination or not mutable[source]:
        raise ValueError("invalid relocation source/destination")
    mutable[destination].append(mutable[source].pop())
    return tuple(tuple(stack) for stack in mutable)


def _pop_target(stacks: Stacks, target: Coordinate) -> Stacks:
    source, tier = target
    mutable = [list(stack) for stack in stacks]
    if tier != len(mutable[source]) - 1:
        raise ValueError("only an exposed target can be retrieved")
    mutable[source].pop()
    return tuple(tuple(stack) for stack in mutable)


def _find_item(stacks: Stacks, item_id: int) -> Coordinate:
    found = [(s, t) for s, stack in enumerate(stacks) for t, value in enumerate(stack) if value == item_id]
    if len(found) != 1:
        raise ValueError(f"item {item_id} must occur exactly once")
    return found[0]


def _greedy_lsgf_cost(
    stacks: Stacks,
    max_tiers: int,
    rpm: Sequence[Sequence[float]],
    first_target: Coordinate | None = None,
    *, pairwise_cache: _PairwiseCache | None = None,
) -> int:
    """LSGF rollout, optionally committing the node's selected target first.

    A decision node is conditioned on a particular next target. The rollout
    must clear and retrieve that target before applying expected-time order
    to later targets.
    """
    bay = stacks
    moves = 0

    def clear_and_retrieve(target: Coordinate) -> None:
        nonlocal bay, moves
        source, tier = target
        if source < 0 or source >= len(bay) or tier < 0 or tier >= len(bay[source]):
            raise ValueError("target coordinate is outside the bay")
        while tier != len(bay[source]) - 1:
            blocker = bay[source][-1]
            legal = _legal_destinations(bay, source, max_tiers)
            destination = lsgf_destination(bay, source, blocker, legal, rpm, pairwise_cache=pairwise_cache)
            bay = _move(bay, source, destination)
            moves += 1
        bay = _pop_target(bay, target)

    if first_target is not None:
        clear_and_retrieve(first_target)
    while any(bay):
        locations = _locations(bay)
        target = min(
            locations,
            key=lambda pos: (_expected_period(rpm, bay[pos[0]][pos[1]]), pos),
        )
        clear_and_retrieve(target)
    return moves


@dataclass
class _Node:
    kind: str
    stacks: Stacks
    target: Coordinate | None = None
    probability: float = 1.0
    parent: "_Node | None" = None
    action: tuple[int, int] | None = None
    children: list["_Node"] = field(default_factory=list)
    unexpanded: list[int] = field(default_factory=list)
    cost: float = 0.0
    visits: int = 0
    expanded: bool = False


@dataclass
class _TreeReuseState:
    """One caller-namespaced selected-action subtree eligible for reuse."""

    anchor: _Node
    previous_root_stacks: Stacks
    max_tiers: int
    rpm: RPM
    pairwise_cache: _PairwiseCache
    namespace: str
    playout_multiplier: int
    budget: int
    cpuct: float
    lsgf_alpha: float


def select_decision_child(node_visits: int, children: Sequence[_Node], cpuct: float = 5.0) -> _Node:
    """Eq. (10)-(11) PUCT selection for cost minimization."""
    if not children:
        raise ValueError("decision node has no expanded children")
    return min(
        children,
        key=lambda child: (
            child.cost - cpuct * math.sqrt(max(0, node_visits)) / (child.visits + 1),
            child.action or (-1, -1),
        ),
    )


def select_chance_child(children: Sequence[_Node]) -> _Node:
    """Eq. (12): visit count divided by transition probability."""
    positive = [child for child in children if child.probability > 0.0]
    if not positive:
        raise ValueError("chance node has no positive-probability child")
    return min(positive, key=lambda child: (child.visits / child.probability, child.target or (-1, -1)))


def update_chance_cost(children: Sequence[_Node]) -> float:
    """Eq. (6) chance-node backpropagation: probability-weighted sum."""
    if not children:
        return 0.0
    total_probability = math.fsum(child.probability for child in children)
    if not math.isclose(total_probability, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("chance child probabilities must sum to one")
    return math.fsum(child.probability * child.cost for child in children)


def update_decision_cost(children: Sequence[_Node]) -> float:
    """Eq. (6) decision-node backpropagation: one move plus minimum child cost."""
    if not children:
        return 0.0
    return 1.0 + min(child.cost for child in children)


def _visit_trace(root: _Node) -> list[dict[str, object]]:
    """Compact deterministic node/edge counters for correctness smoke checks."""
    rows: list[dict[str, object]] = []
    pending: list[tuple[tuple[int, ...], _Node]] = [((), root)]
    while pending:
        path, node = pending.pop()
        rows.append({
            "path": path,
            "kind": node.kind,
            "action": node.action,
            "target": node.target,
            "probability": node.probability,
            "cost": node.cost,
            "visits": node.visits,
            "children": len(node.children),
        })
        for index in reversed(range(len(node.children))):
            pending.append((path + (index,), node.children[index]))
    return rows


class AMCTSPolicy:
    """AMCTS with optional exact-state reuse for stable-ID generic RPM calls.

    The public H20 adapter deliberately omits a stable-ID namespace and thus
    always rebuilds. Generic callers may opt into reuse only when IDs and RPM
    semantics remain stable across observations.
    """

    def __init__(self, *, playout_multiplier: int = 2500) -> None:
        if playout_multiplier <= 0:
            raise ValueError("playout_multiplier must be positive")
        self.playout_multiplier = int(playout_multiplier)
        self._tree_reuse: _TreeReuseState | None = None

    @staticmethod
    def _matching_decision_node(
        anchor: _Node, stacks: Stacks, target: Coordinate,
    ) -> _Node | None:
        """Find the nearest exact state/target only inside the saved branch."""
        pending = [anchor]
        cursor = 0
        while cursor < len(pending):
            node = pending[cursor]
            cursor += 1
            if node.kind == "decision" and node.stacks == stacks and node.target == target:
                return node
            pending.extend(node.children)
        return None

    def _reuse_root(
        self,
        stacks: Stacks,
        target: Coordinate,
        max_tiers: int,
        rpm: RPM,
        namespace: str,
        budget: int,
    ) -> tuple[_Node | None, _PairwiseCache | None, str]:
        cached = self._tree_reuse
        if cached is None:
            return None, None, "no_previous_tree"
        compatible = (
            cached.namespace == namespace
            and cached.max_tiers == max_tiers
            and len(cached.anchor.stacks) == len(stacks)
            and cached.rpm == rpm
            and cached.playout_multiplier == self.playout_multiplier
            and cached.budget == budget
            and cached.cpuct == 5.0
            and cached.lsgf_alpha == 0.05
        )
        if not compatible:
            self._tree_reuse = None
            return None, None, "incompatible_namespace_rpm_geometry_or_parameters"
        if cached.previous_root_stacks == stacks:
            self._tree_reuse = None
            return None, None, "unchanged_previous_root"
        candidate = self._matching_decision_node(cached.anchor, stacks, target)
        if candidate is None:
            self._tree_reuse = None
            return None, None, "no_exact_descendant_in_selected_branch"
        candidate.parent = None
        candidate.action = None
        candidate.probability = 1.0
        self._tree_reuse = None
        return candidate, cached.pairwise_cache, "reused_exact_descendant"

    def select_destination(self, frame: object, horizon: int = 20, *, playouts: int | None = None) -> tuple[int, dict[str, object]]:
        groups = h20_groups_from_frame(frame, horizon)
        target_item = groups.period_groups.index(1)
        destination, details = self.select_state(
            groups.stacks,
            groups.rpm,
            groups.max_tiers,
            target_item,
            playouts=playouts,
        )
        if destination not in tuple(int(x) for x in frame.legal_destinations):
            raise AssertionError("AMCTS selected an illegal public destination")
        details.update({
            "method": "ZHANG_AMCTS_LSGF_H20_ADAPTED",
            "horizon": groups.horizon,
            "adaptation": "visible ranks are point-mass RPM rows; anonymous tail is one shared point-mass period",
        })
        return destination, details

    def select_state(
        self,
        stacks: Stacks,
        rpm: Sequence[Sequence[float]],
        max_tiers: int,
        target_item: int,
        *,
        playouts: int | None = None,
        include_visit_trace: bool = False,
        reuse_namespace: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        """Run the generic-RPM core; reuse requires stable IDs and an explicit namespace."""
        stacks = tuple(tuple(int(item) for item in stack) for stack in stacks)
        all_items = _items_in(stacks)
        matrix = np.asarray(rpm, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("RPM must be a nonempty item-by-period matrix")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError("RPM entries must be finite and nonnegative")
        if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("every RPM row must sum to one")
        if len(set(all_items)) != len(all_items):
            raise ValueError("each item ID must occur at most once in the bay")
        if any(item < 0 or item >= matrix.shape[0] for item in all_items):
            raise ValueError("bay item ID has no corresponding RPM row")
        target = _find_item(stacks, int(target_item))
        source, tier = target
        if tier == len(stacks[source]) - 1:
            raise ValueError("AMCTS is only called when the current target is blocked")
        legal = _legal_destinations(stacks, source, int(max_tiers))
        if not legal:
            raise ValueError("AMCTS current target has no legal destination")
        budget = int(playouts if playouts is not None else self.playout_multiplier * len(stacks))
        if budget <= 0:
            raise ValueError("playouts must be positive")
        rpm_tuple = tuple(tuple(float(x) for x in row) for row in matrix)
        reusable_call = reuse_namespace is not None and playouts is None
        if reuse_namespace is not None and (not isinstance(reuse_namespace, str) or not reuse_namespace.strip()):
            raise ValueError("reuse_namespace must be a nonempty stable-ID namespace")
        reuse_status = "disabled_no_stable_namespace"
        root: _Node | None = None
        pairwise_cache: _PairwiseCache | None = None
        if reusable_call:
            root, pairwise_cache, reuse_status = self._reuse_root(
                stacks, target, int(max_tiers), rpm_tuple,
                str(reuse_namespace), budget,
            )
        else:
            self._tree_reuse = None
            if reuse_namespace is not None:
                reuse_status = "disabled_explicit_playout_override"
        reused = root is not None
        if root is None:
            root = _Node("decision", stacks, target=target)
            root.unexpanded = list(legal)
        if pairwise_cache is None:
            pairwise_cache = _PairwiseCache.from_rpm(matrix)
        root_visits_before = root.visits
        self._expand_tree(
            root, int(max_tiers), rpm_tuple, budget,
            pairwise_cache,
        )
        if not root.children:
            raise RuntimeError("AMCTS failed to expand a legal root action")
        chosen = min(root.children, key=lambda child: (child.cost, child.action[1] if child.action else -1))
        destination = int(chosen.action[1])
        if destination not in legal:
            raise AssertionError("AMCTS selected an illegal destination")
        if reusable_call:
            anchor = chosen
            anchor.parent = None
            self._tree_reuse = _TreeReuseState(
                anchor=anchor,
                previous_root_stacks=stacks,
                max_tiers=int(max_tiers),
                rpm=rpm_tuple,
                pairwise_cache=pairwise_cache,
                namespace=str(reuse_namespace),
                playout_multiplier=self.playout_multiplier,
                budget=budget,
                cpuct=5.0,
                lsgf_alpha=0.05,
            )
        details: dict[str, object] = {
            "method": "ZHANG_AMCTS_LSGF_GENERAL_RPM_CORE",
            "playouts": budget,
            "playout_budget_rule": "2500 * number_of_stacks" if playouts is None else "explicit correctness-test override",
            "cpuct": 5.0,
            "lsgf_alpha": 0.05,
            "selected_root_cost": float(chosen.cost),
            "root_cost_includes_selected_relocation": False,
            "probability_stage": "Eq. (7)-(8) pairwise approximation on supplied general RPM",
            "tree_reuse": reused,
            "tree_reuse_status": reuse_status,
            "root_visits_before": root_visits_before,
            "root_visits_after": root.visits,
            "playouts_added": budget,
            "reuse_root_parent_detached": root.parent is None,
        }
        if include_visit_trace:
            details["tree_visit_trace"] = _visit_trace(root)
        return destination, details

    def _as_node(
        self,
        stacks: Stacks,
        target: Coordinate | None,
        parent: _Node,
        action: tuple[int, int] | None,
        max_tiers: int,
        rpm: Sequence[Sequence[float]],
        pairwise_cache: _PairwiseCache,
        probability: float = 1.0,
    ) -> _Node:
        if target is None:
            if not any(stacks):
                return _Node("terminal", stacks, parent=parent, action=action, probability=probability, cost=0.0, expanded=True)
            return _Node("chance", stacks, parent=parent, action=action, probability=probability)
        source, tier = target
        if tier == len(stacks[source]) - 1:
            return _Node("retrieval", stacks, target=target, parent=parent, action=action, probability=probability)
        node = _Node("decision", stacks, target=target, parent=parent, action=action, probability=probability)
        node.unexpanded = list(_legal_destinations(stacks, source, max_tiers))
        return node

    def _heuristic_chance_cost(
        self, stacks: Stacks, max_tiers: int, rpm: Sequence[Sequence[float]], pairwise_cache: _PairwiseCache
    ) -> float:
        probabilities = next_group_probabilities(stacks, rpm, pairwise_cache=pairwise_cache)
        total = 0.0
        for target, probability in probabilities:
            if probability <= 0.0:
                continue
            remaining = _pop_target(stacks, target) if target[1] == len(stacks[target[0]]) - 1 else stacks
            target_after = None if remaining != stacks else target
            if target_after is None:
                value = _greedy_lsgf_cost(remaining, max_tiers, rpm, pairwise_cache=pairwise_cache)
            else:
                value = self._heuristic_decision_cost(remaining, target_after, max_tiers, rpm, pairwise_cache)
            total += probability * value
        return total

    def _heuristic_decision_cost(
        self, stacks: Stacks, target: Coordinate, max_tiers: int, rpm: Sequence[Sequence[float]],
        pairwise_cache: _PairwiseCache,
    ) -> float:
        return float(_greedy_lsgf_cost(
            stacks, max_tiers, rpm, first_target=target, pairwise_cache=pairwise_cache,
        ))

    def _expand_one_decision(
        self, node: _Node, max_tiers: int, rpm: Sequence[Sequence[float]], pairwise_cache: _PairwiseCache
    ) -> _Node | None:
        if not node.unexpanded:
            node.expanded = True
            return None
        source, _tier = node.target  # type: ignore[misc]
        destination = node.unexpanded.pop(0)
        child_stacks = _move(node.stacks, source, destination)
        target = node.target
        child = self._as_node(child_stacks, target, node, (source, destination), max_tiers, rpm, pairwise_cache)
        child.cost = self._heuristic_node_cost(child, max_tiers, rpm, pairwise_cache)
        node.children.append(child)
        if not node.unexpanded:
            node.expanded = True
        return child

    def _heuristic_node_cost(
        self, node: _Node, max_tiers: int, rpm: Sequence[Sequence[float]], pairwise_cache: _PairwiseCache
    ) -> float:
        if node.kind == "terminal":
            return 0.0
        if node.kind == "retrieval":
            next_stacks = _pop_target(node.stacks, node.target)  # type: ignore[arg-type]
            return self._heuristic_chance_cost(next_stacks, max_tiers, rpm, pairwise_cache)
        if node.kind == "chance":
            return self._heuristic_chance_cost(node.stacks, max_tiers, rpm, pairwise_cache)
        return self._heuristic_decision_cost(node.stacks, node.target, max_tiers, rpm, pairwise_cache)  # type: ignore[arg-type]

    def _expand_retrieval(
        self, node: _Node, max_tiers: int, rpm: Sequence[Sequence[float]], pairwise_cache: _PairwiseCache
    ) -> None:
        if node.expanded:
            return
        next_stacks = _pop_target(node.stacks, node.target)  # type: ignore[arg-type]
        child = self._as_node(next_stacks, None, node, None, max_tiers, rpm, pairwise_cache)
        node.children.append(child)
        node.expanded = True
        if child.kind == "chance":
            self._expand_chance(child, max_tiers, rpm, pairwise_cache)
        node.cost = child.cost

    def _expand_chance(
        self, node: _Node, max_tiers: int, rpm: Sequence[Sequence[float]], pairwise_cache: _PairwiseCache
    ) -> None:
        if node.expanded:
            return
        outcomes = next_group_probabilities(node.stacks, rpm, pairwise_cache=pairwise_cache)
        positive = [(target, probability) for target, probability in outcomes if probability > 0.0]
        if not positive:
            raise RuntimeError("AMCTS chance node has no positive-probability target")
        for target, probability in positive:
            child = self._as_node(node.stacks, target, node, None, max_tiers, rpm, pairwise_cache, probability)
            child.cost = self._heuristic_node_cost(child, max_tiers, rpm, pairwise_cache)
            node.children.append(child)
        node.expanded = True
        node.cost = update_chance_cost(node.children)

    def _backpropagate(self, path: Sequence[_Node]) -> None:
        for node in reversed(path):
            node.visits += 1
            if node.kind == "chance" and node.expanded:
                node.cost = update_chance_cost(node.children)
            elif node.kind == "decision" and node.children:
                node.cost = update_decision_cost(node.children)
            elif node.kind == "retrieval" and node.children:
                node.cost = node.children[0].cost

    def _expand_tree(
        self,
        root: _Node,
        max_tiers: int,
        rpm: Sequence[Sequence[float]],
        budget: int,
        pairwise_cache: _PairwiseCache,
    ) -> None:
        for _ in range(budget):
            node = root
            path = [root]
            while node.expanded and node.kind != "terminal":
                if node.kind == "decision":
                    node = select_decision_child(node.visits, node.children)
                elif node.kind == "chance":
                    node = select_chance_child(node.children)
                elif node.kind == "retrieval":
                    node = node.children[0]
                else:
                    break
                path.append(node)
            if node.kind == "chance" and not node.expanded:
                self._expand_chance(node, max_tiers, rpm, pairwise_cache)
            elif node.kind == "decision" and not node.expanded:
                child = self._expand_one_decision(node, max_tiers, rpm, pairwise_cache)
                if child is not None:
                    path.append(child)
            elif node.kind == "retrieval" and not node.expanded:
                self._expand_retrieval(node, max_tiers, rpm, pairwise_cache)
                path.append(node.children[0])
            self._backpropagate(path)


class ZhangPublicActor:
    """Online actor wrapper; retrieval sequence and noise metadata are absent."""

    def __init__(self, *, playout_multiplier: int = 2500) -> None:
        self.search = AMCTSPolicy(playout_multiplier=playout_multiplier)

    def select(self, frame: object, *, horizon: int = 20, playouts: int | None = None) -> tuple[int, dict[str, object]]:
        require_public_frame(frame)
        return self.search.select_destination(frame, horizon, playouts=playouts)


__all__ = [
    "AMCTSPolicy", "H20Groups", "ZhangPublicActor", "approximate_next_probabilities",
    "h20_groups_from_frame", "lsgf_destination", "lsgf_score", "next_group_probabilities",
    "pairwise_before", "select_chance_child", "select_decision_child", "update_chance_cost",
    "update_decision_cost",
]
