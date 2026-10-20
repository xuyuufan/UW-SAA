"""Sequence commitment for public EG blocker groups."""
from __future__ import annotations
from typing import Any, Callable
from environment.public_frame import require_public_frame
from environment.public_history import validate_sequence
from environment.sequence_transition import _after_move, _assert_snapshot

_CRANE_TRAVEL_TIME=1.2
_PICK_DROP_TIME=30.0

class SequenceCommitter:
    """Commit an already-selected full EG blocker group and verify every step."""

    def __init__(self, algorithm: str, selector: Callable[[object], tuple[tuple[int, ...], dict[str, Any]]]):
        self.algorithm = algorithm
        self.selector = selector
        self.sequence: tuple[int, ...] | None = None
        self.next_index = 0
        self.expected_full: Any | None = None
        self._cached_moves = 0

    @property
    def is_group_start(self) -> bool:
        return self.algorithm != "EG" or self.sequence is None

    @property
    def cached_moves(self) -> int:
        return self._cached_moves

    def reset(self) -> None:
        self._clear_pending()
        self._cached_moves = 0

    def _clear_pending(self) -> None:
        self.sequence = None
        self.next_index = 0
        self.expected_full = None

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        require_public_frame(frame)
        from environment.public_history import validate_sequence

        if self.algorithm == "EG" and self.sequence is not None:
            from environment.sequence_transition import _after_move, _assert_snapshot

            if self.expected_full is None:
                raise RuntimeError("EG sequence cache lost its expected public frame")
            _assert_snapshot(frame, self.expected_full)
            action = int(self.sequence[self.next_index])
            self._cached_moves += 1
            self.next_index += 1
            cached_trace = {
                "group_start": False,
                "cached_move": True,
                "committed_sequence": list(self.sequence),
                "sequence_index": self.next_index,
            }
            if self.next_index < len(self.sequence):
                self.expected_full = _after_move(
                    frame, action,
                    crane_travel_time=_CRANE_TRAVEL_TIME,
                    pick_drop_time=_PICK_DROP_TIME,
                )
            else:
                self._clear_pending()
            return action, cached_trace

        sequence, trace = self.selector(frame)
        sequence = validate_sequence(frame, sequence, self.algorithm)
        if self.algorithm != "EG":
            if len(sequence) != 1:
                raise ValueError("non-EG selectors must commit one destination")
            return sequence[0], {**trace, "group_start": True, "cached_move": False}
        if self.sequence is not None:
            raise RuntimeError("EG selector attempted to replace a committed blocker group")
        self.sequence = sequence
        self.next_index = 1
        if len(sequence) > 1:
            from environment.sequence_transition import _after_move

            self.expected_full = _after_move(
                frame, sequence[0],
                crane_travel_time=_CRANE_TRAVEL_TIME,
                pick_drop_time=_PICK_DROP_TIME,
            )
        else:
            self._clear_pending()
        return sequence[0], {**trace, "group_start": True, "cached_move": False, "committed_sequence": list(sequence)}
