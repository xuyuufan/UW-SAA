"""H20 warm-start policy with the PLAN-001 public A1 feature extension."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .scale_diverse_policy import (
    SCALE_DIVERSE_POLICY_V1,
    ScaleDiverseRelocationPolicy,
)

H20_A1_ARCHITECTURE = "online_crp_h20_a1_v1"
H20_A1_FEATURE_VERSION = "h20_a1_public_features_v1"
H20_BASE_SHA256 = "aedd5e744607359318352f699cb4320f85dbbfc2c12b510047e04bd0013ef2d0"
H20_A1_ABLATION_GROUPS = frozenset({"container", "local_stack", "global_context"})
_H20_BASE_ARCHITECTURE = SCALE_DIVERSE_POLICY_V1


class _ZeroExtendedInputLinear(nn.Linear):
    """Apply old and added input columns separately to preserve step-zero math."""

    def __init__(self, in_features: int, out_features: int, *, old_features: int) -> None:
        super().__init__(in_features, out_features)
        if not 0 < old_features < in_features:
            raise ValueError("old_features must split the expanded input columns")
        self.old_features = old_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        old_output = F.linear(inputs[..., : self.old_features], self.weight[:, : self.old_features], self.bias)
        added_output = F.linear(inputs[..., self.old_features :], self.weight[:, self.old_features :], None)
        return old_output + added_output


def build_h20_a1_feature_tensors(
    priorities: torch.Tensor,
    *,
    ablation_groups: Sequence[str] = (),
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return added container channels, added layout channels, and source mask.

    The only input is the public priority tensor. No episode or hidden retrieval
    sequence is accepted here.
    """

    if priorities.ndim != 3:
        raise ValueError("priorities must have shape [batch, stacks, tiers]")
    if not torch.is_floating_point(priorities):
        raise TypeError("priorities must be a floating point public tensor")
    groups = frozenset(ablation_groups)
    invalid = groups - H20_A1_ABLATION_GROUPS
    if invalid:
        raise ValueError(f"unsupported H20-A1 ablation groups: {sorted(invalid)}")

    batch, stacks, tiers = priorities.shape
    if stacks <= 1 or tiers <= 0:
        raise ValueError("active public state needs at least two stacks and one tier")
    occupied = priorities != 0
    visible = priorities > 0
    source = (visible & (priorities == 1)).any(dim=2)
    if validate and not torch.all(source.sum(dim=1) == 1):
        raise ValueError("each active state must expose exactly one current target")
    source_indices = source.to(torch.int64).argmax(dim=1)

    heights = occupied.sum(dim=2).to(priorities.dtype)
    denominator = heights.clamp_min(1.0)
    unknown_ratio = (priorities < 0).sum(dim=2).to(priorities.dtype) / denominator
    has_known_bool = visible.any(dim=2)
    has_known = has_known_bool.to(priorities.dtype)
    positive_inf = torch.full_like(priorities, float("inf"))
    earliest = torch.where(visible, priorities, positive_inf).amin(dim=2)
    cell_count = float(stacks * tiers)
    earliest_normalized = torch.where(
        has_known_bool, earliest / cell_count, torch.zeros_like(earliest)
    )

    batch_indices = torch.arange(batch, device=priorities.device)
    source_height = heights[batch_indices, source_indices].to(torch.int64)
    source_top_index = (source_height - 1).clamp_min(0)
    blocker = priorities[batch_indices, source_indices, source_top_index]
    blocker_known = blocker > 0
    blocker_limit = torch.where(
        blocker_known, blocker, torch.full_like(blocker, float("inf"))
    )
    covered = visible & (priorities < blocker_limit[:, None, None])
    stack_indices = torch.arange(stacks, device=priorities.device)[None, :]
    is_destination = stack_indices != source_indices[:, None]
    covered = covered & is_destination.unsqueeze(-1)
    covered_count = covered.sum(dim=2).to(priorities.dtype)
    covered_ratio = covered_count / denominator
    has_covered = covered.any(dim=2)
    nearest_covered = torch.where(covered, priorities, positive_inf).amin(dim=2)
    covered_nearest = torch.where(
        has_covered, nearest_covered / cell_count, torch.zeros_like(nearest_covered)
    )

    max_public_priority = priorities.amax(dim=(1, 2))
    effective_h_ratio = (max_public_priority - 1.0).clamp_min(0.0) / cell_count
    occupied_count = occupied.sum(dim=(1, 2)).to(priorities.dtype)
    visible_count = visible.sum(dim=(1, 2)).to(priorities.dtype)
    remaining_ratio = occupied_count / cell_count
    known_fraction = visible_count / occupied_count.clamp_min(1.0)
    global_features = torch.stack(
        (effective_h_ratio, remaining_ratio, known_fraction), dim=-1
    )[:, None, :].expand(-1, stacks, -1)

    local_features = torch.stack(
        (unknown_ratio, earliest_normalized, has_known, covered_ratio, covered_nearest),
        dim=-1,
    )
    if "local_stack" in groups:
        local_features = torch.zeros_like(local_features)
    if "global_context" in groups:
        global_features = torch.zeros_like(global_features)

    container_features = torch.stack(
        (occupied.to(priorities.dtype), visible.to(priorities.dtype)), dim=-1
    )
    if "container" in groups:
        container_features = torch.zeros_like(container_features)
    return container_features, torch.cat((local_features, global_features), dim=-1), source


class H20A1RelocationPolicy(ScaleDiverseRelocationPolicy):
    """H20 LSTM–Attention–pointer policy with eight public layout features."""

    architecture = H20_A1_ARCHITECTURE
    feature_version = H20_A1_FEATURE_VERSION

    def __init__(
        self,
        *,
        lookahead_horizon: int = 20,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        clip_constant: float = 5.0,
        validate_inputs: bool = True,
        ablation_groups: Sequence[str] = (),
    ) -> None:
        super().__init__(
            lookahead_horizon=lookahead_horizon,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            clip_constant=clip_constant,
            validate_inputs=validate_inputs,
        )
        groups = frozenset(ablation_groups)
        invalid = groups - H20_A1_ABLATION_GROUPS
        if invalid:
            raise ValueError(f"unsupported H20-A1 ablation groups: {sorted(invalid)}")
        self.ablation_groups = groups
        self.container_embedding = nn.Sequential(
            _ZeroExtendedInputLinear(4, embed_dim // 2, old_features=2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.layout_projection = nn.Sequential(
            _ZeroExtendedInputLinear(12, embed_dim // 2, old_features=4),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def encode(
        self, priorities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if priorities.ndim != 3:
            raise ValueError("priorities must have shape [batch, stacks, tiers]")
        batch, stacks, tiers = priorities.shape
        if tiers <= 0:
            raise ValueError("priorities must contain at least one tier")

        occupied = priorities != 0
        visible = priorities > 0
        values = torch.where(
            priorities < 0,
            self.unknown_token.to(priorities.dtype),
            priorities,
        )
        denominator = torch.maximum(
            values.amax(dim=(1, 2), keepdim=True),
            torch.ones((), dtype=values.dtype, device=values.device),
        )
        normalized = torch.where(
            occupied,
            1.0 - (values - 1.0) / denominator,
            torch.zeros_like(values),
        )
        tier_grid = torch.linspace(
            0.0, 1.0, tiers, dtype=priorities.dtype, device=priorities.device
        ).view(1, 1, tiers, 1)
        tier_features = self.tier_position(tier_grid).expand(batch, stacks, -1, -1)
        added_container, added_layout, source = build_h20_a1_feature_tensors(
            priorities,
            ablation_groups=tuple(self.ablation_groups),
            validate=self.validate_inputs,
        )
        container_features = torch.cat(
            (normalized.unsqueeze(-1), tier_features, added_container), dim=-1
        )
        embedded = self.container_embedding(container_features)
        embedded = embedded.reshape(batch * stacks, tiers, self.embed_dim)
        outputs, (hidden, _) = self.stack_lstm(embedded)
        priority_embedding = self.priority_projection(
            torch.cat((outputs.mean(dim=1), hidden[-1]), dim=-1)
        ).reshape(batch, stacks, self.embed_dim)

        heights = occupied.sum(dim=2).to(priorities.dtype)
        height_ratio = heights / max(tiers, 1)
        free_ratio = 1.0 - height_ratio
        source_marker = (visible & (priorities == 1)).any(dim=2).to(priorities.dtype)
        fill_ratio = occupied.sum(dim=(1, 2)).to(priorities.dtype) / (stacks * tiers)
        fill_ratio = fill_ratio[:, None].expand(-1, stacks)
        old_layout = torch.stack(
            (height_ratio, free_ratio, source_marker, fill_ratio), dim=-1
        )
        layout_embedding = self.layout_projection(
            torch.cat((old_layout, added_layout), dim=-1)
        )
        encoded = self.stack_projection(
            torch.cat((priority_embedding, layout_embedding), dim=-1)
        )
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        return encoded, source_marker.argmax(dim=1)


def tensor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().contiguous().cpu()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()


def _policy_kwargs(config: Mapping[str, object]) -> dict[str, object]:
    names = (
        "lookahead_horizon",
        "embed_dim",
        "num_encoder_layers",
        "num_heads",
        "ffn_dim",
        "clip_constant",
    )
    return {name: config[name] for name in names if name in config}


def load_h20_base_checkpoint(
    path: str | Path,
) -> tuple[dict[str, object], ScaleDiverseRelocationPolicy, str]:
    """Load and strictly verify the one frozen H20 base used by PLAN-001."""

    checkpoint_path = Path(path)
    actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_sha != H20_BASE_SHA256:
        raise ValueError(
            f"H20 base SHA256 mismatch: expected {H20_BASE_SHA256}, got {actual_sha}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_architecture") != _H20_BASE_ARCHITECTURE:
        raise ValueError("base checkpoint architecture is not the documented H20 V1")
    config = checkpoint.get("online_crp_training_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("base checkpoint lacks H20 config or model_state_dict")
    if int(config.get("lookahead_horizon", -1)) != 20:
        raise ValueError("base checkpoint is not configured for H=20")
    base = ScaleDiverseRelocationPolicy(**_policy_kwargs(config))
    base.load_state_dict(state_dict, strict=True)
    if base.clip_constant != 5.0:
        raise ValueError("base H20 score clip must be 5.0")
    return dict(checkpoint), base, actual_sha


def expand_h20_state_dict(
    base_state_dict: Mapping[str, torch.Tensor],
    target_policy: H20A1RelocationPolicy,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Copy all H20 weights and zero-extend only the two input matrices."""

    target_state = target_policy.state_dict()
    if set(base_state_dict) != set(target_state):
        missing = sorted(set(target_state) - set(base_state_dict))
        extra = sorted(set(base_state_dict) - set(target_state))
        raise ValueError(f"state names differ; missing={missing}, extra={extra}")

    changes = {
        "container_embedding.0.weight": (64, 2, 4),
        "layout_projection.0.weight": (64, 4, 12),
    }
    expanded: dict[str, torch.Tensor] = {}
    copied: list[str] = []
    for name, target in target_state.items():
        old = base_state_dict[name]
        if name in changes:
            rows, old_columns, new_columns = changes[name]
            if tuple(old.shape) != (rows, old_columns):
                raise ValueError(f"unexpected H20 shape for {name}: {tuple(old.shape)}")
            if tuple(target.shape) != (rows, new_columns):
                raise ValueError(f"unexpected H20-A1 shape for {name}: {tuple(target.shape)}")
            value = torch.zeros_like(target, device="cpu")
            value[:, :old_columns].copy_(old.detach().cpu())
            if torch.count_nonzero(value[:, old_columns:]).item() != 0:
                raise AssertionError(f"new columns for {name} are not zero")
            expanded[name] = value
        else:
            if old.shape != target.shape:
                raise ValueError(
                    f"unexpected shape change for {name}: {tuple(old.shape)} -> {tuple(target.shape)}"
                )
            value = old.detach().to(device="cpu").clone()
            if not torch.equal(value, target.detach().cpu()):
                # target's random initialization is irrelevant; exactness is
                # verified against the source after strict load below.
                pass
            expanded[name] = value
            copied.append(name)

    result = target_policy.load_state_dict(expanded, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict H20-A1 load failed: {result}")
    loaded_state = target_policy.state_dict()
    for name in copied:
        if not torch.equal(loaded_state[name].detach().cpu(), base_state_dict[name].detach().cpu()):
            raise AssertionError(f"H20 parameter {name} changed during transfer")
    for name, (_, old_columns, _) in changes.items():
        if not torch.equal(
            loaded_state[name][:, :old_columns].detach().cpu(),
            base_state_dict[name].detach().cpu(),
        ):
            raise AssertionError(f"old columns changed during transfer for {name}")

    added_count = sum(
        loaded_state[name].numel() - base_state_dict[name].numel()
        for name in changes
    )
    parameter_count = sum(parameter.numel() for parameter in target_policy.parameters())
    if added_count != 640 or parameter_count != 1_106_482:
        raise AssertionError(
            f"unexpected H20-A1 parameter count: added={added_count}, total={parameter_count}"
        )
    report = {
        "architecture": H20_A1_ARCHITECTURE,
        "feature_version": H20_A1_FEATURE_VERSION,
        "strict_load": "passed",
        "changed_weight_shapes": {
            name: {
                "old": list(base_state_dict[name].shape),
                "new": list(loaded_state[name].shape),
                "old_columns_bitwise_copied": True,
                "new_columns_zero": True,
                "new_column_count": loaded_state[name].numel() - base_state_dict[name].numel(),
            }
            for name in changes
        },
        "copied_parameter_tensors": len(copied),
        "all_other_parameters_bitwise_copied": True,
        "added_parameter_count": added_count,
        "total_parameter_count": parameter_count,
        "expanded_model_state_sha256": tensor_state_sha256(loaded_state),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in target_policy.parameters() if parameter.requires_grad
        ),
    }
    return expanded, report


def build_method_policy(
    method: str,
    base_checkpoint: Mapping[str, object],
) -> tuple[nn.Module, dict[str, object]]:
    """Build an H20-CONT or H20-A1 model from one already verified checkpoint."""

    config = base_checkpoint.get("online_crp_training_config")
    base_state = base_checkpoint.get("model_state_dict")
    if not isinstance(config, Mapping) or not isinstance(base_state, Mapping):
        raise ValueError("base checkpoint is missing config or state_dict")
    kwargs = _policy_kwargs(config)
    if method == "h20_cont":
        policy = ScaleDiverseRelocationPolicy(**kwargs)
        result = policy.load_state_dict(base_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise AssertionError(f"strict CONT load failed: {result}")
        return policy, {
            "architecture": _H20_BASE_ARCHITECTURE,
            "strict_load": "passed",
            "base_sha256": H20_BASE_SHA256,
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "state_sha256": tensor_state_sha256(policy.state_dict()),
        }
    if method == "h20_a1":
        policy = H20A1RelocationPolicy(**kwargs)
        _, report = expand_h20_state_dict(base_state, policy)
        report["base_sha256"] = H20_BASE_SHA256
        return policy, report
    raise ValueError("method must be 'h20_cont' or 'h20_a1'")
