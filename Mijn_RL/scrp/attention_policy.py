"""Pure-attention pointer policy for multiscale online container relocation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


ATTENTION_POLICY_V1 = "online_crp_attention_v1"


class _AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, values: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attended, _ = self.attention(
            values,
            values,
            values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        values = self.norm1(values + attended)
        return self.norm2(values + self.ffn(values))


@dataclass(frozen=True)
class AttentionPolicyConfig:
    """Serialized architecture settings used for checkpoint validation."""

    max_lookahead: int = 20
    embed_dim: int = 128
    intra_stack_layers: int = 1
    inter_stack_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 512
    clip_constant: float = 5.0

    def __post_init__(self) -> None:
        if self.max_lookahead < 0:
            raise ValueError("max_lookahead must be non-negative")
        if self.embed_dim <= 0 or self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if self.intra_stack_layers <= 0 or self.inter_stack_layers <= 0:
            raise ValueError("attention layer counts must be positive")
        if self.ffn_dim <= 0 or self.clip_constant <= 0:
            raise ValueError("ffn_dim and clip_constant must be positive")


class AttentionRelocationPolicy(nn.Module):
    """Size-agnostic pure-attention policy with a masked pointer decoder.

    ``priorities`` uses the repository's public tensor contract: zero is an
    empty tier, -1 is occupied but unknown, 1 is the current target, and
    positive values above 1 are revealed future priorities. ``forbidden`` is
    true for the source stack and every full stack.
    """

    architecture = ATTENTION_POLICY_V1

    def __init__(
        self,
        config: AttentionPolicyConfig | None = None,
        *,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or AttentionPolicyConfig()
        self.validate_inputs = bool(validate_inputs)
        dim = self.config.embed_dim

        # The token represents the furthest value hidden under the configured
        # maximum observation window; it is never reset for smaller n.
        self.unknown_token = nn.Parameter(
            torch.tensor(float(self.config.max_lookahead + 2))
        )
        self.tier_position = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.container_embedding = nn.Sequential(
            nn.Linear(2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim)
        )
        self.stack_summary = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.stack_summary, mean=0.0, std=dim**-0.5)
        self.intra_stack = nn.ModuleList(
            _AttentionBlock(dim, self.config.num_heads, self.config.ffn_dim)
            for _ in range(self.config.intra_stack_layers)
        )
        self.layout_projection = nn.Sequential(
            nn.Linear(4, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim)
        )
        self.stack_projection = nn.Sequential(
            nn.Linear(2 * dim, self.config.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.config.ffn_dim, dim),
        )
        self.inter_stack = nn.ModuleList(
            _AttentionBlock(dim, self.config.num_heads, self.config.ffn_dim)
            for _ in range(self.config.inter_stack_layers)
        )
        self.context_projection = nn.Linear(2 * dim, dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            dim, self.config.num_heads, dropout=0.0, batch_first=True
        )
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)

    def architecture_config(self) -> dict[str, int | float | str]:
        return {
            "architecture": self.architecture,
            "max_lookahead": self.config.max_lookahead,
            "embed_dim": self.config.embed_dim,
            "intra_stack_layers": self.config.intra_stack_layers,
            "inter_stack_layers": self.config.inter_stack_layers,
            "num_heads": self.config.num_heads,
            "ffn_dim": self.config.ffn_dim,
            "clip_constant": self.config.clip_constant,
            "dropout": 0.0,
        }

    def encode(
        self, priorities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if priorities.ndim != 3:
            raise ValueError("priorities must have shape [batch, stacks, tiers]")
        batch, stacks, tiers = priorities.shape
        if batch == 0 or stacks < 2 or tiers <= 0:
            raise ValueError("active policy input needs rows, stacks, and tiers")
        if self.validate_inputs and (
            not torch.isfinite(priorities).all()
            or torch.any(priorities < -1)
            or torch.any((priorities > 0) & (priorities < 1))
            or torch.any(priorities > self.config.max_lookahead + 1)
        ):
            raise ValueError("priorities contain invalid public values")

        occupied = priorities.ne(0)
        values = torch.where(
            priorities.lt(0),
            self.unknown_token.to(dtype=priorities.dtype),
            priorities,
        )
        denominator = values.amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        normalized = torch.where(
            occupied,
            1.0 - (values - 1.0) / denominator,
            torch.zeros_like(values),
        )
        tier_grid = torch.linspace(
            0.0, 1.0, tiers, dtype=priorities.dtype, device=priorities.device
        ).view(1, 1, tiers, 1)
        tier_values = self.tier_position(tier_grid).expand(batch, stacks, tiers, 1)
        container_features = torch.cat((normalized.unsqueeze(-1), tier_values), dim=-1)
        embedded = self.container_embedding(container_features)

        flat = embedded.reshape(batch * stacks, tiers, self.config.embed_dim)
        summary = self.stack_summary.expand(batch * stacks, -1, -1)
        stack_tokens = torch.cat((summary, flat), dim=1)
        # Empty tiers are padding. Unknown occupied positions remain visible.
        empty_mask = torch.cat(
            (
                torch.zeros((batch * stacks, 1), dtype=torch.bool, device=priorities.device),
                ~occupied.reshape(batch * stacks, tiers),
            ),
            dim=1,
        )
        for layer in self.intra_stack:
            stack_tokens = layer(stack_tokens, key_padding_mask=empty_mask)
        stack_summary = stack_tokens[:, 0].reshape(
            batch, stacks, self.config.embed_dim
        )

        heights = occupied.sum(dim=2).to(priorities.dtype)
        height_ratio = heights / float(tiers)
        source_mark = priorities.eq(1).any(dim=2)
        if self.validate_inputs and not torch.all(
            priorities.eq(1).sum(dim=(1, 2)) == 1
        ):
            raise ValueError("each active state must expose exactly one current target")
        fill_ratio = occupied.sum(dim=(1, 2)).to(priorities.dtype) / float(stacks * tiers)
        fill_ratio = fill_ratio[:, None].expand(-1, stacks)
        layout_features = torch.stack(
            (height_ratio, 1.0 - height_ratio, source_mark.to(priorities.dtype), fill_ratio),
            dim=-1,
        )
        layout = self.layout_projection(layout_features)
        encoded = self.stack_projection(torch.cat((stack_summary, layout), dim=-1))
        for layer in self.inter_stack:
            encoded = layer(encoded)
        return encoded, source_mark.to(torch.int64).argmax(dim=1)

    def action_log_probabilities(
        self, priorities: torch.Tensor, forbidden: torch.Tensor
    ) -> torch.Tensor:
        if forbidden.shape != priorities.shape[:2]:
            raise ValueError("forbidden must have shape [batch, stacks]")
        encoded, source_indices = self.encode(priorities)
        forbidden = forbidden.to(device=encoded.device, dtype=torch.bool)
        if self.validate_inputs:
            has_legal = (~forbidden).any(dim=1)
            source_is_forbidden = forbidden.gather(1, source_indices[:, None]).squeeze(1)
            if not bool(torch.all(has_legal & source_is_forbidden).item()):
                raise ValueError("each active row needs legal destinations and a forbidden source")
            full = priorities.ne(0).sum(dim=2).eq(priorities.shape[2])
            expected_forbidden = full | torch.nn.functional.one_hot(
                source_indices, num_classes=priorities.shape[1]
            ).to(torch.bool)
            if not torch.equal(forbidden, expected_forbidden):
                raise ValueError("forbidden mask must exclude exactly the source and full stacks")

        batch = torch.arange(encoded.shape[0], device=encoded.device)
        source = encoded[batch, source_indices]
        global_summary = encoded.mean(dim=1)
        query_context = self.context_projection(
            torch.cat((source, global_summary), dim=-1)
        )
        context, _ = self.cross_attention(
            query_context.unsqueeze(1), encoded, encoded, need_weights=False
        )
        query = self.query_projection(context).squeeze(1)
        keys = self.key_projection(encoded)

        # Explicitly leave autocast for score calculation, masking, and
        # normalization so forbidden actions are exactly -inf in every mode.
        with torch.autocast(device_type=query.device.type, enabled=False):
            raw_scores = torch.einsum("bd,bsd->bs", query.float(), keys.float())
            raw_scores = raw_scores / math.sqrt(self.config.embed_dim)
            scores = (self.config.clip_constant * torch.tanh(raw_scores)).masked_fill(
                forbidden, float("-inf")
            )
            if self.validate_inputs and torch.any(torch.isneginf(scores).all(dim=1)):
                raise ValueError("every state must have at least one legal destination")
            log_probabilities = torch.log_softmax(scores, dim=-1)
        return log_probabilities

    def forward(
        self,
        priorities: torch.Tensor,
        forbidden: torch.Tensor,
        *,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probabilities = self.action_log_probabilities(priorities, forbidden)
        actions = (
            log_probabilities.argmax(dim=-1)
            if greedy
            else Categorical(logits=log_probabilities).sample()
        )
        selected = log_probabilities.gather(1, actions[:, None]).squeeze(1)
        return actions, selected


__all__ = [
    "ATTENTION_POLICY_V1",
    "AttentionPolicyConfig",
    "AttentionRelocationPolicy",
]
