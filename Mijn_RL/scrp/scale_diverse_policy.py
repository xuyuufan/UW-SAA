"""Size-agnostic 2D online relocation policy inspired by Shin et al. (2026).

The policy consumes a stack-by-tier tensor.  Positive values are relative
retrieval priorities in the revealed online window, -1 is an unrevealed
container, and 0 is an empty tier.  It combines an intra-stack LSTM with
inter-stack self-attention and a masked pointer decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .online_crp import OnlineCRPPublicState


SCALE_DIVERSE_POLICY_V1 = "scale_diverse_online_2d_v1"


@dataclass(frozen=True)
class Online2DPriorityAdapter:
    """Convert a leakage-safe public state to a padded stack-by-tier tensor."""

    max_tiers: int

    def build(self, state: OnlineCRPPublicState) -> np.ndarray:
        if self.max_tiers <= 0:
            raise ValueError("max_tiers must be positive")
        result = np.zeros((len(state.stacks), self.max_tiers), dtype=np.float32)
        for stack_id, stack in enumerate(state.stacks):
            if len(stack) > self.max_tiers:
                raise ValueError("public stack exceeds max_tiers")
            for tier, priority in enumerate(stack):
                # Public 0 means an occupied but unrevealed position.  Empty
                # tiers are the zero padding outside the tuple.
                result[stack_id, tier] = -1.0 if priority == 0 else float(priority)
        return result


class _AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class ScaleDiverseRelocationPolicy(nn.Module):
    """LSTM-attention pointer policy for variable-size 2D online CRP."""

    def __init__(
        self,
        *,
        lookahead_horizon: int = 0,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        clip_constant: float = 10.0,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if lookahead_horizon < 0:
            raise ValueError("lookahead_horizon must be non-negative")
        if embed_dim <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.lookahead_horizon = lookahead_horizon
        self.embed_dim = embed_dim
        self.clip_constant = clip_constant
        self.validate_inputs = validate_inputs
        self.unknown_token = nn.Parameter(
            torch.tensor(float(lookahead_horizon + 2))
        )
        self.tier_position = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.container_embedding = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.stack_lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.priority_projection = nn.Linear(2 * embed_dim, embed_dim)

        # Relocation-only 2D layout features: height, free capacity, source
        # marker, and global fill ratio.  Crane travel features are excluded.
        self.layout_projection = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.stack_projection = nn.Sequential(
            nn.Linear(2 * embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.encoder_layers = nn.ModuleList(
            _AttentionBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_encoder_layers)
        )
        self.context_projection = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.query_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def encode(
        self,
        priorities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if priorities.ndim != 3:
            raise ValueError("priorities must have shape [batch, stacks, tiers]")
        batch, stacks, tiers = priorities.shape
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
            occupied, 1.0 - (values - 1.0) / denominator, torch.zeros_like(values)
        )
        tier_grid = torch.linspace(
            0.0, 1.0, tiers, dtype=priorities.dtype, device=priorities.device
        ).view(1, 1, tiers, 1)
        tier_features = self.tier_position(tier_grid).expand(batch, stacks, -1, -1)
        container_features = torch.cat((normalized.unsqueeze(-1), tier_features), dim=-1)
        embedded = self.container_embedding(container_features)
        embedded = embedded.view(batch * stacks, tiers, self.embed_dim)
        outputs, (hidden, _) = self.stack_lstm(embedded)
        priority_embedding = self.priority_projection(torch.cat(
            (outputs.mean(dim=1), hidden[-1]), dim=-1
        )).view(batch, stacks, self.embed_dim)

        heights = occupied.sum(dim=2).to(priorities.dtype)
        height_ratio = heights / max(tiers, 1)
        free_ratio = 1.0 - height_ratio
        source = (visible & (priorities == 1)).any(dim=2).to(priorities.dtype)
        if self.validate_inputs and not torch.all(source.sum(dim=1) == 1):
            raise ValueError("each active state must expose exactly one current target")
        fill_ratio = occupied.sum(dim=(1, 2)).to(priorities.dtype) / (stacks * tiers)
        fill_ratio = fill_ratio[:, None].expand(-1, stacks)
        layout = torch.stack((height_ratio, free_ratio, source, fill_ratio), dim=-1)
        layout_embedding = self.layout_projection(layout)
        encoded = self.stack_projection(torch.cat(
            (priority_embedding, layout_embedding), dim=-1
        ))
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        return encoded, source.argmax(dim=1)

    def action_log_probabilities(
        self, priorities: torch.Tensor, forbidden: torch.Tensor
    ) -> torch.Tensor:
        encoded, source_indices = self.encode(priorities)
        batch = torch.arange(encoded.shape[0], device=encoded.device)
        source = encoded[batch, source_indices]
        global_embedding = encoded.mean(dim=1)
        context = self.context_projection(torch.cat((source, global_embedding), dim=-1))
        context, _ = self.cross_attention(
            context.unsqueeze(1), encoded, encoded, need_weights=False
        )
        query = self.query_projection(context).squeeze(1)
        keys = self.key_projection(encoded)
        # Casting alone does not disable AMP for the final dot product.
        with torch.autocast(device_type=query.device.type, enabled=False):
            raw_scores = torch.einsum(
                "bd,bsd->bs", query.float(), keys.float()
            ) / math.sqrt(self.embed_dim)
            scores = (self.clip_constant * torch.tanh(raw_scores)).masked_fill(
                forbidden.bool(), float("-inf")
            )
            if self.validate_inputs and torch.any(torch.isinf(scores).all(dim=1)):
                raise ValueError("every state must have at least one legal destination")
            return torch.log_softmax(scores, dim=-1)

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
