"""DSwin-style cross-modal MSI detail routing for Stage 2.

This module is a controlled replacement for the SFSR/NSP hard reliability gate.
It keeps the existing Stage-2 chain intact:

    Stage-1 spectral basis -> SRF analytical anchor -> observable/null heads
    -> residual-of-residual multiscale pyramid.

Only the high-frequency detail branch is changed. Instead of using

    reliable_high = Q * Delta F^H

from Sobel/NSP as the only high-frequency detail, a lightweight
token-centric deformable sliding-window router lets each coefficient token
sample MSI frequency-difference features from a content-adaptive local window.

The router is initialized as an identity replacement:

    routed_high = high_difference
    route_confidence = old_reliability_map

so a pretrained multiscale Stage-2 checkpoint can be loaded without changing
its initial output. The new deformable residual path then learns extra routing
during fine-tuning.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import _group_count
from .stage2_multiscale_pyramid import Stage2MultiScalePyramidNet


class DSwinCrossModalDetailRouter(nn.Module):
    """Lightweight token-centric deformable sliding-window detail router."""

    def __init__(
        self,
        basis_rank: int,
        feature_channels: int,
        window_size: int = 3,
        offset_scale: float = 1.0,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError("window_size must be an odd integer >= 3")
        if offset_scale < 0:
            raise ValueError("offset_scale must be non-negative")

        self.basis_rank = int(basis_rank)
        self.feature_channels = int(feature_channels)
        self.window_size = int(window_size)
        self.num_samples = int(window_size * window_size)
        self.offset_scale = float(offset_scale)
        hidden_channels = int(hidden_channels or feature_channels)
        groups = _group_count(feature_channels)
        hidden_groups = _group_count(hidden_channels)

        self.query_embed = nn.Sequential(
            nn.Conv2d(basis_rank, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )
        self.detail_embed = nn.Sequential(
            nn.Conv2d(feature_channels * 3, feature_channels, 1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )
        self.value_proj = nn.Conv2d(feature_channels, feature_channels, 1)
        self.offset_logits = nn.Sequential(
            nn.Conv2d(feature_channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(hidden_groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.num_samples * 3, 1),
        )
        self.routed_residual = nn.Conv2d(feature_channels, feature_channels, 1)
        self.confidence_residual = nn.Conv2d(feature_channels, 1, 1)

        nn.init.zeros_(self.routed_residual.weight)
        nn.init.zeros_(self.routed_residual.bias)
        nn.init.zeros_(self.confidence_residual.weight)
        nn.init.zeros_(self.confidence_residual.bias)

        radius = window_size // 2
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append([dx, dy])
        self.register_buffer(
            "base_offsets",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _base_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).unsqueeze(0)

    def _sample_values(
        self,
        values: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = values.shape
        base_grid = self._base_grid(height, width, values.device, values.dtype)
        base_offsets = self.base_offsets.to(device=values.device, dtype=values.dtype)
        x_scale = 2.0 / max(width - 1, 1)
        y_scale = 2.0 / max(height - 1, 1)
        static = base_offsets.view(1, self.num_samples, 2, 1, 1)
        learned = torch.tanh(offsets) * self.offset_scale
        total = static + learned
        norm = values.new_tensor([x_scale, y_scale]).view(1, 1, 2, 1, 1)
        total = total * norm

        sampled = []
        for index in range(self.num_samples):
            grid = base_grid + total[:, index].permute(0, 2, 3, 1)
            sampled.append(
                F.grid_sample(
                    values,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            )
        return torch.stack(sampled, dim=1)

    def forward(
        self,
        normalized_coefficients: torch.Tensor,
        low_difference: torch.Tensor,
        mid_difference: torch.Tensor,
        high_difference: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if (
            low_difference.shape != mid_difference.shape
            or low_difference.shape != high_difference.shape
        ):
            raise ValueError("Frequency-difference features must share shape")
        if reliability_map.ndim != 4 or reliability_map.size(1) != 1:
            raise ValueError("reliability_map must be [N, 1, H, W]")

        query = self.query_embed(normalized_coefficients)
        detail = self.detail_embed(
            torch.cat([low_difference, mid_difference, high_difference], dim=1)
        )
        values = self.value_proj(detail)
        logits = self.offset_logits(torch.cat([query, detail], dim=1))
        offsets, attention_logits = torch.split(
            logits,
            [self.num_samples * 2, self.num_samples],
            dim=1,
        )
        batch, _, height, width = offsets.shape
        offsets = offsets.view(batch, self.num_samples, 2, height, width)
        attention_logits = attention_logits.view(batch, self.num_samples, height, width)
        attention = torch.softmax(attention_logits, dim=1)
        sampled = self._sample_values(values, offsets)
        routed = (sampled * attention.unsqueeze(2)).sum(dim=1)

        residual = self.routed_residual(routed)
        routed_high = high_difference + residual
        confidence_delta = self.confidence_residual(routed)
        route_confidence = (reliability_map + 0.25 * torch.tanh(confidence_delta)).clamp(
            0.0,
            1.0,
        )

        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / float(torch.log(values.new_tensor(self.num_samples)))
        return {
            "dswin_query_feature": query,
            "dswin_detail_feature": detail,
            "dswin_routed_feature": routed,
            "dswin_routed_residual": residual,
            "dswin_routed_high_feature": routed_high,
            "dswin_route_confidence_map": route_confidence,
            "dswin_attention_entropy": entropy.mean(),
            "dswin_offset_abs": offsets.detach().abs().mean(),
            "dswin_route_residual_abs": residual.detach().abs().mean(),
            "dswin_route_confidence_mean": route_confidence.detach().mean(),
        }


class Stage2DSwinDetailRoutingNet(Stage2MultiScalePyramidNet):
    """Multiscale Stage 2 with DSwin-style cross-modal detail routing."""

    def __init__(
        self,
        *args,
        dswin_window_size: int = 3,
        dswin_offset_scale: float = 1.0,
        dswin_hidden_channels: int | None = None,
        **kwargs,
    ):
        feature_channels = int(kwargs.get("feature_channels", 64))
        super().__init__(*args, **kwargs)
        self.detail_router = DSwinCrossModalDetailRouter(
            basis_rank=self.basis_rank,
            feature_channels=feature_channels,
            window_size=dswin_window_size,
            offset_scale=dswin_offset_scale,
            hidden_channels=dswin_hidden_channels,
        )

    def _predict_normalized_residual(
        self,
        normalized_upsampled_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_discrepancy_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        route = self.detail_router(
            normalized_coefficients=normalized_upsampled_coefficients,
            low_difference=low_discrepancy_feature,
            mid_difference=mid_feature,
            high_difference=reliable_high_feature,
            reliability_map=reliability_map,
        )
        output = super()._predict_normalized_residual(
            normalized_upsampled_coefficients=normalized_upsampled_coefficients,
            physical_feature=physical_feature,
            low_discrepancy_feature=low_discrepancy_feature,
            mid_feature=mid_feature,
            reliable_high_feature=route["dswin_routed_high_feature"],
            reliability_map=route["dswin_route_confidence_map"],
        )
        output.update(route)
        output["reliable_high_feature_after_routing"] = route[
            "dswin_routed_high_feature"
        ]
        output["reliability_map_after_routing"] = route[
            "dswin_route_confidence_map"
        ]
        return output
