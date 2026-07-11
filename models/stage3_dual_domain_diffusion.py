"""Shared diffusion utilities required by Stage-3 refinement models.

RAPD-DSwin uses the deterministic part of the Stage-3 refiner for the current
experiment, but the existing Stage-3 module still imports the schedule and
`t`-conditioned residual block definitions.  This file keeps those lightweight
utilities available without restoring the old full-field diffusion refiner.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import _group_count


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create standard sinusoidal embeddings for integer diffusion timesteps."""
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeConditionedResidualBlock(nn.Module):
    """Small residual block with additive timestep conditioning."""

    def __init__(
        self,
        channels: int,
        time_channels: int,
        dilation: int = 1,
    ):
        super().__init__()
        if channels <= 0 or time_channels <= 0:
            raise ValueError("Channel counts must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        groups = _group_count(channels)
        padding = int(dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=dilation,
        )
        self.time_projection = nn.Linear(time_channels, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        time_bias = self.time_projection(time_embedding).view(
            time_embedding.size(0),
            -1,
            1,
            1,
        )
        hidden = hidden + time_bias
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return x + hidden


class GaussianDiffusionSchedule(nn.Module):
    """Linear-beta Gaussian diffusion schedule with x0 prediction helper."""

    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1")
        self.timesteps = int(timesteps)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gathered = values.to(device=target.device, dtype=target.dtype)[timesteps]
        return gathered.view(-1, *([1] * (target.ndim - 1)))

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alpha_bars, timesteps, clean)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            clean,
        )
        return sqrt_alpha * clean + sqrt_one_minus * noise

    def predict_clean_from_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alpha_bars, timesteps, noisy)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            noisy,
        )
        return (noisy - sqrt_one_minus * predicted_noise) / sqrt_alpha.clamp_min(1e-8)
