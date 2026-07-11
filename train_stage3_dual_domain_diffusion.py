"""Stage-3 helper utilities for loading a frozen Stage-2 model.

Only the helper functions are needed in RAPD-DSwin.  The old full-field
Stage-3 diffusion training entrypoint is intentionally not restored.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch

from models.stage2_multiscale_pyramid import Stage2MultiScalePyramidNet
from train_stage2_coefficients import (
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from utils import move_to_device


def _checkpoint_state(path: str, device: torch.device) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_stage2_model(cfg, info: dict, device: torch.device) -> Tuple[Stage2MultiScalePyramidNet, dict, int]:
    """Build and strictly load the frozen multiscale Stage-2 checkpoint."""
    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        int(info["n_bands"]),
        device,
    )
    spectral_response = build_spectral_response(info).to(device)
    stage2 = Stage2MultiScalePyramidNet(
        stage1_model=stage1,
        spectral_response=spectral_response,
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        anchor_normalized_clip=cfg.anchor_normalized_clip,
        projector_tolerance=cfg.projector_tolerance,
        feature_channels=cfg.stage2_feature_channels,
        encoder_blocks=cfg.stage2_encoder_blocks,
        fusion_channels=cfg.stage2_fusion_channels,
        fusion_blocks=cfg.stage2_fusion_blocks,
        max_normalized_residual=cfg.stage2_max_normalized_residual,
        coefficient_scale_floor=cfg.stage2_coefficient_scale_floor,
        num_frequency_bands=cfg.stage2_num_frequency_bands,
        init_low_boundary=cfg.stage2_init_low_boundary,
        init_high_boundary=cfg.stage2_init_high_boundary,
        boundary_temperature=cfg.stage2_boundary_temperature,
        edge_threshold_mode=cfg.stage2_edge_threshold_mode,
        edge_mask_threshold=cfg.stage2_edge_mask_threshold,
        edge_reference_quantile=cfg.stage2_edge_reference_quantile,
        noise_quantile=cfg.stage2_noise_quantile,
        hard_partition=not cfg.stage2_soft_frequency_partition,
        pyramid_quarter_scale=cfg.pyramid_quarter_scale,
        pyramid_half_scale=cfg.pyramid_half_scale,
    ).to(device)

    state = _checkpoint_state(cfg.stage2_checkpoint, device)
    model_state = state.get("model", state)
    missing, unexpected = stage2.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Stage-2 checkpoint does not match Stage2MultiScalePyramidNet: "
            f"missing={missing}, unexpected={unexpected}"
        )
    stage2.eval()
    for parameter in stage2.parameters():
        parameter.requires_grad_(False)
    return stage2, state, int(state.get("epoch", 0))


@torch.no_grad()
def estimate_residual_scales(
    model,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, torch.Tensor]:
    """Estimate Stage-3 residual RMS scales and store them in the refiner."""
    coefficient_sum = None
    coefficient_count = 0
    orthogonal_sum = torch.zeros((), device=device)
    orthogonal_count = 0

    model.eval()
    for index, batch in enumerate(loader):
        if max_batches > 0 and index >= max_batches:
            break
        batch = move_to_device(batch, device)
        stage2_outputs = model.stage2_forward(batch["lr_hsi"], batch["hr_msi"])
        residual = batch["gt"] - stage2_outputs["reconstructed_hsi"]
        coefficient, _, orthogonal = model.decompose_residual(
            residual,
            stage2_outputs["basis"],
        )
        coeff_energy = coefficient.square().sum(dim=(0, 2, 3))
        if coefficient_sum is None:
            coefficient_sum = torch.zeros_like(coeff_energy)
        coefficient_sum += coeff_energy
        coefficient_count += coefficient.size(0) * coefficient.size(2) * coefficient.size(3)
        orthogonal_sum += orthogonal.square().sum()
        orthogonal_count += orthogonal.numel()

    if coefficient_sum is None or coefficient_count == 0 or orthogonal_count == 0:
        coefficient_scale = torch.ones(
            model.basis_rank,
            device=device,
            dtype=model.coefficient_residual_scale.dtype,
        )
        orthogonal_scale = torch.ones(
            1,
            device=device,
            dtype=model.orthogonal_residual_scale.dtype,
        )
    else:
        coefficient_scale = torch.sqrt(
            coefficient_sum / float(coefficient_count)
        ).clamp_min(model.residual_scale_floor)
        orthogonal_scale = torch.sqrt(
            orthogonal_sum / float(orthogonal_count)
        ).reshape(1).clamp_min(model.residual_scale_floor)

    model.set_residual_scales(coefficient_scale, orthogonal_scale)
    return {
        "coefficient_scale": coefficient_scale.detach(),
        "orthogonal_scale": orthogonal_scale.detach(),
    }
