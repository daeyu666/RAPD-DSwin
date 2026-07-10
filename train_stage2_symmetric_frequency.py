"""Compatibility helpers for dual-source symmetric SSP frequency differencing.

RAPD-DSwin keeps the model implementation in ``models/frequency_decomposition.py``.
This file preserves the old training-helper import path used by the multiscale
Stage-2 script.
"""

from __future__ import annotations

from typing import Dict

import torch

from losses import SAMLoss
from models.frequency_decomposition import Stage2SymmetricFrequencyNet
from train_stage2_coefficients import FixedSpatialDegradation
from train_stage2_dual_space import evaluate_dual
from utils import AverageMeter, move_to_device


SYMMETRIC_NAMES = [
    "symmetric_low_abs",
    "symmetric_mid_abs",
    "symmetric_high_abs",
    "symmetric_reliable_high_abs",
    "symmetric_low_share",
    "symmetric_mid_share",
    "symmetric_high_share",
    "physical_freq_low",
    "physical_freq_mid",
    "physical_freq_high",
    "reference_freq_low",
    "reference_freq_mid",
    "reference_freq_high",
    "physical_partition_loss",
    "reference_partition_loss",
]


@torch.no_grad()
def symmetric_diagnostics(
    model: Stage2SymmetricFrequencyNet,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    meters = {name: AverageMeter() for name in SYMMETRIC_NAMES}
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=False,
        )
        low = float(outputs["low_difference_feature"].abs().mean().item())
        mid = float(outputs["mid_difference_feature"].abs().mean().item())
        high = float(outputs["high_difference_feature"].abs().mean().item())
        reliable_high = float(
            outputs["reliable_high_difference_feature"].abs().mean().item()
        )
        total = max(low + mid + reliable_high, 1e-12)
        physical = outputs["physical_frequency_activation_ratio"].detach()
        reference = outputs["reference_frequency_activation_ratio"].detach()
        values = {
            "symmetric_low_abs": low,
            "symmetric_mid_abs": mid,
            "symmetric_high_abs": high,
            "symmetric_reliable_high_abs": reliable_high,
            "symmetric_low_share": low / total,
            "symmetric_mid_share": mid / total,
            "symmetric_high_share": reliable_high / total,
            "physical_freq_low": float(physical[0].item()),
            "physical_freq_mid": float(physical[1].item()),
            "physical_freq_high": float(physical[2].item()),
            "reference_freq_low": float(reference[0].item()),
            "reference_freq_mid": float(reference[1].item()),
            "reference_freq_high": float(reference[2].item()),
            "physical_partition_loss": float(
                outputs["physical_partition_reconstruction_loss"].item()
            ),
            "reference_partition_loss": float(
                outputs["reference_partition_reconstruction_loss"].item()
            ),
        }
        batch_size = batch["lr_hsi"].size(0)
        for name, value in values.items():
            meters[name].update(value, batch_size)
    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate_symmetric(
    model: Stage2SymmetricFrequencyNet,
    loader,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    result = evaluate_dual(
        model,
        loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    result.update(symmetric_diagnostics(model, loader, device))
    return result
