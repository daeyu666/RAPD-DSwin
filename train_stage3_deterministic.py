"""Run Stage 3 with only deterministic dual-domain residual completion.

This wrapper reuses ``train_stage3_uncertainty_guided_diffusion.py`` but forces
all epochs to stay in the deterministic warm-up phase. Diffusion learning rates
are zero, inference starts from zero, and the saved/evaluated final output is
therefore equal to the deterministic Stage-3 output.
"""

from __future__ import annotations

import sys

import train_stage3_uncertainty_guided_diffusion as base


def _has_option(arguments: list[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def _inject_defaults() -> None:
    arguments = sys.argv[1:]
    defaults = {
        "--checkpoint_root": "./checkpoints_stage3_deterministic",
        "--output_root": "./outputs_stage3_deterministic",
        "--log_root": "./logs_stage3_deterministic",
        "--stage3_det_warmup_epochs": "100000",
        "--stage3_joint_start_epoch": "100001",
        "--stage3_diff_lr": "0.0",
        "--stage3_joint_diff_lr": "0.0",
        "--stage3_inference_steps": "1",
        "--stage3_initial_noise": "zero",
        "--stage3_scale_estimation_batches": "20",
    }
    injected: list[str] = []
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            injected.extend([option, value])
    if injected:
        sys.argv = [sys.argv[0], *injected, *arguments]


def main() -> None:
    _inject_defaults()
    base.main()


if __name__ == "__main__":
    main()
