"""Train Stage 2 with DSwin-style cross-modal detail routing.

This entrypoint reuses the residual-pyramid Stage-2 training code while swapping
only the model class and warm-start logic. It is a controlled replacement for the
NSP hard reliability gate:

    Q * Delta F^H  ->  DSwin-routed high-frequency MSI detail.

A residual-pyramid Stage-2 checkpoint can be loaded directly. The DSwin routed
residual and confidence delta are zero-initialized, so the initial prediction
matches the source checkpoint before fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch

import train_stage2_multiscale_pyramid as base
from models.stage2_dswin_detail_routing import Stage2DSwinDetailRoutingNet


@dataclass(frozen=True)
class DSwinRoutingConfig:
    window_size: int = 3
    offset_scale: float = 1.0
    hidden_channels: int | None = None


def _has_option(arguments: list[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def _extract_dswin_arguments() -> DSwinRoutingConfig:
    """Parse DSwin-only arguments and remove them before base.parse_args.

    The reused multiscale training parser does not know DSwin-specific options.
    Leaving them in ``sys.argv`` causes an ``unrecognized arguments`` failure.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dswin_window_size", type=int, default=3)
    parser.add_argument("--dswin_offset_scale", type=float, default=1.0)
    parser.add_argument("--dswin_hidden_channels", type=int, default=None)
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    return DSwinRoutingConfig(
        window_size=parsed.dswin_window_size,
        offset_scale=parsed.dswin_offset_scale,
        hidden_channels=parsed.dswin_hidden_channels,
    )


def _inject_default_arguments() -> None:
    arguments = sys.argv[1:]
    defaults = {
        "--symmetric_frequency_checkpoint": (
            "./checkpoints/stage2_multiscale_residual_pyramid/PaviaU/"
            "residual_pyramid_best_psnr.pth"
        ),
        "--checkpoint_root": "./checkpoints_dswin",
        "--output_root": "./outputs_dswin",
        "--log_root": "./logs_dswin",
        "--pyramid_new_lr": "2e-5",
        "--pyramid_source_lr": "5e-7",
        "--pyramid_warmup_epochs": "5",
    }
    injected: list[str] = []
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            injected.extend([option, value])
    if injected:
        sys.argv = [sys.argv[0], *injected, *arguments]


def load_dswin_warm_start(
    model: Stage2DSwinDetailRoutingNet,
    path: str,
    device: torch.device,
) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage-2 warm-start checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    source = state.get("model", state)
    destination = model.state_dict()
    transferable = {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    has_pyramid_source = any(key.startswith("quarter_branch.") for key in source)
    allowed_missing_prefixes = ["detail_router."]
    if not has_pyramid_source:
        allowed_missing_prefixes.extend(
            ["quarter_branch.", "half_branch.", "full_correction_branch."]
        )
    problematic_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    skipped_source = [key for key in source if key not in transferable]
    if unexpected or problematic_missing or skipped_source:
        raise RuntimeError(
            "DSwin detail routing warm-start mismatch: "
            f"unexpected={unexpected}, missing={problematic_missing}, "
            f"skipped_source={skipped_source}"
        )
    if not has_pyramid_source:
        model.initialize_pyramid_from_full()
    return state


def main() -> None:
    dswin_config = _extract_dswin_arguments()
    _inject_default_arguments()

    class ConfiguredStage2DSwinDetailRoutingNet(Stage2DSwinDetailRoutingNet):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                dswin_window_size=dswin_config.window_size,
                dswin_offset_scale=dswin_config.offset_scale,
                dswin_hidden_channels=dswin_config.hidden_channels,
                **kwargs,
            )

    ConfiguredStage2DSwinDetailRoutingNet.__name__ = "ConfiguredStage2DSwinDetailRoutingNet"
    base.Stage2MultiScalePyramidNet = ConfiguredStage2DSwinDetailRoutingNet
    base.load_symmetric_warm_start = load_dswin_warm_start
    base.main()


if __name__ == "__main__":
    main()
