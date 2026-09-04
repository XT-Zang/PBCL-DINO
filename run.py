#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

CONFIG = Path(__file__).parent / "config" / "aquadataset_stage_adapter_fpn_ocr" / "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head.py"
DEFAULT_SEED = 42


def _configure_dataset_root() -> Path:
    value = os.environ.get("AQUADATASET_ROOT")
    if not value:
        raise SystemExit("Set AQUADATASET_ROOT to the Aquadataset root before running this command.")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"AQUADATASET_ROOT is not a directory: {root}")
    os.environ["AQUADATASET_ROOT"] = str(root)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PBCL-DINO minimal Aquadataset runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "preflight", "train"):
        command = sub.add_parser(name)
        command.add_argument(
            "--seed",
            type=int,
            choices=(42, 43, 44),
            default=DEFAULT_SEED,
            help="reproducibility seed (default: %(default)s)",
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _configure_dataset_root()

    if args.command == "inspect":
        from geoseg.runtime import load_experiment

        experiment = load_experiment(CONFIG, build=False, seed=args.seed)
        print(experiment.spec.to_json(), end="")
        return 0
    if args.command == "preflight":
        from geoseg.runtime import load_experiment, preflight_experiment

        experiment = load_experiment(CONFIG, build=False, seed=args.seed)
        print(preflight_experiment(experiment))
        return 0
    if args.command == "train":
        from train_supervision import run_training

        run_training(CONFIG, seed=args.seed)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
