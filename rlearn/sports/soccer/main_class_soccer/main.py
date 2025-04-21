"""
Unified RLearn CLI
==================

Split / preprocess / train / visualize for StatsBomb, DataStadium, RoboCup‑2D.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Callable

import pytorch_lightning as pl # type: ignore
import torch

# --- local imports -----------------------------------------------------------
from rlearn.adapters import ( # type: ignore
    StatsBombAdapter,
    DataStadiumAdapter,
    RoboCup2DAdapter,
)
from rlearn.dataset.splitter import DatasetSplitter # type: ignore
from rlearn.dataset.preprocessor import ObservationPreprocessor # type: ignore
from rlearn.trainer.runner import TrainerRunner # type: ignore
from rlearn.visualizer.viewer import Visualizer # type: ignore
from rlearn.utils.file_utils import load_json # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("RLearnCLI")

# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RLearn unified CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # common arguments
    def add_common(sub_p: argparse.ArgumentParser) -> None:
        sub_p.add_argument("--provider", required=True, choices=["statsbomb", "datastadium", "robocup2d"])
        sub_p.add_argument("--input", required=True, type=Path, help="Input directory")
        sub_p.add_argument("--output", required=True, type=Path, help="Output directory")
        sub_p.add_argument("--config", type=Path, help="YAML / JSON config (optional)")

    # split
    sp_split = sub.add_parser("split", help="train/val/test split")
    add_common(sp_split)
    sp_split.add_argument("--seed", type=int, default=42)
    sp_split.add_argument("--val-ratio", type=float, default=0.1)
    sp_split.add_argument("--test-ratio", type=float, default=0.5)

    # preprocess
    sp_prep = sub.add_parser("preprocess", help="event → observation/action")
    add_common(sp_prep)
    sp_prep.add_argument("--batch-size", type=int, default=64)
    sp_prep.add_argument("--num-proc", type=int, default=4)

    # train
    sp_train = sub.add_parser("train", help="train RL model")
    add_common(sp_train)
    sp_train.add_argument("--run-name", required=True)
    sp_train.add_argument("--accelerator", default="gpu")
    sp_train.add_argument("--devices", default=1, type=int)
    sp_train.add_argument("--strategy", default="ddp")

    # visualize
    sp_vis = sub.add_parser("visualize", help="visualize Q‑values movie")
    add_common(sp_vis)
    sp_vis.add_argument("--checkpoint", required=True, type=Path)
    sp_vis.add_argument("--match-id", required=True)
    sp_vis.add_argument("--sequence-id", type=int, required=True)

    return p


# -----------------------------------------------------------------------------


def get_adapter(provider: str):
    if provider == "statsbomb":
        return StatsBombAdapter
    if provider == "datastadium":
        return DataStadiumAdapter
    if provider == "robocup2d":
        return RoboCup2DAdapter
    raise ValueError(f"Unsupported provider: {provider}")


# -----------------------------------------------------------------------------


def cmd_split(args):
    adapter_cls = get_adapter(args.provider)
    splitter = DatasetSplitter(
        adapter=adapter_cls(),
        input_dir=args.input,
        output_dir=args.output,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    splitter.run()


def cmd_preprocess(args):
    adapter_cls = get_adapter(args.provider)
    preprocessor = ObservationPreprocessor(
        adapter=adapter_cls(),
        input_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        config_path=args.config,
    )
    preprocessor.run()


def cmd_train(args):
    cfg = load_json(args.config) if args.config else {}
    runner = TrainerRunner(
        config=cfg,
        data_dir=args.input,
        output_dir=args.output / args.run_name,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
    )
    runner.run()


def cmd_visualize(args):
    visualizer = Visualizer(
        data_dir=args.input,
        checkpoint=args.checkpoint,
        match_id=args.match_id,
        sequence_id=args.sequence_id,
        output_dir=args.output,
    )
    visualizer.run()


# -----------------------------------------------------------------------------


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    dispatch: Dict[str, Callable[[argparse.Namespace], None]] = {
        "split": cmd_split,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "visualize": cmd_visualize,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
