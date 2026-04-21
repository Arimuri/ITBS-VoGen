"""Command-line entry point for ITBS-VoGen."""

from __future__ import annotations

import argparse
from pathlib import Path

from .infer import InferConfig, run as run_inference
from .train import TrainConfig, run as run_training


def _build_infer_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("infer", help="Run vocal restoration inference.")
    p.add_argument("input", type=Path, help="Input audio file (wav/mp3/flac).")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output wav path.")
    p.add_argument(
        "-s", "--speaker", required=True,
        help="Speaker name under models/speakers/<name>/ (requires model.pth, optionally model.index).",
    )
    p.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "harvest", "pm", "crepe"])
    p.add_argument("--f0-up-key", type=int, default=0)
    p.add_argument("--index-rate", type=float, default=0.66)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--rms-mix-rate", type=float, default=1.0)
    p.add_argument("--device", default=None)


def _build_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train a speaker model from a directory of vocals.")
    p.add_argument("-s", "--speaker", required=True, help="Speaker/experiment name.")
    p.add_argument(
        "-d", "--trainset-dir", type=Path, required=True,
        help="Directory of training wavs (recursed).",
    )
    p.add_argument("--sr", default="48k", choices=["32k", "40k", "48k"])
    p.add_argument("--version", default="v2", choices=["v1", "v2"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-proc", type=int, default=2, help="Parallel workers for preprocess/f0.")
    p.add_argument("--device", default=None, help="Torch device. Auto-detects if omitted.")
    p.add_argument(
        "--stages", default=None,
        help="Comma-separated subset of stages to run: preprocess,f0,features,filelist,config,train,index,export.",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="itbs-vogen")
    sub = p.add_subparsers(dest="command", required=True)
    _build_infer_parser(sub)
    _build_train_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "infer":
        cfg = InferConfig(
            input_path=args.input,
            output_path=args.output,
            speaker=args.speaker,
            f0_method=args.f0_method,
            f0_up_key=args.f0_up_key,
            index_rate=args.index_rate,
            protect=args.protect,
            rms_mix_rate=args.rms_mix_rate,
            device=args.device,
        )
        run_inference(cfg)
    elif args.command == "train":
        cfg = TrainConfig(
            speaker=args.speaker,
            trainset_dir=args.trainset_dir,
            sr=args.sr,
            version=args.version,
            total_epoch=args.epochs,
            save_every_epoch=args.save_every,
            batch_size=args.batch_size,
            n_proc=args.n_proc,
            device=args.device,
        )
        stages = args.stages.split(",") if args.stages else None
        run_training(cfg, stages=stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
