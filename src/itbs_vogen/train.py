"""Orchestrate RVC's multi-stage training pipeline as subprocess calls.

Mirrors the order that RVC's own infer-web.py executes:

    1. preprocess.py                 — slice + resample source audio
    2. extract_f0_rmvpe_dml.py       — F0 extraction (single-process variant)
    3. extract_feature_print.py      — HuBERT content features
    4. generate filelist + config    — training index
    5. train.py                      — the actual fine-tune loop
    6. train_index (faiss)           — build retrieval index from features

We invoke RVC's scripts from the RVC repo root and keep all intermediate
artifacts under ``third_party/rvc/logs/<experiment>/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import shuffle

REPO_ROOT = Path(__file__).resolve().parents[2]
RVC_DIR = REPO_ROOT / "third_party" / "rvc"

# Sample rates RVC understands.
SR_CHOICES = {"32k": 32000, "40k": 40000, "48k": 48000}


@dataclass
class TrainConfig:
    speaker: str
    trainset_dir: Path
    sr: str = "48k"
    version: str = "v2"
    total_epoch: int = 200
    save_every_epoch: int = 50
    batch_size: int = 4
    n_proc: int = 2  # CPU processes for preprocess / feature extraction
    device: str | None = None
    save_every_weights: bool = True
    save_latest_only: bool = True
    cache_all_in_memory: bool = False


def detect_device() -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed. Run `pip install -r third_party/rvc/requirements.txt` first."
        ) from exc
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _run(cmd: list[str], step: str) -> None:
    print(f"\n[itbs-vogen/train] ===== {step} =====")
    print(f"[itbs-vogen/train] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=RVC_DIR, check=True)


def _exp_dir(speaker: str) -> Path:
    return RVC_DIR / "logs" / speaker


# ---- Pipeline steps ----------------------------------------------------------


def preprocess(cfg: TrainConfig) -> None:
    sr_val = SR_CHOICES[cfg.sr]
    exp_dir = _exp_dir(cfg.speaker)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "preprocess.log").write_text("")

    # preprocess.py expects: <inp_root> <sr> <n_p> <exp_dir> <noparallel> <per>
    cmd = [
        sys.executable,
        "infer/modules/train/preprocess.py",
        str(cfg.trainset_dir.resolve()),
        str(sr_val),
        str(cfg.n_proc),
        str(exp_dir),
        "False",  # noparallel
        "3.7",    # per (seconds per slice)
    ]
    _run(cmd, "1/5 preprocess (slice + resample)")


def extract_f0(cfg: TrainConfig) -> None:
    """Extract F0 using RMVPE.

    Picks the right RVC script based on device:
    - CUDA  -> extract_f0_rmvpe.py       (single-process, uses CUDA_VISIBLE_DEVICES)
    - CPU   -> extract_f0_print.py with method "rmvpe" (no GPU required)

    The ``_dml`` variant (DirectML, Windows) is intentionally avoided; it imports
    ``torch_directml`` unconditionally and fails on Linux/Mac.
    """
    exp_dir = _exp_dir(cfg.speaker)
    device = cfg.device or detect_device()

    if device.startswith("cuda"):
        # extract_f0_rmvpe.py args: n_part, i_part, i_gpu, exp_dir, is_half
        cmd = [
            sys.executable,
            "infer/modules/train/extract/extract_f0_rmvpe.py",
            "1",      # n_part
            "0",      # i_part
            "0",      # i_gpu (sets CUDA_VISIBLE_DEVICES)
            str(exp_dir),
            "False",  # is_half
        ]
    else:
        # CPU / MPS path: the generic extractor supports "rmvpe" as an f0 method.
        # args: exp_dir, n_p, f0method
        cmd = [
            sys.executable,
            "infer/modules/train/extract/extract_f0_print.py",
            str(exp_dir),
            str(cfg.n_proc),
            "rmvpe",
        ]
    _run(cmd, "2/5 extract F0 (RMVPE)")


def extract_features(cfg: TrainConfig) -> None:
    """Extract HuBERT content features."""
    device = cfg.device or detect_device()
    exp_dir = _exp_dir(cfg.speaker)
    # extract_feature_print.py expects: <device> <leng> <idx> <n_g> <exp_dir> <version> <is_half>
    cmd = [
        sys.executable,
        "infer/modules/train/extract_feature_print.py",
        device,
        "1",  # leng (number of parallel workers)
        "0",  # idx
        "0",  # n_g (gpu id, unused for cpu/mps)
        str(exp_dir),
        cfg.version,
        "False",  # is_half (False on Mac is safest)
    ]
    _run(cmd, "3/5 extract HuBERT features")


def generate_filelist(cfg: TrainConfig) -> None:
    exp_dir = _exp_dir(cfg.speaker)
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if cfg.version == "v1" else "3_feature768")
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"

    names = {p.stem for p in gt_wavs_dir.iterdir() if p.is_file()}
    names &= {p.stem for p in feature_dir.iterdir() if p.is_file()}
    names &= {p.stem for p in f0_dir.iterdir() if p.is_file()}
    names &= {p.stem for p in f0nsf_dir.iterdir() if p.is_file()}

    fea_dim = 256 if cfg.version == "v1" else 768
    spk_id = 0
    lines = []
    for name in sorted(names):
        lines.append(
            f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|"
            f"{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
        )
    # Duplicate mute reference a couple of times (mirrors RVC's infer-web.py).
    mute_gt = RVC_DIR / "logs" / "mute" / "0_gt_wavs" / f"mute{cfg.sr}.wav"
    mute_fea = RVC_DIR / "logs" / "mute" / f"3_feature{fea_dim}" / "mute.npy"
    mute_f0 = RVC_DIR / "logs" / "mute" / "2a_f0" / "mute.wav.npy"
    mute_f0nsf = RVC_DIR / "logs" / "mute" / "2b-f0nsf" / "mute.wav.npy"
    for _ in range(2):
        lines.append(f"{mute_gt}|{mute_fea}|{mute_f0}|{mute_f0nsf}|{spk_id}")
    shuffle(lines)

    (exp_dir / "filelist.txt").write_text("\n".join(lines))
    print(f"[itbs-vogen/train] wrote filelist with {len(lines)} entries")


def save_config(cfg: TrainConfig) -> None:
    """Write experiment config.json by copying RVC's template for the target sr."""
    src = RVC_DIR / "configs" / cfg.version / f"{cfg.sr}.json"
    dst = _exp_dir(cfg.speaker) / "config.json"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise FileNotFoundError(f"RVC config template not found: {src}")


def train_model(cfg: TrainConfig) -> None:
    pretrain_g = RVC_DIR / "assets" / "pretrained_v2" / f"f0G{cfg.sr}.pth"
    pretrain_d = RVC_DIR / "assets" / "pretrained_v2" / f"f0D{cfg.sr}.pth"
    if not pretrain_g.exists() or not pretrain_d.exists():
        raise FileNotFoundError(
            f"Pretrained weights missing: {pretrain_g}, {pretrain_d}\n"
            f"Run scripts/download_models.sh first."
        )

    cmd = [
        sys.executable,
        "infer/modules/train/train.py",
        "-e", cfg.speaker,
        "-sr", cfg.sr,
        "-f0", "1",
        "-bs", str(cfg.batch_size),
        "-te", str(cfg.total_epoch),
        "-se", str(cfg.save_every_epoch),
        "-pg", str(pretrain_g),
        "-pd", str(pretrain_d),
        "-l", "1" if cfg.save_latest_only else "0",
        "-c", "1" if cfg.cache_all_in_memory else "0",
        "-sw", "1" if cfg.save_every_weights else "0",
        "-v", cfg.version,
    ]
    _run(cmd, "4/5 train (fine-tune)")


def train_index(cfg: TrainConfig) -> None:
    """Build the FAISS retrieval index from extracted features.

    We inline the logic rather than shelling out because RVC's webui keeps
    this step inside a generator function not exposed as a standalone script.
    """
    import numpy as np  # local import: requires RVC env
    import faiss

    exp_dir = _exp_dir(cfg.speaker)
    feature_dir = exp_dir / ("3_feature256" if cfg.version == "v1" else "3_feature768")
    if not feature_dir.exists():
        raise RuntimeError(f"Features not found at {feature_dir}")

    npys = []
    for fp in sorted(feature_dir.iterdir()):
        if fp.suffix == ".npy":
            npys.append(np.load(fp))
    if not npys:
        raise RuntimeError("No feature .npy files found")
    big_npy = np.concatenate(npys, axis=0)

    # Optional subsampling if too large (matches RVC's behavior).
    if big_npy.shape[0] > 2e5:
        rng = np.random.default_rng(0)
        idx = rng.choice(big_npy.shape[0], int(2e5), replace=False)
        big_npy = big_npy[idx]

    np.save(exp_dir / "total_fea.npy", big_npy)

    # IVF index sized to feature count (same heuristic as RVC).
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(n_ivf, 1)
    index = faiss.index_factory(big_npy.shape[1], f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    index.add(big_npy)

    out = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{cfg.speaker}_{cfg.version}.index"
    faiss.write_index(index, str(out))
    print(f"[itbs-vogen/train] wrote index: {out}")


# ---- Orchestration + export --------------------------------------------------


def export_artifacts(cfg: TrainConfig) -> Path:
    """Copy the final .pth + .index to ``models/speakers/<name>/`` for inference."""
    exp_dir = _exp_dir(cfg.speaker)

    # RVC saves the consumable weight to assets/weights/<name>.pth at the last save step
    # when save_every_weights is enabled. The name may include the epoch; pick the last.
    weights_dir = RVC_DIR / "assets" / "weights"
    candidates = sorted(weights_dir.glob(f"{cfg.speaker}*.pth"))
    if not candidates:
        raise RuntimeError(f"No trained weight found under {weights_dir} for {cfg.speaker}")
    latest_pth = candidates[-1]

    index_candidates = sorted(exp_dir.glob(f"trained_IVF*_{cfg.speaker}_*.index"))
    if not index_candidates:
        raise RuntimeError(f"No index file found under {exp_dir}")
    latest_index = index_candidates[-1]

    dest_dir = REPO_ROOT / "models" / "speakers" / cfg.speaker
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "model.pth").write_bytes(latest_pth.read_bytes())
    (dest_dir / "model.index").write_bytes(latest_index.read_bytes())
    print(f"[itbs-vogen/train] exported to {dest_dir}")
    return dest_dir


def run(cfg: TrainConfig, stages: list[str] | None = None) -> None:
    """Run the pipeline. ``stages`` limits which steps execute (default: all)."""
    all_stages = ["preprocess", "f0", "features", "filelist", "config", "train", "index", "export"]
    run_set = set(stages) if stages else set(all_stages)

    if "preprocess" in run_set:
        preprocess(cfg)
    if "f0" in run_set:
        extract_f0(cfg)
    if "features" in run_set:
        extract_features(cfg)
    if "filelist" in run_set:
        generate_filelist(cfg)
    if "config" in run_set:
        save_config(cfg)
    if "train" in run_set:
        train_model(cfg)
    if "index" in run_set:
        train_index(cfg)
    if "export" in run_set:
        export_artifacts(cfg)
