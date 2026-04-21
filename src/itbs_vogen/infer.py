"""Thin wrapper around RVC's CLI for vocal restoration.

We deliberately invoke RVC's existing ``tools/infer_cli.py`` as a subprocess
rather than importing its internals. This keeps the licensing boundary clean
(no code derivation) and insulates us from RVC's cwd-sensitive import layout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RVC_DIR = REPO_ROOT / "third_party" / "rvc"
RVC_WEIGHTS_DIR = RVC_DIR / "assets" / "weights"
LOCAL_SPEAKERS_DIR = REPO_ROOT / "models" / "speakers"


@dataclass
class InferConfig:
    input_path: Path
    output_path: Path
    speaker: str
    f0_method: str = "rmvpe"
    f0_up_key: int = 0
    index_rate: float = 0.66
    protect: float = 0.33
    rms_mix_rate: float = 1.0
    filter_radius: int = 3
    resample_sr: int = 0
    device: str | None = None


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


def _stage_speaker(speaker: str) -> tuple[str, Path | None]:
    """Ensure RVC can find the speaker model under its expected location.

    Our convention: speaker assets live under ``models/speakers/<name>/`` with
    ``model.pth`` (required) and ``model.index`` (optional). We symlink the
    ``.pth`` into RVC's ``assets/weights/`` so RVC's VC loader can resolve it
    by filename, and we return the ``.index`` path separately.
    """
    speaker_dir = LOCAL_SPEAKERS_DIR / speaker
    pth_src = speaker_dir / "model.pth"
    index_src = speaker_dir / "model.index"

    if not pth_src.exists():
        raise FileNotFoundError(
            f"Speaker model not found: {pth_src}\n"
            f"Place ``model.pth`` (and optionally ``model.index``) under "
            f"``models/speakers/{speaker}/``."
        )

    RVC_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    pth_link_name = f"{speaker}.pth"
    pth_link = RVC_WEIGHTS_DIR / pth_link_name
    if pth_link.is_symlink() or pth_link.exists():
        pth_link.unlink()
    pth_link.symlink_to(pth_src.resolve())

    index_path = index_src if index_src.exists() else None
    return pth_link_name, index_path


def run(cfg: InferConfig) -> Path:
    """Run RVC inference. Returns the output path on success."""
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name, index_path = _stage_speaker(cfg.speaker)
    device = cfg.device or detect_device()

    cmd = [
        sys.executable,
        "tools/infer_cli.py",
        "--input_path", str(cfg.input_path.resolve()),
        "--opt_path", str(cfg.output_path.resolve()),
        "--model_name", model_name,
        "--f0method", cfg.f0_method,
        "--f0up_key", str(cfg.f0_up_key),
        "--index_rate", str(cfg.index_rate),
        "--protect", str(cfg.protect),
        "--rms_mix_rate", str(cfg.rms_mix_rate),
        "--filter_radius", str(cfg.filter_radius),
        "--resample_sr", str(cfg.resample_sr),
        "--device", device,
    ]
    if index_path is not None:
        cmd.extend(["--index_path", str(index_path.resolve())])

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"[itbs-vogen] device={device} speaker={cfg.speaker} f0={cfg.f0_method}")
    print(f"[itbs-vogen] input={cfg.input_path}")
    print(f"[itbs-vogen] output={cfg.output_path}")
    subprocess.run(cmd, cwd=RVC_DIR, check=True, env=env)
    return cfg.output_path
