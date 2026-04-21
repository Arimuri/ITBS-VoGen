"""Compatibility shim loaded at Python startup when this directory is on PYTHONPATH.

Covers the moving-target ecosystem issues that break RVC (frozen at 2023)
when run against modern PyTorch / matplotlib:

1. PyTorch 2.6 flipped ``torch.load(weights_only=...)`` default to ``True``.
   fairseq 0.12.2 calls ``torch.load`` without the flag and its HuBERT
   checkpoint contains a ``fairseq.data.dictionary.Dictionary`` instance
   that isn't in torch's default safe-globals allowlist — triggering
   ``_pickle.UnpicklingError``.

2. matplotlib 3.10 removed ``FigureCanvasAgg.tostring_rgb``. RVC's
   ``infer/lib/train/utils.py`` calls it to turn training spectrograms into
   images for tensorboard logging. Without a shim, the first log flush in
   training kills the training subprocess.

Python automatically imports this module ("sitecustomize") as part of ``site``
initialization when it is found on sys.path. train.py / infer.py set
PYTHONPATH to this directory before spawning RVC subprocesses so these
patches apply.
"""

# --- torch.load weights_only=False default ----------------------------------
try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    pass
else:
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore[assignment]


# --- matplotlib FigureCanvasAgg.tostring_rgb shim ---------------------------
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore[import-not-found]
except ImportError:
    pass
else:
    if not hasattr(FigureCanvasAgg, "tostring_rgb"):
        import numpy as _np

        def _tostring_rgb(self):  # type: ignore[no-untyped-def]
            self.draw()
            arr = _np.asarray(self.buffer_rgba(), dtype=_np.uint8)
            return arr[..., :3].tobytes()

        FigureCanvasAgg.tostring_rgb = _tostring_rgb  # type: ignore[attr-defined]
