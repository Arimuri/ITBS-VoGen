"""Compatibility shim loaded at Python startup when this directory is on PYTHONPATH.

Problem: PyTorch 2.6 flipped the default of ``torch.load(weights_only=...)`` from
``False`` to ``True``. fairseq 0.12.2 (used by RVC) calls ``torch.load`` without
passing that flag, and its HuBERT checkpoints contain a
``fairseq.data.dictionary.Dictionary`` instance that isn't in torch's default
safe-globals allowlist — triggering ``_pickle.UnpicklingError``.

Since pinning torch to <2.6 can fail on platforms where only newer CUDA wheels
are published, we instead restore the old behavior globally by wrapping
``torch.load`` at interpreter startup.

Python automatically imports this module ("sitecustomize") as part of ``site``
initialization when it is found on sys.path — which we arrange by setting
PYTHONPATH before invoking RVC's subprocesses.
"""

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
