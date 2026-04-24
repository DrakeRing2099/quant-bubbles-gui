from __future__ import annotations

import os
from functools import lru_cache
from itertools import product

import numpy as np


_FORCED_BACKEND = os.environ.get("SIG_BACKEND", "").strip().lower()
_VALID_BACKENDS = {"", "roughpy", "signatory", "iisignature"}
if _FORCED_BACKEND not in _VALID_BACKENDS:
    raise ValueError(
        f"Unknown SIG_BACKEND={_FORCED_BACKEND!r}. Expected one of: roughpy, signatory, iisignature."
    )

_roughpy = None
_signatory = None
_torch = None
_iisignature = None

if _FORCED_BACKEND in {"", "roughpy"}:
    try:
        import roughpy as _roughpy  # type: ignore[assignment]
    except ImportError:
        _roughpy = None

if _FORCED_BACKEND in {"", "signatory"}:
    try:
        import torch as _torch  # type: ignore[assignment]
        import signatory as _signatory  # type: ignore[assignment]
    except ImportError:
        _torch = None
        _signatory = None

if _FORCED_BACKEND in {"", "iisignature"}:
    try:
        import iisignature as _iisignature  # type: ignore[assignment]
    except ImportError:
        _iisignature = None

if _FORCED_BACKEND == "roughpy" and _roughpy is None:
    raise ImportError("SIG_BACKEND=roughpy was requested but roughpy is not installed.")
if _FORCED_BACKEND == "signatory" and _signatory is None:
    raise ImportError("SIG_BACKEND=signatory was requested but signatory is not installed.")
if _FORCED_BACKEND == "iisignature" and _iisignature is None:
    raise ImportError("SIG_BACKEND=iisignature was requested but iisignature is not installed.")

if _roughpy is not None:
    ACTIVE_BACKEND = "roughpy"
elif _signatory is not None:
    ACTIVE_BACKEND = "signatory"
elif _iisignature is not None:
    ACTIVE_BACKEND = "iisignature"
else:
    raise ImportError("No signature backend is available. Install roughpy, signatory, or iisignature.")


@lru_cache(maxsize=None)
def _prepared_logsignature(channels: int, depth: int):
    if _iisignature is None:
        raise RuntimeError("iisignature backend is not available")
    return _iisignature.prepare(channels, depth)


@lru_cache(maxsize=None)
def _roughpy_context(channels: int, depth: int):
    if _roughpy is None:
        raise RuntimeError("roughpy backend is not available")
    return _roughpy.get_context(width=channels, depth=depth, coeffs=_roughpy.DPReal)


def _validate_device(device: str) -> None:
    if device == "cpu":
        return
    if ACTIVE_BACKEND != "signatory":
        raise RuntimeError(f"Backend '{ACTIVE_BACKEND}' only supports device='cpu'.")
    if device != "cuda":
        raise RuntimeError(f"Unsupported device '{device}'.")
    if _torch is None or not _torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")


def _is_lyndon(word: tuple[int, ...]) -> bool:
    return all(word < word[i:] for i in range(1, len(word)))


@lru_cache(maxsize=None)
def _roughpy_lyndon_tensor_indices(channels: int, depth: int) -> tuple[int, ...]:
    if _roughpy is None:
        raise RuntimeError("roughpy backend is not available")

    indices: list[int] = []
    for length in range(1, depth + 1):
        for word in product(range(1, channels + 1), repeat=length):
            if _is_lyndon(word):
                key = _roughpy.TensorKey(list(word), width=channels, depth=depth)
                indices.append(int(key.to_index()))
    return tuple(indices)


def _roughpy_stream_from_path(path: np.ndarray, depth: int):
    ctx = _roughpy_context(path.shape[1], depth)
    increments = np.diff(path.astype(np.float64, copy=False), axis=0)
    return _roughpy.LieIncrementStream.from_increments(increments, ctx=ctx)


def _roughpy_signature_one(path: np.ndarray, depth: int) -> np.ndarray:
    stream = _roughpy_stream_from_path(path, depth)
    interval = _roughpy.RealInterval(0.0, float(path.shape[0] - 1))
    # RoughPy includes the scalar term 1.0; Signatory does not.
    return np.asarray(stream.signature(interval), dtype=np.float64)[1:]


def _roughpy_logsignature_one(path: np.ndarray, depth: int) -> np.ndarray:
    stream = _roughpy_stream_from_path(path, depth)
    interval = _roughpy.RealInterval(0.0, float(path.shape[0] - 1))
    # Signatory's default mode='words' corresponds to Lyndon-word coefficients
    # extracted from the expanded tensor logsignature and ordered by degree then
    # lexicographically within each degree.
    expanded = np.asarray(stream.signature(interval).log(), dtype=np.float64)
    indices = _roughpy_lyndon_tensor_indices(path.shape[1], depth)
    return expanded[list(indices)]


def compute_signature(
    paths: np.ndarray,
    depth: int,
    device: str = "cpu",
) -> np.ndarray:
    path_array = np.asarray(paths, dtype=np.float32)

    if ACTIVE_BACKEND == "signatory":
        _validate_device(device)
        tensor = _torch.from_numpy(path_array).to(device)
        with _torch.no_grad():
            feat = _signatory.signature(tensor, depth=depth)
        return feat.detach().cpu().numpy().astype(np.float32, copy=False)

    if ACTIVE_BACKEND == "roughpy":
        _validate_device(device)
        feats = [_roughpy_signature_one(path, depth) for path in path_array]
        return np.asarray(feats, dtype=np.float32)

    _validate_device(device)
    feats = [
        _iisignature.sig(path.astype(np.float64, copy=False), depth)
        for path in path_array
    ]
    return np.asarray(feats, dtype=np.float32)


def compute_logsignature(
    paths: np.ndarray,
    depth: int,
    device: str = "cpu",
) -> np.ndarray:
    path_array = np.asarray(paths, dtype=np.float32)

    if ACTIVE_BACKEND == "signatory":
        _validate_device(device)
        tensor = _torch.from_numpy(path_array).to(device)
        with _torch.no_grad():
            try:
                feat = _signatory.logsignature(tensor, depth=depth)
            except TypeError:
                feat = _signatory.logsignature(tensor, depth=depth, mode="words")
        return feat.detach().cpu().numpy().astype(np.float32, copy=False)

    if ACTIVE_BACKEND == "roughpy":
        _validate_device(device)
        feats = [_roughpy_logsignature_one(path, depth) for path in path_array]
        return np.asarray(feats, dtype=np.float32)

    _validate_device(device)
    prepared = _prepared_logsignature(path_array.shape[2], depth)
    feats = [
        _iisignature.logsig(path.astype(np.float64, copy=False), prepared)
        for path in path_array
    ]
    return np.asarray(feats, dtype=np.float32)


def backend_info() -> dict[str, object]:
    return {
        "active_backend": ACTIVE_BACKEND,
        "forced_backend": _FORCED_BACKEND or None,
        "roughpy_available": _roughpy is not None,
        "signatory_available": _signatory is not None,
        "iisignature_available": _iisignature is not None,
    }
