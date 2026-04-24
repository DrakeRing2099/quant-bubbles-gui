from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import signatory


SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_label: str
    raw_dir: str
    paths_dir: str
    features_dir: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cev": DatasetSpec(
        name="cev",
        dataset_label="cev",
        raw_dir="data/raw/paths",
        paths_dir="data/processed/systemB_paths",
        features_dir="data/processed/systemB_signatures_mod",
    ),
    "shifted_cev": DatasetSpec(
        name="shifted_cev",
        dataset_label="shifted_cev_random_displacement",
        raw_dir="data/raw/shifted_cev",
        paths_dir="data/processed/shifted_cev_systemB_paths",
        features_dir="data/processed/shifted_cev_systemB_signatures",
    ),
    "sin": DatasetSpec(
        name="sin",
        dataset_label="sin_stochastic_volatility",
        raw_dir="data/raw/sin",
        paths_dir="data/processed/sin_systemB_paths",
        features_dir="data/processed/sin_systemB_signatures",
    ),
    "switching_cev": DatasetSpec(
        name="switching_cev",
        dataset_label="switching_cev_poisson_regimes",
        raw_dir="data/raw/switching_cev",
        paths_dir="data/processed/switching_cev_systemB_paths",
        features_dir="data/processed/switching_cev_systemB_signatures",
    ),
}


FEATURE_VARIANTS: dict[str, dict[str, Any]] = {
    "base_sig": {
        "path_variant": "base",
        "transform": "signature",
        "channels": ["t_norm", "log_price"],
    },
    "base_log": {
        "path_variant": "base",
        "transform": "logsignature",
        "channels": ["t_norm", "log_price"],
    },
    "ll_sig": {
        "path_variant": "ll",
        "transform": "signature",
        "channels": ["t_norm", "lead(log_price)", "lag(log_price)"],
    },
    "ll_log": {
        "path_variant": "ll",
        "transform": "logsignature",
        "channels": ["t_norm", "lead(log_price)", "lag(log_price)"],
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {valid}") from exc


def resolve_dir(root: Path, override: str | None, default_rel: str) -> Path:
    target = Path(override) if override else Path(default_rel)
    return target if target.is_absolute() else root / target


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def raw_split_path(raw_dir: Path, split: str) -> Path:
    return raw_dir / f"{split}_paths.npz"


def path_output_name(split: str, variant: str) -> str:
    if variant == "base":
        return f"{split}_paths_for_signature.npz"
    if variant == "ll":
        return f"{split}_ll_paths_for_signature.npz"
    raise ValueError(f"Unknown path variant: {variant}")


def load_raw_split(raw_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = raw_split_path(raw_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw split: {path}")

    data = np.load(path, allow_pickle=True)
    required = {"prices", "times", "labels"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")

    prices = np.asarray(data["prices"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64).reshape(-1)
    payload = {key: data[key] for key in data.files if key not in {"prices", "times"}}
    return prices, times, payload


def load_processed_paths(
    paths_dir: Path,
    split: str,
    variant: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = paths_dir / path_output_name(split, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing processed paths: {path}")

    data = np.load(path, allow_pickle=True)
    if "paths" not in data.files:
        raise KeyError(f"Missing 'paths' in {path}")

    paths = np.asarray(data["paths"], dtype=np.float64)
    payload = {key: data[key] for key in data.files if key != "paths"}
    return paths, payload


def normalize_times(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"Expected 1D time grid with at least 2 points, got shape {times.shape}")

    span = float(times[-1] - times[0])
    if span <= 0.0:
        raise ValueError("Time grid must be strictly increasing")
    return (times - times[0]) / span


def build_base_paths(
    prices: np.ndarray,
    times: np.ndarray,
    eps: float = 1e-6,
    batch_size: int = 4096,
) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError(f"Expected prices shape (N, L), got {prices.shape}")

    n_paths, path_len = prices.shape
    if times.shape[0] != path_len:
        raise ValueError(f"times length {times.shape[0]} != prices length {path_len}")

    t_norm = normalize_times(times)
    out = np.empty((n_paths, path_len, 2), dtype=np.float64)
    out[:, :, 0] = t_norm[None, :]

    chunk = max(1, int(batch_size))
    for start in range(0, n_paths, chunk):
        end = min(start + chunk, n_paths)
        batch = np.maximum(prices[start:end], eps)
        s0 = np.maximum(batch[:, [0]], eps)
        out[start:end, :, 1] = np.log(batch / s0)

    return out


def lead_lag_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    length = x.shape[0]
    if length < 2:
        raise ValueError("Lead-lag requires at least 2 points")

    out = np.empty((2 * length - 1, 2), dtype=np.float64)
    out[0, 0] = x[0]
    out[0, 1] = x[0]

    idx = 1
    for k in range(1, length):
        out[idx, 0] = x[k]
        out[idx, 1] = x[k - 1]
        idx += 1
        out[idx, 0] = x[k]
        out[idx, 1] = x[k]
        idx += 1

    return out


def build_lead_lag_paths(
    prices: np.ndarray,
    times: np.ndarray,
    eps: float = 1e-6,
    batch_size: int = 2048,
) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError(f"Expected prices shape (N, L), got {prices.shape}")

    n_paths, path_len = prices.shape
    if times.shape[0] != path_len:
        raise ValueError(f"times length {times.shape[0]} != prices length {path_len}")

    if path_len < 2:
        raise ValueError("Lead-lag requires paths with at least 2 points")

    out = np.empty((n_paths, 2 * path_len - 1, 3), dtype=np.float64)
    out[:, :, 0] = np.linspace(0.0, 1.0, 2 * path_len - 1, dtype=np.float64)[None, :]

    chunk = max(1, int(batch_size))
    for start in range(0, n_paths, chunk):
        end = min(start + chunk, n_paths)
        batch = np.maximum(prices[start:end], eps)
        s0 = np.maximum(batch[:, [0]], eps)
        log_prices = np.log(batch / s0)

        for offset, series in enumerate(log_prices):
            ll = lead_lag_1d(series)
            out[start + offset, :, 1] = ll[:, 0]
            out[start + offset, :, 2] = ll[:, 1]

    return out


def validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available. Use --device cpu.")


def compute_signature(paths: np.ndarray, depth: int, device: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(paths, dtype=np.float32)).to(device)
    with torch.no_grad():
        feat = signatory.signature(tensor, depth=depth)
    return feat.detach().cpu().numpy()


def compute_logsignature(paths: np.ndarray, depth: int, device: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(paths, dtype=np.float32)).to(device)
    with torch.no_grad():
        try:
            feat = signatory.logsignature(tensor, depth=depth)
        except TypeError:
            feat = signatory.logsignature(tensor, depth=depth, mode="words")
    return feat.detach().cpu().numpy()


def compute_features(
    paths: np.ndarray,
    depth: int,
    device: str,
    batch_size: int,
    transform: str,
) -> np.ndarray:
    validate_device(device)

    chunk = max(1, int(batch_size))
    features: list[np.ndarray] = []
    for start in range(0, paths.shape[0], chunk):
        batch = np.asarray(paths[start : start + chunk], dtype=np.float32)
        if transform == "signature":
            features.append(compute_signature(batch, depth=depth, device=device))
        elif transform == "logsignature":
            features.append(compute_logsignature(batch, depth=depth, device=device))
        else:
            raise ValueError(f"Unknown transform: {transform}")
    return np.vstack(features)


def save_paths_file(path: Path, paths: np.ndarray, payload: dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    bundle = {"paths": paths}
    bundle.update(payload)
    np.savez_compressed(path, **bundle)


def save_feature_file(path: Path, features: np.ndarray, payload: dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    bundle = {"X_sig": features}
    bundle.update(payload)
    np.savez_compressed(path, **bundle)


def feature_output_name(split: str, variant: str, depth: int) -> str:
    return f"{split}_{variant}_depth{depth}.npz"


def compatibility_feature_aliases(dataset: str, split: str, variant: str, depth: int) -> list[str]:
    if variant == "base_sig":
        aliases = [f"{split}_base_signatures_depth{depth}.npz"]
        if dataset == "cev":
            aliases.append(f"{split}_signatures_depth{depth}.npz")
        return aliases

    if variant == "ll_sig":
        return [f"{split}_ll_signatures_depth{depth}.npz"]

    return []
