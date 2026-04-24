from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

try:
    from src.systemB_pipeline_common import (
        compute_logsignature,
        load_raw_split,
        project_root,
        resolve_dataset_spec,
        resolve_dir,
        validate_device,
        write_json,
    )
except ImportError:
    from systemB_pipeline_common import (
        compute_logsignature,
        load_raw_split,
        project_root,
        resolve_dataset_spec,
        resolve_dir,
        validate_device,
        write_json,
    )


DEFAULT_MULTISCALE_WINDOWS: tuple[int, ...] = (21, 50, 108, 252)
DEFAULT_LOCAL_LOOKBACK = 10
CHANNEL_NAMES = [
    "time",
    "normalized_log_price",
    "rolling_realized_volatility",
    "momentum",
    "drawdown",
]


def require_xgboost():
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required for the multiscale_xgb pipeline. "
            "Install it from requirements.txt / gui/requirements_gui.txt first."
        ) from exc
    return xgboost


def normalize_scales(scales: Sequence[int] | str | None) -> tuple[int, ...]:
    if scales is None:
        return DEFAULT_MULTISCALE_WINDOWS
    if isinstance(scales, str):
        items = [part.strip() for part in scales.split(",") if part.strip()]
        values = tuple(int(part) for part in items)
    else:
        values = tuple(int(v) for v in scales)
    if not values:
        raise ValueError("multiscale windows cannot be empty")
    if any(v < 2 for v in values):
        raise ValueError(f"all multiscale windows must be >= 2, got {values}")
    return tuple(dict.fromkeys(values))


def multiscale_feature_dir(root: Path, dataset: str) -> Path:
    return root / "data" / "processed" / "multiscale_xgb_features" / dataset


def multiscale_model_dir(root: Path, dataset: str) -> Path:
    return root / "models" / "multiscale_xgb" / dataset


def multiscale_model_dir_roughpy(root: Path, dataset: str) -> Path:
    return root / "models" / "multiscale_xgb_roughpy" / dataset


def feature_split_filename(split: str, depth: int) -> str:
    return f"{split}_multiscale_logsig_depth{depth}.npz"


def model_filename(depth: int) -> str:
    return f"model_depth{depth}.joblib"


def rolling_realized_volatility(returns: np.ndarray, local_lookback: int) -> np.ndarray:
    out = np.zeros_like(returns, dtype=np.float64)
    for i in range(1, returns.shape[0]):
        start = max(1, i - local_lookback + 1)
        window = returns[start : i + 1]
        out[i] = float(np.sqrt(np.mean(window * window))) if window.size else 0.0
    return out


def momentum_channel(log_prices: np.ndarray, local_lookback: int) -> np.ndarray:
    out = np.zeros_like(log_prices, dtype=np.float64)
    for i in range(1, log_prices.shape[0]):
        ref = max(0, i - local_lookback)
        out[i] = float(log_prices[i] - log_prices[ref])
    return out


def drawdown_channel(prices: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(prices)
    return prices / running_max - 1.0


def build_multichannel_path(prices_window: np.ndarray, local_lookback: int, eps: float = 1e-6) -> np.ndarray:
    prices = np.asarray(prices_window, dtype=np.float64).reshape(-1)
    if prices.shape[0] < 2:
        raise ValueError("window must contain at least 2 points")
    if not np.all(np.isfinite(prices)):
        raise ValueError("window contains non-finite prices")

    prices = np.maximum(prices, eps)
    time = np.linspace(0.0, 1.0, prices.shape[0], dtype=np.float64)
    log_prices = np.log(prices / prices[0])

    returns = np.zeros_like(log_prices, dtype=np.float64)
    returns[1:] = np.diff(np.log(prices))

    vol = rolling_realized_volatility(returns, local_lookback=local_lookback)
    mom = momentum_channel(log_prices, local_lookback=local_lookback)
    dd = drawdown_channel(prices)

    return np.column_stack((time, log_prices, vol, mom, dd))


def build_multichannel_paths(price_windows: np.ndarray, local_lookback: int) -> np.ndarray:
    prices = np.asarray(price_windows, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError(f"expected price windows with shape (N, L), got {prices.shape}")
    paths = [build_multichannel_path(prices[i], local_lookback=local_lookback) for i in range(prices.shape[0])]
    return np.stack(paths, axis=0)


def scale_logsignature_features(
    price_windows: np.ndarray,
    depth: int,
    device: str,
    batch_size: int,
    local_lookback: int,
) -> np.ndarray:
    validate_device(device)
    feats: list[np.ndarray] = []
    chunk = max(1, int(batch_size))
    for start in range(0, price_windows.shape[0], chunk):
        batch_prices = price_windows[start : start + chunk]
        batch_paths = build_multichannel_paths(batch_prices, local_lookback=local_lookback)
        feats.append(compute_logsignature(batch_paths, depth=depth, device=device))
    return np.vstack(feats)


def multiscale_logsignature_features(
    prices: np.ndarray,
    scales: Sequence[int] | str | None,
    depth: int,
    device: str,
    batch_size: int,
    local_lookback: int,
) -> np.ndarray:
    scales_tuple = normalize_scales(scales)
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError(f"expected prices with shape (N, L), got {prices.shape}")
    if prices.shape[1] < max(scales_tuple):
        raise ValueError(f"path length {prices.shape[1]} is smaller than max scale {max(scales_tuple)}")

    blocks = []
    for scale in scales_tuple:
        scale_prices = prices[:, -scale:]
        blocks.append(
            scale_logsignature_features(
                scale_prices,
                depth=depth,
                device=device,
                batch_size=batch_size,
                local_lookback=local_lookback,
            )
        )
    return np.concatenate(blocks, axis=1)


def rolling_multiscale_logsignature_features(
    prices: np.ndarray,
    scales: Sequence[int] | str | None,
    depth: int,
    device: str,
    batch_size: int,
    local_lookback: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    scales_tuple = normalize_scales(scales)
    price_array = np.asarray(prices, dtype=np.float64).reshape(-1)
    max_scale = max(scales_tuple)
    ends = np.arange(max_scale, price_array.shape[0] + 1, int(step), dtype=int)
    if ends.size == 0:
        raise ValueError(f"not enough history for multiscale windows {scales_tuple}")

    blocks = []
    for scale in scales_tuple:
        scale_windows = np.stack([price_array[end - scale : end] for end in ends], axis=0)
        blocks.append(
            scale_logsignature_features(
                scale_windows,
                depth=depth,
                device=device,
                batch_size=batch_size,
                local_lookback=local_lookback,
            )
        )
    return np.concatenate(blocks, axis=1), ends


def save_feature_split(path: Path, features: np.ndarray, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"X_feat": features}
    bundle.update(payload)
    np.savez_compressed(path, **bundle)


def load_feature_split(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"missing features file: {path}")
    data = np.load(path, allow_pickle=True)
    if "X_feat" not in data.files:
        raise KeyError(f"missing X_feat in {path}")
    return np.asarray(data["X_feat"], dtype=np.float32), {key: data[key] for key in data.files if key != "X_feat"}


def save_xgb_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_xgb_model(path: Path):
    require_xgboost()
    if not path.exists():
        raise FileNotFoundError(f"missing XGBoost model: {path}")
    return joblib.load(path)


def default_feature_paths(dataset: str, in_dir: str | None = None, out_dir: str | None = None) -> tuple[Path, Path]:
    root = project_root()
    spec = resolve_dataset_spec(dataset)
    raw_dir = resolve_dir(root, in_dir, spec.raw_dir)
    features_dir = resolve_dir(root, out_dir, str(multiscale_feature_dir(root, dataset).relative_to(root)))
    return raw_dir, features_dir


def default_model_dir(dataset: str, out_dir: str | None = None) -> Path:
    root = project_root()
    if out_dir is not None:
        return resolve_dir(root, out_dir, str(multiscale_model_dir(root, dataset).relative_to(root)))

    flavor = os.environ.get("MULTISCALE_XGB_MODEL_FLAVOR", "auto").strip().lower()
    roughpy_dir = multiscale_model_dir_roughpy(root, dataset)
    legacy_dir = multiscale_model_dir(root, dataset)

    if flavor == "roughpy":
        return roughpy_dir
    if flavor == "legacy":
        return legacy_dir
    if flavor != "auto":
        raise ValueError(
            f"Unknown MULTISCALE_XGB_MODEL_FLAVOR={flavor!r}. Expected one of: auto, roughpy, legacy."
        )

    if roughpy_dir.exists():
        return roughpy_dir
    return legacy_dir


def write_feature_metadata(
    path: Path,
    *,
    dataset: str,
    dataset_label: str,
    raw_dir: Path,
    out_dir: Path,
    depth: int,
    device: str,
    batch_size: int,
    scales: Sequence[int],
    local_lookback: int,
    payload_keys: dict[str, list[str]],
) -> None:
    write_json(
        path,
        {
            "model_family": "multiscale_xgb",
            "dataset": dataset,
            "dataset_label": dataset_label,
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
            "feature_type": "logsignature",
            "classifier": "XGBoost",
            "channels": CHANNEL_NAMES,
            "scales": list(scales),
            "local_lookback": int(local_lookback),
            "signature_depth": int(depth),
            "compute": {"device": device, "batch_size": int(batch_size)},
            "payload_keys": payload_keys,
        },
    )
