from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.multiscale_xgb_common import (
        DEFAULT_LOCAL_LOOKBACK,
        DEFAULT_MULTISCALE_WINDOWS,
        default_model_dir,
        load_xgb_model,
        normalize_scales,
        project_root,
        require_xgboost,
        rolling_multiscale_logsignature_features,
    )
    from src.realdata_systemB_variants import fetch_adj_close
except ImportError:
    from multiscale_xgb_common import (
        DEFAULT_LOCAL_LOOKBACK,
        DEFAULT_MULTISCALE_WINDOWS,
        default_model_dir,
        load_xgb_model,
        normalize_scales,
        project_root,
        require_xgboost,
        rolling_multiscale_logsignature_features,
    )
    from realdata_systemB_variants import fetch_adj_close


def run_scan_return_series(
    ticker: str,
    start: str,
    end: str | None,
    step: int,
    depth: int,
    dataset: str = "cev",
    device: str = "cpu",
    local_lookback: int = DEFAULT_LOCAL_LOOKBACK,
    multiscale_windows: tuple[int, ...] | list[int] | str | None = DEFAULT_MULTISCALE_WINDOWS,
    window: int | None = None,
    threshold: float = 0.8,
):
    require_xgboost()
    root = project_root()
    scales = normalize_scales(multiscale_windows)
    s = fetch_adj_close(ticker, start, end)
    X_feat, ends = rolling_multiscale_logsignature_features(
        s.values,
        scales=scales,
        depth=depth,
        device=device,
        batch_size=256,
        local_lookback=local_lookback,
        step=step,
    )
    model_dir = default_model_dir(dataset)
    model = load_xgb_model(model_dir / f"model_depth{depth}.joblib")
    probs = model.predict_proba(X_feat)[:, 1]
    dates = pd.DatetimeIndex(s.index[ends - 1], name="Date")
    probs_df = pd.DataFrame({"bubble_prob": probs}, index=dates)

    return {
        "price": s,
        "probs": probs_df,
        "meta": {
            "ticker": ticker,
            "dataset": dataset,
            "model_family": "multiscale_xgb",
            "feature_type": "logsignature",
            "classifier": "XGBoost",
            "depth": int(depth),
            "step": int(step),
            "scales": list(scales),
            "local_lookback": int(local_lookback),
            "start": start,
            "end": end,
            "threshold": float(threshold),
            "model_dir": str(model_dir.relative_to(root)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-data inference with the multiscale_xgb model family.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--dataset", default="cev", choices=["cev", "shifted_cev", "sin"])
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local_lookback", type=int, default=DEFAULT_LOCAL_LOOKBACK)
    parser.add_argument("--multiscale_windows", default="21,50,108,252")
    args = parser.parse_args()

    out = run_scan_return_series(
        ticker=args.ticker,
        dataset=args.dataset,
        start=args.start,
        end=args.end,
        step=args.step,
        depth=args.depth,
        device=args.device,
        local_lookback=args.local_lookback,
        multiscale_windows=args.multiscale_windows,
    )
    print(out["meta"])
    print(out["probs"].tail(10))


if __name__ == "__main__":
    main()
