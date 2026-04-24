# realdata_systemB_iforest.py
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import signatory

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ------------------------
# Utilities
# ------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fetch_adj_close(ticker: str, start: str, end: str | None) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker}")

    s = df["Close"].dropna()
    s = s[~s.index.duplicated(keep="last")]

    if len(s) < 200:
        raise ValueError(f"Too few points for {ticker}: {len(s)}")

    return s


def make_paths(
    s: pd.Series,
    window: int,
    step: int,
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Build System B paths:
      channels = [t_norm, log(price / price_0)]
    """
    x = np.asarray(s.values, dtype=float)
    dates = s.index

    ends = list(range(window, len(x) + 1, step))
    if len(ends) < 20:
        raise ValueError("Too few rolling windows.")

    t = np.linspace(0.0, 1.0, window, dtype=float)

    paths = []
    end_dates = []

    for e in ends:
        w = x[e - window : e]
        if np.any(w <= 0.0):
            continue

        lp = np.log(w / w[0])
        if not np.all(np.isfinite(lp)):
            continue

        path = np.column_stack((t, lp))
        paths.append(path)
        end_dates.append(dates[e - 1])

    if not paths:
        raise ValueError("No valid windows constructed.")

    return np.stack(paths, axis=0), end_dates


def compute_signatures(
    paths: np.ndarray,
    depth: int,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    X = torch.from_numpy(paths).to(torch.float32).to(device)

    feats = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X[i : i + batch_size]
            sig = signatory.signature(batch, depth=depth)
            feats.append(sig.cpu().numpy())

    return np.vstack(feats)


# ------------------------
# Isolation Forest logic
# ------------------------

def isolation_forest_scores(
    X: np.ndarray,
    fit_frac: float,
    seed: int,
) -> Tuple[np.ndarray, dict]:
    """
    X : signature features, shape (N, D)
    """
    n = X.shape[0]
    fit_n = max(50, int(fit_frac * n))

    # Scale on ALL data (important)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_fit = Xs[:fit_n]

    iso = IsolationForest(
        n_estimators=500,
        contamination=0.05,
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_fit)

    score = -iso.decision_function(Xs)  # higher = more anomalous

    meta = {
        "method": "IsolationForest",
        "fit_n": fit_n,
        "contamination": 0.05,
        "n_estimators": 500,
    }

    return score, meta


def run_scan_return_series(
    ticker: str,
    start: str,
    end: str | None,
    window: int,
    step: int,
    depth: int,
    device: str = "cpu",
    fit_frac: float = 0.25,
    seed: int = 0,
):
    """
    GUI-compatible entrypoint.

    Returns:
      price: pd.Series (adj close)
      probs_df: pd.DataFrame indexed by Date with column 'bubble_prob'
               (here bubble_prob := normalized anomaly score in [0,1])
      meta: dict
    """
    s = fetch_adj_close(ticker, start, end)
    paths, end_dates = make_paths(s, window=window, step=step)
    Xsig = compute_signatures(paths, depth=depth, device=device)

    scores, meta = isolation_forest_scores(Xsig, fit_frac=fit_frac, seed=seed)

    a = scores
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-12)

    probs_df = pd.DataFrame(
        {"bubble_prob": a_norm},
        index=pd.DatetimeIndex(end_dates, name="Date"),
    )

    meta = {
        **meta,
        "ticker": ticker,
        "start": start,
        "end": end,
        "window": int(window),
        "step": int(step),
        "depth": int(depth),
        "device": device,
        "fit_frac": float(fit_frac),
        "seed": int(seed),
        "note": "bubble_prob is normalized IsolationForest anomaly score",
    }
    return s, probs_df, meta


# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="System B — Isolation Forest anomaly detection on signature features."
    )
    parser.add_argument("--ticker", type=str, default="YESBANK.NS")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--window", type=int, default=108)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fit_frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/systemB_iforest",
    )

    args = parser.parse_args()

    root = project_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data → paths → signatures
    s = fetch_adj_close(args.ticker, args.start, args.end)
    paths, dates = make_paths(s, args.window, args.step)
    Xsig = compute_signatures(paths, depth=args.depth, device=args.device)

    # 2) Isolation Forest
    scores, meta = isolation_forest_scores(
        Xsig,
        fit_frac=args.fit_frac,
        seed=args.seed,
    )

    # 3) Normalize for GUI friendliness
    a = scores
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-12)

    out = pd.DataFrame(
        {
            "anomaly_score": a,
            "anomaly_score_norm": a_norm,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )

    tag = f"{args.ticker}_w{args.window}_s{args.step}_d{args.depth}_iforest"
    csv_path = out_dir / f"{tag}.csv"
    out.to_csv(csv_path)

    print(f"Saved: {csv_path}")
    print("Meta:", meta)
    print(out.tail(10))


if __name__ == "__main__":
    main()
