from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional, Tuple
from pathlib import Path
import json

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from backend import ScanConfig, run_scan, REPO_ROOT, OUTPUTS_DIR


st.set_page_config(page_title="Quant Bubbles Inspector", layout="wide")


# ----------------------------
# Universes (ticker selection)
# ----------------------------
UNIVERSES_PATH = Path(__file__).parent / "universes" / "universes.json"


@st.cache_data(show_spinner=False)
def load_universes(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for universe, items in data.items():
        norm_items = []
        for it in items:
            sym = str(it.get("symbol", "")).strip().upper()
            name = str(it.get("name", "")).strip()
            if sym:
                norm_items.append({"symbol": sym, "name": name})
        out[universe] = norm_items
    return out


def format_item(it: dict) -> str:
    name = it.get("name") or ""
    sym = it.get("symbol") or ""
    return f"{name} ({sym})" if name else sym


def filter_items(items: list[dict], q: str) -> list[dict]:
    q = (q or "").strip().lower()
    if not q:
        return items
    out = []
    for it in items:
        sym = (it.get("symbol") or "").lower()
        name = (it.get("name") or "").lower()
        if q in sym or q in name:
            out.append(it)
    return out


# ----------------------------
# Helpers
# ----------------------------
def _preset_windows() -> List[int]:
    return [21, 50, 108, 252]


def _parse_tickers(text: str) -> List[str]:
    tickers = [t.strip().upper() for t in text.replace("\n", ",").split(",")]
    return [t for t in tickers if t]


def _parse_windows(text: str) -> tuple[int, ...]:
    items = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    values = tuple(int(part) for part in items)
    if not values or any(v < 2 for v in values):
        raise ValueError("Multiscale windows must be integers >= 2.")
    return values


AVAILABLE_SUPERVISED_OPTIONS: dict[str, list[int]] = {
    "cev": [3, 4],
    "shifted_cev": [3],
    "sin": [3],
}


def _safe_selectbox(label: str, options: list, key: str):
    if not options:
        raise ValueError(f"No options available for {label}.")
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return st.selectbox(label, options, key=key)


def _slice_by_date(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out


def _slice_series(s: pd.Series, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.Series:
    out = s
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out


@st.cache_data(show_spinner=False)
def cached_scan(cfg_dict: dict) -> dict:
    cfg = ScanConfig(**cfg_dict)
    res = run_scan(cfg)
    return {
        "price": res.price,
        "probs": res.probs,
        "meta": res.meta,
    }


def _plot_price(ax, price: pd.Series, title: str):
    ax.plot(price.index, price.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    _format_date_axis(ax)


def _plot_prob(ax, probs: pd.DataFrame, threshold: float, title: str, autoscale: bool):
    y = probs["bubble_prob"].astype(float)
    ax.plot(probs.index, y.values)
    ax.axhline(threshold)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("P(bubble)")
    _format_date_axis(ax)
    if autoscale and len(y) > 0 and np.isfinite(y).any():
        peak = float(np.nanmax(y.values))
        ymax = min(1.0, peak * 1.05) if peak > 0 else 1.0
        ax.set_ylim(-0.02, max(0.2, ymax))
    else:
        ax.set_ylim(-0.02, 1.02)


def _format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _make_figure(figsize: Tuple[float, float] = (9.5, 4.8)):
    return plt.subplots(figsize=figsize, constrained_layout=True)


def _summary_stats(probs: pd.DataFrame, threshold: float) -> dict:
    y = probs["bubble_prob"].astype(float)
    out = {
        "n_points": int(len(y)),
        "max_prob": float(np.nanmax(y.values)) if len(y) else float("nan"),
        "mean_prob": float(np.nanmean(y.values)) if len(y) else float("nan"),
        "frac_ge_threshold": float(np.mean(y.values >= threshold)) if len(y) else float("nan"),
    }
    hit = probs.index[y.values >= threshold]
    out["first_hit_date"] = hit[0].date().isoformat() if len(hit) else None
    return out


st.title("Quant Bubbles Inspector")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Run settings")

    # normalize start/end ONCE and reuse everywhere
    start_raw = st.text_input("Start date (YYYY-MM-DD)", value="2014-01-01")
    end_raw = st.text_input("End date (YYYY-MM-DD, optional)", value="")
    start = start_raw.strip() or None
    end = end_raw.strip() or None

    model_family = _safe_selectbox(
        "Model family",
        ["supervised_systemB"],
        key="model_family",
    )
    st.caption("Only working roughpy supervised models are enabled in this GUI.")
    model_kind = "supervised"

    dataset = _safe_selectbox(
        "Dataset",
        list(AVAILABLE_SUPERVISED_OPTIONS.keys()),
        key="dataset",
    )
    path_type = _safe_selectbox("Path type", ["base", "lead-lag"], key="path_type")
    feature_type = _safe_selectbox("Feature type", ["signature", "logsignature"], key="feature_type")
    variant = None
    local_lookback = 10
    multiscale_windows = (21, 50, 108, 252)

    depth = _safe_selectbox("Depth", AVAILABLE_SUPERVISED_OPTIONS[dataset], key="depth")
    if len(AVAILABLE_SUPERVISED_OPTIONS[dataset]) == 1:
        st.caption(f"Only depth {depth} is available for {dataset}.")

    colw, cols = st.columns(2)
    with colw:
        window = st.number_input(
            "Window (trading days)",
            min_value=10,
            max_value=2000,
            value=108,
            step=1,
        )
    with cols:
        step = st.number_input(
            "Step (days)",
            min_value=1,
            max_value=250,
            value=5,
            step=1,
        )

    threshold = st.slider("Threshold", min_value=0.10, max_value=0.99, value=0.80, step=0.01)
    fit_frac = 0.25
    seed = 0

    st.divider()
    st.header("Tickers")

    universes = load_universes(UNIVERSES_PATH)
    if not universes:
        st.warning(f"Missing universes file: {UNIVERSES_PATH}")
        universes = {"(empty)": []}

    universe_names = list(universes.keys())
    universe = st.selectbox("Universe", universe_names, index=0)

    items = universes.get(universe, [])
    q = st.text_input("Search (name or symbol)", value="")

    filtered = filter_items(items, q)

    MAX_SHOW = 300
    if len(filtered) > MAX_SHOW:
        st.warning(f"Too many matches ({len(filtered)}). Narrow the search.")
        filtered = filtered[:MAX_SHOW]

    if len(filtered) == 0:
        st.info("No matches. Type a custom ticker below.")
        selected_symbol = ""
    else:
        labels = [format_item(it) for it in filtered]
        selected_label = st.selectbox("Select ticker", labels, index=0)
        selected_symbol = filtered[labels.index(selected_label)]["symbol"]

    custom = st.text_input("Or enter custom ticker", value="")
    ticker_single = (custom.strip().upper() or selected_symbol).strip()

    st.caption("Compare: select multiple tickers (same universe), or paste custom tickers below.")
    multi_labels = [format_item(it) for it in filtered] if filtered else []
    selected_multi_labels = st.multiselect("Select tickers to compare", multi_labels, default=[])

    selected_multi = []
    label_to_symbol = {format_item(it): it["symbol"] for it in filtered}
    for lab in selected_multi_labels:
        if lab in label_to_symbol:
            selected_multi.append(label_to_symbol[lab])

    tickers_multi_text = st.text_area(
        "Or paste tickers (comma/newline)",
        value="",
        height=90,
    )

    pasted = _parse_tickers(tickers_multi_text)
    tickers_multi = list(dict.fromkeys([*selected_multi, *pasted]))  # unique, preserve order

    st.divider()
    st.header("View controls")

    autoscale_prob = st.checkbox("Auto-scale probability y-axis to visible peak", value=False)

    zoom_preset = st.selectbox(
        "Zoom preset",
        ["Full range", "Mid-2019 → Mid-2020", "Custom"],
        index=0,
    )

    custom_start_ts: Optional[pd.Timestamp] = None
    custom_end_ts: Optional[pd.Timestamp] = None
    if zoom_preset == "Custom":
        use_custom = st.checkbox("Enable custom zoom dates", value=True)
        if use_custom:
            cs = st.date_input("Start date", value=pd.Timestamp("2019-06-01").date())
            ce = st.date_input("End date", value=pd.Timestamp("2020-06-30").date())
            custom_start_ts = pd.Timestamp(cs) if cs else None
            custom_end_ts = pd.Timestamp(ce) if ce else None

    run_button = st.button("Run", type="primary")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["Single ticker", "Compare tickers"])


def resolve_zoom_range(index: pd.DatetimeIndex) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if zoom_preset == "Full range":
        return None, None
    if zoom_preset == "Mid-2019 → Mid-2020":
        return pd.Timestamp("2019-06-01"), pd.Timestamp("2020-06-30")
    return custom_start_ts, custom_end_ts


# ----------------------------
# Tab 1
# ----------------------------
with tab1:
    st.subheader("Single ticker view")

    if not run_button:
        st.info("Set parameters on the left and click **Run**.")
    else:
        cfg = ScanConfig(
            ticker=ticker_single.strip(),
            depth=int(depth),
            window=int(window),
            step=int(step),
            threshold=float(threshold),
            variant=variant,
            start=start,
            end=end,
            dataset=dataset,
            path_type=path_type,
            feature_type=feature_type,
            model_family=model_family,
            local_lookback=int(local_lookback),
            multiscale_windows=multiscale_windows,
            model_kind=model_kind,      # NEW
            fit_frac=float(fit_frac),   # NEW
            seed=int(seed),             # NEW
        )

        if step >= window:
            st.warning("Step should be smaller than window (otherwise windows barely overlap).")

        cfg_dict = asdict(cfg)

        with st.spinner("Running scan..."):
            try:
                out = cached_scan(cfg_dict)
            except Exception as e:
                st.error(f"Scan failed for {cfg.ticker}: {e}")
                st.stop()

        price = out["price"]
        probs = out["probs"]
        meta = out["meta"]

        z0, z1 = resolve_zoom_range(probs.index)
        price_v = _slice_series(price, z0, z1)
        probs_v = _slice_by_date(probs, z0, z1)

        colA, colB = st.columns([2, 1], gap="large")

        # Title fragments
        model_tag = f"{dataset} | {path_type} | {feature_type}"
        config_label = f"Dataset: {dataset} | Path: {path_type} | Feature: {feature_type} | Depth: {depth}"
        config_slug = f"{dataset}_{'ll' if path_type == 'lead-lag' else 'base'}_{'log' if feature_type == 'logsignature' else 'sig'}"

        with colA:
            st.caption(config_label)
            fig, ax = _make_figure()
            _plot_prob(
                ax,
                probs_v,
                threshold,
                f"{cfg.ticker} — bubble probability ({model_tag}, depth={depth})",
                autoscale_prob,
            )
            st.pyplot(fig, clear_figure=True)

            fig2, ax2 = _make_figure()
            _plot_price(ax2, price_v, f"{cfg.ticker} adjusted close")
            st.pyplot(fig2, clear_figure=True)

        with colB:
            stats = _summary_stats(probs_v, threshold)
            st.markdown("**Summary (visible range)**")
            st.json(stats)

            if autoscale_prob and len(probs_v) > 0:
                st.caption(f"Visible peak probability: {stats['max_prob']:.3f}")

            st.markdown("**Run metadata**")
            st.json(meta if meta else {"meta": None})

        st.divider()
        save_col1, save_col2 = st.columns(2)

        with save_col2:
            csv_bytes = probs_v.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download visible probs CSV",
                data=csv_bytes,
                file_name=f"{cfg.ticker}_{config_slug}_d{depth}_w{window}_s{step}_probs.csv",
                mime="text/csv",
            )


# ----------------------------
# Tab 2
# ----------------------------
with tab2:
    st.subheader("Compare tickers (probability overlay)")

    tickers = tickers_multi

    if not run_button:
        st.info("Set parameters on the left and click **Run** to compute, then view comparisons here.")
    else:
        if len(tickers) == 0:
            st.warning("No tickers provided.")
        else:
            z0, z1 = resolve_zoom_range(pd.DatetimeIndex([]))

            series = {}
            errors = []

            for t in tickers:
                cfg = ScanConfig(
                    ticker=t,
                    depth=int(depth),
                    window=int(window),
                    step=int(step),
                    threshold=float(threshold),
                    variant=variant,
                    start=start,
                    end=end,
                    dataset=dataset,
                    path_type=path_type,
                    feature_type=feature_type,
                    model_family=model_family,
                    local_lookback=int(local_lookback),
                    multiscale_windows=multiscale_windows,
                    model_kind=model_kind,      # NEW
                    fit_frac=float(fit_frac),   # NEW
                    seed=int(seed),             # NEW
                )

                try:
                    out = cached_scan(asdict(cfg))
                    probs = out["probs"]
                    probs_v = _slice_by_date(probs, z0, z1)
                    series[t] = probs_v["bubble_prob"].astype(float)
                except Exception as e:
                    errors.append(f"{t}: {e}")

            if errors:
                st.warning("Some tickers failed:\n" + "\n".join(errors))

            if len(series) == 0:
                st.error("All tickers failed. Check symbols / date range / data source.")
                st.stop()

            df = pd.concat(series, axis=1).sort_index()

            model_tag = f"{dataset} | {path_type} | {feature_type}"
            config_label = f"Dataset: {dataset} | Path: {path_type} | Feature: {feature_type} | Depth: {depth}"

            st.caption(config_label)
            fig, ax = _make_figure()
            for col in df.columns:
                ax.plot(df.index, df[col].values, label=col)
            ax.axhline(threshold)
            ax.set_title(f"Bubble probability comparison ({model_tag}, depth={depth}, window={window}, step={step})")
            ax.set_xlabel("Date")
            ax.set_ylabel("P(bubble)")
            _format_date_axis(ax)

            if autoscale_prob:
                peak = float(np.nanmax(df.values)) if df.size else 1.0
                ymax = min(1.0, peak * 1.05) if peak > 0 else 1.0
                ax.set_ylim(-0.02, max(0.2, ymax))
            else:
                ax.set_ylim(-0.02, 1.02)

            ax.legend(loc="upper left")
            st.pyplot(fig, clear_figure=True)

            st.markdown("**Quick stats (visible range)**")
            rows = []
            for t in df.columns:
                y = df[t].values
                rows.append({
                    "ticker": t,
                    "max": float(np.nanmax(y)),
                    "mean": float(np.nanmean(y)),
                    "frac>=thr": float(np.mean(y >= threshold)),
                })
            st.dataframe(pd.DataFrame(rows).sort_values("max", ascending=False), use_container_width=True)
