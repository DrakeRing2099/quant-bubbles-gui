# Quant Bubbles — Asset Bubble Detection via Path Signatures

> **Live demo →** [quant-bubbles.streamlit.app](https://quant-bubbles.streamlit.app)  
> Detects asset price bubbles in real market data using path signature features and multiple ML classifiers, trained on synthetic SDE-based datasets.

---

## Update: current GUI status

This README now reflects two true parts of the project history:

- the broader research/development work, which included multiple model families and earlier signatory-based experiments
- the current hosted GUI path, which I switched later to a narrower roughpy-backed supervised inference setup

So the older work described below is still real project work. I have not removed it. But the current GUI in this repo is intentionally narrower and currently exposes:

- backend: `roughpy`
- model family: supervised `System B` variants
- datasets: `cev`, `shifted_cev`, `sin`
- depths:
  - `cev`: `3`, `4`
  - `shifted_cev`: `3`
  - `sin`: `3`

This later change was made to keep the hosted inference app simpler and more reproducible.

---

## What this project does

This system detects whether a financial asset is in a **price bubble** — specifically, whether its price process is a *strict local martingale* (SLM) rather than a true martingale. The distinction has rigorous mathematical grounding in stochastic calculus and is not observable from the price path directly, making it a non-trivial ML problem.

The pipeline:
1. **Simulates** synthetic price paths under several SDE families with known bubble/no-bubble labels
2. **Extracts** path signature features from rolling windows of the price path
3. **Trains** classifiers on the synthetic data
4. **Applies** the trained models to real market data (via Yahoo Finance) to produce a rolling bubble probability score

---

## The mathematics

### True martingales vs strict local martingales

A price process $S_t$ is a **bubble** in the mathematical finance sense if it is a strict local martingale under the risk-neutral measure — i.e., $\mathbb{E}[S_t] < S_0$ for some $t > 0$. This manifests in the CEV model:

$$dS_t = \gamma_0 \, S_t^{\gamma_1} \, dW_t$$

- $\gamma_1 \leq 1$ → true martingale (no bubble)
- $\gamma_1 > 1$ → strict local martingale (bubble)

The challenge: both regimes produce superficially similar price paths. The difference lives in the *fine structure* of the path, not its gross shape.

### Path signatures

The **signature** of a path $X : [0,T] \to \mathbb{R}^d$ is the collection of iterated integrals:

$$S(X)^{i_1, \ldots, i_k} = \int_{0 < t_1 < \cdots < t_k < T} dX^{i_1}_{t_1} \cdots dX^{i_k}_{t_k}$$

truncated at depth $N$. Key properties that make signatures ideal for this problem:

- **Universal non-linearity** — any continuous function of a path can be approximated by a linear function of its signature (Chen's theorem)
- **Captures fine path structure** — sensitive to the quadratic variation structure that distinguishes SLMs from martingales
- **Fixed-length feature vector** — regardless of path length, enabling standard classifiers

The **lead-lag transform** embeds a 1D price path into 2D before computing the signature, preserving quadratic variation information that would otherwise be lost.

### Synthetic datasets

| Dataset | SDE | Label mechanism | Size |
|---|---|---|---|
| `cev` | Standard CEV | Fixed $\gamma_1$ per path | 1k paths/model × 5 models |
| `shifted_cev` | Displaced CEV $dS = \sigma(S+d)^\beta dW$ | $\beta > 1$ → bubble | 80k / 10k / 10k |
| `sin` | Stochastic vol (2-factor) | $\sigma_1 > 1/3$ → bubble | 80k / 10k / 10k |
| `switching_cev` | Regime-switching CEV (Poisson switches) | Per-timestep label | 80k / 10k / 10k |

The `switching_cev` dataset is original to this project — paths switch between bubble and normal regimes at Poisson-random times, with $\gamma_1$ re-sampled per regime spell.

---

## Model families

### 1. Supervised (Signature + Logistic Regression)
The baseline supervised approach. Signature features fed into a `StandardScaler` → `LogisticRegression`. Four path/feature variants:

| Variant | Path transform | Feature |
|---|---|---|
| `ll_sig` | Lead-lag | Signature |
| `ll_log` | Lead-lag | Log-signature |
| `base_sig` | Base (time + log-price) | Signature |
| `base_log` | Base | Log-signature |

This is the currently enabled GUI path, now run through the `roughpy` backend for hosted inference.

### 2. Signature + MLP *(new)*
Replaces logistic regression with a neural network:
```
LayerNorm(D) → Linear(D→256) → GELU → Dropout(0.3)
             → Linear(256→64) → GELU → Linear(64→1) → Sigmoid
```
Trained with cosine LR decay, early stopping, and balanced class sampling via `WeightedRandomSampler`. Substantially more expressive than logistic regression on complex path distributions.

This remains part of the project work, but is not currently exposed in the hosted GUI.

### 3. Multiscale XGBoost
This remains part of the project work, but is not currently enabled in the hosted roughpy GUI.

Computes log-signature features at **four time scales simultaneously** (21, 50, 108, 252 trading days), concatenates them, and trains an XGBoost classifier. Captures both short-term volatility structure and long-term trend behaviour.

Each window's multichannel path includes: normalised time, log-price, rolling realised volatility, momentum, and drawdown — a richer feature set than single-scale variants.

### 4. Isolation Forest *(unsupervised baseline)*
Fits an Isolation Forest on signature features from an early baseline window of the price history, then scores subsequent windows as anomalous relative to that baseline. No labels required — useful as a reference signal and for assets with no analogous training data.

---

## Architecture

```
QuantBubblesInspector/
├── src/
│   ├── simulate_paths.py                    # CEV data generation
│   ├── simulate_shifted_cev_random_displacement.py
│   ├── simulate_sin_paths.py
│   ├── simulate_switching_cev.py            # NEW: Poisson regime-switching CEV
│   ├── build_systemB_paths_all.py           # Base + lead-lag path construction
│   ├── build_switching_cev_paths.py         # NEW: paths for switching_cev
│   ├── compute_systemB_features_all.py      # Signature / log-signature features
│   ├── build_multiscale_logsig_features.py  # Multiscale feature builder
│   ├── train_systemB_classifier_variants.py # Logistic regression training
│   ├── train_sig_mlp.py                     # NEW: MLP training
│   ├── train_multiscale_xgb.py              # XGBoost training
│   ├── realdata_systemB_variants.py         # Real-data inference (supervised)
│   ├── realdata_sig_mlp.py                  # NEW: Real-data inference (MLP)
│   ├── realdata_multiscale_xgb.py           # Real-data inference (XGBoost)
│   ├── realdata_systemB_iforest.py          # Real-data inference (iForest)
│   ├── systemB_pipeline_common.py           # Shared path/feature utilities
│   └── multiscale_xgb_common.py             # Multiscale utilities
├── gui/
│   ├── app.py                               # Streamlit frontend
│   ├── backend.py                           # Model routing layer
│   └── universes/universes.json             # 80+ tickers across asset classes
├── models/                                  # Pre-trained model artifacts
├── data/
│   └── raw/                                 # Sim configs + generated data
└── outputs/
```

**Backend routing** — `backend.py` acts as a clean adapter layer. `ScanConfig` captures all run parameters; `run_scan()` routes to the correct inference module based on `model_family`, returning a unified `ScanResult(price, probs, meta)` regardless of which model ran. Adding a new model family requires touching exactly two files.

---

## Results

The lead-lag signature model (trained on CEV, depth=3, window=108) correctly identifies the **Yes Bank bubble and collapse** (2018–2020):

- Bubble probability rises from ~0.05 (2014–2017) to **>0.9** at the peak in early 2020
- Crosses the 0.8 threshold approximately 6–8 months before the March 2020 collapse
- Returns cleanly to near-zero post-collapse, confirming the signal is not a permanent bias

This is consistent with the mathematical prediction: Yes Bank's price trajectory during this period exhibits the quadratic variation signature of a strict local martingale.

---

## GUI features

The current hosted GUI is intentionally restricted to the working roughpy supervised `System B` variants, even though the broader project explored additional model families.

- **Single ticker view** — rolling bubble probability + price chart, summary statistics, CSV export
- **Compare tickers** — overlay bubble probability for multiple tickers simultaneously
- **80+ tickers** across NSE India, US equities, global indices, ETFs, commodities, FX, and crypto
- **Model switcher** — compare all four model families on the same ticker interactively
- **Zoom presets** — quickly focus on the 2019–2020 period or custom date ranges
- **Autoscale** — probability y-axis auto-scales to visible peak for low-signal assets

---

## Setup and running locally

### Requirements
- Python 3.9+
- Windows / Linux / macOS

### Install
```bash
git clone https://github.com/yourusername/QuantBubblesInspector
cd QuantBubblesInspector
pip install -r requirements.txt
```

### Run the GUI (inference only, pre-trained models included)
```bash
streamlit run gui/app.py
```

Current GUI status:

- uses the `roughpy` backend at inference time
- loads the bundled roughpy supervised model artifacts from `models/`
- currently exposes only the supervised `System B` variants for `cev`, `shifted_cev`, and `sin`

### Reproduce training from scratch

```bash
# 1. Generate synthetic data
python src/simulate_paths.py
python src/simulate_shifted_cev_random_displacement.py
python src/simulate_sin_paths.py
python src/simulate_switching_cev.py          # new dataset

# 2. Build paths
python src/build_systemB_paths_all.py --dataset cev
python src/build_switching_cev_paths.py

# 3. Compute signature features
python src/compute_systemB_features_all.py --dataset cev --depth 3

# 4. Train classifiers
python src/train_systemB_classifier_variants.py --depth 3   # logistic regression
python src/train_sig_mlp.py --depth 3 --run_all_variants    # MLP
python src/train_multiscale_xgb.py --dataset cev --depth 3  # XGBoost
```

---

## Key dependencies

| Package | Purpose |
|---|---|
| `roughpy` | Current hosted GUI backend for signature and log-signature computation |
| `signatory` | Earlier PyTorch-based signature backend used in prior experiments |
| `torch` | MLP training and earlier GPU/signatory-based experiments |
| `xgboost` | Multiscale XGBoost classifier used in earlier project variants |
| `scikit-learn` | Logistic regression, Isolation Forest, preprocessing |
| `yfinance` | Real market data fetch |
| `streamlit` | Interactive GUI |
| `joblib` | Model serialisation |

For the current hosted GUI, the active deployment path is `roughpy` + supervised inference. The `signatory`, `torch`, and `xgboost` entries remain here because they were genuinely used in earlier stages of the project.

---

## Theoretical background

This project draws on:

* **Rough path theory and signature methods** — Terry Lyons (1998), Ilya Chevyrev & Andrey Kormilitzin (2016), and the generalized signature framework of James Morrill et al. (2021), which motivated using path signatures/logsignatures as structured features for time-series classification.

* **Strict local martingales and bubble theory** — Robert A. Jarrow, Philip Protter & Kenji Shimbo (2010), who characterize certain finite-horizon asset price bubbles through strict local martingales under equivalent martingale measures.

* **CEV-type diffusion models and bubbles** — the classical Constant Elasticity of Variance framework emphasized in later bubble literature, used here as one of the synthetic data-generating processes for separating bubble vs non-bubble regimes. Your code literally simulates CEV paths with bubble labels tied to the elasticity parameter. Humans do love turning theory into CSV files.

* **Machine-learning bubble detection** — recent work using deep learning for bubble identification, including Oksana Bashchenko & Alexis Marchal (2020), which inspired the broader idea of data-driven bubble classification from simulated price dynamics.

* **This project’s original contribution** — a reproducible pipeline that simulates multiple SDE regimes, converts paths into signature-ready representations, extracts signature/logsignature features, and trains classifiers (logistic regression / XGBoost variants) for bubble regime detection on synthetic and real equity data. Because naturally, instead of resting, you built a stochastic-geometry bubble detector.


---

## Disclaimer

This software is for research and educational purposes only. It does not constitute financial advice or a trading system.

---

*Built as part of a quantitative finance research project. All synthetic data generation, feature pipelines, model training, and the GUI were implemented from scratch.*
