# Thermodynamic Spectral Market Engine

This repository contains a high-performance, deterministic, and neural-network-free market analysis framework. It maps the global stock market as a coupled network of linear and quasi-linear oscillators using advanced pure signal processing and topological mathematics.

The algorithm treats capital flow not as discrete daily steps, but as a continuous fluid diffusing across a multi-dimensional tensor matrix. It tracks over **934 unique market nodes** across equities, ETFs, forex, and commodities, analyzing **871,422 continuous directed flow vectors** simultaneously.

---

## 🧠 Core Mathematical Architecture

This engine abandons standard portfolio theory (Modern Portfolio Theory, CAPM) and black-box Machine Learning (LSTMs, Transformers). Instead, it utilizes strict deterministic physics and signal processing to map structural capital migration.

### 1. The 3rd-Order Flow Tensor & Volume Friction
Rather than analyzing stocks in a vacuum, the system represents the entire market as a dynamic tensor network $\mathcal{F}_{t} \in \mathbb{R}^{N \times N \times D}$.
*   Capital cannot flow instantly or seamlessly between nodes without encountering liquidity resistance. To capture this, the engine applies a **Volume Flow Tensor** filter.
*   The true realized wealth flow velocity ($\mathcal{W}_{i \to j}$) from stock $i$ to stock $j$ is regulated by the rolling volume friction matrix, preventing false allocation signals from low-liquidity price spikes.

### 2. Spectral Market Engine (Fourier Decomposition)
Instead of looking at instantaneous daily speeds, the algorithm uses a Fast Fourier Transform (FFT) pipeline to separate structural baseline flows from short-term market noise.
*   The **Continuous Derivative ($dh/dt$)** of asset momentum is transformed from the time domain into the frequency domain $H(f)$.
*   Frequencies are segmented into chronological horizons: **High-Frequency (1-14d)** noise, **Mid-Frequency (15-90d)** cyclical flow, and **Low-Frequency (90d+)** structural baseline trends.
*   This output is rendered into a **Bode Magnitude Plot** saved locally as `bode_plots.pdf`.

### 3. Kuramoto Phase Synchronization (Systemic Crash Detection)
Before a massive macro crash (e.g., 2008 Financial Crisis, 2020 COVID shock), assets across completely independent sectors begin to mathematically synchronize their phases.
*   The algorithm extracts the instantaneous phase $\theta_i(t)$ of every asset using the **Hilbert Transform**.
*   By tracking the global **Kuramoto Order Parameter $r(t)$**, the system can predict structural vaporization *before* prices drop heavily. If the global synchronization breaches a critical threshold ($r > 0.85$), the system mathematically forces a rotation entirely into safe-haven nodes (e.g., Treasuries and Gold).

### 4. Eigenvector Flow Centrality (Identifying Capital Sinks)
An asset is only considered a "safe capital sink" if the capital flowing into it is coming from *other highly resilient capital sinks* (analogous to Google's PageRank).
*   The algorithm calculates the principal eigenvector of the asymmetric $N \times N$ Transfer Matrix. This prevents the system from buying a decaying asset merely because speculative capital is temporarily rotating into it.

### 5. Continuous-Time Fractional Kelly Sizing
Target portfolio weights are not hardcoded. The matrix dynamically calculates the exact optimal bet size using a bounded continuous-time fractional Kelly Criterion.
*   $f^* = \frac{\mu - r}{\sigma^2}$
*   The algorithm uses the **All-Time Velocity Score** as a proxy for the drift/variance ratio, heavily weighted by the node's Eigenvector Centrality, ensuring that capital is only allocated to mathematically verified structural sinks.

---

## 🚀 Installation & Setup

1. **Install Dependencies**
The environment strictly requires standard quantitative libraries. There are no heavy ML dependencies (like PyTorch or TensorFlow).

```bash
pip install -r requirements.txt
```

*(Core dependencies: `numpy`, `pandas`, `scipy`, `yfinance`, `matplotlib`)*

---

## 📊 Running the Engine

### 1. The Core Execution Ledger
To process the current market state and generate your active execution parameters, run the primary analyzer:

```bash
python market_analyzer.py
```

**Outputs:**
1.  **Terminal Ledger:** A tabular report divided into 4 Phases.
    *   **Phase 1:** Core Portfolio Matrix (Exact target allocation weights).
    *   **Phase 2:** Velocity Trajectory Profiles (The Multi-Year conviction buys).
    *   **Phase 3:** Systemic Wealth Vaporization Zones (The Absolute No-Go/Liquidation List).
    *   **Phase 4:** Spectral Analysis & Frequency-to-Horizon Banding.
2.  **PDF Report (`bode_plots.pdf`):** Visual Bode Magnitude Plots of the top conviction nodes separating structural accumulation from cyclical noise.

### 2. Historical Rigorous Backtesting
To validate the model's thermodynamic crash-detection and structural growth capabilities against historical regimes, run the backtest suite. The suite output includes granular month-over-month performance metrics, historical Best/Worst month tracking, and calendar seasonality averages:

```bash
python rigorous_backtest.py
```

This will execute a forward-walking simulation using a 50,000 NIS starting balance and a monthly rebalancing frequency, testing the algorithm's performance vs. the SPY benchmark across:
*   **The 2008 Financial Crisis** (Testing the Kuramoto Phase Defense / Crash Rotation)
*   **The 2010s Bull Run** (Testing Eigenvector Centrality compounding)
*   **The 2022 Tech Bear Market** (Testing Fractional Kelly Sizing scaling)
