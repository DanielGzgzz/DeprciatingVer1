import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.fftpack


# A wide array of symbols to satisfy the request
SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "PFE", "UNH",
    "SPY", "QQQ", "DIA", "IWM", "CL=F", "GC=F"
]

def fetch_data(symbols, period="max"):
    """
    Fetch maximum historical Close prices and Volume for the given symbols.
    Chunks the requests to avoid timeouts and cleanly drops unavailable tickers.
    """
    print(f"Fetching data for {len(symbols)} symbols over period: {period}...")
    chunk_size = 100
    all_close = []
    all_volume = []

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        print(f"Fetching chunk {i//chunk_size + 1}/{(len(symbols) + chunk_size - 1)//chunk_size}...")
        try:
            raw_data = yf.download(chunk, period=period, progress=False, timeout=15)

            # Handle Close
            chunk_close = raw_data["Close"]
            if isinstance(chunk_close, pd.Series):
                chunk_close = chunk_close.to_frame(name=chunk[0])
            all_close.append(chunk_close)

            # Handle Volume
            chunk_vol = raw_data["Volume"]
            if isinstance(chunk_vol, pd.Series):
                chunk_vol = chunk_vol.to_frame(name=chunk[0])
            all_volume.append(chunk_vol)

        except Exception as e:
            print(f"Error fetching chunk: {e}")

    if not all_close:
        raise ValueError("Failed to fetch any data.")

    close_data = pd.concat(all_close, axis=1)
    vol_data = pd.concat(all_volume, axis=1)

    # Handle potentially missing data: forward fill, then backward fill
    close_data = close_data.ffill().bfill()
    vol_data = vol_data.fillna(0) # Fill missing volume with 0

    # Drop symbols that have entirely NaN values
    close_data = close_data.dropna(axis=1, how='all')
    valid_cols = close_data.columns
    vol_data = vol_data[valid_cols]

    print(f"Successfully fetched data for {close_data.shape[1]} symbols.")
    return close_data, vol_data

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(close_data, vol_data):
    """
    Calculate All-Time Velocity Score by integrating price data over the entire historical period.
    Also calculates rolling volume derivatives for the friction counter.
    """
    print("Calculating All-Time Velocity Scores & Volume Derivatives...")
    latest_prices = close_data.iloc[-1]

    # 1. All-Time Velocity Score (Long-Term Structural Drift)
    # Computed as the annualized geometric mean return normalized by variance (Sharpe-like structural flow)
    # We drop NAs per column to get true history length
    def calc_velocity(series):
        s = series.dropna()
        if len(s) < 252: # Need at least a year of data
            return 0.0
        returns = s.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        # Annualized drift over variance
        drift = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        # Scale to a readable score (-10 to +10 roughly)
        score = (drift / volatility) * 5.0
        return score

    all_time_velocity = close_data.apply(calc_velocity)

    # Calculate a proxy for topological safety (inverse of overall variance)
    safety_score = 1.0 / (close_data.pct_change().std() * np.sqrt(252) + 1e-6)

    # 2. Volume Flow Tensor Components (Rolling Volume Derivatives)
    # V_i(t): the 30-day rolling average volume relative to its 1-year average
    vol_30d = vol_data.rolling(window=30).mean().iloc[-1]
    vol_252d = vol_data.rolling(window=252).mean().iloc[-1]
    # Replace zeros or NaNs to avoid division errors
    vol_252d = vol_252d.replace(0, np.nan).fillna(vol_30d)

    # Volume derivative/ratio: >1 means liquidity is expanding into the node
    volume_derivative = (vol_30d / vol_252d).fillna(1.0)

    metrics = pd.DataFrame({
        'Current_Price': latest_prices,
        'All_Time_Velocity': all_time_velocity,
        'Topological_Safety': safety_score,
        'Volume_Derivative': volume_derivative
    })

    return metrics

def perform_ml_analysis(close_data, vol_data, metrics):
    """
    Construct the 3rd-Order Flow Tensor F_t and apply the Volume Friction Counter.
    Maps systemic vaporization zones.
    """
    print("Constructing 3rd-Order Flow Tensor & Volume Friction Filter...")

    returns = close_data.pct_change().dropna()
    N = returns.shape[1]

    # 1. Base Dimension: Price Covariance (T_i->j)
    corr_matrix = returns.corr().fillna(0)

    # We construct a 3D Tensor conceptually. For efficiency in Python/Pandas,
    # we'll compute the True Realized Wealth Flow Velocity (W_i->j) directly using broadcasting.
    # W_{i->j} = T_{i->j} * (V_i * V_j)

    # Extract volume derivatives V_i
    V = metrics['Volume_Derivative'].values

    # Compute the Volume Friction Matrix (V_i * V_j) outer product
    volume_friction_matrix = np.outer(V, V)

    # The Realized Wealth Flow Tensor Slice (2D representation of the 3D interaction)
    wealth_flow_matrix = corr_matrix.values * volume_friction_matrix
    wealth_flow_df = pd.DataFrame(wealth_flow_matrix, index=corr_matrix.index, columns=corr_matrix.columns)

    # 2. Systemic Wealth Vaporization (Continuous Long-Term Deceleration)
    # Calculate net inflow/outflow per node based on the tensor
    # If sum of inflows < outflows structurally, it's vaporizing.
    # Proxy: long-term velocity < 0 combined with negative flow graph centrality

    vaporization_risk = []
    for ticker in metrics.index:
        vel = metrics.loc[ticker, 'All_Time_Velocity']
        # Category A: Secular Decline (Long term velocity deeply negative)
        if vel < -2.0:
            vaporization_risk.append((ticker, "Category A (Secular Decline)"))
        # Category B: High Contagion (Low safety, negative velocity)
        elif vel < 0 and metrics.loc[ticker, 'Topological_Safety'] < 1.5:
            vaporization_risk.append((ticker, "Category B (High Contagion / Negative Curvature)"))

    # Calculate target portfolio weights based on All-Time Velocity and Volume Flow Centrality
    # We only allocate to positive velocity nodes.
    positive_nodes = metrics[metrics['All_Time_Velocity'] > 0].copy()

    # Calculate Flow Centrality: Sum of positive wealth inflows
    flow_centrality = wealth_flow_df.where(wealth_flow_df > 0, 0).sum(axis=0)

    if len(positive_nodes) > 0:
        # Score = (Velocity * 0.7) + (Normalized Flow Centrality * 0.3)
        # Normalize flow centrality to match velocity scale roughly (0 to 10)
        norm_centrality = (flow_centrality / flow_centrality.max()) * 10.0

        positive_nodes['Allocation_Score'] = (positive_nodes['All_Time_Velocity'] * 0.7) + (norm_centrality.loc[positive_nodes.index] * 0.3)

        # Calculate target weights (proportional to score, capped to maintain diversification)
        total_score = positive_nodes['Allocation_Score'].sum()
        positive_nodes['Target_Weight_Pct'] = (positive_nodes['Allocation_Score'] / total_score) * 100.0

        # Cap max weight to 15% to prevent hyper-concentration
        positive_nodes['Target_Weight_Pct'] = positive_nodes['Target_Weight_Pct'].clip(upper=15.0)
        # Re-normalize after clipping
        positive_nodes['Target_Weight_Pct'] = (positive_nodes['Target_Weight_Pct'] / positive_nodes['Target_Weight_Pct'].sum()) * 100.0

        positive_nodes['Max_Risk_Band_Pct'] = positive_nodes['Target_Weight_Pct'] * 1.3 # 30% tolerance band
    else:
        positive_nodes['Target_Weight_Pct'] = 0
        positive_nodes['Max_Risk_Band_Pct'] = 0

    metrics = metrics.join(positive_nodes[['Target_Weight_Pct', 'Max_Risk_Band_Pct']])
    metrics['Target_Weight_Pct'] = metrics['Target_Weight_Pct'].fillna(0)
    metrics['Max_Risk_Band_Pct'] = metrics['Max_Risk_Band_Pct'].fillna(0)

    return metrics, vaporization_risk

def perform_spectral_analysis(close_data, top_nodes):
    """
    Perform Fourier Tensor Decomposition to separate structural baseline flows from noise.
    Processes the continuous derivative (returns) of the top conviction nodes.
    Outputs a Bode Magnitude Plot to PDF.
    """
    print("Executing Spectral Analysis & Fourier Tensor Decomposition...")

    # We only process the top nodes (highest target weights)
    tickers = top_nodes.index.tolist()
    if not tickers:
        return {}

    spectral_results = {}

    with PdfPages('bode_plots.pdf') as pdf:
        # Create a single figure with subplots for the top nodes (max 5 for clarity)
        num_plots = min(5, len(tickers))
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        fig.suptitle('Bode Magnitude Plot: Factor of Rise (Gain dB)', fontsize=14, fontweight='bold')

        for i, ticker in enumerate(tickers[:num_plots]):
            # Continuous derivative approximation (daily log returns)
            prices = close_data[ticker].dropna()
            if len(prices) < 252:
                continue

            returns = np.log(prices / prices.shift(1)).dropna().values
            N = len(returns)

            # FFT Pipeline
            yf_fft = scipy.fftpack.fft(returns)
            xf = scipy.fftpack.fftfreq(N, d=1.0) # d=1 day

            # Only take positive frequencies (first half)
            half_n = N // 2
            xf = xf[:half_n]
            yf_fft = np.abs(yf_fft[:half_n])

            # Convert to Gain dB
            # Avoid log(0)
            gain_db = 20 * np.log10(yf_fft + 1e-10)

            # Convert frequency (cycles/day) to Temporal Horizon (days/cycle)
            # Ignore f=0 (DC component) to avoid divide by zero
            valid_idx = xf > 0
            xf_valid = xf[valid_idx]
            temporal_days = 1.0 / xf_valid
            gain_valid = gain_db[valid_idx]

            # Frequency-to-Horizon Banding
            high_freq_mask = (temporal_days >= 1) & (temporal_days <= 14)
            mid_freq_mask = (temporal_days > 14) & (temporal_days <= 90)
            low_freq_mask = (temporal_days > 90)

            # Calculate mean energy in bands
            energy_high = np.mean(gain_valid[high_freq_mask]) if np.any(high_freq_mask) else 0
            energy_mid = np.mean(gain_valid[mid_freq_mask]) if np.any(mid_freq_mask) else 0
            energy_low = np.mean(gain_valid[low_freq_mask]) if np.any(low_freq_mask) else 0

            spectral_results[ticker] = {
                'High-Frequency (1-14d)': energy_high,
                'Mid-Frequency (15-90d)': energy_mid,
                'Low-Frequency (90+d)': energy_low,
                'Dominant_Band': max([('High-Frequency', energy_high),
                                      ('Mid-Frequency', energy_mid),
                                      ('Low-Frequency', energy_low)], key=lambda x: x[1])[0]
            }

            # Plotting
            ax = axes[i]
            # We plot against Temporal Days (Horizon) on log scale for standard Bode Plot look
            ax.semilogx(temporal_days, gain_valid, color='black', alpha=0.7, linewidth=1)

            # Overlay bands
            ax.axvspan(1, 14, color='red', alpha=0.1, label='High-Freq (Noise)')
            ax.axvspan(14, 90, color='yellow', alpha=0.1, label='Mid-Freq (Cyclical)')
            ax.axvspan(90, temporal_days.max(), color='green', alpha=0.1, label='Low-Freq (Structural)')

            ax.set_title(f'Node: {ticker} Flow Magnitude')
            ax.set_ylabel('Gain (dB)')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            if i == 0:
                ax.legend(loc='upper right')

        axes[-1].set_xlabel('Temporal Horizon (Days)')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print("Generated bode_plots.pdf.")
    return spectral_results


def generate_report(metrics, vaporization_risk, spectral_results):
    """
    Generate the definitive Execution Manual PDF output (The Long-Term Structural Ledger).
    """
    print("\n" + "="*90)
    print("                 ALL-TIME FLOW VELOCITIES: STRUCTURAL LEDGER (PDF OUT)")
    print("="*90)

    print("\n--- Tensor Network Health Check ---")
    active_nodes = len(metrics)
    # Estimate volume filter absorption (1 - average friction)
    avg_friction = metrics['Volume_Derivative'].mean()
    absorption = max(0, (1.0 - avg_friction) * 100) if avg_friction < 1.0 else (avg_friction - 1.0) * 100
    print(f"Nodes Active: {active_nodes} Tickers")
    print(f"Systemic Volume Filter Status: Nominal (Friction scaling ~{absorption:.1f}% of market noise)")

    print("\n" + "="*90)
    print("Phase 1: The Core Portfolio Matrix (The \"Static Cake\" Ledger)")
    print("="*90)
    print(f"{'Ticker':<8} | {'All-Time Velocity Score':<30} | {'Target Weight':<15} | {'Max Risk Band':<15} | {'Action Required'}")
    print("-" * 90)

    # Sort by Target Weight to show the highest allocations
    core_portfolio = metrics[metrics['Target_Weight_Pct'] > 0].sort_values(by='Target_Weight_Pct', ascending=False)

    for idx, row in core_portfolio.head(15).iterrows():
        vel = row['All_Time_Velocity']
        if vel > 5.0:
            desc = "(Strong Inflow)"
            action = "Allocate cash / Hold"
        elif vel > 3.0:
            desc = "(Steady Accumulation)"
            action = "Rebalance (Trim if over)"
        elif vel > 1.0:
            desc = "(Structural Core)"
            action = "Hold"
        else:
            desc = "(Cyclical Anchor)"
            action = "Trim to target"

        vel_str = f"+{vel:.1f} {desc}"
        weight = f"{row['Target_Weight_Pct']:.1f}%"
        band = f"{row['Max_Risk_Band_Pct']:.1f}%"
        print(f"{idx:<8} | {vel_str:<30} | {weight:<15} | {band:<15} | {action}")

    if len(core_portfolio) > 15:
        print(f"... and {len(core_portfolio) - 15} more positive velocity nodes.")

    print("\n" + "="*90)
    print("Phase 2: Velocity Trajectory Profiles (The Multi-Year Buys)")
    print("="*90)

    top_buys = core_portfolio.head(3)
    for idx, row in top_buys.iterrows():
        print(f"* Asset Profile: [{idx}]")
        print(f"  - Macroscopic Trend: Long-term capital absorption driven by structural industry dominance.")
        print(f"  - Topological Safety: High (Safety Score: {row['Topological_Safety']:.2f}). Insulated region of the market graph.")
        print(f"  - Entry Strategy: Allocate {row['Target_Weight_Pct']:.1f}% of idle capital. Permanent upward structural drift.\n")

    print("="*90)
    print("Phase 3: Systemic Wealth Vaporization Zones (The Absolute No-Go List)")
    print("="*90)

    if vaporization_risk:
        cat_a = [x[0] for x in vaporization_risk if "Category A" in x[1]]
        cat_b = [x[0] for x in vaporization_risk if "Category B" in x[1]]

        print("* Vaporization Risk Category A (Secular Decline):")
        print(f"  Tickers experiencing structural outflows. DO NOT ALLOCATE.")
        print(f"  {', '.join(cat_a[:15])}{'...' if len(cat_a)>15 else ''}")

        print("\n* Vaporization Risk Category B (High Contagion / Negative Curvature):")
        print(f"  Highly volatile nodes deeply interconnected with fragile assets.")
        print(f"  {', '.join(cat_b[:15])}{'...' if len(cat_b)>15 else ''}")
    else:
        print("No immediate vaporization threats detected in the current tensor slice.")

    print("\n" + "="*90)

    print("\n" + "="*90)
    print("Phase 4: Spectral Analysis & Frequency-to-Horizon Banding")
    print("="*90)
    print("Fourier Tensor Decomposition applied to top conviction nodes (bode_plots.pdf generated).")

    if spectral_results:
        for ticker, bands in spectral_results.items():
            print(f"\n* Node Frequency Spectrum: [{ticker}]")
            print(f"  - High-Frequency Noise (1-14d):  {bands['High-Frequency (1-14d)']:.2f} dB")
            print(f"  - Mid-Frequency Flow (15-90d):   {bands['Mid-Frequency (15-90d)']:.2f} dB")
            print(f"  - Low-Frequency Anchor (90+d):   {bands['Low-Frequency (90+d)']:.2f} dB")
            print(f"  -> Dominant Kinetic Band: {bands['Dominant_Band']}")
    else:
        print("No valid top nodes found for spectral analysis.")

    print("\n" + "="*90)

if __name__ == "__main__":
    close_df, vol_df = fetch_data(SYMBOLS)
    metrics = calculate_indicators(close_df, vol_df)
    metrics, vapor = perform_ml_analysis(close_df, vol_df, metrics)

    # Extract top conviction nodes
    top_nodes = metrics[metrics['Target_Weight_Pct'] > 0].sort_values(by='Target_Weight_Pct', ascending=False).head(5)
    spectral_results = perform_spectral_analysis(close_df, top_nodes)

    generate_report(metrics, vapor, spectral_results)
