import numpy as np
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from scipy.signal import hilbert
from market_analyzer import SYMBOLS, fetch_data, calculate_indicators, perform_ml_analysis, perform_spectral_analysis

def evaluate_ticker(ticker):
    core_benchmarks = ["SPY", "QQQ", "TLT", "GLD", "XOM", "JNJ", "MSFT", "AAPL"]
    if ticker not in core_benchmarks:
        temp_symbols = core_benchmarks + [ticker]
    else:
        temp_symbols = core_benchmarks

    import sys, os
    old_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            close_df, vol_df = fetch_data(temp_symbols)
            metrics = calculate_indicators(close_df, vol_df)
            metrics, vapor = perform_ml_analysis(close_df, vol_df, metrics)
        finally:
            sys.stdout = old_stdout

    if ticker in metrics.index:
        row = metrics.loc[ticker]
        current_price = row['Current_Price']

        # Calculate Real Value V_tensor
        try:
            real_value = close_df[ticker].rolling(window=126).mean().iloc[-1]
        except:
            real_value = current_price

        # Pull global network data
        global_r = metrics.loc[metrics.index[0], 'Global_Kuramoto_Sync'] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
        kuramoto_accel = metrics.loc[metrics.index[0], 'Kuramoto_Phase_Accel'] if 'Kuramoto_Phase_Accel' in metrics.columns else 0.0

        # Retrieve Fundamental Data via yfinance
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        # A. Dividend Yield Worth (D_i)
        div_yield = info.get('dividendYield', 0)
        if div_yield is None: div_yield = 0
        payout_ratio = info.get('payoutRatio', 0)
        if payout_ratio is None: payout_ratio = 0

        if payout_ratio > 1.0:
            D_i = 0.01
            div_status = "UNSUSTAINABLE"
        else:
            D_i = max(0.01, 1.0 + (div_yield * 10 * (1 - payout_ratio)))
            div_status = "OPTIMAL" if div_yield > 0 else "N/A"

        # B. Earnings Surprise Momentum (E_i)
        E_i = 1.0
        e_str = ""
        try:
            edates = yf_ticker.earnings_dates
            if edates is not None and len(edates) > 0:
                past_edates = edates.dropna(subset=['Reported EPS']).head(4)
                if len(past_edates) == 4:
                    e_sum = 0
                    weights = [0.4, 0.3, 0.2, 0.1]
                    e_strs = []
                    for i in range(4):
                        est = past_edates.iloc[i]['EPS Estimate']
                        act = past_edates.iloc[i]['Reported EPS']
                        if pd.isna(est) or est == 0: est = 1e-5
                        diff = act - est
                        e_sum += weights[i] * np.sign(diff) * abs(diff / est)
                        e_strs.append("[HIT]" if diff >= 0 else "[MISS]")
                    e_strs.reverse()
                    e_str = "".join(e_strs)
                    E_i = max(0.01, 1.0 + e_sum)
                else:
                    E_i = 1.0
                    e_str = "INSUFFICIENT DATA"
        except Exception:
            E_i = 1.0
            e_str = "ERROR RETRIEVING DATA"

        # C. Structural Growth vs. Fade (G_i)
        G_i = 1.0
        g_status = "UNKNOWN"
        try:
            rev = yf_ticker.quarterly_financials.loc['Total Revenue'] if 'Total Revenue' in yf_ticker.quarterly_financials.index else None
            fcf = yf_ticker.quarterly_cashflow.loc['Free Cash Flow'] if 'Free Cash Flow' in yf_ticker.quarterly_cashflow.index else None

            if rev is not None and len(rev.dropna()) >= 2 and fcf is not None and len(fcf.dropna()) >= 2:
                rev = rev.dropna()
                fcf = fcf.dropna()
                delta_rev = rev.iloc[0] - rev.iloc[1]
                delta_fcf = fcf.iloc[0] - fcf.iloc[1]
                sig_rev = rev.std()
                if sig_rev == 0: sig_rev = 1e-5
                sig_fcf = fcf.std()
                if sig_fcf == 0: sig_fcf = 1e-5
                G_i = np.tanh((delta_rev / sig_rev) + (delta_fcf / sig_fcf)) + 1.0
                g_status = "EXPANDING" if G_i > 1.0 else "FADING"
        except Exception:
            pass

        # D. Sentiment Velocity (S_i)
        vel = row['All_Time_Velocity']
        S_i = max(0.01, 1.0 + (vel / 10.0))
        s_status = "POSITIVE" if S_i > 1.0 else "NEGATIVE"

        # Base Compute Dynamic Fundamental Mass
        m_i = (S_i * D_i * E_i * G_i) ** 0.25

        # --- Dynamic Execution Logic Gates ---

        # We need historical SL state for Markov Penalty. In this CLI runner, we will simulate it.
        # Check if a local cache file exists for Markov state
        markov_file = f".{ticker}_markov.txt"
        strike_count = 0
        if os.path.exists(markov_file):
            with open(markov_file, "r") as mf:
                try:
                    strike_count = int(mf.read().strip())
                except:
                    strike_count = 0

        # Apply Game-Theoretic Penalty
        k = 0.5
        m_i = m_i * np.exp(-k * strike_count)

        # Calculate Sigma (Volatility)
        sigma_i = close_df[ticker].pct_change().rolling(window=14).std().iloc[-1] * current_price
        if pd.isna(sigma_i) or sigma_i == 0: sigma_i = current_price * 0.02

        # Calculate Trailing SL
        sl_file = f".{ticker}_sl.txt"
        previous_SL = 0.0
        if os.path.exists(sl_file):
            with open(sl_file, "r") as sf:
                try:
                    previous_SL = float(sf.read().strip())
                except:
                    previous_SL = 0.0

        calculated_SL = real_value - (sigma_i / m_i)
        trailing_SL = max(previous_SL, calculated_SL)

        # Write new SL state
        with open(sl_file, "w") as sf:
            sf.write(str(trailing_SL))

        # Friction & Take Profit Check
        F = 0.01
        alpha_target = 0.02

        # We need an entry price to determine TP min threshold. Mocking entry price to real value for demonstration
        entry_price = real_value
        TP_min_threshold = entry_price * (1 + F + alpha_target)

        # Calculate Phase Acceleration for this specific node
        node_returns = close_df[ticker].pct_change().dropna()
        node_phase_accel = 0.0
        if len(node_returns) > 30:
            detrended = node_returns - node_returns.mean()
            analytic_signal = hilbert(detrended.values)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            node_r_t = instantaneous_phase # simplified single node phase velocity proxy
            node_phase_vel = np.diff(node_r_t)
            if len(node_phase_vel) > 1:
                node_phase_accel = np.diff(node_phase_vel)[-1]

        # Directive Calculation
        if m_i < 0.4 and kuramoto_accel > 0.05:
            base_directive = "EVACUATE"
            inertia_status = "LOW INERTIA"
        else:
            base_directive = "ENGAGE"
            inertia_status = "HIGH INERTIA" if m_i > 1.0 else "MODERATE INERTIA"

        # Gate 1: Trailing SL Check
        if current_price < trailing_SL:
            final_verdict = "[bold red]HARD STOP: CAPITAL PRESERVED. PENALTY APPLIED.[/bold red]"
            with open(markov_file, "w") as mf:
                mf.write(str(strike_count + 1))

        # Gate 2: Friction & Take Profit Check
        elif current_price > TP_min_threshold:
            if node_phase_accel < 0:
                final_verdict = "[bold green]TAKE PROFIT: MOMENTUM DECAY CAPTURED.[/bold green]"
                if os.path.exists(markov_file): os.remove(markov_file)
            else:
                final_verdict = "[bold yellow]MAINTAIN: MOMENTUM RISING. SL TRAILED UP.[/bold yellow]"

        # Gate 3: Entry Check
        elif base_directive == "ENGAGE" and strike_count == 0:
            final_verdict = "[bold cyan]ENGAGE: NEW NODE ALLOCATED.[/bold cyan]"

        else:
            final_verdict = "[bold magenta]MAINTAIN: WAITING FOR THRESHOLD.[/bold magenta]"

        # Draw Output Console
        console = Console()
        pad = "─" * (76 - len(f" ── [ TACTICAL ENGINE NODE: {ticker} ] "))

        content_lines = [
            f" ── [bold white]TACTICAL ENGINE NODE: {ticker}[/bold white] " + pad,
            f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip()}                | Health Matrix: Operational",
            f" ───────────────────────────────────────────────────────────────────────────",
            "",
            f"  [bold cyan][LAYER 1: FUNDAMENTAL INERTIA MATRIX][/bold cyan]",
            f"  ├─ Sentiment Stream      : {s_status:<9} (S_i = {S_i:.2f})",
            f"  ├─ Dividend Sustainability: {div_status:<9} (D_i = {D_i:.2f})",
            f"  ├─ Earnings Track (Q1-Q4): {e_str} (E_i = {E_i:.2f})",
            f"  ├─ Corporate Regime      : {g_status:<9} (G_i = {G_i:.2f})",
            f"  └─ [bold]COMPUTED NODE MASS    : m_i = {m_i:.2f}  [{inertia_status}][/bold]",
            "",
            f"  [bold magenta][LAYER 2: TOPOLOGICAL NETWORK FLOW][/bold magenta]",
            f"  ├─ Global Phase Lock     : {'CRITICAL' if global_r > 0.85 else 'NOMINAL'}   (r = {global_r:.2f})",
            f"  └─ Network Coherence Vel : {'ACCELERATING' if kuramoto_accel > 0.1 else 'STABLE'}    (r_dot = {kuramoto_accel:+.2f})",
            "",
            f" ───────────────────────────────────────────────────────────────────────────",
            f"  SYSTEM DIRECTIVE:  {final_verdict}",
            f" ───────────────────────────────────────────────────────────────────────────"
        ]

        content_str = "\n".join(content_lines)
        content = Text.from_markup(content_str)
        print("\n")
        console.print(content)
        print("\n")
    else:
        print(f"Data for {ticker} could not be resolved.")

def evaluate_portfolio():
    print("\n--- Enter Portfolio Holdings ---")
    print("Enter 'END' when finished.")
    holdings = {}
    total_value = 0.0

    while True:
        ticker = input("Ticker Symbol (e.g. AAPL): ").strip().upper()
        if ticker == 'END':
            break

        try:
            value = float(input(f"Estimated $ Value of {ticker}: "))
            holdings[ticker] = value
            total_value += value
        except ValueError:
            print("Invalid value. Please enter numbers only.")

    if total_value == 0:
        print("Empty portfolio.")
        return

    print("\nProcessing Localized Tensor Network to assess portfolio...")
    core_benchmarks = ["SPY", "QQQ", "TLT", "GLD", "XOM", "JNJ", "MSFT", "AAPL"]
    custom_symbols = list(set(core_benchmarks + list(holdings.keys())))

    import sys, os
    old_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            close_df, vol_df = fetch_data(custom_symbols)
            metrics = calculate_indicators(close_df, vol_df)
            metrics, vapor = perform_ml_analysis(close_df, vol_df, metrics)
        finally:
            sys.stdout = old_stdout

    print(f"\n--- PORTFOLIO ASSESSMENT (Total Value: ${total_value:,.2f}) ---")

    for ticker, val in holdings.items():
        if ticker in metrics.index:
            current_weight = (val / total_value) * 100.0
            optimal_weight = metrics.loc[ticker, 'Target_Weight_Pct']
            vel = metrics.loc[ticker, 'All_Time_Velocity']
            current_price = metrics.loc[ticker, 'Current_Price']

            print(f"\n[{ticker}] Current Weight: {current_weight:.1f}% | Optimal Weight: {optimal_weight:.1f}% | Velocity: {vel:.2f}")
            if vel <= 0:
                print(f"   => VERDICT: SELL ENTIRE POSITION. Asset is in structural decay.")
            elif current_weight > (optimal_weight * 1.3):
                print(f"   => VERDICT: TRIM. You are over-exposed beyond the mathematical risk band.")
            elif current_weight < optimal_weight:
                print(f"   => VERDICT: ACCUMULATE. Asset has capacity to safely absorb more capital.")
            else:
                print(f"   => VERDICT: HOLD. Position is optimally sized.")
        else:
            print(f"\n[{ticker}] => VERDICT: UNKNOWN. Insufficient data to map in tensor.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            evaluate_ticker(sys.argv[2])
        elif sys.argv[1] == "--portfolio":
            evaluate_portfolio()

def evaluate_portfolio():
    print("\n--- Enter Portfolio Holdings ---")
    print("Enter 'END' when finished.")
    holdings = {}
    total_value = 0.0

    while True:
        ticker = input("Ticker Symbol (e.g. AAPL): ").strip().upper()
        if ticker == 'END':
            break

        try:
            value = float(input(f"Estimated $ Value of {ticker}: "))
            holdings[ticker] = value
            total_value += value
        except ValueError:
            print("Invalid value. Please enter numbers only.")

    if total_value == 0:
        print("Empty portfolio.")
        return

    print("\nProcessing Localized Tensor Network to assess portfolio...")
    core_benchmarks = ["SPY", "QQQ", "TLT", "GLD", "XOM", "JNJ", "MSFT", "AAPL"]
    custom_symbols = list(set(core_benchmarks + list(holdings.keys())))

    import sys, os
    old_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            close_df, vol_df = fetch_data(custom_symbols)
            metrics = calculate_indicators(close_df, vol_df)
            metrics, vapor = perform_ml_analysis(close_df, vol_df, metrics)
        finally:
            sys.stdout = old_stdout

    print(f"\n--- PORTFOLIO ASSESSMENT (Total Value: ${total_value:,.2f}) ---")

    for ticker, val in holdings.items():
        if ticker in metrics.index:
            current_weight = (val / total_value) * 100.0
            optimal_weight = metrics.loc[ticker, 'Target_Weight_Pct']
            vel = metrics.loc[ticker, 'All_Time_Velocity']
            current_price = metrics.loc[ticker, 'Current_Price']

            print(f"\n[{ticker}] Current Weight: {current_weight:.1f}% | Optimal Weight: {optimal_weight:.1f}% | Velocity: {vel:.2f}")
            if vel <= 0:
                print(f"   => VERDICT: SELL ENTIRE POSITION. Asset is in structural decay.")
            elif current_weight > (optimal_weight * 1.3):
                print(f"   => VERDICT: TRIM. You are over-exposed beyond the mathematical risk band.")
            elif current_weight < optimal_weight:
                print(f"   => VERDICT: ACCUMULATE. Asset has capacity to safely absorb more capital.")
            else:
                print(f"   => VERDICT: HOLD. Position is optimally sized.")
        else:
            print(f"\n[{ticker}] => VERDICT: UNKNOWN. Insufficient data to map in tensor.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            evaluate_ticker(sys.argv[2])
        elif sys.argv[1] == "--portfolio":
            evaluate_portfolio()
