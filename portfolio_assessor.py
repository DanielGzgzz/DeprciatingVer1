import numpy as np
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
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

        price = row['Current_Price']

        try:
            real_value = close_df[ticker].rolling(window=126).mean().iloc[-1]
        except:
            real_value = price

        vel = row['All_Time_Velocity']
        target = row['Target_Weight_Pct']
        centrality = row.get('Eigenvector_Centrality', 0.1)
        global_r = metrics.loc[metrics.index[0], 'Global_Kuramoto_Sync'] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
        lambda2 = metrics.loc[metrics.index[0], 'Fiedler_Value'] if 'Fiedler_Value' in metrics.columns else 0.5

        # Calculate stops
        z_score = 2.0 + (min(1.0, centrality) * 2.0)
        sigma_hf = close_df[ticker].pct_change().rolling(window=14).std().iloc[-1] * price
        if pd.isna(sigma_hf) or sigma_hf == 0: sigma_hf = price * 0.02

        sl = price - (z_score * sigma_hf)

        projected_gain_pct = min(0.40, max(0.05, vel * centrality * 0.10))
        tp = price * (1.0 + projected_gain_pct)

        # Calculate deterministic time to target (Phase cycle duration)
        # We estimate how many days it will take to travel the distance (tp - price)
        # given the daily volatility velocity (sigma_hf).
        target_distance = tp - price
        daily_drift_estimate = max(sigma_hf * 0.15, price * 0.001) # Assume 15% of daily vol is directional drift
        estimated_days = int(target_distance / daily_drift_estimate)
        estimated_days = min(365, max(1, estimated_days)) # Cap between 1 and 365 days


        drawdown_pct = ((price - sl) / price) * 100.0

        # Calculate Confidence derived from fiedler lambda2 (scaling 0.0 to 1.0 into 0-100%)
        # Cap lambda2 to a reasonable max scale of ~2.0 for standard graph
        confidence = min(99, max(1, int((lambda2 / 1.5) * 100)))

        # Verdict Logic
        if drawdown_pct > 10.0:
            verdict = "EXIT / REJECT (Risk Factor > 10%)"
        elif vel <= 0:
            verdict = "EXIT / REJECT (Structural Vaporization)"
        else:
            verdict = "BUY / ACCUMULATE"

        # Draw Panel
        console = Console()
        content = Text()
        content.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Status: Market Open\n", style="dim")
        content.append("─" * 58 + "\n", style="dim")

        content.append(f"  Current Price:   ", style="bold")
        content.append(f"${price:.2f}\n", style="cyan")

        content.append(f"  Real Value:      ", style="bold")
        price_delta_pct = ((price - real_value) / real_value) * 100.0
        pricing_status = f"{abs(price_delta_pct):.1f}% OVERPRICED" if price_delta_pct > 0 else f"{abs(price_delta_pct):.1f}% UNDERPRICED"
        content.append(f"${real_value:.2f}  ({pricing_status})\n", style="magenta")

        content.append("─" * 58 + "\n", style="dim")

        content.append(f"  Take Profit:     ", style="bold")
        content.append(f"${tp:.2f}  (Phase Target in ~{estimated_days} Days)\n", style="green")

        content.append(f"  Stop Loss:       ", style="bold")
        content.append(f"${sl:.2f}  (Topological Support)\n", style="red")

        content.append("─" * 58 + "\n", style="dim")

        content.append(f"  Confidence vs SP500:  ", style="bold")

        # Win probability string formulation
        # Using the base confidence scaled slightly by velocity for precision
        decimal_prob = min(0.99, max(0.01, (confidence / 100.0) + (vel * 0.01)))
        prob_pct = decimal_prob * 100
        attempts = int(decimal_prob * 10)

        content.append(f"{prob_pct:.1f}%  ({attempts} of 10 attempts profit by ~{estimated_days} Days)\n", style="yellow")

        v_style = "bold green" if "BUY" in verdict else "bold red"
        content.append(f"  Verdict:              ", style="bold")
        content.append(f"{verdict}\n", style=v_style)

        content.append("─" * 58 + "\n", style="dim")
        content.append(f" [SYSTEM] Global Phase Lock r(t) = {global_r:.2f}. ", style="dim")
        if global_r > 0.8:
            content.append("CRITICAL FLIGHT RISK DETECTED.", style="bold red")
        else:
            content.append("System stable. No crash footprint.", style="green")

        panel = Panel(
            content,
            title=f"[bold white]TSME TICKER MONITOR: {ticker}[/bold white]",
            expand=False,
            border_style="blue" if global_r <= 0.8 else "red"
        )
        print("\n")
        console.print(panel)
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
