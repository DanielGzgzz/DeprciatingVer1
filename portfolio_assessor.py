import numpy as np
import sys
import pandas as pd
import yfinance as yf
from market_analyzer import SYMBOLS, fetch_data, calculate_indicators, perform_ml_analysis, perform_spectral_analysis

def evaluate_ticker(ticker):
    print(f"\n--- Thermodynamic Assessment for [{ticker}] ---")

    # To evaluate quickly without doing the 900+ tensor, we build a localized tensor
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

            # Run Spectral Analysis to get time horizon
            target_node = metrics.loc[[ticker]] if ticker in metrics.index else pd.DataFrame()
            if not target_node.empty:
                spectral_results = perform_spectral_analysis(close_df, target_node)
            else:
                spectral_results = {}
        finally:
            sys.stdout = old_stdout

    if ticker in metrics.index:
        row = metrics.loc[ticker]
        vel = row['All_Time_Velocity']
        target = row['Target_Weight_Pct']
        safety = row['Topological_Safety']

        # Calculate Execution Parameters
        current_price = row['Current_Price']

        # Compute SMAs locally since they were removed from the core matrix calculate_indicators
        sma_50 = close_df[ticker].rolling(window=50).mean().iloc[-1]
        sma_200 = close_df[ticker].rolling(window=200).mean().iloc[-1]

        # Estimated Real Value is a weighted average of long-term and mid-term moving averages
        est_real_value = (sma_50 * 0.4) + (sma_200 * 0.6)

        # LOGICAL FIX: Buy parameters must be distinct from current position management
        buy_target = min(current_price, est_real_value) # Maximum price to safely acquire shares

        # Execute Dynamic Volatility Multiplier based on chunk training (M=2.0)
        volatility = close_df[ticker].pct_change().rolling(window=30).std().iloc[-1] * np.sqrt(252)
        dynamic_stop_pct = min(0.30, max(0.05, volatility * 2.0))

        # Stop Loss and Take Profit should be calculated based on the *entry* price (buy_target),
        # not the current price (which might be massively over-extended).
        stop_loss = buy_target * (1.0 - dynamic_stop_pct)
        take_profit = buy_target * (1.0 + 0.15 + volatility)

        # Estimated Time Horizon from Spectral Analysis
        time_horizon = "Unknown"
        if ticker in spectral_results:
            dom_band = spectral_results[ticker]['Dominant_Band']
            if dom_band == "High-Frequency":
                time_horizon = "1 to 14 Days (Noise/Tactical)"
            elif dom_band == "Mid-Frequency":
                time_horizon = "15 to 90 Days (Cyclical Flow)"
            elif dom_band == "Low-Frequency":
                time_horizon = "90+ Days (Structural Baseline)"

        print(f"All-Time Velocity Score: {vel:.2f}")
        print(f"Topological Safety:      {safety:.2f}")
        print(f"Optimal Matrix Weight:   {target:.2f}%")

        print(f"\n--- PRINTABLE EXECUTION PLAN ---")
        print(f"Current Price:           ${current_price:.2f}")
        print(f"Estimated Real Value:    ${est_real_value:.2f}")
        print(f"Optimal Buy Zone:        < ${buy_target:.2f}")
        print(f"Dynamic Trailing Stop:   ${stop_loss:.2f} (Updates daily based on {dynamic_stop_pct*100:.1f}% rolling volatility)")
        print(f"Adaptive Take Profit:    ${take_profit:.2f}")
        print(f"Estimated Time Horizon:  {time_horizon}")

        print(f"\n--- Systemic Verdict ---")
        if vel > 3.0:
            verdict = "ACCUMULATE (Strong Structural Inflow)"
        elif vel > 0:
            verdict = "HOLD (Positive Flow)"
        else:
            verdict = "LIQUIDATE (Structural Vaporization Detected)"
        print(f"Verdict: {verdict}")

        # Save Printable Execution Plan
        with open(f"Execution_Plan_{ticker}.txt", "w") as plan:
            plan.write(f"THERMODYNAMIC EXECUTION PLAN: {ticker}\n")
            plan.write("=========================================\n")
            plan.write(f"Optimal Matrix Weight:   {target:.2f}%\n")
            plan.write(f"Current Price:           ${current_price:.2f}\n")
            plan.write(f"Estimated Real Value:    ${est_real_value:.2f}\n")
            plan.write(f"Optimal Buy Zone:        < ${buy_target:.2f}\n")
            plan.write(f"Dynamic Trailing Stop:   ${stop_loss:.2f} (Updates daily based on {dynamic_stop_pct*100:.1f}% rolling volatility)\n")
            plan.write(f"Adaptive Take Profit:    ${take_profit:.2f}\n")
            plan.write(f"Estimated Time Horizon:  {time_horizon}\n")
            plan.write(f"Verdict:                 {verdict}\n")
        print(f"\n[+] Saved Printable Execution Plan to Execution_Plan_{ticker}.txt")
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

            # Run Spectral Analysis to get time horizon
            target_nodes = metrics[metrics.index.isin(holdings.keys())]
            if not target_nodes.empty:
                spectral_results = perform_spectral_analysis(close_df, target_nodes)
            else:
                spectral_results = {}
        finally:
            sys.stdout = old_stdout

    print(f"\n--- PORTFOLIO ASSESSMENT (Total Value: ${total_value:,.2f}) ---")

    for ticker, val in holdings.items():
        if ticker in metrics.index:
            current_weight = (val / total_value) * 100.0
            optimal_weight = metrics.loc[ticker, 'Target_Weight_Pct']
            vel = metrics.loc[ticker, 'All_Time_Velocity']
            safety = metrics.loc[ticker, 'Topological_Safety']
            current_price = metrics.loc[ticker, 'Current_Price']

            # Compute SMAs locally
            sma_50 = close_df[ticker].rolling(window=50).mean().iloc[-1]
            sma_200 = close_df[ticker].rolling(window=200).mean().iloc[-1]
            est_real_value = (sma_50 * 0.4) + (sma_200 * 0.6)

            volatility = close_df[ticker].pct_change().rolling(window=30).std().iloc[-1] * np.sqrt(252)
            dynamic_stop_pct = min(0.30, max(0.05, volatility * 2.0))

            buy_target = min(current_price, est_real_value)
            stop_loss = buy_target * (1.0 - dynamic_stop_pct)
            take_profit = buy_target * (1.0 + 0.15 + volatility)

            time_horizon = "Unknown"
            if ticker in spectral_results:
                dom_band = spectral_results[ticker]['Dominant_Band']
                if dom_band == "High-Frequency":
                    time_horizon = "1 to 14 Days"
                elif dom_band == "Mid-Frequency":
                    time_horizon = "15 to 90 Days"
                elif dom_band == "Low-Frequency":
                    time_horizon = "90+ Days"

            print(f"\n[{ticker}] Current Weight: {current_weight:.1f}% | Optimal Weight: {optimal_weight:.1f}% | Velocity: {vel:.2f}")
            print(f"   => Current Price: ${current_price:.2f} | Est. Real Value: ${est_real_value:.2f}")
            print(f"   => Dynamic Limits: Trailing Stop @ ${stop_loss:.2f} (-{dynamic_stop_pct*100:.1f}%) | Take Profit @ ${take_profit:.2f}")
            print(f"   => Flow Horizon: {time_horizon}")

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

    # Save Printable Portfolio Execution Plan
    with open("Execution_Plan_Portfolio.txt", "w") as plan:
        plan.write("THERMODYNAMIC PORTFOLIO EXECUTION PLAN\n")
        plan.write("=========================================\n")
        for ticker, val in holdings.items():
            if ticker in metrics.index:
                current_weight = (val / total_value) * 100.0
                optimal_weight = metrics.loc[ticker, 'Target_Weight_Pct']
                current_price = metrics.loc[ticker, 'Current_Price']
                volatility = close_df[ticker].pct_change().rolling(window=30).std().iloc[-1] * np.sqrt(252)
                dynamic_stop_pct = min(0.30, max(0.05, volatility * 2.0))
                sma_50 = close_df[ticker].rolling(window=50).mean().iloc[-1]
                sma_200 = close_df[ticker].rolling(window=200).mean().iloc[-1]
                est_real_value = (sma_50 * 0.4) + (sma_200 * 0.6)
                buy_target = min(current_price, est_real_value)
                stop_loss = buy_target * (1.0 - dynamic_stop_pct)
                take_profit = buy_target * (1.0 + 0.15 + volatility)

                plan.write(f"\n[{ticker}]\n")
                plan.write(f"  Target Weight: {optimal_weight:.1f}% (Current: {current_weight:.1f}%)\n")
                plan.write(f"  Buy Zone:      < ${buy_target:.2f}\n")
                plan.write(f"  Trailing Stop: ${stop_loss:.2f} (-{dynamic_stop_pct*100:.1f}%)\n")
                plan.write(f"  Take Profit:   ${take_profit:.2f}\n")

    print("\n[+] Saved Printable Portfolio Execution Plan to Execution_Plan_Portfolio.txt")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            evaluate_ticker(sys.argv[2])
        elif sys.argv[1] == "--portfolio":
            evaluate_portfolio()
