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

        # The optimizer sweet spots: Stop Loss -8.0%, Take Profit +15.0% to +20.0%
        # We dynamically adjust based on topological safety. Safer assets = tighter stops, looser profit taking.
        safety_multiplier = max(0.5, min(1.5, 1.0 / (safety + 1e-6)))

        buy_target = min(current_price, est_real_value) # Buy at or below real value
        stop_loss = current_price * (1.0 - (0.08 * safety_multiplier))
        take_profit = current_price * (1.0 + (0.175 / safety_multiplier))

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

        print(f"\n--- Execution Parameters ---")
        print(f"Current Price:           ${current_price:.2f}")
        print(f"Estimated Real Value:    ${est_real_value:.2f}")
        print(f"Optimal Buy Zone:        < ${buy_target:.2f}")
        print(f"Mathematical Stop Loss:  ${stop_loss:.2f}")
        print(f"Projected Take Profit:   ${take_profit:.2f}")
        print(f"Estimated Time Horizon:  {time_horizon}")

        print(f"\n--- Systemic Verdict ---")
        if vel > 3.0:
            print("Verdict: ACCUMULATE (Strong Structural Inflow)")
        elif vel > 0:
            print("Verdict: HOLD (Positive Flow)")
        else:
            print("Verdict: LIQUIDATE (Structural Vaporization Detected)")
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

            safety_multiplier = max(0.5, min(1.5, 1.0 / (safety + 1e-6)))
            stop_loss = current_price * (1.0 - (0.08 * safety_multiplier))
            take_profit = current_price * (1.0 + (0.175 / safety_multiplier))

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
            print(f"   => Execution Limits: Stop Loss @ ${stop_loss:.2f} | Take Profit @ ${take_profit:.2f}")
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            evaluate_ticker(sys.argv[2])
        elif sys.argv[1] == "--portfolio":
            evaluate_portfolio()
