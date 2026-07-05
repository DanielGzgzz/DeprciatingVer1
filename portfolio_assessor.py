import numpy as np
import sys
import pandas as pd
import yfinance as yf
from market_analyzer import SYMBOLS, fetch_data, calculate_indicators, perform_ml_analysis, perform_spectral_analysis

def evaluate_ticker(ticker):
    print(f"\n--- Thermodynamic Assessment for [{ticker}] ---")

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

        current_price = row['Current_Price']

        # Real Current Price Estimation & Mathematical meaning
        # Calculate a robust real value using weighted SMAs representing Structural Baseline Value
        try:
            real_estimated_value = close_df[ticker].rolling(window=126).mean().iloc[-1]
            price_delta = current_price - real_estimated_value
            price_state = "ABOVE" if price_delta > 0 else "BELOW"
            pct_diff = (abs(price_delta) / real_estimated_value) * 100

            print(f"\n   => ESTIMATED REAL VALUE: {real_estimated_value:,.2f}")
            print(f"   => CURRENT PRICE:        {current_price:,.2f}")
            print(f"   => MATHEMATICAL MEANING: Price is {pct_diff:.2f}% {price_state} fair structural value.")
        except:
            pass

        flow_deriv = row.get('Flow_Derivative', 0.0)
        vol_sat = row.get('Volume_Saturation', 1.0)
        centrality = row.get('Eigenvector_Centrality', 0.1)

        emergent_sectors = row.get('Emergent_Sectors', {})
        if isinstance(emergent_sectors, dict) and emergent_sectors:
            top_sector = max(emergent_sectors.items(), key=lambda x: x[1])
            sector_str = f"{top_sector[0]} (Weight: {top_sector[1]:.2f})"
        else:
            sector_str = "Unknown"

        z_score = 2.0 + (min(1.0, centrality) * 2.0)

        sigma_hf = close_df[ticker].pct_change().rolling(window=14).std().iloc[-1] * current_price
        if pd.isna(sigma_hf) or sigma_hf == 0:
            sigma_hf = current_price * 0.02

        hard_stop = current_price - (z_score * sigma_hf)
        drawdown_pct = ((current_price - hard_stop) / current_price) * 100.0

        projected_gain_pct = min(0.40, max(0.05, vel * centrality * 0.10))
        hard_take_profit = current_price * (1.0 + projected_gain_pct)

        flow_state = "Accelerating" if flow_deriv > 0 else "Decelerating"

        from datetime import datetime, timedelta
        target_date_1 = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
        target_date_2 = (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d')
        target_date_3 = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')

        print(f"Current Market Price: ${current_price:.2f}")
        print(f"All-Time Velocity:    {vel:.2f}")
        print(f"Emergent Sector:      {sector_str}")
        print(f"Optimal Matrix Wt:    {target:.2f}%")

        print(f"\n--- ABSOLUTE CIRCUIT BREAKERS (Calculated Failsafes) ---")
        print(f"Max Noise Variance (σ_HF):  ${sigma_hf:.2f}")
        print(f"Calculated Hard Stop:       ${hard_stop:.2f} (P_curr - {z_score:.1f}σ_HF) -> [ROUTE: SELL STOP MARKET]")
        print(f"Calculated Take-Profit:     ${hard_take_profit:.2f} (Integral Peak) -> [ROUTE: SELL LIMIT]")
        print(f"Risk Factor:                {drawdown_pct:.1f}% Drawdown to Stop")

        print(f"\n--- CHRONOLOGICAL WAYPOINT MATRIX (3 Horizons) ---")
        print(f"Current Flow State: {flow_state} (dW/dt = {flow_deriv:+.4f}) | Saturation Index: {vol_sat:.2f}")

        print(f"\nWAYPOINT 1 (High-Freq Noise Clearance) [{target_date_1}]:")
        print(f"  IF Price < ${current_price:.2f} BUT Flow Acceleration (dW/dt) > 0.01:")
        print(f"     -> ACTION: HOLD (Price is lagging continuous inflow).")
        print(f"  ELSE (Flow is decelerating while price is down):")
        print(f"     -> ACTION: ROUTE SELL MARKET (Thermodynamic thesis failed; leave early).")

        print(f"\nWAYPOINT 2 (Mid-Freq Cyclical Check) [{target_date_2}]:")
        print(f"  IF Saturation Index < 1.80 AND Eigenvector Centrality > 0.70:")
        print(f"     -> ACTION: HOLD (Capital sink hasn't reached structural exhaustion).")
        print(f"  ELSE (Saturation breached or Centrality decaying):")
        print(f"     -> ACTION: ROUTE SELL 50% LIMIT @ CURRENT BID (Scale out).")

        print(f"\nWAYPOINT 3 (Low-Freq Structural Target) [{target_date_3}]:")
        print(f"  IF Integral Volume Target Achieved:")
        print(f"     -> ACTION: ROUTE SELL LIMIT 100% (Rotate capital to new sink).")
        print(f"  ELSE:")
        print(f"     -> ACTION: RECALCULATE TENSOR AND GENERATE NEW 3-WAYPOINT TREE.")

        print(f"\n--- Systemic Verdict ---")
        if drawdown_pct > 10.0:
            print(f"Verdict: REJECT (Risk Factor {drawdown_pct:.1f}% > 10.0% Portfolio Tolerance)")
        elif vel <= 0:
            print(f"Verdict: REJECT (Structural Vaporization Detected. Velocity {vel:.2f} <= 0)")
        else:
            print(f"Verdict: EXECUTE BUY LIMIT @ ${current_price:.2f} (Target: {target:.2f}% Portfolio Weight)")

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
