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

        sma_50 = close_df[ticker].rolling(window=50).mean().iloc[-1]
        sma_200 = close_df[ticker].rolling(window=200).mean().iloc[-1]
        est_real_value = (sma_50 * 0.4) + (sma_200 * 0.6)
        buy_target = min(current_price, est_real_value)

        volatility = close_df[ticker].pct_change().rolling(window=30).std().iloc[-1] * np.sqrt(252)
        dynamic_stop_pct = min(0.30, max(0.05, volatility * 2.0))

        # Stop Loss and Take Profit replaced by Chronological Waypoints
        flow_deriv = row['Flow_Derivative']
        vol_sat = row['Volume_Saturation']
        flow_state = "Accelerating" if flow_deriv > 0 else "Decelerating"
        sat_state = "Saturating" if vol_sat > 1.2 else "Stable"

        time_horizon = "Unknown"
        days_horizon = 30
        if ticker in spectral_results:
            dom_band = spectral_results[ticker]['Dominant_Band']
            if dom_band == "High-Frequency":
                time_horizon = "1 to 14 Days (Noise/Tactical)"
                days_horizon = 14
            elif dom_band == "Mid-Frequency":
                time_horizon = "15 to 90 Days (Cyclical Flow)"
                days_horizon = 90
            elif dom_band == "Low-Frequency":
                time_horizon = "90+ Days (Structural Baseline)"
                days_horizon = 180

        from datetime import datetime, timedelta
        target_date_1 = (datetime.now() + timedelta(days=min(14, days_horizon//2))).strftime('%Y-%m-%d')
        target_date_2 = (datetime.now() + timedelta(days=days_horizon)).strftime('%Y-%m-%d')

        waypoint_logic = f"\n--- CHRONOLOGICAL WAYPOINT MATRIX ---\n"
        waypoint_logic += f"Current Flow State: {flow_state} (dW/dt = {flow_deriv:.4f}) | Volume: {sat_state} (Sat Index = {vol_sat:.2f})\n"
        waypoint_logic += f"\nWAYPOINT 1 [{target_date_1}]:\n"
        if flow_deriv < -0.05:
            waypoint_logic += f"  IF Price < ${buy_target:.2f} AND Flow remains Decelerating -> SELL (Derivative Stop-Loss Triggered).\n"
        else:
            waypoint_logic += f"  IF Price < ${buy_target:.2f} BUT Flow is Accelerating -> HOLD. Capital is rotating inward.\n"

        waypoint_logic += f"\nWAYPOINT 2 [{target_date_2}]:\n"
        if vol_sat > 1.2:
            waypoint_logic += f"  IF Volume Saturation persists (>1.2) -> SCALE OUT (Integral Take-Profit Triggered). Peak absorption reached.\n"
        else:
            waypoint_logic += f"  IF Target not met and Volume is Stable -> HOLD. Allow structural drift to continue.\n"

        print(f"All-Time Velocity Score: {vel:.2f}")
        print(f"Topological Safety:      {safety:.2f}")
        print(f"Optimal Matrix Weight:   {target:.2f}%")
        print(waypoint_logic)

        print(f"\n--- Systemic Verdict ---")
        if vel > 3.0:
            verdict = "ACCUMULATE (Strong Structural Inflow)"
        elif vel > 0:
            verdict = "HOLD (Positive Flow)"
        else:
            verdict = "LIQUIDATE (Structural Vaporization Detected)"
        print(f"Verdict: {verdict}")

        with open(f"Execution_Plan_{ticker}.txt", "w") as plan:
            plan.write(f"THERMODYNAMIC EXECUTION PLAN: {ticker}\n")
            plan.write("=========================================\n")
            plan.write(f"Optimal Matrix Weight:   {target:.2f}%\n")
            plan.write(f"Current Price:           ${current_price:.2f}\n")
            plan.write(f"Estimated Real Value:    ${est_real_value:.2f}\n")
            plan.write(waypoint_logic)
            plan.write(f"\nVerdict:                 {verdict}\n")
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
