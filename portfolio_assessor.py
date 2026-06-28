import sys
import pandas as pd
import yfinance as yf
from market_analyzer import SYMBOLS, fetch_data, calculate_indicators, perform_ml_analysis

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
        finally:
            sys.stdout = old_stdout

    if ticker in metrics.index:
        row = metrics.loc[ticker]
        vel = row['All_Time_Velocity']
        target = row['Target_Weight_Pct']
        safety = row['Topological_Safety']

        print(f"All-Time Velocity Score: {vel:.2f}")
        print(f"Topological Safety:      {safety:.2f}")
        print(f"Optimal Matrix Weight:   {target:.2f}%")

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
        finally:
            sys.stdout = old_stdout

    print(f"\n--- PORTFOLIO ASSESSMENT (Total Value: ${total_value:,.2f}) ---")

    for ticker, val in holdings.items():
        if ticker in metrics.index:
            current_weight = (val / total_value) * 100.0
            optimal_weight = metrics.loc[ticker, 'Target_Weight_Pct']
            vel = metrics.loc[ticker, 'All_Time_Velocity']

            print(f"\n[{ticker}] Current Weight: {current_weight:.1f}% | Optimal Weight: {optimal_weight:.1f}% | Velocity: {vel:.2f}")
            if vel <= 0:
                print(f"   => ACTION: SELL ENTIRE POSITION. Asset is in structural decay.")
            elif current_weight > (optimal_weight * 1.3):
                print(f"   => ACTION: TRIM. You are over-exposed beyond the mathematical risk band.")
            elif current_weight < optimal_weight:
                print(f"   => ACTION: ACCUMULATE. Asset has capacity to safely absorb more capital.")
            else:
                print(f"   => ACTION: HOLD. Position is optimally sized.")
        else:
            print(f"\n[{ticker}] => ACTION: UNKNOWN. Insufficient data to map in tensor.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            evaluate_ticker(sys.argv[2])
        elif sys.argv[1] == "--portfolio":
            evaluate_portfolio()
