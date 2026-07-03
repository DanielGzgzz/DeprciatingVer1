import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from market_analyzer import calculate_indicators, perform_ml_analysis

# We will test on a tight, representative matrix to allow the heavy calibration to run in reasonable time
CALIBRATION_NODES = ["SPY", "QQQ", "TLT", "GLD", "AAPL", "MSFT", "XOM", "JNJ", "JPM", "BAC", "NVDA"]

def fetch_decade_data():
    print("Fetching 12 years of structural market data (2012-2024)...")
    data = yf.download(CALIBRATION_NODES, start="2012-01-01", end="2024-01-01", progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_calibration_chunk(close_df, vol_df, train_start, train_end, test_start, test_end, base_sl, base_tp):
    """
    Simulates a 4-year train to calibrate the volatility multiplier,
    then executes a 4-year blind test using the calibrated trailing parameters.
    Returns the OOS performance.
    """
    # 1. In-Sample Training (Determine optimal volatility scalar for stops)
    train_close = close_df.loc[train_start:train_end]
    train_vol = vol_df.loc[train_start:train_end]

    # We calibrate a multiplier (M) applied to the asset's trailing 30-day volatility
    # to set the trailing stop distance.
    # High M = Loose stops (survives whipsaws, larger drawdowns)
    # Low M = Tight stops (preserves capital, gets hunted easily)
    M_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]
    best_M = 2.0
    best_train_ret = -999.0

    for M in M_candidates:
        ret = execute_trailing_backtest(train_close, train_vol, M, base_tp, "Training")
        if ret > best_train_ret:
            best_train_ret = ret
            best_M = M

    # 2. Out-of-Sample Testing
    test_close = close_df.loc[test_start:test_end]
    test_vol = vol_df.loc[test_start:test_end]
    test_ret = execute_trailing_backtest(test_close, test_vol, best_M, base_tp, "Testing")

    spy_start = test_close.iloc[0]['SPY']
    spy_end = test_close.iloc[-1]['SPY']
    spy_ret = ((spy_end / spy_start) - 1.0) * 100.0

    return best_M, test_ret, spy_ret

def execute_trailing_backtest(close_df, vol_df, vol_multiplier, base_tp, phase="Testing"):
    """
    Executes a backtest using a dynamic trailing stop based on rolling volatility.
    """
    INITIAL_CASH = 50000.0
    cash = INITIAL_CASH
    holdings = {t: 0.0 for t in CALIBRATION_NODES}
    trailing_peaks = {t: 0.0 for t in CALIBRATION_NODES}
    entry_prices = {t: 0.0 for t in CALIBRATION_NODES}

    # Pre-calculate rolling 30-day volatility (standard deviation of log returns)
    returns = np.log(close_df / close_df.shift(1))
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) # Annualized
    rolling_vol = rolling_vol.bfill()

    total_days = close_df.shape[0]
    # Rebalance every 21 days
    rebalance_dates = list(range(30, total_days, 21))
    rebalance_dict = {}

    # Generate signals
    for idx in rebalance_dates:
        h_close = close_df.iloc[:idx]
        h_vol = vol_df.iloc[:idx]

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            m = calculate_indicators(h_close, h_vol)
            m, v = perform_ml_analysis(h_close, h_vol, m)

            global_sync = m['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in m.columns else 0.0
            if global_sync > 0.85:
                targets = {t: 0.0 for t in CALIBRATION_NODES}
                targets['TLT'] = 0.5
                targets['GLD'] = 0.5
            else:
                targets = (m['Target_Weight_Pct'] / 100.0).to_dict()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        rebalance_dict[idx] = targets

    # Walk forward
    portfolio_history = []

    for day_idx in range(30, total_days):
        current_prices = close_df.iloc[day_idx]
        current_vols = rolling_vol.iloc[day_idx]

        # Check Trailing Stops & Take Profits dynamically
        for ticker in list(holdings.keys()):
            if holdings[ticker] > 0 and ticker in current_prices:
                price = current_prices[ticker]
                volatility = current_vols[ticker]

                # Update trailing peak
                if price > trailing_peaks[ticker]:
                    trailing_peaks[ticker] = price

                # Dynamic Trailing Stop = Peak Price * (1 - (Volatility * Multiplier))
                # Capped to ensure it's a realistic percentage (e.g. max 30% drawdown allowed)
                dynamic_stop_pct = min(0.30, max(0.05, volatility * vol_multiplier))
                trailing_stop_price = trailing_peaks[ticker] * (1.0 - dynamic_stop_pct)

                # Dynamic Take Profit (scales with volatility, highly volatile assets need wider targets)
                take_profit_price = entry_prices[ticker] * (1.0 + base_tp + volatility)

                if price <= trailing_stop_price or price >= take_profit_price:
                    # Liquidate
                    cash += holdings[ticker] * price
                    holdings[ticker] = 0.0
                    trailing_peaks[ticker] = 0.0
                    entry_prices[ticker] = 0.0

        holdings_value = sum(holdings.get(t, 0) * current_prices.get(t, 0) for t in CALIBRATION_NODES if t in current_prices)
        total_value = cash + holdings_value

        # Rebalance
        if day_idx in rebalance_dict:
            targets = rebalance_dict[day_idx]
            cash = total_value
            holdings = {t: 0.0 for t in CALIBRATION_NODES}

            for ticker, weight in targets.items():
                if weight > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    alloc = total_value * weight
                    shares = alloc / current_prices[ticker]
                    holdings[ticker] = shares
                    trailing_peaks[ticker] = current_prices[ticker]
                    entry_prices[ticker] = current_prices[ticker]
                    cash -= alloc

        portfolio_history.append(total_value)

    return ((portfolio_history[-1] / INITIAL_CASH) - 1.0) * 100.0

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONTINUOUS HEAVY-CALIBRATOR ENGINE (4Yr Train -> 4Yr Test Chunking)")
    print("="*80)
    close_df, vol_df = fetch_decade_data()

    # Define overlapping epochs
    # Epoch 1: Train 2012-2015 -> Test 2016-2019
    # Epoch 2: Train 2016-2019 -> Test 2020-2023

    epochs = [
        ("Epoch 1", "2012-01-01", "2015-12-31", "2016-01-01", "2019-12-31"),
        ("Epoch 2", "2016-01-01", "2019-12-31", "2020-01-01", "2023-12-31")
    ]

    for label, tr_start, tr_end, te_start, test_end in epochs:
        print(f"\nProcessing {label}...")
        print(f"  Training Window: {tr_start} to {tr_end}")
        print(f"  Testing Window:  {te_start} to {test_end}")

        best_M, algo_ret, spy_ret = run_calibration_chunk(close_df, vol_df, tr_start, tr_end, te_start, test_end, base_sl=0.08, base_tp=0.15)

        print(f"  => Discovered Optimal Volatility Multiplier: {best_M}x")
        print(f"  => Algorithm Blind OOS Return: {algo_ret:+.2f}%")
        print(f"  => Benchmark (SPY) OOS Return: {spy_ret:+.2f}%")
        diff = algo_ret - spy_ret
        if diff > 0:
            print(f"  => RESULT: Outperformed SPY by {diff:.2f}%")
        else:
            print(f"  => RESULT: Trailed SPY by {-diff:.2f}%")
