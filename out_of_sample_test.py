import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from market_analyzer import calculate_indicators, perform_ml_analysis

# Use a tight list of high-liquidity nodes for faster backtest iteration
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "CVX", "BAC", "SPY", "QQQ", "TLT", "GLD"]
INITIAL_BALANCE_NIS = 50000.0
SAFE_HAVENS = ["TLT", "GLD"]

def fetch_capsule_data(start_date, end_date):
    # Fetch data with a 1-year buffer before the start date so indicators can prime
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    data = yf.download(TEST_SYMBOLS, start=fetch_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_allocations(close_df, vol_df, test_start_date, crash_threshold):
    try:
        start_idx = close_df.index.get_indexer([pd.to_datetime(test_start_date)], method='nearest')[0]
    except KeyError:
        start_idx = 252

    total_days = close_df.shape[0]
    # Standard rebalance frequency from our previous grid search
    rebalance_dates = list(range(start_idx, total_days, 14))
    allocations_over_time = []

    for current_day_idx in rebalance_dates:
        historical_close = close_df.iloc[:current_day_idx]
        historical_vol = vol_df.iloc[:current_day_idx]

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            metrics = calculate_indicators(historical_close, historical_vol)
            metrics, vapor = perform_ml_analysis(historical_close, historical_vol, metrics)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        global_sync = metrics['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
        is_crashing = global_sync > crash_threshold

        if is_crashing:
            targets = {ticker: 0.0 for ticker in TEST_SYMBOLS}
            for safe_asset in SAFE_HAVENS:
                targets[safe_asset] = 0.5 if safe_asset in metrics.index else 0.0
        else:
            targets = (metrics['Target_Weight_Pct'] / 100.0).to_dict()

        allocations_over_time.append((current_day_idx, targets))

    return allocations_over_time

def execute_backtest(close_df, allocations_over_time, stop_loss_pct, take_profit_pct):
    current_cash = INITIAL_BALANCE_NIS
    holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS}
    buy_prices = {ticker: 0.0 for ticker in TEST_SYMBOLS}
    portfolio_value_history = []

    rebalance_dict = {day_idx: allocs for day_idx, allocs in allocations_over_time}

    start_idx = allocations_over_time[0][0]
    total_days = close_df.shape[0]

    for current_day_idx in range(start_idx, total_days):
        current_prices = close_df.iloc[current_day_idx]

        # Check stops
        for ticker in list(holdings.keys()):
            if holdings[ticker] > 0 and ticker in current_prices:
                price = current_prices[ticker]
                buy_price = buy_prices[ticker]
                if buy_price > 0:
                    pct_change = (price / buy_price) - 1.0
                    if pct_change <= -stop_loss_pct or pct_change >= take_profit_pct:
                        # Liquidate position
                        current_cash += holdings[ticker] * price
                        holdings[ticker] = 0.0

        holdings_value = sum(holdings.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in TEST_SYMBOLS if ticker in current_prices)
        total_portfolio_value = current_cash + holdings_value

        if current_day_idx in rebalance_dict:
            target_weights = rebalance_dict[current_day_idx]

            # Liquidate everything
            current_cash = total_portfolio_value
            holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS}

            for ticker, weight in target_weights.items():
                if weight > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    allocated_cash = total_portfolio_value * weight
                    shares_to_buy = allocated_cash / current_prices[ticker]
                    holdings[ticker] = shares_to_buy
                    buy_prices[ticker] = current_prices[ticker]
                    current_cash -= allocated_cash

        portfolio_value_history.append(total_portfolio_value)

    final_value = portfolio_value_history[-1]
    return ((final_value / INITIAL_BALANCE_NIS) - 1.0) * 100.0

if __name__ == "__main__":
    print("\n================================================================================")
    print("PHASE 1: IN-SAMPLE TRAINING CAPSULE (2008 to 2018)")
    print("================================================================================")
    print("Fetching training data...")
    train_close, train_vol = fetch_capsule_data("2008-01-01", "2018-12-31")

    # We will "train" the algorithm by testing a grid of Kuramoto Crash Thresholds
    # on the in-sample data. Because this is deterministic physics, we are just calibrating
    # the structural elasticity of the tensor, not fitting weights.

    thresholds = [0.80, 0.85, 0.90]
    best_threshold = 0.85
    best_return = -999.0

    # Using fixed stops from our previous single-window optimizer
    fixed_sl = 0.08
    fixed_tp = 0.15

    for t in thresholds:
        print(f"Testing Matrix Rigidity (Kuramoto Threshold = {t})...")
        allocs = run_allocations(train_close, train_vol, "2008-01-01", crash_threshold=t)
        ret = execute_backtest(train_close, allocs, fixed_sl, fixed_tp)
        print(f" -> Training Return: {ret:+.2f}%")
        if ret > best_return:
            best_return = ret
            best_threshold = t

    print(f"\n=> Training Complete. Optimal Calibrated Threshold: {best_threshold}")

    print("\n================================================================================")
    print("PHASE 2: BLIND OUT-OF-SAMPLE TESTING CAPSULE (2019 to 2024)")
    print("================================================================================")
    print("Fetching blind testing data...")
    test_close, test_vol = fetch_capsule_data("2019-01-01", "2024-01-01")

    print(f"Running deterministic tensor flow using strictly locked parameters (Threshold={best_threshold})...")
    test_allocs = run_allocations(test_close, test_vol, "2019-01-01", crash_threshold=best_threshold)
    test_return = execute_backtest(test_close, test_allocs, fixed_sl, fixed_tp)

    # Compute SPY benchmark for the exact same OOS window
    test_start_idx = test_allocs[0][0]
    spy_start = test_close.iloc[test_start_idx]['SPY']
    spy_end = test_close.iloc[-1]['SPY']
    spy_return = ((spy_end / spy_start) - 1.0) * 100.0

    print("\n" + "="*80)
    print("OUT-OF-SAMPLE BLIND TEST RESULTS (2019 - 2024)")
    print("="*80)
    print(f"Algorithm OOS Return:   {test_return:+.2f}%")
    print(f"Benchmark (SPY):        {spy_return:+.2f}%")

    diff = test_return - spy_return
    if diff > 0:
        print(f"=> SUCCESS: Outperformed the SPY benchmark out-of-sample by {diff:.2f}%")
    else:
        print(f"=> UNDERPERFORMANCE: Trailed the SPY benchmark out-of-sample by {-diff:.2f}%")
    print("="*80)
