import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from market_analyzer import calculate_indicators, perform_ml_analysis

# Expanded list including explicit safe havens and benchmarks
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "CVX", "BAC", "SPY", "QQQ", "TLT", "GLD"]
INITIAL_BALANCE_NIS = 50000.0
SAFE_HAVENS = ["TLT", "GLD"]

def fetch_period_data(start_date, end_date):
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=2)
    print(f"\nFetching historical data for {start_date} to {end_date} (with buffer)...")
    data = yf.download(TEST_SYMBOLS, start=fetch_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_monthly_rebalancing(close_df, vol_df, test_start_date):
    try:
        start_idx = close_df.index.get_indexer([pd.to_datetime(test_start_date)], method='nearest')[0]
    except KeyError:
        start_idx = 252 # Fallback

    total_days = close_df.shape[0]
    rebalance_dates = list(range(start_idx, total_days, 21))
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

        # Check Kuramoto Crash Flag
        global_sync = metrics['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
        CRASH_THRESHOLD = 0.85
        is_crashing = global_sync > CRASH_THRESHOLD

        # Determine targets
        if is_crashing:
            # Shift 100% to Safe Havens (50/50 split)
            targets = {ticker: 0.0 for ticker in TEST_SYMBOLS}
            for safe_asset in SAFE_HAVENS:
                targets[safe_asset] = 0.5 if safe_asset in metrics.index else 0.0
        else:
            targets = (metrics['Target_Weight_Pct'] / 100.0).to_dict()

        allocations_over_time.append((current_day_idx, targets, is_crashing, global_sync))

    return allocations_over_time

def execute_backtest(close_df, allocations_over_time, label):
    current_cash = INITIAL_BALANCE_NIS
    holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS}
    portfolio_value_history = []

    rebalance_dict = {day_idx: (allocs, crash, sync) for day_idx, allocs, crash, sync in allocations_over_time}

    start_idx = allocations_over_time[0][0]
    total_days = close_df.shape[0]

    for current_day_idx in range(start_idx, total_days):
        current_prices = close_df.iloc[current_day_idx]

        holdings_value = sum(holdings.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in TEST_SYMBOLS if ticker in current_prices)
        total_portfolio_value = current_cash + holdings_value

        if current_day_idx in rebalance_dict:
            target_weights, is_crashing, sync = rebalance_dict[current_day_idx]

            # Liquidate
            current_cash = total_portfolio_value
            holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS}

            # Buy
            for ticker, weight in target_weights.items():
                if weight > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    allocated_cash = total_portfolio_value * weight
                    shares_to_buy = allocated_cash / current_prices[ticker]
                    holdings[ticker] = shares_to_buy
                    current_cash -= allocated_cash

        portfolio_value_history.append(total_portfolio_value)

    final_value = portfolio_value_history[-1]
    total_return = ((final_value / INITIAL_BALANCE_NIS) - 1.0) * 100.0

    spy_start_price = close_df.iloc[start_idx]['SPY']
    spy_end_price = close_df.iloc[-1]['SPY']
    benchmark_return = ((spy_end_price / spy_start_price) - 1.0) * 100.0

    print("\n" + "="*80)
    print(f"RIGOROUS BACKTEST RESULTS: {label}")
    print("="*80)
    print(f"Algorithm Return:   {total_return:+.2f}%")
    print(f"Benchmark (SPY):    {benchmark_return:+.2f}%")

    diff = total_return - benchmark_return
    if diff > 0:
        print(f"=> MASSIVE SUCCESS: Thermodynamic Model Outperformed the SPY benchmark by {diff:.2f}%")
    else:
        print(f"=> UNDERPERFORMANCE: Trailed the SPY benchmark by {-diff:.2f}%")
    print("="*80)

if __name__ == "__main__":
    # Test 1: The 2008 Financial Crisis (Sept 2007 to March 2009)
    close_df_08, vol_df_08 = fetch_period_data("2007-09-01", "2009-03-31")
    allocs_08 = run_monthly_rebalancing(close_df_08, vol_df_08, "2007-09-01")
    execute_backtest(close_df_08, allocs_08, "2008 Financial Crisis (Kuramoto Phase Defense Test)")

    # Test 2: The 2010s Bull Run (Jan 2013 to Dec 2019)
    close_df_bull, vol_df_bull = fetch_period_data("2013-01-01", "2019-12-31")
    allocs_bull = run_monthly_rebalancing(close_df_bull, vol_df_bull, "2013-01-01")
    execute_backtest(close_df_bull, allocs_bull, "2010s Bull Run (Eigenvector Centrality Growth Test)")

    # Test 3: The 2022 Tech Bear Market (Jan 2022 to Dec 2022)
    close_df_22, vol_df_22 = fetch_period_data("2022-01-01", "2022-12-31")
    allocs_22 = run_monthly_rebalancing(close_df_22, vol_df_22, "2022-01-01")
    execute_backtest(close_df_22, allocs_22, "2022 Tech Bear Market (Kelly Fraction Scaling Test)")
