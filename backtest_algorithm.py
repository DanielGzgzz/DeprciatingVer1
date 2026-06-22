import yfinance as yf
import pandas as pd
import numpy as np
from market_analyzer import calculate_indicators, perform_ml_analysis

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "SPY", "QQQ"]
INITIAL_BALANCE_NIS = 50000.0

def fetch_1y_data():
    print(f"Fetching 2 years of historical data for rolling backtest...")
    data = yf.download(TEST_SYMBOLS, period="2y", progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_monthly_rebalancing(close_df, vol_df):
    """
    Simulates walking forward through the last 1 year (252 trading days).
    Rebalances the portfolio every 21 trading days (approx 1 month).
    """
    total_days = close_df.shape[0]
    backtest_window = 252 # Last 1 year
    start_idx = total_days - backtest_window

    rebalance_dates = list(range(start_idx, total_days, 21))

    allocations_over_time = []

    print("\n--- Running Monthly Algorithm Allocations ---")
    for i, current_day_idx in enumerate(rebalance_dates):
        # Data available up to current day
        historical_close = close_df.iloc[:current_day_idx]
        historical_vol = vol_df.iloc[:current_day_idx]
        current_date = close_df.index[current_day_idx].strftime('%Y-%m-%d')

        # Suppress prints from market_analyzer functions
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            metrics = calculate_indicators(historical_close, historical_vol)
            metrics, vapor = perform_ml_analysis(historical_close, historical_vol, metrics)
        finally:
            sys.stdout = old_stdout

        # Extract target weights
        targets = metrics['Target_Weight_Pct'] / 100.0 # Convert to decimal

        print(f"[{current_date}] Calculated Model Allocations for {len(targets[targets>0])} assets.")
        allocations_over_time.append((current_day_idx, targets))

    return allocations_over_time

def execute_backtest(close_df, allocations_over_time):
    print("\n--- Executing Simulated Trades (Starting Balance: 50,000 NIS) ---")

    current_cash = INITIAL_BALANCE_NIS
    holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS}
    portfolio_value_history = []

    # Track the dates of rebalancing
    rebalance_dict = {day_idx: allocs for day_idx, allocs in allocations_over_time}

    # We walk through the backtest window day by day to track value
    start_idx = allocations_over_time[0][0]
    total_days = close_df.shape[0]

    for current_day_idx in range(start_idx, total_days):
        current_prices = close_df.iloc[current_day_idx]
        current_date = close_df.index[current_day_idx].strftime('%Y-%m-%d')

        # 1. Update Portfolio Value based on current holdings
        holdings_value = sum(holdings.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in TEST_SYMBOLS)
        total_portfolio_value = current_cash + holdings_value

        # 2. Rebalance if it's a rebalance day
        if current_day_idx in rebalance_dict:
            target_weights = rebalance_dict[current_day_idx]

            # Liquidate everything (simplest mocked rebalance logic)
            current_cash = total_portfolio_value
            holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS}

            # Buy new targets
            for ticker, weight in target_weights.items():
                if weight > 0 and ticker in current_prices:
                    allocated_cash = total_portfolio_value * weight
                    shares_to_buy = allocated_cash / current_prices[ticker]
                    holdings[ticker] = shares_to_buy
                    current_cash -= allocated_cash

            print(f"[{current_date}] Executed Rebalance. Total Value: {total_portfolio_value:,.2f} NIS")

        portfolio_value_history.append(total_portfolio_value)

    final_value = portfolio_value_history[-1]
    total_return = ((final_value / INITIAL_BALANCE_NIS) - 1.0) * 100.0

    # Benchmark against SPY (Buy and Hold)
    spy_start_price = close_df.iloc[start_idx]['SPY']
    spy_end_price = close_df.iloc[-1]['SPY']
    benchmark_return = ((spy_end_price / spy_start_price) - 1.0) * 100.0

    print("\n" + "="*60)
    print("BACKTEST RESULTS (1-Year Window)")
    print("="*60)
    print(f"Starting Balance:   {INITIAL_BALANCE_NIS:,.2f} NIS")
    print(f"Final Balance:      {final_value:,.2f} NIS")
    print(f"Algorithm Return:   {total_return:+.2f}%")
    print(f"Benchmark (SPY):    {benchmark_return:+.2f}%")

    if total_return > benchmark_return:
        print(f"\n=> SUCCESS: Algorithm outperformed the SPY benchmark by {total_return - benchmark_return:.2f}%")
    else:
        print(f"\n=> UNDERPERFORMANCE: Algorithm trailed the SPY benchmark by {benchmark_return - total_return:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    close_df, vol_df = fetch_1y_data()
    allocs = run_monthly_rebalancing(close_df, vol_df)
    execute_backtest(close_df, allocs)