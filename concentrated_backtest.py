import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
from market_analyzer import calculate_indicators, perform_ml_analysis

# A diverse set of symbols including forex for conversion
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "CVX", "BAC", "SPY", "QQQ", "TLT", "GLD", "ILS=X"]
INITIAL_BALANCE_NIS = 50000.0
SAFE_HAVENS = ["TLT", "GLD"]
MAX_POSITIONS = 5

def fetch_data_with_fx(start_date, end_date):
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    data = yf.download(TEST_SYMBOLS, start=fetch_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_concentrated_allocations(close_df, vol_df, test_start_date):
    try:
        start_idx = close_df.index.get_indexer([pd.to_datetime(test_start_date)], method='nearest')[0]
    except KeyError:
        start_idx = 252

    total_days = close_df.shape[0]
    rebalance_dates = list(range(start_idx, total_days, 21)) # Approx Monthly
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

        # We only want tradeable assets, not the forex tracker itself
        tradeable_metrics = metrics.drop("ILS=X", errors='ignore')

        # Check Kuramoto Crash Flag
        global_sync = metrics['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
        is_crashing = global_sync > 0.85

        targets = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}

        if is_crashing:
            # Dynamic Topological Haven: Select assets furthest from the synchronizing liquidating cluster
            if 'Topological_Distance' in tradeable_metrics.columns:
                dynamic_havens = tradeable_metrics.sort_values(by='Topological_Distance', ascending=False).head(2)
                weight_per_haven = 1.0 / len(dynamic_havens) if len(dynamic_havens) > 0 else 0
                for ticker in dynamic_havens.index:
                    targets[ticker] = weight_per_haven
            else:
                for safe_asset in SAFE_HAVENS:
                    if safe_asset in targets:
                        targets[safe_asset] = 0.5
        else:
            # Picky Top 5 methodology
            # Rank strictly by target weight (which encompasses velocity, safety, and centrality)
            top_5 = tradeable_metrics[tradeable_metrics['Target_Weight_Pct'] > 0].sort_values(by='Target_Weight_Pct', ascending=False).head(MAX_POSITIONS)

            # Equal weight the top 5 (20% each, approx 10,000 NIS each)
            weight_per_asset = 1.0 / len(top_5) if len(top_5) > 0 else 0

            for ticker in top_5.index:
                targets[ticker] = weight_per_asset

        allocations_over_time.append((current_day_idx, targets))

    return allocations_over_time

def execute_concentrated_backtest(close_df, allocations_over_time):
    # Account tracked in NIS
    current_cash_nis = INITIAL_BALANCE_NIS
    # Holdings tracked in shares
    holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}

    portfolio_value_history_nis = []
    rebalance_dict = {day_idx: allocs for day_idx, allocs in allocations_over_time}

    start_idx = allocations_over_time[0][0]
    total_days = close_df.shape[0]

    months_elapsed = 0

    for current_day_idx in range(start_idx, total_days):
        current_prices = close_df.iloc[current_day_idx]

        # Get exact daily conversion rate (1 USD = X ILS)
        usd_to_ils = current_prices.get("ILS=X", 3.7) # Fallback to 3.7 if missing

        # Calculate current holdings value in NIS
        holdings_value_usd = sum(holdings.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in holdings.keys() if ticker in current_prices)
        holdings_value_nis = holdings_value_usd * usd_to_ils

        total_portfolio_value_nis = current_cash_nis + holdings_value_nis

        if current_day_idx in rebalance_dict:
            target_weights = rebalance_dict[current_day_idx]

            # Liquidate to NIS
            current_cash_nis = total_portfolio_value_nis
            holdings = {ticker: 0.0 for ticker in holdings.keys()}

            # Reinvest
            for ticker, weight in target_weights.items():
                if weight > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    allocated_cash_nis = total_portfolio_value_nis * weight
                    allocated_cash_usd = allocated_cash_nis / usd_to_ils
                    shares_to_buy = allocated_cash_usd / current_prices[ticker]

                    holdings[ticker] = shares_to_buy
                    current_cash_nis -= allocated_cash_nis

            months_elapsed += 1

        portfolio_value_history_nis.append(total_portfolio_value_nis)

    final_value_nis = portfolio_value_history_nis[-1]
    total_net_profit_nis = final_value_nis - INITIAL_BALANCE_NIS
    total_return_pct = (total_net_profit_nis / INITIAL_BALANCE_NIS) * 100.0

    avg_monthly_profit_nis = total_net_profit_nis / months_elapsed if months_elapsed > 0 else 0
    avg_monthly_profit_pct = total_return_pct / months_elapsed if months_elapsed > 0 else 0

    # Benchmark SPY in NIS
    spy_start_usd = close_df.iloc[start_idx]['SPY']
    spy_end_usd = close_df.iloc[-1]['SPY']
    fx_start = close_df.iloc[start_idx].get("ILS=X", 3.7)
    fx_end = close_df.iloc[-1].get("ILS=X", 3.7)

    spy_start_nis = spy_start_usd * fx_start
    spy_end_nis = spy_end_usd * fx_end
    benchmark_return_pct = ((spy_end_nis / spy_start_nis) - 1.0) * 100.0

    print("\n" + "="*80)
    print("CONCENTRATED 'TOP 5' FOREX-ADJUSTED BACKTEST RESULTS")
    print("="*80)
    print(f"Starting Balance:           {INITIAL_BALANCE_NIS:,.2f} NIS")
    print(f"Final Balance:              {final_value_nis:,.2f} NIS")
    print(f"Total Net Profit:           {total_net_profit_nis:,.2f} NIS")
    print(f"Average Monthly Profit:     {avg_monthly_profit_nis:,.2f} NIS/mo ({avg_monthly_profit_pct:+.2f}%/mo)")
    print("-" * 80)
    print(f"Algorithm Total Return:     {total_return_pct:+.2f}%")
    print(f"Benchmark SPY (in NIS):     {benchmark_return_pct:+.2f}%")

    diff = total_return_pct - benchmark_return_pct
    if diff > 0:
        print(f"\n=> SUCCESS: Outperformed the SPY benchmark by {diff:.2f}%")
    else:
        print(f"\n=> UNDERPERFORMANCE: Trailed the SPY benchmark by {-diff:.2f}%")
    print("="*80)

if __name__ == "__main__":
    print("Fetching last 2 years of data...")
    # 1 Year backtest window
    close_df, vol_df = fetch_data_with_fx("2023-01-01", "2024-01-01")
    allocs = run_concentrated_allocations(close_df, vol_df, "2023-01-01")
    execute_concentrated_backtest(close_df, allocs)
