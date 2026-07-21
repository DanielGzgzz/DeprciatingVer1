import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
from market_analyzer import calculate_indicators, perform_ml_analysis

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "CVX", "BAC", "SPY", "QQQ", "TLT", "GLD", "ILS=X"]
INITIAL_BALANCE_NIS = 50000.0
FEE_PER_TRANSACTION_NIS = 60.0
MAX_POSITIONS = 5

def fetch_data_with_fx(start_date, end_date):
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    data = yf.download(TEST_SYMBOLS, start=fetch_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_network_allocations(close_df, vol_df, test_start_date):
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

        tradeable_metrics = metrics.drop("ILS=X", errors='ignore')

        targets = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}

        # STRATEGY: Allocate strictly by Eigenvector Centrality
        if 'Eigenvector_Centrality' in tradeable_metrics.columns:
            top_central = tradeable_metrics.sort_values(by='Eigenvector_Centrality', ascending=False).head(MAX_POSITIONS)
            total_centrality = top_central['Eigenvector_Centrality'].sum()

            if total_centrality > 0:
                for ticker in top_central.index:
                    targets[ticker] = top_central.loc[ticker, 'Eigenvector_Centrality'] / total_centrality
            else:
                for safe_asset in ["TLT", "GLD"]:
                    if safe_asset in targets: targets[safe_asset] = 0.5
        else:
            for safe_asset in ["TLT", "GLD"]:
                if safe_asset in targets: targets[safe_asset] = 0.5

        allocations_over_time.append((current_day_idx, targets))

    return allocations_over_time

def execute_network_backtest(close_df, allocations_over_time, label):
    current_cash_nis = INITIAL_BALANCE_NIS
    holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}
    peak_prices = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}

    portfolio_value_history_nis = []
    rebalance_dict = {day_idx: allocs for day_idx, allocs in allocations_over_time}

    start_idx = allocations_over_time[0][0]
    total_days = close_df.shape[0]

    total_fees_paid_nis = 0.0
    high_water_mark = INITIAL_BALANCE_NIS
    max_drawdown_pct = 0.0

    for current_day_idx in range(start_idx, total_days):
        current_prices = close_df.iloc[current_day_idx]
        usd_to_ils = current_prices.get("ILS=X", 3.7)

        # Trailing stop loss logic (10%)
        for ticker, shares in list(holdings.items()):
            if shares > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                current_price = current_prices[ticker]

                if current_price > peak_prices[ticker]:
                    peak_prices[ticker] = current_price

                elif current_price < 0.9 * peak_prices[ticker]:
                    # Liquidate position (Sell Fee)
                    value_usd = shares * current_price
                    value_nis = value_usd * usd_to_ils

                    # Deduct fee
                    current_cash_nis += value_nis
                    current_cash_nis -= FEE_PER_TRANSACTION_NIS
                    total_fees_paid_nis += FEE_PER_TRANSACTION_NIS

                    holdings[ticker] = 0.0
                    peak_prices[ticker] = 0.0

        # Calculate current holdings value in NIS
        holdings_value_usd = sum(holdings.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in holdings.keys() if ticker in current_prices)
        holdings_value_nis = holdings_value_usd * usd_to_ils

        total_portfolio_value_nis = current_cash_nis + holdings_value_nis

        # Max Drawdown tracking
        if total_portfolio_value_nis > high_water_mark:
            high_water_mark = total_portfolio_value_nis
        else:
            drawdown = (high_water_mark - total_portfolio_value_nis) / high_water_mark
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown

        if current_day_idx in rebalance_dict:
            target_weights = rebalance_dict[current_day_idx]

            # Liquidate to NIS
            current_cash_nis = total_portfolio_value_nis
            for ticker, qty in holdings.items():
                if qty > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    current_cash_nis -= FEE_PER_TRANSACTION_NIS # Sell Fee
                    total_fees_paid_nis += FEE_PER_TRANSACTION_NIS

            holdings = {ticker: 0.0 for ticker in holdings.keys()}
            peak_prices = {ticker: 0.0 for ticker in peak_prices.keys()}

            # Reinvest
            total_investable = current_cash_nis
            for ticker, weight in target_weights.items():
                if weight > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    allocated_cash_nis = total_investable * weight
                    # Buy Fee
                    allocated_cash_nis -= FEE_PER_TRANSACTION_NIS
                    total_fees_paid_nis += FEE_PER_TRANSACTION_NIS

                    if allocated_cash_nis > 0:
                        allocated_cash_usd = allocated_cash_nis / usd_to_ils
                        shares_to_buy = allocated_cash_usd / current_prices[ticker]

                        holdings[ticker] = shares_to_buy
                        peak_prices[ticker] = current_prices[ticker]
                        current_cash_nis -= (allocated_cash_nis + FEE_PER_TRANSACTION_NIS)

        portfolio_value_history_nis.append(total_portfolio_value_nis)

    final_value_nis = portfolio_value_history_nis[-1]
    total_net_profit_nis = final_value_nis - INITIAL_BALANCE_NIS
    total_return_pct = (total_net_profit_nis / INITIAL_BALANCE_NIS) * 100.0

    print("\n" + "="*80)
    print(f"RESULTS FOR: {label}")
    print("="*80)
    print(f"Starting Balance:           {INITIAL_BALANCE_NIS:,.2f} NIS")
    print(f"Final Balance:              {final_value_nis:,.2f} NIS")
    print(f"Total Net Profit:           {total_net_profit_nis:,.2f} NIS")
    print(f"Total Fees Paid:            {total_fees_paid_nis:,.2f} NIS")
    print(f"Max Drawdown:               {max_drawdown_pct*100:.2f}%")
    print("-" * 80)
    print(f"Algorithm Net Return:       {total_return_pct:+.2f}%")
    print("="*80)

    return final_value_nis, total_net_profit_nis, total_fees_paid_nis, max_drawdown_pct

def run_scenarios():
    now = datetime.now() - timedelta(days=30)
    end_date_str = now.strftime('%Y-%m-%d')

    # Last 1 Month
    start_1m = (now - timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"Fetching data for Last 1 Month ({start_1m} to {end_date_str})...")
    c1, v1 = fetch_data_with_fx(start_1m, end_date_str)
    a1 = run_network_allocations(c1, v1, start_1m)
    execute_network_backtest(c1, a1, "Last 1 Month")

    # Last 1 Year
    start_1y = (now - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"\nFetching data for Last 1 Year ({start_1y} to {end_date_str})...")
    c2, v2 = fetch_data_with_fx(start_1y, end_date_str)
    a2 = run_network_allocations(c2, v2, start_1y)
    execute_network_backtest(c2, a2, "Last 1 Year")

    # Last 10 Years
    start_10y = (now - timedelta(days=365*10)).strftime('%Y-%m-%d')
    print(f"\nFetching data for Last 10 Years ({start_10y} to {end_date_str})...")
    c3, v3 = fetch_data_with_fx(start_10y, end_date_str)
    a3 = run_network_allocations(c3, v3, start_10y)
    execute_network_backtest(c3, a3, "Last 10 Years")

if __name__ == "__main__":
    run_scenarios()
