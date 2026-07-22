import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
from market_analyzer import calculate_indicators, perform_ml_analysis

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "CVX", "BAC", "SPY", "VOO", "QQQ", "TLT", "GLD", "ILS=X"]
INITIAL_BALANCE_NIS = 50000.0
FEE_PER_TRANSACTION_NIS = 60.0
MAX_POSITIONS = 5
KURAMOTO_THRESHOLD = 0.85

def fetch_data_with_fx(start_date, end_date):
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    data = yf.download(TEST_SYMBOLS, start=fetch_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_kuramoto_allocations(close_df, vol_df, test_start_date):
    try:
        start_idx = close_df.index.get_indexer([pd.to_datetime(test_start_date)], method='nearest')[0]
    except KeyError:
        start_idx = 252

    total_days = close_df.shape[0]
    # We assess the market every 21 days just to check the Kuramoto state.
    # We do NOT force a rebalance.
    rebalance_dates = list(range(start_idx, total_days, 21))
    allocations = {}

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

        global_sync = metrics['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
        is_crashing = global_sync > KURAMOTO_THRESHOLD

        targets = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}

        if is_crashing:
            for safe_asset in ["TLT", "GLD"]:
                if safe_asset in targets: targets[safe_asset] = 0.5
        else:
            if 'Eigenvector_Centrality' in tradeable_metrics.columns:
                top = tradeable_metrics.sort_values(by='Eigenvector_Centrality', ascending=False).head(MAX_POSITIONS)
                tot = top['Eigenvector_Centrality'].sum()
                if tot > 0:
                    for ticker in top.index:
                        targets[ticker] = top.loc[ticker, 'Eigenvector_Centrality'] / tot
            else:
                for safe_asset in ["TLT", "GLD"]:
                    if safe_asset in targets: targets[safe_asset] = 0.5

        allocations[current_day_idx] = (targets, is_crashing)

    return allocations

def execute_kuramoto_backtest(close_df, allocations_dict, label):
    current_cash_nis = INITIAL_BALANCE_NIS
    holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}

    portfolio_value_history_nis = []

    start_idx = list(allocations_dict.keys())[0]
    total_days = close_df.shape[0]

    total_fees_paid_nis = 0.0
    high_water_mark = INITIAL_BALANCE_NIS
    max_drawdown_pct = 0.0

    # We track the current state to only trade on state transitions
    current_state_crashing = None

    for current_day_idx in range(start_idx, total_days):
        current_prices = close_df.iloc[current_day_idx]
        usd_to_ils = current_prices.get("ILS=X", 3.7)

        if current_day_idx in allocations_dict:
            target_weights, is_crashing = allocations_dict[current_day_idx]

            # Initial setup or State Change (Normal <-> Crashing)
            if current_state_crashing is None or is_crashing != current_state_crashing:
                # Full liquidation
                for ticker, shares in list(holdings.items()):
                    if shares > 0:
                        value_usd = shares * current_prices.get(ticker, 0)
                        current_cash_nis += (value_usd * usd_to_ils) - FEE_PER_TRANSACTION_NIS
                        total_fees_paid_nis += FEE_PER_TRANSACTION_NIS
                        holdings[ticker] = 0.0

                # Full reinvestment
                total_portfolio_value_nis = current_cash_nis
                for ticker, weight in target_weights.items():
                    if weight > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                        allocated_nis = (total_portfolio_value_nis * weight) - FEE_PER_TRANSACTION_NIS
                        if allocated_nis > 0:
                            current_cash_nis -= (allocated_nis + FEE_PER_TRANSACTION_NIS)
                            total_fees_paid_nis += FEE_PER_TRANSACTION_NIS
                            holdings[ticker] = (allocated_nis / usd_to_ils) / current_prices[ticker]

                current_state_crashing = is_crashing

        # Calculate current holdings value in NIS
        current_holdings_nis = {ticker: (shares * current_prices.get(ticker, 0) * usd_to_ils) for ticker, shares in holdings.items() if shares > 0 and ticker in current_prices}
        total_portfolio_value_nis = current_cash_nis + sum(current_holdings_nis.values())

        # Max Drawdown tracking
        if total_portfolio_value_nis > high_water_mark:
            high_water_mark = total_portfolio_value_nis
        else:
            drawdown = (high_water_mark - total_portfolio_value_nis) / high_water_mark
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown

        portfolio_value_history_nis.append(total_portfolio_value_nis)

    final_value_nis = portfolio_value_history_nis[-1]
    total_net_profit_nis = final_value_nis - INITIAL_BALANCE_NIS
    total_return_pct = (total_net_profit_nis / INITIAL_BALANCE_NIS) * 100.0

    # Calculate Benchmark Returns (in NIS)
    fx_start = close_df.iloc[start_idx].get("ILS=X", 3.7)
    fx_end = close_df.iloc[-1].get("ILS=X", 3.7)

    spy_start = close_df.iloc[start_idx].get("SPY", 1.0) * fx_start
    spy_end = close_df.iloc[-1].get("SPY", 1.0) * fx_end
    spy_return_pct = ((spy_end / spy_start) - 1.0) * 100.0 if spy_start > 0 else 0.0

    voo_start = close_df.iloc[start_idx].get("VOO", 1.0) * fx_start
    voo_end = close_df.iloc[-1].get("VOO", 1.0) * fx_end
    voo_return_pct = ((voo_end / voo_start) - 1.0) * 100.0 if voo_start > 0 else 0.0

    print("\n" + "="*80)
    print(f"RESULTS FOR: {label} (Pure Kuramoto Crisis Trader)")
    print("="*80)
    print(f"Starting Balance:           {INITIAL_BALANCE_NIS:,.2f} NIS")
    print(f"Final Balance:              {final_value_nis:,.2f} NIS")
    print(f"Total Net Profit:           {total_net_profit_nis:,.2f} NIS")
    print(f"Total Fees Paid:            {total_fees_paid_nis:,.2f} NIS")
    print(f"Max Drawdown:               {max_drawdown_pct*100:.2f}%")
    print("-" * 80)
    print(f"Algorithm Net Return:       {total_return_pct:+.2f}%")
    print(f"Benchmark SPY Net Return:   {spy_return_pct:+.2f}%")
    print(f"Benchmark VOO Net Return:   {voo_return_pct:+.2f}%")

    diff_spy = total_return_pct - spy_return_pct
    diff_voo = total_return_pct - voo_return_pct

    if diff_spy > 0:
        print(f"\n=> Outperformed SPY by:       +{diff_spy:.2f}%")
    else:
        print(f"\n=> Trailed SPY by:            {diff_spy:.2f}%")

    if diff_voo > 0:
        print(f"=> Outperformed VOO by:       +{diff_voo:.2f}%")
    else:
        print(f"=> Trailed VOO by:            {diff_voo:.2f}%")
    print("="*80)

    return final_value_nis, total_net_profit_nis, total_fees_paid_nis, max_drawdown_pct

def run_scenarios():
    now = datetime.now() - timedelta(days=30)
    end_date_str = now.strftime('%Y-%m-%d')

    start_1m = (now - timedelta(days=30)).strftime('%Y-%m-%d')
    print(f"Fetching data for Last 1 Month ({start_1m} to {end_date_str})...")
    c1, v1 = fetch_data_with_fx(start_1m, end_date_str)
    a1 = run_kuramoto_allocations(c1, v1, start_1m)
    execute_kuramoto_backtest(c1, a1, "Last 1 Month")

    start_1y = (now - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"\nFetching data for Last 1 Year ({start_1y} to {end_date_str})...")
    c2, v2 = fetch_data_with_fx(start_1y, end_date_str)
    a2 = run_kuramoto_allocations(c2, v2, start_1y)
    execute_kuramoto_backtest(c2, a2, "Last 1 Year")

    start_10y = (now - timedelta(days=365*10)).strftime('%Y-%m-%d')
    print(f"\nFetching data for Last 10 Years ({start_10y} to {end_date_str})...")
    c3, v3 = fetch_data_with_fx(start_10y, end_date_str)
    a3 = run_kuramoto_allocations(c3, v3, start_10y)
    execute_kuramoto_backtest(c3, a3, "Last 10 Years")

if __name__ == "__main__":
    run_scenarios()
