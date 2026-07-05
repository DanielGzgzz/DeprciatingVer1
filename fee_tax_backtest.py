import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime
from market_analyzer import calculate_indicators, perform_ml_analysis

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "CVX", "BAC", "SPY", "QQQ", "TLT", "GLD", "ILS=X"]
INITIAL_BALANCE_NIS = 50000.0
SAFE_HAVENS = ["TLT", "GLD"]
FEE_PER_TRANSACTION_NIS = 60.0
TAX_RATE_PROFIT = 0.25

def fetch_data_with_fx(start_date, end_date):
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    data = yf.download(TEST_SYMBOLS, start=fetch_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_allocations(close_df, vol_df, test_start_date, rebalance_freq_days, max_positions):
    try:
        start_idx = close_df.index.get_indexer([pd.to_datetime(test_start_date)], method='nearest')[0]
    except KeyError:
        start_idx = 252

    total_days = close_df.shape[0]
    rebalance_dates = list(range(start_idx, total_days, rebalance_freq_days))
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

        global_sync = metrics['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
        is_crashing = global_sync > 0.85

        targets = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}

        if is_crashing:
            for safe_asset in SAFE_HAVENS:
                if safe_asset in targets:
                    targets[safe_asset] = 0.5
        else:
            top = tradeable_metrics[tradeable_metrics['Target_Weight_Pct'] > 0].sort_values(by='Target_Weight_Pct', ascending=False).head(max_positions)
            weight_per_asset = 1.0 / len(top) if len(top) > 0 else 0
            for ticker in top.index:
                targets[ticker] = weight_per_asset

        allocations_over_time.append((current_day_idx, targets))

    return allocations_over_time

def execute_backtest(close_df, allocations_over_time):
    current_cash_nis = INITIAL_BALANCE_NIS
    holdings = {ticker: 0.0 for ticker in TEST_SYMBOLS if ticker != "ILS=X"}
    rebalance_dict = {day_idx: allocs for day_idx, allocs in allocations_over_time}

    start_idx = allocations_over_time[0][0]
    total_days = close_df.shape[0]
    total_fees_paid_nis = 0.0

    for current_day_idx in range(start_idx, total_days):
        current_prices = close_df.iloc[current_day_idx]
        usd_to_ils = current_prices.get("ILS=X", 3.7)

        if current_day_idx in rebalance_dict:
            target_weights = rebalance_dict[current_day_idx]

            # Liquidate current holdings
            holdings_value_usd = 0
            for ticker, qty in holdings.items():
                if qty > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    holdings_value_usd += qty * current_prices[ticker]
                    # Pay sell fee
                    current_cash_nis -= FEE_PER_TRANSACTION_NIS
                    total_fees_paid_nis += FEE_PER_TRANSACTION_NIS

            current_cash_nis += holdings_value_usd * usd_to_ils
            holdings = {ticker: 0.0 for ticker in holdings.keys()}

            # Reinvest
            total_investable_nis = current_cash_nis
            for ticker, weight in target_weights.items():
                if weight > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                    allocated_cash_nis = total_investable_nis * weight
                    # Pay buy fee
                    allocated_cash_nis -= FEE_PER_TRANSACTION_NIS
                    total_fees_paid_nis += FEE_PER_TRANSACTION_NIS

                    if allocated_cash_nis > 0:
                        allocated_cash_usd = allocated_cash_nis / usd_to_ils
                        shares_to_buy = allocated_cash_usd / current_prices[ticker]
                        holdings[ticker] = shares_to_buy
                        current_cash_nis -= (allocated_cash_nis + FEE_PER_TRANSACTION_NIS)

    # Final value
    final_prices = close_df.iloc[-1]
    final_usd_to_ils = final_prices.get("ILS=X", 3.7)
    final_holdings_value_usd = sum(qty * final_prices.get(ticker, 0) for ticker, qty in holdings.items() if ticker in final_prices)
    final_holdings_value_nis = final_holdings_value_usd * final_usd_to_ils

    gross_final_value_nis = current_cash_nis + final_holdings_value_nis
    gross_profit_nis = gross_final_value_nis - INITIAL_BALANCE_NIS

    # Apply taxes on profit
    tax_paid_nis = 0.0
    if gross_profit_nis > 0:
        tax_paid_nis = gross_profit_nis * TAX_RATE_PROFIT

    net_final_value_nis = gross_final_value_nis - tax_paid_nis
    net_profit_nis = net_final_value_nis - INITIAL_BALANCE_NIS
    net_return_pct = (net_profit_nis / INITIAL_BALANCE_NIS) * 100.0

    # SPY Benchmark
    spy_start_nis = close_df.iloc[start_idx]['SPY'] * close_df.iloc[start_idx].get("ILS=X", 3.7)
    spy_end_nis = final_prices['SPY'] * final_usd_to_ils
    spy_return_pct = ((spy_end_nis / spy_start_nis) - 1.0) * 100.0

    return net_return_pct, spy_return_pct, total_fees_paid_nis, tax_paid_nis, net_profit_nis

def optimize_strategy():
    print("Fetching data for optimization...")
    close_df, vol_df = fetch_data_with_fx("2023-01-01", "2024-01-01")

    frequencies = [21, 63, 126] # Monthly, Quarterly, Bi-Annually
    positions = [1, 2, 3, 5]

    best_return = -999
    best_params = {}

    results = []

    for freq in frequencies:
        for pos in positions:
            print(f"Testing Method -> Freq: {freq} days, Max Pos: {pos}...")
            allocs = run_allocations(close_df, vol_df, "2023-01-01", freq, pos)
            net_ret, spy_ret, fees, tax, net_prof = execute_backtest(close_df, allocs)

            results.append({
                "Freq": freq, "Pos": pos, "NetRet": net_ret, "SpyRet": spy_ret,
                "Fees": fees, "Tax": tax, "NetProf": net_prof
            })

            if net_ret > best_return:
                best_return = net_ret
                best_params = {"Freq": freq, "Pos": pos, "NetRet": net_ret, "NetProf": net_prof, "SpyRet": spy_ret}

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (After 60 NIS per trade + 25% Tax)")
    print("="*80)
    for r in results:
        print(f"Freq: {r['Freq']:3d} | Pos: {r['Pos']} | Fees: {r['Fees']:6.0f} NIS | Tax: {r['Tax']:6.0f} NIS | Net Profit: {r['NetProf']:8.2f} NIS ({r['NetRet']:+6.2f}%)")

    print("-" * 80)
    print(f"BEST METHOD: Rebalance every {best_params['Freq']} days with Top {best_params['Pos']} positions.")
    print(f"Best Net Return: {best_params['NetRet']:+.2f}% | Best Net Profit: {best_params['NetProf']:.2f} NIS")
    print(f"SPY Benchmark Return: {best_params['SpyRet']:+.2f}%")

    diff = best_params['NetRet'] - best_params['SpyRet']
    if diff > 0:
        print(f"\n=> SUCCESS: Outperformed the SPY benchmark by {diff:.2f}% (After Fees & Taxes)")
    else:
        print(f"\n=> UNDERPERFORMANCE: Trailed the SPY benchmark by {-diff:.2f}% (After Fees & Taxes)")
    print("="*80)


def run_capital_scaling_test():
    print("\n" + "="*80)
    print("CAPITAL SCALING TEST (Fee Drag Analysis)")
    print("="*80)
    close_df, vol_df = fetch_data_with_fx("2023-01-01", "2024-01-01")
    allocs = run_allocations(close_df, vol_df, "2023-01-01", 63, 5)

    capitals = [50000.0, 250000.0, 1000000.0]
    global INITIAL_BALANCE_NIS

    for cap in capitals:
        INITIAL_BALANCE_NIS = cap
        net_ret, spy_ret, fees, tax, net_prof = execute_backtest(close_df, allocs)
        print(f"Starting Capital: {cap:10,.0f} NIS")
        print(f"Fees Paid:        {fees:10,.0f} NIS")
        print(f"Taxes Paid:       {tax:10,.0f} NIS")
        print(f"Algorithm Net Return: {net_ret:+.2f}%")
        print(f"SPY Net Return:       {spy_ret:+.2f}%")
        diff = net_ret - spy_ret
        print(f"Diff vs SPY:          {diff:+.2f}%\n")

if __name__ == "__main__":
    run_capital_scaling_test()
    optimize_strategy()
