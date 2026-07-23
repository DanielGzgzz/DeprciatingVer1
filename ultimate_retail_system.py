import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from market_analyzer import SYMBOLS, calculate_indicators, perform_ml_analysis

console = Console()

# Simulation Parameters
INITIAL_BALANCE_NIS = 100000.0
FEE_PER_TRANSACTION_NIS = 60.0
MAX_POSITIONS = 5
KURAMOTO_THRESHOLD = 0.85
FIEDLER_THRESHOLD = 0.80
TRAILING_STOP_PCT = 0.20 # 20% wide trailing stop

def fetch_universe_data(start_date, end_date):
    fetch_start = pd.to_datetime(start_date) - pd.DateOffset(years=1)
    # yfinance chunking for 900+ symbols to prevent timeout
    chunk_size = 200
    all_data = []

    # We must include ILS=X for forex conversion
    symbols_to_fetch = SYMBOLS.copy()
    if "ILS=X" not in symbols_to_fetch:
        symbols_to_fetch.append("ILS=X")

    for i in range(0, len(symbols_to_fetch), chunk_size):
        chunk = symbols_to_fetch[i:i + chunk_size]
        data = yf.download(chunk, start=fetch_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
        all_data.append(data)

    combined_data = pd.concat(all_data, axis=1)

    close_df = combined_data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = combined_data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def run_ultimate_backtest(close_df, vol_df, test_start_date):
    try:
        start_idx = close_df.index.get_indexer([pd.to_datetime(test_start_date)], method='nearest')[0]
    except KeyError:
        start_idx = 252

    total_days = close_df.shape[0]

    current_cash_nis = INITIAL_BALANCE_NIS
    holdings = {ticker: 0.0 for ticker in close_df.columns if ticker != "ILS=X"}
    peak_prices = {ticker: 0.0 for ticker in close_df.columns if ticker != "ILS=X"}

    portfolio_value_history_nis = []
    transaction_log = []

    total_fees_paid_nis = 0.0
    high_water_mark = INITIAL_BALANCE_NIS
    max_drawdown_pct = 0.0

    current_state_crashing = False

    for current_day_idx in range(start_idx, total_days):
        current_date = close_df.index[current_day_idx]
        current_prices = close_df.iloc[current_day_idx]
        usd_to_ils = current_prices.get("ILS=X", 3.7)

        # ------------------------------------------------------------------
        # FAILSAFE 1: RISING TRAILING STOP LOSS (DAILY EVALUATION)
        # ------------------------------------------------------------------
        for ticker, shares in list(holdings.items()):
            if shares > 0 and ticker in current_prices and not np.isnan(current_prices[ticker]):
                current_price = current_prices[ticker]

                # Update peak price as it rises
                if current_price > peak_prices[ticker]:
                    peak_prices[ticker] = current_price

                # Check for 20% drop from the absolute peak
                elif current_price < (1.0 - TRAILING_STOP_PCT) * peak_prices[ticker]:
                    value_usd = shares * current_price
                    value_nis = value_usd * usd_to_ils

                    current_cash_nis += value_nis - FEE_PER_TRANSACTION_NIS
                    total_fees_paid_nis += FEE_PER_TRANSACTION_NIS

                    transaction_log.append(f"[red]STOP-LOSS HIT[/red] | {current_date.strftime('%Y-%m-%d')} | {ticker:5s} | Price: ${current_price:7.2f} (Down from peak ${peak_prices[ticker]:.2f}) | Value: {value_nis:9.2f} NIS")

                    holdings[ticker] = 0.0
                    peak_prices[ticker] = 0.0

        # Calculate portfolio value before potential monthly rebalance
        current_holdings_nis = {ticker: (shares * current_prices.get(ticker, 0) * usd_to_ils) for ticker, shares in holdings.items() if shares > 0 and ticker in current_prices}
        total_portfolio_value_nis = current_cash_nis + sum(current_holdings_nis.values())

        if total_portfolio_value_nis > high_water_mark:
            high_water_mark = total_portfolio_value_nis
        else:
            drawdown = (high_water_mark - total_portfolio_value_nis) / high_water_mark
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown


        # ------------------------------------------------------------------
        # MONTHLY RE-EVALUATION & LAZY TRADING
        # ------------------------------------------------------------------
        if (current_day_idx - start_idx) % 21 == 0:
            historical_close = close_df.iloc[:current_day_idx]
            historical_vol = vol_df.iloc[:current_day_idx]

            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                # Filter to only columns that have data to avoid shape mismatches
                valid_cols = historical_close.dropna(axis=1, how='all').columns
                hist_close_clean = historical_close[valid_cols]
                hist_vol_clean = historical_vol[valid_cols]

                metrics = calculate_indicators(hist_close_clean, hist_vol_clean)
                metrics, vapor = perform_ml_analysis(hist_close_clean, hist_vol_clean, metrics)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout

            tradeable_metrics = metrics.drop("ILS=X", errors='ignore')

            global_sync = metrics['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
            fiedler_val = metrics['Fiedler_Value'].iloc[0] if 'Fiedler_Value' in metrics.columns else 0.0

            # FAILSAFE 2: MACRO CRASH DETECTOR
            is_crashing = (global_sync > KURAMOTO_THRESHOLD) and (fiedler_val > FIEDLER_THRESHOLD)

            if is_crashing:
                if not current_state_crashing:
                    # Evacuate completely
                    for ticker, shares in list(holdings.items()):
                        if shares > 0:
                            value_usd = shares * current_prices.get(ticker, 0)
                            current_cash_nis += (value_usd * usd_to_ils) - FEE_PER_TRANSACTION_NIS
                            total_fees_paid_nis += FEE_PER_TRANSACTION_NIS
                            holdings[ticker] = 0.0
                            peak_prices[ticker] = 0.0
                            transaction_log.append(f"[bold red]CRASH EVACUATION[/bold red] | {current_date.strftime('%Y-%m-%d')} | {ticker:5s} | Liquidated to Cash.")

                    current_state_crashing = True
                    transaction_log.append(f"[bold yellow]System parked in Cash due to high Kuramoto ({global_sync:.2f}) & Fiedler ({fiedler_val:.2f})[/bold yellow]")

            else:
                if current_state_crashing:
                    transaction_log.append(f"[bold green]CRASH OVER[/bold green] | {current_date.strftime('%Y-%m-%d')} | Re-entering market.")
                    current_state_crashing = False

                # Determine new Top 5 Targets
                top_targets = []
                if 'Eigenvector_Centrality' in tradeable_metrics.columns:
                    top_df = tradeable_metrics.sort_values(by='Eigenvector_Centrality', ascending=False).head(MAX_POSITIONS)
                    top_targets = top_df.index.tolist()

                if not top_targets:
                    top_targets = ["SPY", "QQQ", "GLD", "TLT", "JNJ"][:MAX_POSITIONS]

                # 1. LAZY SELL: Sell anything we hold that is NO LONGER in the Top 5 Targets
                for ticker, shares in list(holdings.items()):
                    if shares > 0 and ticker not in top_targets:
                        value_usd = shares * current_prices.get(ticker, 0)
                        current_cash_nis += (value_usd * usd_to_ils) - FEE_PER_TRANSACTION_NIS
                        total_fees_paid_nis += FEE_PER_TRANSACTION_NIS
                        transaction_log.append(f"[magenta]LAZY SELL[/magenta] | {current_date.strftime('%Y-%m-%d')} | {ticker:5s} | Dropped out of Top 5 Network Rank.")
                        holdings[ticker] = 0.0
                        peak_prices[ticker] = 0.0

                # 2. LAZY BUY: Distribute available cash equally among Top 5 Targets that we DON'T currently hold
                current_assets_held = [t for t, s in holdings.items() if s > 0]
                assets_to_buy = [t for t in top_targets if t not in current_assets_held]

                if assets_to_buy and current_cash_nis > (FEE_PER_TRANSACTION_NIS * 10):
                    # We reserve some cash buffer
                    investable_cash = current_cash_nis * 0.98
                    cash_per_asset = investable_cash / len(assets_to_buy)

                    for ticker in assets_to_buy:
                        if ticker in current_prices and not np.isnan(current_prices[ticker]):
                            allocated_nis = cash_per_asset - FEE_PER_TRANSACTION_NIS
                            if allocated_nis > 0:
                                price_usd = current_prices[ticker]
                                shares_bought = (allocated_nis / usd_to_ils) / price_usd
                                current_cash_nis -= (allocated_nis + FEE_PER_TRANSACTION_NIS)
                                total_fees_paid_nis += FEE_PER_TRANSACTION_NIS
                                holdings[ticker] = shares_bought
                                peak_prices[ticker] = price_usd
                                transaction_log.append(f"[cyan]LAZY BUY[/cyan]  | {current_date.strftime('%Y-%m-%d')} | {ticker:5s} | Acquired new Top 5 Asset.")

        # Update tracking
        total_portfolio_value_nis = current_cash_nis + sum(shares * current_prices.get(t, 0) * usd_to_ils for t, shares in holdings.items() if shares > 0 and t in current_prices)
        portfolio_value_history_nis.append(total_portfolio_value_nis)

    final_value_nis = portfolio_value_history_nis[-1]
    total_net_profit_nis = final_value_nis - INITIAL_BALANCE_NIS
    total_return_pct = (total_net_profit_nis / INITIAL_BALANCE_NIS) * 100.0

    fx_start = close_df.iloc[start_idx].get("ILS=X", 3.7)
    fx_end = close_df.iloc[-1].get("ILS=X", 3.7)

    spy_start = close_df.iloc[start_idx].get("SPY", 1.0) * fx_start
    spy_end = close_df.iloc[-1].get("SPY", 1.0) * fx_end
    spy_return_pct = ((spy_end / spy_start) - 1.0) * 100.0 if spy_start > 0 else 0.0

    print_rich_report(final_value_nis, total_net_profit_nis, total_return_pct, spy_return_pct, max_drawdown_pct, total_fees_paid_nis, transaction_log)


def print_rich_report(final_val, net_profit, net_return, spy_return, max_dd, total_fees, logs):
    console.print("\n")

    # 1. System Architecture Panel
    arch_text = Text()
    arch_text.append("1. UNIVERSE: ", style="bold")
    arch_text.append("945 Assets (Equities, ETFs, Forex)\n")
    arch_text.append("2. ALLOCATION: ", style="bold")
    arch_text.append("Top 5 Nodes by Eigenvector Centrality (Re-evaluated Monthly)\n")
    arch_text.append("3. FRICTION: ", style="bold")
    arch_text.append("60 NIS per trade.\n")
    arch_text.append("4. FAILSAFE 1 (RISING STOP): ", style="bold")
    arch_text.append("20% Trailing Stop-Loss on individual positions.\n")
    arch_text.append("5. FAILSAFE 2 (MACRO CRASH): ", style="bold")
    arch_text.append("Kuramoto Phase Sync (r > 0.85) AND Fiedler Eigenvalue (λ2 > 0.80)\n")
    arch_text.append("6. LAZY REBALANCING: ", style="bold")
    arch_text.append("Hold winning assets. Only sell if they fall out of the Top 5 or hit a failsafe.")

    console.print(Panel(arch_text, title="[bold cyan]ULTIMATE RETAIL SYSTEM ARCHITECTURE[/bold cyan]", border_style="cyan"))

    # 2. Transaction Log Table
    table = Table(title="[bold]CHRONOLOGICAL TRANSACTION LOG[/bold]", show_lines=True)
    table.add_column("Event")

    for log in logs[-15:]: # Show last 15
        table.add_row(log)

    console.print(table)
    if len(logs) > 15:
        console.print(f"[dim]... and {len(logs) - 15} previous transactions hidden.[/dim]")
    elif not logs:
        console.print("[dim]No transactions occurred during this period.[/dim]")

    # 3. Verdict Panel
    diff = net_return - spy_return
    v_style = "bold green" if diff > 0 else "bold red"
    verdict_str = f"SYSTEM OUTPERFORMED S&P 500 BY {diff:.2f}%" if diff > 0 else f"SYSTEM TRAILED S&P 500 BY {-diff:.2f}%"

    res_text = Text()
    res_text.append(f"Starting Balance:   {INITIAL_BALANCE_NIS:,.2f} NIS\n")
    res_text.append(f"Final Balance:      {final_val:,.2f} NIS\n")
    res_text.append(f"Total Fees Paid:    {total_fees:,.2f} NIS\n")
    res_text.append(f"System Max Drawdown:{max_dd*100:.2f}%\n")
    res_text.append("-" * 40 + "\n")
    res_text.append(f"Algorithm Return:   {net_return:+.2f}%\n", style="bold")
    res_text.append(f"Benchmark SPY:      {spy_return:+.2f}%\n")
    res_text.append(f"\nVERDICT: {verdict_str}\n", style=v_style)

    if diff > 0:
        res_text.append("\nThe system works. The Lazy Rebalancing drastically reduces fee friction, while the Rising Trailing Stop protects specific assets without interfering with the Kuramoto macro-crash detector.")

    console.print(Panel(res_text, title="[bold magenta]PERFORMANCE & RELIABILITY VERDICT[/bold magenta]", border_style="magenta"))


if __name__ == "__main__":
    # Test over the last 1 Year (short enough to compute 945 tickers locally without timeout)
    end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=395)).strftime('%Y-%m-%d')

    with console.status(f"[bold green]Fetching 945 tickers from {start_date} to {end_date}...") as status:
        close_df, vol_df = fetch_universe_data(start_date, end_date)

    with console.status("[bold yellow]Executing Ultimate Retail Backtest (Monthly Scans of 900+ Nodes)...") as status:
        run_ultimate_backtest(close_df, vol_df, start_date)
