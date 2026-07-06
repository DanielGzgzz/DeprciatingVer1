import time
import os
import sys
import math
import yfinance as yf
from market_analyzer import SYMBOLS
import pandas as pd

# ANSI Color Codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
RESET = '\033[0m'
CLEAR = '\033[2J'
HOME = '\033[H'

def draw_ascii_chart(data, width=50, height=10):
    if not data:
        return [""] * height

    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1

    chart_lines = [[" " for _ in range(width)] for _ in range(height)]

    for x in range(min(width, len(data))):
        val = data[x]
        y = int((val - min_val) / range_val * (height - 1))
        # Invert Y to draw from top down in console
        y = (height - 1) - y
        chart_lines[y][x] = "*"

    return ["".join(line) for line in chart_lines]

def print_dashboard(cycle):
    sys.stdout.write(CLEAR + HOME)

    # 1. Sectorial Sync Table
    print(f"{CYAN}{BOLD}========================================================================{RESET}")
    print(f"{CYAN}{BOLD}                KURAMOTO SECTOR FLOW (LIVE SYNC MATRIX)                 {RESET}")
    print(f"{CYAN}{BOLD}========================================================================{RESET}")

    print(f"{BOLD}{'Ticker':<10} | {'Sector':<15} | {'Phase (θ)':<15} | {'Lag (τ)':<8} | {'Status':<15}{RESET}")
    print("-" * 72)

    # Simulated Mock Data for the TUI rendering loop
    # In full production this connects to `calculate_dynamic_lags_and_coupling`
    sync_val = math.sin(cycle * 0.5)
    lag_val = abs(math.cos(cycle * 0.3))

    print(f"XOM        | Energy          | 3.14 (Leader)   | 0s       | {GREEN}SYNCED{RESET}")
    print(f"CVX        | Energy          | {2.81 + sync_val:.2f} (Lagging) | 45m      | {RED if lag_val > 0.5 else YELLOW}{'BUY LEAD' if lag_val > 0.5 else 'WATCH'}{RESET}")
    print(f"NVDA       | Semiconductors  | 1.55 (Leader)   | 0s       | {GREEN}SYNCED{RESET}")
    print(f"AMD        | Semiconductors  | {1.20 - sync_val:.2f} (Lagging) | 12m      | {YELLOW}WATCH{RESET}")
    print(f"JPM        | Financials      | 0.05 (Leader)   | 0s       | {GREEN}SYNCED{RESET}")
    print(f"BAC        | Financials      | {-0.10 + sync_val:.2f} (Lagging) | 30m      | {RED if lag_val > 0.3 else YELLOW}{'BUY LEAD' if lag_val > 0.3 else 'WATCH'}{RESET}")

    print("\n")

    # 2. Native ASCII Phase Chart
    print(f"{MAGENTA}{BOLD}--- SYSTEM COHERENCE (PHASE CONVERGENCE) ---{RESET}")
    chart_data = [math.sin(x * 0.5 + cycle * 0.2) * 5 + 5 for x in range(50)]
    rendered_chart = draw_ascii_chart(chart_data, 50, 8)
    for line in rendered_chart:
        print(f"{CYAN}{line}{RESET}")

    print("\n")

    # 3. Alpha Trigger Panel
    print(f"{RED}{BOLD}>>> ALPHA ALPHA TRIGGER ALERTS <<<{RESET}")
    print("-" * 72)

    if cycle % 5 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {BOLD}STAT-ARB ALERT:{RESET} SPY Phase Divergence Exceeds 2σ.")
    if cycle % 7 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {BOLD}COILED SPRING:{RESET} Ticker AMPX detected High Centrality + Low Vol.")
    if cycle % 3 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {BOLD}HIGH-BETA LAGGARD:{RESET} AMD heavily desynchronized from NVDA phase.")

    print("\nPress Ctrl+C to exit dashboard.")
    sys.stdout.flush()

def run_tui():
    cycle = 0
    try:
        while True:
            print_dashboard(cycle)
            cycle += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{CYAN}Shutting down TUI Dashboard...{RESET}")

if __name__ == "__main__":
    run_tui()
