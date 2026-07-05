#!/bin/bash

# Ensure terminal colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}      THERMODYNAMIC SPECTRAL MARKET ENGINE V1.0      ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Update the Universe
echo -e "\n[SYSTEM] Updating Global Tensor Matrix (Fetching S&P 500, S&P 400, ETFs, Forex)..."
python3 expand_symbols.py

echo -e "\n${GREEN}[SYSTEM] Matrix updated successfully.${NC}"

# Show the top 10 tickers by grabbing the output safely from python
echo -e "\n--- Top 10 Core Matrix Nodes (Sample) ---"
python3 -c "from market_analyzer import SYMBOLS; print(' '.join(SYMBOLS[:10]))"
echo -e "\n-----------------------------------------"

while true; do
    echo -e "
${CYAN}MAIN MENU:${NC}"
    echo "  [1] Scan the Entire Market (Generate Full Structural Ledger & Bode Plots)"
    echo "  [2] Assess Current Portfolio (Interactive Evaluation)"
    echo "  [3] Check Specific Ticker (Single Node Vector Analysis)"
    echo "  [4] Run Historical Backtest (Dirac Shocks & Monte Carlo)"
    echo "  [5] Export Previous Scan Results (TXT & Deep-Dive PDF)"
    echo "  [6] Quick Test (Simulate Last 30 Days)"
    echo "  [7] Train & Calibrate (Matrix Roaming & Optimization)"
    echo "  [8] Run Concentrated Top-5 NIS Backtest"
    echo "  [9] Quit"

    read -p "Select an option [1-9]: " option

    case $option in
        1)
            echo -e "
${GREEN}Executing Full Market Scan (This will process 900+ vectors)...${NC}"
            python3 market_analyzer.py
            ;;
        2)
            echo -e "
${GREEN}Initializing Interactive Portfolio Assessor...${NC}"
            python3 portfolio_assessor.py --portfolio
            ;;
        3)
            read -p "Enter Ticker Symbol (e.g. AMPX): " ticker
            if [ -n "$ticker" ]; then
                echo -e "
${GREEN}Scanning Node: $ticker...${NC}"
                python3 portfolio_assessor.py --single "$ticker"
                export LAST_TICKER="$ticker"
            else
                echo -e "${RED}Invalid input.${NC}"
            fi
            ;;
        4)
            echo -e "
${GREEN}Executing Dirac Shocks & Monte Carlo Validation...${NC}"
            python3 stress_test.py
            ;;
        5)
            if [ -n "$LAST_TICKER" ]; then
                echo -e "
${GREEN}Exporting Granular Data for $LAST_TICKER...${NC}"
                # Generate the PDF
                python3 -c "from ticker_pdf_engine import generate_ticker_pdf; generate_ticker_pdf('$LAST_TICKER')"
            else
                echo -e "${RED}No ticker has been scanned yet. Please run Option 3 first.${NC}"
            fi
            ;;
        6)
            echo -e "
${GREEN}Simulating Last 30 Days...${NC}"
            # A quick wrapper to run rigorous_backtest logic over a short period
            python3 -c "
import yfinance as yf, pandas as pd
from rigorous_backtest import fetch_period_data, run_monthly_rebalancing, execute_backtest
from datetime import datetime, timedelta
start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
end = datetime.now().strftime('%Y-%m-%d')
try:
    close_df, vol_df = fetch_period_data(start, end)
    allocs = run_monthly_rebalancing(close_df, vol_df, start)
    execute_backtest(close_df, allocs, 'Last 30 Days Simulation')
except Exception as e:
    print('Quick Test Error:', e)
"
            ;;
        7)
            echo -e "
${GREEN}Executing Heavy Continuous Calibration...${NC}"
            python3 continuous_calibrator.py
            ;;

        8)
            echo -e "
${GREEN}Running 50,000 NIS Top 5 Concentrated Backtest...${NC}"
            python3 concentrated_backtest.py
            ;;
        9)
            echo -e "
${CYAN}Shutting down Thermodynamic Engine...${NC}"
            break
            ;;
        *)
            echo -e "${RED}Invalid option. Please select 1-9.${NC}"
            ;;
    esac
done
