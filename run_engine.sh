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
    echo -e "\n${CYAN}MAIN MENU:${NC}"
    echo "  [1] Scan the Entire Market (Generate Full Structural Ledger & Bode Plots)"
    echo "  [2] Assess Current Portfolio (Interactive Evaluation)"
    echo "  [3] Check Specific Ticker (Single Node Vector Analysis)"
    echo "  [4] Quit"

    read -p "Select an option [1-4]: " option

    case $option in
        1)
            echo -e "\n${GREEN}Executing Full Market Scan (This will process 900+ vectors)...${NC}"
            python3 market_analyzer.py
            ;;
        2)
            echo -e "\n${GREEN}Initializing Interactive Portfolio Assessor...${NC}"
            python3 portfolio_assessor.py --portfolio
            ;;
        3)
            read -p "Enter Ticker Symbol (e.g. AMPX): " ticker
            if [ -n "$ticker" ]; then
                echo -e "\n${GREEN}Scanning Node: $ticker...${NC}"
                python3 portfolio_assessor.py --single "$ticker"
            else
                echo -e "${RED}Invalid input.${NC}"
            fi
            ;;
        4)
            echo -e "\n${CYAN}Shutting down Thermodynamic Engine...${NC}"
            break
            ;;
        *)
            echo -e "${RED}Invalid option. Please select 1, 2, 3, or 4.${NC}"
            ;;
    esac
done
