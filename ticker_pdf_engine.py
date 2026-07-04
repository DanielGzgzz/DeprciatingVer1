import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from market_analyzer import SYMBOLS, fetch_data, calculate_indicators, perform_ml_analysis, perform_spectral_analysis

def generate_ticker_pdf(ticker):
    print(f"\nGenerating Granular Ticker PDF Engine for [{ticker}]...")

    # 1. Fetch Data
    core_benchmarks = ["SPY", "QQQ", "TLT", "GLD", "XOM", "JNJ", "MSFT", "AAPL"]
    if ticker not in core_benchmarks:
        temp_symbols = core_benchmarks + [ticker]
    else:
        temp_symbols = core_benchmarks

    import os
    old_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            close_df, vol_df = fetch_data(temp_symbols)
            metrics = calculate_indicators(close_df, vol_df)
            metrics, vapor = perform_ml_analysis(close_df, vol_df, metrics)
            target_node = metrics.loc[[ticker]] if ticker in metrics.index else pd.DataFrame()
            if not target_node.empty:
                spectral_results = perform_spectral_analysis(close_df, target_node)
            else:
                spectral_results = {}
        finally:
            sys.stdout = old_stdout

    if ticker not in metrics.index:
        print(f"Failed to resolve {ticker}.")
        return

    # 2. Extract Data
    row = metrics.loc[ticker]
    current_price = row['Current_Price']
    sma_50 = close_df[ticker].rolling(window=50).mean().iloc[-1]
    sma_200 = close_df[ticker].rolling(window=200).mean().iloc[-1]
    est_real_value = (sma_50 * 0.4) + (sma_200 * 0.6)

    volatility = close_df[ticker].pct_change().rolling(window=30).std().iloc[-1] * np.sqrt(252)
    dynamic_stop_pct = min(0.30, max(0.05, volatility * 2.0))
    buy_target = min(current_price, est_real_value)
    stop_loss = buy_target * (1.0 - dynamic_stop_pct)
    take_profit = buy_target * (1.0 + 0.15 + volatility)

    flow_deriv = row['Flow_Derivative']
    vol_sat = row['Volume_Saturation']
    flow_state = "Accelerating" if flow_deriv > 0 else "Decelerating"
    sat_state = "Saturating" if vol_sat > 1.2 else "Stable"

    # 3. Create PDF
    pdf_filename = f"{ticker}_Granular_Analysis.pdf"
    with PdfPages(pdf_filename) as pdf:

        # PAGE 1: Overview & Waypoint Tree
        fig = plt.figure(figsize=(10, 8))
        fig.text(0.5, 0.92, f"GRANULAR THERMODYNAMIC NODE ANALYSIS: {ticker}", fontsize=18, fontweight='bold', ha='center')
        fig.text(0.5, 0.88, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10, ha='center', color='gray')

        body = f"""
1. NODE STATE SUMMARY
--------------------------------------------------------------------------------
Current Price:           ${current_price:.2f}
Estimated Real Value:    ${est_real_value:.2f}
All-Time Velocity:       {row['All_Time_Velocity']:.2f}
Topological Safety:      {row['Topological_Safety']:.2f}
Current Flow State:      {flow_state} (dW/dt = {flow_deriv:.4f})
Volume Accumulation:     {sat_state} (Sat Index = {vol_sat:.2f})

2. CHRONOLOGICAL WAYPOINT MATRIX
--------------------------------------------------------------------------------
Optimal Buy Zone:        < ${buy_target:.2f}
Dynamic Trailing Stop:   ${stop_loss:.2f} (-{dynamic_stop_pct*100:.1f}% rolling vol)
Adaptive Take Profit:    ${take_profit:.2f}

[WAYPOINT 1 - Short Term]:
"""
        if flow_deriv < -0.05:
            body += f"  IF Price < ${buy_target:.2f} AND Flow remains Decelerating -> SELL (Derivative Stop-Loss Triggered).\n"
        else:
            body += f"  IF Price < ${buy_target:.2f} BUT Flow is Accelerating -> HOLD. Capital is rotating inward.\n"

        body += f"\n[WAYPOINT 2 - Integral Flow Exit]:\n"
        if vol_sat > 1.2:
            body += f"  IF Volume Saturation persists (>1.2) -> SCALE OUT (Integral Take-Profit Triggered). Peak absorption reached.\n"
        else:
            body += f"  IF Target not met and Volume is Stable -> HOLD. Allow structural drift to continue.\n"

        # Fetch Earnings Dates
        try:
            ticker_obj = yf.Ticker(ticker)
            earnings = ticker_obj.calendar
            if earnings is not None and not earnings.empty:
                # the new yfinance calendar returns a DataFrame, format varies
                body += f"\n3. NEXT EARNINGS CATALYST\n--------------------------------------------------------------------------------\n"
                # Simple extraction, just stringify the first row to be safe
                body += str(earnings.head(1))
        except Exception:
            body += "\n[Earnings data unavailable]"

        fig.text(0.1, 0.75, body, fontsize=11, va='top', ha='left', family='monospace')
        plt.axis('off')
        pdf.savefig(fig)

        # PAGE 2: Emergent Sectorial Weights
        fig.clf()
        emergent_sectors = row['Emergent_Sectors']

        fig.text(0.5, 0.9, f"EMERGENT SECTORIAL MAPPING (SOFT CLUSTERING)", fontsize=14, fontweight='bold', ha='center')

        if emergent_sectors and isinstance(emergent_sectors, dict) and sum(emergent_sectors.values()) > 0:
            labels = []
            sizes = []
            for k, v in emergent_sectors.items():
                if v > 0.01: # Only show weights > 1%
                    labels.append(k)
                    sizes.append(v)

            if sizes:
                ax = fig.add_subplot(111)
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
            else:
                fig.text(0.5, 0.5, "Node has zero correlation to standard Sector Seeds.", ha='center')
        else:
            fig.text(0.5, 0.5, "Emergent sector data not available.", ha='center')

        pdf.savefig(fig)
        plt.close('all')

    print(f"Generated {pdf_filename} successfully.")

if __name__ == "__main__":
    generate_ticker_pdf("AAPL")
