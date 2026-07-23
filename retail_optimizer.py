import yfinance as yf
import pandas as pd
import numpy as np
import sys

def calculate_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def test_sma_crossover(data, short_window, long_window):
    df = pd.DataFrame({'Close': data})
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    df['Signal'] = 0.0

    # Fix ChainedAssignmentError by using loc
    mask = slice(short_window, None)
    df.loc[df.index[short_window:], 'Signal'] = np.where(
        df['SMA_short'].iloc[short_window:] > df['SMA_long'].iloc[short_window:], 1.0, 0.0
    )

    df['Position'] = df['Signal'].shift(1)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'] * df['Daily_Return']
    cumulative_return = (1 + df['Strategy_Return'].fillna(0)).prod() - 1
    return cumulative_return

def test_rsi_strategy(data, window, lower_bound, upper_bound):
    df = pd.DataFrame({'Close': data})
    df['RSI'] = calculate_rsi(df['Close'], window)
    df['Signal'] = 0.0
    # Buy when RSI crosses above lower bound, sell when crosses below upper bound
    # Simple logic: stay long when RSI is between lower and upper

    # Let's use a simpler one for optimization: Buy below lower_bound, sell above upper_bound
    position = 0
    positions = []
    for rsi_val in df['RSI']:
        if pd.isna(rsi_val):
            positions.append(0)
            continue

        if rsi_val < lower_bound:
            position = 1
        elif rsi_val > upper_bound:
            position = 0

        positions.append(position)

    df['Position'] = positions
    df['Position'] = df['Position'].shift(1)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'] * df['Daily_Return']
    cumulative_return = (1 + df['Strategy_Return'].fillna(0)).prod() - 1
    return cumulative_return


def run_retail_optimizer(ticker):
    print(f"Fetching historical data for {ticker}...")
    try:
        # Fetching a solid period for testing
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            print(f"No data found for {ticker}")
            return

        close_prices = df['Close'].squeeze()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print(f"\n--- Testing Moving Average Crossovers for {ticker} ---")
    best_sma = {'short': 0, 'long': 0, 'return': -np.inf}

    sma_shorts = [10, 20, 50]
    sma_longs = [50, 100, 200]

    for short_w in sma_shorts:
        for long_w in sma_longs:
            if short_w >= long_w:
                continue
            ret = test_sma_crossover(close_prices, short_w, long_w)
            print(f"SMA {short_w}/{long_w} Return: {ret*100:.2f}%")
            if ret > best_sma['return']:
                best_sma = {'short': short_w, 'long': long_w, 'return': ret}

    print(f"\n[OPTIMAL SMA] Short: {best_sma['short']}, Long: {best_sma['long']} -> Return: {best_sma['return']*100:.2f}%")


    print(f"\n--- Testing RSI Strategies for {ticker} ---")
    best_rsi = {'window': 0, 'lower': 0, 'upper': 0, 'return': -np.inf}

    rsi_windows = [14, 21]
    rsi_lowers = [30, 40]
    rsi_uppers = [60, 70]

    for w in rsi_windows:
        for low in rsi_lowers:
            for high in rsi_uppers:
                ret = test_rsi_strategy(close_prices, w, low, high)
                print(f"RSI (W:{w} L:{low} U:{high}) Return: {ret*100:.2f}%")
                if ret > best_rsi['return']:
                    best_rsi = {'window': w, 'lower': low, 'upper': high, 'return': ret}

    print(f"\n[OPTIMAL RSI] Window: {best_rsi['window']}, Lower: {best_rsi['lower']}, Upper: {best_rsi['upper']} -> Return: {best_rsi['return']*100:.2f}%")

    print("\nBenchmark Buy & Hold Return:", ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100, "%")


if __name__ == "__main__":
    ticker = "SPY"
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    run_retail_optimizer(ticker)
