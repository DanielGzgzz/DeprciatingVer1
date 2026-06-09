import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# A wide array of symbols to satisfy the request
SYMBOLS = [
    # Tech
    "NVDA", "AMD", "INTC", "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    # Commodities
    "CL=F", # Crude Oil
    "GC=F", # Gold
    "SI=F", # Silver
    "HG=F", # Copper
    # Currencies/Forex
    "ILS=X", # USD to ILS
    "EURUSD=X",
    "GBPUSD=X",
    "JPY=X",
    # Indices/ETFs
    "SPY", "QQQ", "DIA", "IWM",
    # Other Sectors
    "JPM", "BAC", "GS", # Financials
    "JNJ", "PFE", "UNH", # Healthcare
    "XOM", "CVX", # Energy
    "TSLA", "F", "GM" # Auto
]

def fetch_data(symbols, period="1y"):
    """
    Fetch historical close prices for the given symbols.
    """
    print(f"Fetching data for {len(symbols)} symbols over period: {period}...")
    data = yf.download(symbols, period=period, progress=False)["Close"]

    # Handle potentially missing data: forward fill, then backward fill
    data = data.ffill().bfill()

    # Drop symbols that have entirely NaN values
    data = data.dropna(axis=1, how='all')
    print(f"Successfully fetched data for {data.shape[1]} symbols.")
    return data

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(data):
    """
    Calculate moving averages and RSI for each symbol to estimate 'real current price'
    and overbought/oversold conditions. Returns a DataFrame of current metrics.
    """
    print("Calculating technical indicators...")
    latest_prices = data.iloc[-1]

    # Simple Moving Averages
    sma_50 = data.rolling(window=50).mean().iloc[-1]
    sma_200 = data.rolling(window=200).mean().iloc[-1]

    # Exponential Moving Averages
    ema_20 = data.ewm(span=20, adjust=False).mean().iloc[-1]

    # RSI
    rsi_14 = data.apply(calculate_rsi, period=14).iloc[-1]

    metrics = pd.DataFrame({
        'Current_Price': latest_prices,
        'SMA_50': sma_50,
        'SMA_200': sma_200,
        'EMA_20': ema_20,
        'RSI_14': rsi_14
    })

    # Mathematical meaning: distance from moving average (estimated real price)
    metrics['Dist_from_SMA_50_pct'] = ((metrics['Current_Price'] - metrics['SMA_50']) / metrics['SMA_50']) * 100
    metrics['Trend'] = np.where(metrics['Current_Price'] > metrics['SMA_50'], "Bullish (Above SMA50)", "Bearish (Below SMA50)")

    # Basic overbought/oversold logic
    conditions = [
        (metrics['RSI_14'] > 70),
        (metrics['RSI_14'] < 30)
    ]
    choices = ['Overbought', 'Oversold']
    metrics['RSI_Signal'] = np.select(conditions, choices, default='Neutral')

    return metrics

def perform_ml_analysis(data):
    """
    Use PCA and KMeans on daily returns to group symbols into 'learned sectors'.
    Also calculates correlation matrix.
    """
    print("Performing ML analysis (clustering and correlation)...")
    returns = data.pct_change().dropna()

    # Calculate Correlation Matrix
    corr_matrix = returns.corr()

    # Transpose so rows are symbols and columns are dates for clustering
    # We want to cluster symbols based on their historical return patterns
    X = returns.T

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use PCA to reduce dimensionality for clustering (e.g. 5 principal components)
    pca = PCA(n_components=min(5, len(X_scaled)))
    X_pca = pca.fit_transform(X_scaled)

    # KMeans Clustering to automatically group into 'sectors'
    # We'll use 5 clusters as a generic starting point
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    learned_sectors = pd.Series(clusters, index=returns.columns, name='Learned_Sector')

    return learned_sectors, corr_matrix

def generate_report(metrics, corr_matrix):
    """
    Generate a text-based analytical report.
    """
    print("\n" + "="*60)
    print("                 MARKET ANALYSIS REPORT")
    print("="*60)

    print("\n1. SECTOR CLUSTERING (Machine Learning Derived)")
    print("-" * 50)
    for cluster_id in sorted(metrics['Learned_Sector'].unique()):
        cluster_symbols = metrics[metrics['Learned_Sector'] == cluster_id].index.tolist()
        print(f"Cluster {cluster_id}: {', '.join(cluster_symbols)}")

    print("\n2. HIGH CORRELATION INFLUENCES")
    print("-" * 50)
    # Check influence of some key symbols if they exist
    key_symbols = ["NVDA", "CL=F", "ILS=X"]
    for sym in key_symbols:
        if sym in corr_matrix.columns:
            corrs = corr_matrix[sym].sort_values(ascending=False)
            # Exclude self
            top_pos = corrs[1:4]
            top_neg = corrs.tail(3)
            print(f"If {sym} moves, watch:")
            print(f"  Highly Positively Correlated: {', '.join([f'{k} ({v:.2f})' for k, v in top_pos.items()])}")
            print(f"  Highly Negatively Correlated: {', '.join([f'{k} ({v:.2f})' for k, v in top_neg.items()])}")
        else:
            print(f"Could not analyze influence for {sym} (data unavailable).")

    print("\n3. PRICE ESTIMATION & TRENDS")
    print("-" * 50)
    # Highlight strong trends
    strong_bulls = metrics[(metrics['Dist_from_SMA_50_pct'] > 5) & (metrics['RSI_14'] < 70)]
    strong_bears = metrics[(metrics['Dist_from_SMA_50_pct'] < -5) & (metrics['RSI_14'] > 30)]

    print(f"Strong Bullish Trend (Price > 5% above SMA50, Not Overbought):")
    if not strong_bulls.empty:
        for idx, row in strong_bulls.iterrows():
            print(f"  {idx}: Current={row['Current_Price']:.2f}, SMA50={row['SMA_50']:.2f} (+{row['Dist_from_SMA_50_pct']:.1f}%)")
    else:
        print("  None")

    print(f"\nStrong Bearish Trend (Price > 5% below SMA50, Not Oversold):")
    if not strong_bears.empty:
        for idx, row in strong_bears.iterrows():
            print(f"  {idx}: Current={row['Current_Price']:.2f}, SMA50={row['SMA_50']:.2f} ({row['Dist_from_SMA_50_pct']:.1f}%)")
    else:
        print("  None")

    print("\n4. OVERBOUGHT/OVERSOLD ALERTS")
    print("-" * 50)
    overbought = metrics[metrics['RSI_Signal'] == 'Overbought'].index.tolist()
    oversold = metrics[metrics['RSI_Signal'] == 'Oversold'].index.tolist()

    print(f"Overbought (RSI > 70): {', '.join(overbought) if overbought else 'None'}")
    print(f"Oversold (RSI < 30): {', '.join(oversold) if oversold else 'None'}")

    print("="*60 + "\n")

if __name__ == "__main__":
    df = fetch_data(SYMBOLS)
    metrics = calculate_indicators(df)
    sectors, corr = perform_ml_analysis(df)
    metrics = metrics.join(sectors)
    generate_report(metrics, corr)
