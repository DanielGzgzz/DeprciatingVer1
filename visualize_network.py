import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
from market_analyzer import SYMBOLS

# We will use a targeted subset to make the graph readable, focusing on key sectors:
# Tech, Energy, Defense, Safe Havens, Financials
TARGET_NODES = [
    "SPY", "QQQ", "TLT", "GLD",
    "AAPL", "MSFT", "NVDA",
    "XOM", "CVX", "CL=F",
    "LMT", "RTX", "NOC",
    "JPM", "GS", "BAC"
]

EVENTS = {
    "COVID-19 Crash (Feb-Mar 2020)": ("2020-01-01", "2020-02-15", "2020-02-20", "2020-04-15"),
    "Russia-Ukraine War (Feb 2022)": ("2021-12-01", "2022-02-15", "2022-02-20", "2022-04-15")
}

def fetch_data_for_event(pre_start, post_end):
    print(f"Fetching data from {pre_start} to {post_end}...")
    data = yf.download(TARGET_NODES, start=pre_start, end=post_end, progress=False)
    close_df = data["Close"].ffill().bfill().dropna(axis=1, how='all')
    vol_df = data["Volume"].fillna(0).dropna(axis=1, how='all')
    return close_df, vol_df

def compute_local_tensor(close_df, vol_df):
    returns = close_df.pct_change().dropna()
    corr_matrix = returns.corr().fillna(0)

    # Simple volume derivative for the window
    vol_mean = vol_df.mean()
    vol_current = vol_df.iloc[-min(10, len(vol_df)):]
    V = (vol_current.mean() / vol_mean).fillna(1.0).values

    volume_friction = np.outer(V, V)
    wealth_flow_matrix = corr_matrix.values * volume_friction
    np.fill_diagonal(wealth_flow_matrix, 0)

    return pd.DataFrame(wealth_flow_matrix, index=corr_matrix.index, columns=corr_matrix.columns)

def draw_network(ax, flow_matrix, title):
    G = nx.Graph()

    # Add nodes
    for node in flow_matrix.columns:
        G.add_node(node)

    # Add strong edges
    # We only plot edges that represent significant structural connection (top 15%)
    threshold = np.percentile(np.abs(flow_matrix.values[flow_matrix.values != 0]), 85)

    for i in flow_matrix.columns:
        for j in flow_matrix.columns:
            if i != j:
                weight = flow_matrix.loc[i, j]
                if abs(weight) > threshold:
                    # Color red if negative correlation (capital flight from i to j), green if positive sync
                    color = 'red' if weight < 0 else 'green'
                    G.add_edge(i, j, weight=abs(weight)*5, color=color)

    # Position nodes using spring layout based on connectivity
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=700, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')

    # Draw edges
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]

    if edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=colors, width=weights, alpha=0.6)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

if __name__ == "__main__":
    print("Generating Event-Based Topological Network Visualizations...")

    with PdfPages('event_topology_maps.pdf') as pdf:
        for event_name, (pre_start, pre_end, post_start, post_end) in EVENTS.items():
            print(f"\nProcessing {event_name}...")
            close_df, vol_df = fetch_data_for_event(pre_start, post_end)

            # Split data
            pre_close = close_df.loc[pre_start:pre_end]
            pre_vol = vol_df.loc[pre_start:pre_end]

            post_close = close_df.loc[post_start:post_end]
            post_vol = vol_df.loc[post_start:post_end]

            # Compute tensors
            tensor_pre = compute_local_tensor(pre_close, pre_vol)
            tensor_post = compute_local_tensor(post_close, post_vol)

            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Thermodynamic Capital Migration: {event_name}", fontsize=16, fontweight='bold', y=0.98)

            draw_network(ax1, tensor_pre, f"Pre-Event Topology\n({pre_start} to {pre_end})")
            draw_network(ax2, tensor_post, f"Post-Event Topology (Shock Migration)\n({post_start} to {post_end})")

            # Add a legend explanation
            fig.text(0.5, 0.02, "Green Edges = Synchronized Capital Growth | Red Edges = Inverse Migration (Safe Haven Flight)", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print("\nVisualization complete. Output saved to 'event_topology_maps.pdf'.")
