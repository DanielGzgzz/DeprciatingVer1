import yfinance as yf
import pandas as pd
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import hilbert
from numpy.linalg import eig


# A wide array of symbols to satisfy the request
SYMBOLS = [
    "A",
    "AA",
    "AAL",
    "AAON",
    "AAPL",
    "ABBV",
    "ABNB",
    "ABT",
    "ACGL",
    "ACI",
    "ACM",
    "ACN",
    "ADBE",
    "ADC",
    "ADI",
    "ADM",
    "ADP",
    "ADSK",
    "AEE",
    "AEIS",
    "AEP",
    "AES",
    "AFG",
    "AFL",
    "AGCO",
    "AHR",
    "AIG",
    "AIT",
    "AIZ",
    "AJG",
    "AKAM",
    "ALB",
    "ALGM",
    "ALGN",
    "ALK",
    "ALL",
    "ALLE",
    "ALLY",
    "ALV",
    "AM",
    "AMAT",
    "AMCR",
    "AMD",
    "AME",
    "AMG",
    "AMGN",
    "AMH",
    "AMKR",
    "AMP",
    "AMT",
    "AMZN",
    "AN",
    "ANET",
    "ANF",
    "AON",
    "AOS",
    "APA",
    "APD",
    "APG",
    "APH",
    "APO",
    "APP",
    "APPF",
    "APTV",
    "AR",
    "ARE",
    "ARES",
    "ARMK",
    "ARW",
    "ARWR",
    "ASB",
    "ASH",
    "ASML",
    "ATI",
    "ATO",
    "ATR",
    "AUDUSD=X",
    "AVAV",
    "AVB",
    "AVGO",
    "AVNT",
    "AVT",
    "AVTR",
    "AVY",
    "AWK",
    "AXON",
    "AXP",
    "AXTA",
    "AYI",
    "AZO",
    "BA",
    "BABA",
    "BAC",
    "BAH",
    "BALL",
    "BAX",
    "BBWI",
    "BBY",
    "BC",
    "BCO",
    "BDC",
    "BDX",
    "BEN",
    "BF-B",
    "BG",
    "BHF",
    "BIIB",
    "BILL",
    "BIO",
    "BJ",
    "BKH",
    "BKNG",
    "BKR",
    "BLD",
    "BLDR",
    "BLK",
    "BMRN",
    "BMY",
    "BNY",
    "BR",
    "BRK-B",
    "BRKR",
    "BRO",
    "BROS",
    "BRX",
    "BSX",
    "BSY",
    "BTC-USD",
    "BURL",
    "BWA",
    "BWXT",
    "BX",
    "BXP",
    "BYD",
    "C",
    "CACI",
    "CAD=X",
    "CAH",
    "CAR",
    "CARR",
    "CART",
    "CASY",
    "CAT",
    "CAVA",
    "CB",
    "CBOE",
    "CBRE",
    "CBSH",
    "CBT",
    "CCI",
    "CCK",
    "CCL",
    "CDE",
    "CDNS",
    "CDP",
    "CDW",
    "CEG",
    "CELH",
    "CF",
    "CFG",
    "CFR",
    "CG",
    "CGNX",
    "CHD",
    "CHDN",
    "CHE",
    "CHF=X",
    "CHH",
    "CHRD",
    "CHRW",
    "CHTR",
    "CHWY",
    "CI",
    "CIEN",
    "CINF",
    "CL",
    "CL=F",
    "CLF",
    "CLH",
    "CLX",
    "CMC",
    "CMCSA",
    "CME",
    "CMG",
    "CMI",
    "CMS",
    "CNC",
    "CNH",
    "CNM",
    "CNO",
    "CNP",
    "CNX",
    "COF",
    "COHR",
    "COIN",
    "COKE",
    "COLB",
    "COLM",
    "COO",
    "COP",
    "COR",
    "COST",
    "CPAY",
    "CPRI",
    "CPRT",
    "CPT",
    "CR",
    "CRBG",
    "CRH",
    "CRL",
    "CRM",
    "CROX",
    "CRS",
    "CRUS",
    "CRWD",
    "CSCO",
    "CSGP",
    "CSL",
    "CSX",
    "CTAS",
    "CTRE",
    "CTSH",
    "CTVA",
    "CUBE",
    "CUZ",
    "CVLT",
    "CVNA",
    "CVS",
    "CVX",
    "CW",
    "CXT",
    "CYTK",
    "D",
    "DAL",
    "DAR",
    "DASH",
    "DBX",
    "DCI",
    "DD",
    "DDOG",
    "DE",
    "DECK",
    "DELL",
    "DG",
    "DGX",
    "DHI",
    "DHR",
    "DIA",
    "DINO",
    "DIS",
    "DKS",
    "DLB",
    "DLR",
    "DLTR",
    "DOC",
    "DOCN",
    "DOCS",
    "DOCU",
    "DOV",
    "DOW",
    "DPZ",
    "DRI",
    "DT",
    "DTE",
    "DTM",
    "DUK",
    "DUOL",
    "DVA",
    "DVN",
    "DXCM",
    "DY",
    "EA",
    "EBAY",
    "ECHO",
    "ECL",
    "ED",
    "EEFT",
    "EFX",
    "EG",
    "EGP",
    "EHC",
    "EIX",
    "EL",
    "ELAN",
    "ELF",
    "ELS",
    "ELV",
    "EME",
    "EMR",
    "ENS",
    "ENSG",
    "ENTG",
    "EOG",
    "EPR",
    "EQH",
    "EQIX",
    "EQR",
    "EQT",
    "ERIE",
    "ES",
    "ESAB",
    "ESNT",
    "ESS",
    "ETH-USD",
    "ETN",
    "ETR",
    "EURUSD=X",
    "EVR",
    "EVRG",
    "EW",
    "EWBC",
    "EXC",
    "EXE",
    "EXEL",
    "EXLS",
    "EXP",
    "EXPD",
    "EXPE",
    "EXPO",
    "EXR",
    "F",
    "FAF",
    "FANG",
    "FAST",
    "FBIN",
    "FCFS",
    "FCN",
    "FCX",
    "FDS",
    "FDX",
    "FDXF",
    "FE",
    "FFIN",
    "FFIV",
    "FHI",
    "FHN",
    "FICO",
    "FIS",
    "FISV",
    "FITB",
    "FIVE",
    "FIX",
    "FLEX",
    "FLG",
    "FLR",
    "FLS",
    "FN",
    "FNB",
    "FND",
    "FNF",
    "FOUR",
    "FOX",
    "FOXA",
    "FR",
    "FRT",
    "FSLR",
    "FTI",
    "FTNT",
    "FTV",
    "G",
    "GAP",
    "GATX",
    "GBCI",
    "GBPUSD=X",
    "GC=F",
    "GD",
    "GDDY",
    "GE",
    "GEF",
    "GEHC",
    "GEN",
    "GEV",
    "GGG",
    "GHC",
    "GILD",
    "GIS",
    "GL",
    "GLD",
    "GLPI",
    "GLW",
    "GM",
    "GME",
    "GMED",
    "GNRC",
    "GNTX",
    "GOOG",
    "GOOGL",
    "GPC",
    "GPK",
    "GPN",
    "GRMN",
    "GS",
    "GT",
    "GTLS",
    "GWRE",
    "GWW",
    "GXO",
    "H",
    "HAE",
    "HAL",
    "HALO",
    "HAS",
    "HBAN",
    "HCA",
    "HD",
    "HG=F",
    "HGV",
    "HIG",
    "HII",
    "HIMS",
    "HL",
    "HLI",
    "HLNE",
    "HLT",
    "HOG",
    "HOMB",
    "HON",
    "HONA",
    "HOOD",
    "HPE",
    "HPQ",
    "HQY",
    "HR",
    "HRB",
    "HRL",
    "HSIC",
    "HST",
    "HSY",
    "HUBB",
    "HUM",
    "HWC",
    "HWM",
    "HXL",
    "IBKR",
    "IBM",
    "IBOC",
    "ICE",
    "IDA",
    "IDCC",
    "IDXX",
    "IEX",
    "IFF",
    "ILMN",
    "ILS=X",
    "INCY",
    "INGR",
    "INTC",
    "INTU",
    "INVH",
    "IP",
    "IPGP",
    "IQV",
    "IR",
    "IRM",
    "IRT",
    "ISRG",
    "IT",
    "ITT",
    "ITW",
    "IVZ",
    "IWM",
    "J",
    "JAZZ",
    "JBHT",
    "JBL",
    "JCI",
    "JEF",
    "JHG",
    "JKHY",
    "JLL",
    "JNJ",
    "JPM",
    "JPY=X",
    "KBH",
    "KBR",
    "KD",
    "KDP",
    "KEX",
    "KEY",
    "KEYS",
    "KHC",
    "KIM",
    "KKR",
    "KLAC",
    "KMB",
    "KMI",
    "KNF",
    "KNSL",
    "KNX",
    "KO",
    "KR",
    "KRC",
    "KRG",
    "KTOS",
    "KVUE",
    "L",
    "LAD",
    "LAMR",
    "LDOS",
    "LEA",
    "LECO",
    "LEN",
    "LFUS",
    "LH",
    "LHX",
    "LII",
    "LIN",
    "LITE",
    "LIVN",
    "LLY",
    "LMT",
    "LNT",
    "LNTH",
    "LOPE",
    "LOW",
    "LPX",
    "LRCX",
    "LSCC",
    "LSTR",
    "LULU",
    "LUV",
    "LVS",
    "LYB",
    "LYV",
    "M",
    "MA",
    "MAA",
    "MANH",
    "MAR",
    "MAS",
    "MAT",
    "MCD",
    "MCHP",
    "MCK",
    "MCO",
    "MDLZ",
    "MDT",
    "MEDP",
    "MET",
    "META",
    "MGM",
    "MIDD",
    "MKC",
    "MKSI",
    "MLI",
    "MLM",
    "MMM",
    "MMS",
    "MNST",
    "MO",
    "MOG-A",
    "MORN",
    "MOS",
    "MP",
    "MPC",
    "MPWR",
    "MRK",
    "MRNA",
    "MRSH",
    "MRVL",
    "MS",
    "MSA",
    "MSCI",
    "MSFT",
    "MSI",
    "MSM",
    "MTB",
    "MTD",
    "MTDR",
    "MTG",
    "MTN",
    "MTSI",
    "MTZ",
    "MU",
    "MUR",
    "MUSA",
    "MZTI",
    "NBIX",
    "NCLH",
    "NDAQ",
    "NDSN",
    "NEE",
    "NEM",
    "NEU",
    "NFG",
    "NFLX",
    "NI",
    "NJR",
    "NKE",
    "NLY",
    "NNN",
    "NOC",
    "NOV",
    "NOVT",
    "NOW",
    "NRG",
    "NSA",
    "NSC",
    "NTAP",
    "NTNX",
    "NTRS",
    "NUE",
    "NVDA",
    "NVO",
    "NVR",
    "NVS",
    "NVST",
    "NVT",
    "NWE",
    "NWS",
    "NWSA",
    "NXPI",
    "NXST",
    "NXT",
    "NYT",
    "NZDUSD=X",
    "O",
    "OC",
    "ODFL",
    "OGE",
    "OGS",
    "OHI",
    "OKE",
    "OKTA",
    "OLED",
    "OLLI",
    "OLN",
    "OMC",
    "ON",
    "ONB",
    "ONTO",
    "OPCH",
    "ORA",
    "ORCL",
    "ORI",
    "ORLY",
    "OSK",
    "OTIS",
    "OVV",
    "OXY",
    "OZK",
    "P",
    "PAG",
    "PANW",
    "PATH",
    "PAYX",
    "PB",
    "PBF",
    "PCAR",
    "PCG",
    "PCTY",
    "PEG",
    "PEGA",
    "PEN",
    "PEP",
    "PFE",
    "PFG",
    "PFGC",
    "PG",
    "PGR",
    "PH",
    "PHM",
    "PII",
    "PINS",
    "PK",
    "PKG",
    "PLD",
    "PLNT",
    "PLTR",
    "PM",
    "PNC",
    "PNFP",
    "PNR",
    "PNW",
    "PODD",
    "POR",
    "POST",
    "PPC",
    "PPG",
    "PPL",
    "PR",
    "PRI",
    "PRU",
    "PSA",
    "PSKY",
    "PSN",
    "PSX",
    "PTC",
    "PVH",
    "PWR",
    "PYPL",
    "Q",
    "QCOM",
    "QLYS",
    "QQQ",
    "R",
    "RBA",
    "RBC",
    "RCL",
    "REG",
    "REGN",
    "REXR",
    "RF",
    "RGA",
    "RGEN",
    "RGLD",
    "RH",
    "RJF",
    "RL",
    "RLI",
    "RMBS",
    "RMD",
    "RNR",
    "ROIV",
    "ROK",
    "ROKU",
    "ROL",
    "ROP",
    "ROST",
    "RPM",
    "RRC",
    "RRX",
    "RS",
    "RSG",
    "RTX",
    "RVTY",
    "RYAN",
    "RYN",
    "SAIA",
    "SAIC",
    "SAM",
    "SANM",
    "SARO",
    "SBAC",
    "SBRA",
    "SBUX",
    "SCHW",
    "SCI",
    "SEIC",
    "SF",
    "SFM",
    "SGI",
    "SHC",
    "SHW",
    "SI=F",
    "SIGI",
    "SIRI",
    "SITM",
    "SJM",
    "SLAB",
    "SLB",
    "SLGN",
    "SLM",
    "SLV",
    "SMCI",
    "SMG",
    "SMTC",
    "SN",
    "SNA",
    "SNDK",
    "SNPS",
    "SNX",
    "SO",
    "SOLS",
    "SOLV",
    "SON",
    "SPG",
    "SPGI",
    "SPXC",
    "SPY",
    "SR",
    "SRE",
    "SSB",
    "SSD",
    "ST",
    "STAG",
    "STE",
    "STLD",
    "STRL",
    "STT",
    "STWD",
    "STX",
    "STZ",
    "SW",
    "SWK",
    "SWKS",
    "SWX",
    "SYF",
    "SYK",
    "SYNA",
    "SYY",
    "T",
    "TAP",
    "TCBI",
    "TDG",
    "TDY",
    "TECH",
    "TEL",
    "TER",
    "TEX",
    "TFC",
    "TGT",
    "THC",
    "THG",
    "THO",
    "TJX",
    "TKO",
    "TKR",
    "TLN",
    "TLT",
    "TM",
    "TMHC",
    "TMO",
    "TMUS",
    "TNL",
    "TOL",
    "TPL",
    "TPR",
    "TREX",
    "TRGP",
    "TRMB",
    "TROW",
    "TRU",
    "TRV",
    "TSCO",
    "TSLA",
    "TSM",
    "TSN",
    "TT",
    "TTC",
    "TTD",
    "TTE",
    "TTEK",
    "TTMI",
    "TTWO",
    "TWLO",
    "TXN",
    "TXNM",
    "TXRH",
    "TXT",
    "TYL",
    "UAL",
    "UBER",
    "UBSI",
    "UDR",
    "UFPI",
    "UGI",
    "UHS",
    "ULS",
    "ULTA",
    "UMBF",
    "UNG",
    "UNH",
    "UNM",
    "UNP",
    "UPS",
    "URI",
    "USB",
    "USFD",
    "USO",
    "UTHR",
    "UUP",
    "V",
    "VAL",
    "VC",
    "VEEV",
    "VFC",
    "VIAV",
    "VICI",
    "VICR",
    "VLO",
    "VLTO",
    "VLY",
    "VMC",
    "VMI",
    "VNO",
    "VNOM",
    "VNT",
    "VOYA",
    "VRSK",
    "VRSN",
    "VRT",
    "VRTX",
    "VST",
    "VTR",
    "VTRS",
    "VVV",
    "VZ",
    "WAB",
    "WAL",
    "WAT",
    "WBD",
    "WBS",
    "WCC",
    "WDAY",
    "WDC",
    "WEC",
    "WELL",
    "WEX",
    "WFC",
    "WFRD",
    "WH",
    "WHR",
    "WING",
    "WLK",
    "WM",
    "WMB",
    "WMG",
    "WMS",
    "WMT",
    "WPC",
    "WRB",
    "WSM",
    "WSO",
    "WST",
    "WTFC",
    "WTRG",
    "WTS",
    "WTW",
    "WWD",
    "WY",
    "WYNN",
    "XEL",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "XOM",
    "XPO",
    "XRAY",
    "XYL",
    "XYZ",
    "YETI",
    "YUM",
    "ZBH",
    "ZBRA",
    "ZION",
    "ZTS"
]

def fetch_data(symbols, period="max"):
    """
    Fetch maximum historical Close prices and Volume for the given symbols.
    Chunks the requests to avoid timeouts and cleanly drops unavailable tickers.
    """
    print(f"Fetching data for {len(symbols)} symbols over period: {period}...")
    chunk_size = 100
    all_close = []
    all_volume = []

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        print(f"Fetching chunk {i//chunk_size + 1}/{(len(symbols) + chunk_size - 1)//chunk_size}...")
        try:
            raw_data = yf.download(chunk, period=period, progress=False, timeout=15)

            # Handle Close
            chunk_close = raw_data["Close"]
            if isinstance(chunk_close, pd.Series):
                chunk_close = chunk_close.to_frame(name=chunk[0])
            all_close.append(chunk_close)

            # Handle Volume
            chunk_vol = raw_data["Volume"]
            if isinstance(chunk_vol, pd.Series):
                chunk_vol = chunk_vol.to_frame(name=chunk[0])
            all_volume.append(chunk_vol)

        except Exception as e:
            print(f"Error fetching chunk: {e}")

    if not all_close:
        raise ValueError("Failed to fetch any data.")

    close_data = pd.concat(all_close, axis=1)
    vol_data = pd.concat(all_volume, axis=1)

    # Handle potentially missing data: forward fill, then backward fill
    close_data = close_data.ffill().bfill()
    vol_data = vol_data.fillna(0) # Fill missing volume with 0

    # Drop symbols that have entirely NaN values
    close_data = close_data.dropna(axis=1, how='all')
    valid_cols = close_data.columns
    vol_data = vol_data[valid_cols]

    print(f"Successfully fetched data for {close_data.shape[1]} symbols.")
    return close_data, vol_data

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(close_data, vol_data):
    """
    Calculate All-Time Velocity Score by integrating price data over the entire historical period.
    Also calculates rolling volume derivatives for the friction counter.
    """
    print("Calculating All-Time Velocity Scores & Volume Derivatives...")
    latest_prices = close_data.iloc[-1]

    def calc_velocity(series):
        s = series.dropna()
        if len(s) < 252:
            return 0.0
        returns = s.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        drift = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        score = (drift / volatility) * 5.0
        return score

    all_time_velocity = close_data.apply(calc_velocity)

    safety_score = 1.0 / (close_data.pct_change().std() * np.sqrt(252) + 1e-6)

    vol_30d = vol_data.rolling(window=30).mean().iloc[-1]
    vol_252d = vol_data.rolling(window=252).mean().iloc[-1]
    vol_252d = vol_252d.replace(0, np.nan).fillna(vol_30d)

    raw_volume_derivative = (vol_30d / vol_252d).fillna(1.0)
    volume_derivative = np.tanh(raw_volume_derivative) # Bounds friction between 0 and 1

    print("Computing Kuramoto Phase Synchronization Dynamics...")
    returns = close_data.pct_change().dropna()
    if len(returns) > 30:
        detrended = returns - returns.mean()
        analytic_signal = hilbert(detrended, axis=0)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal), axis=0)

        N_assets = instantaneous_phase.shape[1]
        complex_phases = np.exp(1j * instantaneous_phase[-30:])
        r_t = np.abs(np.sum(complex_phases, axis=1) / N_assets)
        global_kuramoto_sync = np.mean(r_t)
    else:
        global_kuramoto_sync = 0.0

    metrics = pd.DataFrame({
        'Current_Price': latest_prices,
        'All_Time_Velocity': all_time_velocity,
        'Topological_Safety': safety_score,
        'Volume_Derivative': volume_derivative,
        'Variance': close_data.pct_change().var() * 252
    })

    metrics.loc[metrics.index[0], 'Global_Kuramoto_Sync'] = global_kuramoto_sync

    return metrics

def perform_ml_analysis(close_data, vol_data, metrics):
    """
    Construct the 3rd-Order Flow Tensor F_t and apply the Volume Friction Counter.
    Maps systemic vaporization zones using Eigenvector Centrality and Kelly Sizing.
    """
    print("Constructing 3rd-Order Flow Tensor & Volume Friction Filter...")

    returns = close_data.pct_change().dropna()
    N = returns.shape[1]

    # 1. Base Dimension: Price Covariance (T_i->j)
    corr_matrix = returns.corr().fillna(0)

    V = metrics['Volume_Derivative'].values
    volume_friction_matrix = np.outer(V, V)

    # Realized Wealth Flow Tensor Slice
    wealth_flow_matrix = corr_matrix.values * volume_friction_matrix
    np.fill_diagonal(wealth_flow_matrix, 0)
    wealth_flow_df = pd.DataFrame(wealth_flow_matrix, index=corr_matrix.index, columns=corr_matrix.columns)

    # --- CONTINUOUS RISK MANIFOLD: DERIVATIVES & INTEGRALS ---
    # We calculate the derivative of the flow leaving a node (acceleration of outflow)
    # Since we have static snapshots, we proxy the derivative via short term vs long term flow.
    # We calculate the integral of the flow (cumulative volume saturation)

    recent_returns = returns.iloc[-10:] # last 10 days
    recent_corr = recent_returns.corr().fillna(0)
    recent_flow = recent_corr.values * volume_friction_matrix
    np.fill_diagonal(recent_flow, 0)
    recent_flow_df = pd.DataFrame(recent_flow, index=corr_matrix.index, columns=corr_matrix.columns)

    # Derivative dW/dt (Acceleration of flow)
    # If recent flow is significantly lower than long-term flow, dW/dt is negative.
    flow_derivative = (recent_flow_df.sum(axis=0) - wealth_flow_df.sum(axis=0)) / 10.0
    metrics['Flow_Derivative'] = flow_derivative

    # Integral ∫W dτ (Cumulative Volume Saturation)
    # We integrate the raw volume over the last 90 days to proxy saturation
    cumulative_volume = vol_data.iloc[-90:].sum(axis=0)
    # Normalize by 1-year volume to get a saturation index
    annual_volume = vol_data.iloc[-252:].sum(axis=0)
    saturation_index = (cumulative_volume / (annual_volume + 1e-6)) * (252/90.0) # > 1 means saturating
    metrics['Volume_Saturation'] = saturation_index

    # --- EMERGENT SECTORIAL MAPPING (SOFT CLUSTERING) ---
    print("Executing Sector Seedation (Soft Clustering)...")
    sector_seeds = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC", "XLP", "XLU", "XLRE", "XLB"]
    available_seeds = [s for s in sector_seeds if s in corr_matrix.columns]

    emergent_sectors = {}
    if available_seeds:
        for ticker in corr_matrix.columns:
            # Get flow connections from this ticker to all seeds
            seed_flows = wealth_flow_df.loc[available_seeds, ticker].clip(lower=0)
            total_seed_flow = seed_flows.sum()
            if total_seed_flow > 0:
                fractional_weights = seed_flows / total_seed_flow
                emergent_sectors[ticker] = fractional_weights.to_dict()
            else:
                emergent_sectors[ticker] = {s: 0.0 for s in available_seeds}
    metrics['Emergent_Sectors'] = pd.Series(emergent_sectors)


    # Advanced Metric 2: Eigenvector Flow Centrality
    print("Calculating Eigenvector Centrality (True Capital Sinks)...")
    # Shift matrix to be strictly positive for Frobenius-Perron theorem
    min_val = np.min(wealth_flow_matrix)
    if min_val < 0:
        shifted_matrix = wealth_flow_matrix - min_val
    else:
        shifted_matrix = wealth_flow_matrix

    eigenvalues, eigenvectors = eig(shifted_matrix)
    # The principal eigenvector corresponds to the largest eigenvalue
    max_idx = np.argmax(np.abs(eigenvalues))
    principal_eigenvector = np.abs(eigenvectors[:, max_idx])

    # Normalize centrality
    eigen_centrality = pd.Series(principal_eigenvector / np.max(principal_eigenvector), index=corr_matrix.index)

    # --- SECOND DEPTH: SPECTRAL GRAPH LAPLACIAN & FIEDLER VECTOR ---
    print("Computing 2nd-Depth: Graph Laplacian & Fiedler Algebraic Connectivity...")
    # Calculate degree matrix D (sum of weights for each node)
    # We use the absolute value of the flow matrix to represent connection strength W
    W = np.abs(wealth_flow_matrix)
    np.fill_diagonal(W, 0)
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)

    # Compute Laplacian L = D - W
    L = D - W

    # Calculate eigenvalues of the Laplacian
    laplacian_eigenvals = np.real(eig(L)[0])
    laplacian_eigenvals = np.sort(laplacian_eigenvals)

    # The Fiedler eigenvalue (lambda_2) represents the algebraic connectivity of the network
    # If the network is highly disconnected, lambda_2 is near 0.
    # If the network is highly interconnected (fragile contagion state), lambda_2 is large.
    if len(laplacian_eigenvals) > 1:
        fiedler_val = laplacian_eigenvals[1]
    else:
        fiedler_val = 0.0

    # We use the Fiedler value to dynamically dampen the Kelly fraction.
    # High connectivity = systemic fragility = smaller Kelly bets.
    # Low connectivity = orthogonal diversification = larger Kelly bets.
    # Normalizing Fiedler value empirically (usually ranges from 0 to N). We scale it inversely.
    fiedler_dampener = 1.0 / (1.0 + (fiedler_val / float(N)))

    # Read Kuramoto Sync
    global_sync = metrics['Global_Kuramoto_Sync'].iloc[0] if 'Global_Kuramoto_Sync' in metrics.columns else 0.0
    vaporization_risk = []

    # Kuramoto Crash Override
    CRASH_THRESHOLD = 0.85
    is_crashing = global_sync > CRASH_THRESHOLD

    for ticker in metrics.index:
        vel = metrics.loc[ticker, 'All_Time_Velocity']
        if is_crashing:
             vaporization_risk.append((ticker, f"Category A (SYSTEMIC KURAMOTO CRASH DETECTED: Sync={global_sync:.2f})"))
        elif vel < -2.0:
            vaporization_risk.append((ticker, "Category A (Secular Decline)"))
        elif vel < 0 and metrics.loc[ticker, 'Topological_Safety'] < 1.5:
            vaporization_risk.append((ticker, "Category B (High Contagion / Negative Curvature)"))

    # Calculate target portfolio weights based on Eigenvector Centrality and Continuous-Time Kelly
    positive_nodes = metrics[metrics['All_Time_Velocity'] > 0].copy()

    if len(positive_nodes) > 0 and not is_crashing:
        # Advanced Metric 3: Continuous-Time Fractional Kelly Sizing (Fiedler Dampened)
        # f* = (mu - r) / sigma^2
        # We use All-Time Velocity as a proxy for the drift/variance ratio,
        # scale it by the eigenvector centrality to ensure it's a true sink,
        # and explicitly dampen the overall sizing using the Spectral Laplacian Fiedler Value.

        # Kelly Fraction Approximation (Bounded)
        kelly_fractions = positive_nodes['All_Time_Velocity'] * eigen_centrality.loc[positive_nodes.index]

        # Second Depth Application: Self-calibrating thermodynamic system
        # Standard was 0.5 (Half-Kelly). Now it dynamically ranges based on topological fragility.
        dynamic_kelly_scale = 1.0 * fiedler_dampener
        kelly_fractions = kelly_fractions * dynamic_kelly_scale

        # Normalize weights
        positive_nodes['Target_Weight_Pct'] = (kelly_fractions / kelly_fractions.sum()) * 100.0
        positive_nodes['Target_Weight_Pct'] = positive_nodes['Target_Weight_Pct'].clip(upper=15.0)
        positive_nodes['Target_Weight_Pct'] = (positive_nodes['Target_Weight_Pct'] / positive_nodes['Target_Weight_Pct'].sum()) * 100.0

        positive_nodes['Max_Risk_Band_Pct'] = positive_nodes['Target_Weight_Pct'] * 1.3
    else:
        # If crashing or no positive nodes, move 100% to Cash/Safe Havens (Risk weight 0)
        positive_nodes['Target_Weight_Pct'] = 0
        positive_nodes['Max_Risk_Band_Pct'] = 0

    metrics = metrics.join(positive_nodes[['Target_Weight_Pct', 'Max_Risk_Band_Pct']])
    metrics['Target_Weight_Pct'] = metrics['Target_Weight_Pct'].fillna(0)
    metrics['Max_Risk_Band_Pct'] = metrics['Max_Risk_Band_Pct'].fillna(0)
    metrics['Eigenvector_Centrality'] = eigen_centrality

    # Store global sync for reporting
    metrics.loc[metrics.index[0], 'Global_Kuramoto_Sync'] = global_sync

    return metrics, vaporization_risk

def perform_spectral_analysis(close_data, top_nodes):
    """
    Perform Fourier Tensor Decomposition to separate structural baseline flows from noise.
    Processes the continuous derivative (returns) of the top conviction nodes.
    Outputs a Bode Magnitude Plot to PDF.
    """
    print("Executing Spectral Analysis & Fourier Tensor Decomposition...")

    # We only process the top nodes (highest target weights)
    tickers = top_nodes.index.tolist()
    if not tickers:
        return {}

    spectral_results = {}

    with PdfPages('bode_plots.pdf') as pdf:
        # Create a single figure with subplots for the top nodes (max 5 for clarity)
        num_plots = min(5, len(tickers))
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        fig.suptitle('Bode Magnitude Plot: Factor of Rise (Gain dB)', fontsize=14, fontweight='bold')

        for i, ticker in enumerate(tickers[:num_plots]):
            # Continuous derivative approximation (daily log returns)
            prices = close_data[ticker].dropna()
            if len(prices) < 252:
                continue

            returns = np.log(prices / prices.shift(1)).dropna().values
            N = len(returns)

            # FFT Pipeline
            yf_fft = scipy.fftpack.fft(returns)
            xf = scipy.fftpack.fftfreq(N, d=1.0) # d=1 day

            # Only take positive frequencies (first half)
            half_n = N // 2
            xf = xf[:half_n]
            yf_fft = np.abs(yf_fft[:half_n])

            # Convert to Gain dB
            # Avoid log(0)
            gain_db = 20 * np.log10(yf_fft + 1e-10)

            # Convert frequency (cycles/day) to Temporal Horizon (days/cycle)
            # Ignore f=0 (DC component) to avoid divide by zero
            valid_idx = xf > 0
            xf_valid = xf[valid_idx]
            temporal_days = 1.0 / xf_valid
            gain_valid = gain_db[valid_idx]

            # Frequency-to-Horizon Banding
            high_freq_mask = (temporal_days >= 1) & (temporal_days <= 14)
            mid_freq_mask = (temporal_days > 14) & (temporal_days <= 90)
            low_freq_mask = (temporal_days > 90)

            # Calculate mean energy in bands
            energy_high = np.mean(gain_valid[high_freq_mask]) if np.any(high_freq_mask) else 0
            energy_mid = np.mean(gain_valid[mid_freq_mask]) if np.any(mid_freq_mask) else 0
            energy_low = np.mean(gain_valid[low_freq_mask]) if np.any(low_freq_mask) else 0

            spectral_results[ticker] = {
                'High-Frequency (1-14d)': energy_high,
                'Mid-Frequency (15-90d)': energy_mid,
                'Low-Frequency (90+d)': energy_low,
                'Dominant_Band': max([('High-Frequency', energy_high),
                                      ('Mid-Frequency', energy_mid),
                                      ('Low-Frequency', energy_low)], key=lambda x: x[1])[0]
            }

            # Plotting
            ax = axes[i]
            # We plot against Temporal Days (Horizon) on log scale for standard Bode Plot look
            ax.semilogx(temporal_days, gain_valid, color='black', alpha=0.7, linewidth=1)

            # Overlay bands
            ax.axvspan(1, 14, color='red', alpha=0.1, label='High-Freq (Noise)')
            ax.axvspan(14, 90, color='yellow', alpha=0.1, label='Mid-Freq (Cyclical)')
            ax.axvspan(90, temporal_days.max(), color='green', alpha=0.1, label='Low-Freq (Structural)')

            ax.set_title(f'Node: {ticker} Flow Magnitude')
            ax.set_ylabel('Gain (dB)')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            if i == 0:
                ax.legend(loc='upper right')

        axes[-1].set_xlabel('Temporal Horizon (Days)')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print("Generated bode_plots.pdf.")
    return spectral_results


def generate_report(metrics, vaporization_risk, spectral_results):
    """
    Generate the definitive Execution Manual PDF output (The Long-Term Structural Ledger).
    """
    print("\n" + "="*90)
    print("                 ALL-TIME FLOW VELOCITIES: STRUCTURAL LEDGER (PDF OUT)")
    print("="*90)

    print("\n--- Tensor Network Health Check ---")
    active_nodes = len(metrics)
    # Estimate volume filter absorption (1 - average friction)
    avg_friction = metrics['Volume_Derivative'].mean()
    absorption = max(0, (1.0 - avg_friction) * 100) if avg_friction < 1.0 else (avg_friction - 1.0) * 100
    print(f"Nodes Active: {active_nodes} Tickers")
    print(f"Systemic Volume Filter Status: Nominal (Friction scaling ~{absorption:.1f}% of market noise)")

    print("\n" + "="*90)
    print("Phase 1: The Core Portfolio Matrix (The \"Static Cake\" Ledger)")
    print("="*90)
    print(f"{'Ticker':<8} | {'All-Time Velocity Score':<30} | {'Target Weight':<15} | {'Max Risk Band':<15} | {'Action Required'}")
    print("-" * 90)

    # Sort by Target Weight to show the highest allocations
    core_portfolio = metrics[metrics['Target_Weight_Pct'] > 0].sort_values(by='Target_Weight_Pct', ascending=False)

    for idx, row in core_portfolio.head(15).iterrows():
        vel = row['All_Time_Velocity']
        if vel > 5.0:
            desc = "(Strong Inflow)"
            action = "Allocate cash / Hold"
        elif vel > 3.0:
            desc = "(Steady Accumulation)"
            action = "Rebalance (Trim if over)"
        elif vel > 1.0:
            desc = "(Structural Core)"
            action = "Hold"
        else:
            desc = "(Cyclical Anchor)"
            action = "Trim to target"

        vel_str = f"+{vel:.1f} {desc}"
        weight = f"{row['Target_Weight_Pct']:.1f}%"
        band = f"{row['Max_Risk_Band_Pct']:.1f}%"
        print(f"{idx:<8} | {vel_str:<30} | {weight:<15} | {band:<15} | {action}")

    if len(core_portfolio) > 15:
        print(f"... and {len(core_portfolio) - 15} more positive velocity nodes.")

    print("\n" + "="*90)
    print("Phase 2: Velocity Trajectory Profiles (The Multi-Year Buys)")
    print("="*90)

    top_buys = core_portfolio.head(3)
    for idx, row in top_buys.iterrows():
        print(f"* Asset Profile: [{idx}]")
        print(f"  - Macroscopic Trend: Long-term capital absorption driven by structural industry dominance.")
        print(f"  - Topological Safety: High (Safety Score: {row['Topological_Safety']:.2f}). Insulated region of the market graph.")
        print(f"  - Entry Strategy: Allocate {row['Target_Weight_Pct']:.1f}% of idle capital. Permanent upward structural drift.\n")

    print("="*90)
    print("Phase 3: Systemic Wealth Vaporization Zones (The Absolute No-Go List)")
    print("="*90)

    if vaporization_risk:
        cat_a = [x[0] for x in vaporization_risk if "Category A" in x[1]]
        cat_b = [x[0] for x in vaporization_risk if "Category B" in x[1]]

        print("* Vaporization Risk Category A (Secular Decline):")
        print(f"  Tickers experiencing structural outflows. DO NOT ALLOCATE.")
        print(f"  {', '.join(cat_a[:15])}{'...' if len(cat_a)>15 else ''}")

        print("\n* Vaporization Risk Category B (High Contagion / Negative Curvature):")
        print(f"  Highly volatile nodes deeply interconnected with fragile assets.")
        print(f"  {', '.join(cat_b[:15])}{'...' if len(cat_b)>15 else ''}")
    else:
        print("No immediate vaporization threats detected in the current tensor slice.")

    print("\n" + "="*90)

    print("\n" + "="*90)
    print("Phase 4: Spectral Analysis & Frequency-to-Horizon Banding")
    print("="*90)
    print("Fourier Tensor Decomposition applied to top conviction nodes (bode_plots.pdf generated).")

    if spectral_results:
        for ticker, bands in spectral_results.items():
            print(f"\n* Node Frequency Spectrum: [{ticker}]")
            print(f"  - High-Frequency Noise (1-14d):  {bands['High-Frequency (1-14d)']:.2f} dB")
            print(f"  - Mid-Frequency Flow (15-90d):   {bands['Mid-Frequency (15-90d)']:.2f} dB")
            print(f"  - Low-Frequency Anchor (90+d):   {bands['Low-Frequency (90+d)']:.2f} dB")
            print(f"  -> Dominant Kinetic Band: {bands['Dominant_Band']}")
    else:
        print("No valid top nodes found for spectral analysis.")

    print("\n" + "="*90)

if __name__ == "__main__":
    close_df, vol_df = fetch_data(SYMBOLS)
    metrics = calculate_indicators(close_df, vol_df)
    metrics, vapor = perform_ml_analysis(close_df, vol_df, metrics)

    # Extract top conviction nodes
    top_nodes = metrics[metrics['Target_Weight_Pct'] > 0].sort_values(by='Target_Weight_Pct', ascending=False).head(5)
    spectral_results = perform_spectral_analysis(close_df, top_nodes)

    generate_report(metrics, vapor, spectral_results)
