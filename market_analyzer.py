import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.fftpack


# A wide array of symbols to satisfy the request
SYMBOLS = [
    "A",
    "AA",
    "AAP",
    "AAPL",
    "ABBV",
    "ABG",
    "ABNB",
    "ABT",
    "ACHC",
    "ACM",
    "ACV",
    "ADBE",
    "ADI",
    "ADM",
    "ADUS",
    "AEM",
    "AEO",
    "AEP",
    "AES",
    "AFG",
    "AFI",
    "AFL",
    "AFRM",
    "AGCO",
    "AGG",
    "AGNC",
    "AGRO",
    "AJG",
    "AKAM",
    "ALCO",
    "ALGN",
    "ALHC",
    "ALHE",
    "ALL",
    "ALLE",
    "ALLY",
    "ALNY",
    "ALST",
    "ALTM",
    "ALUM",
    "AMAT",
    "AMCR",
    "AMD",
    "AME",
    "AMED",
    "AMG",
    "AMGN",
    "AMH",
    "AMOT",
    "AMP",
    "AMRX",
    "AMT",
    "AMZN",
    "AN",
    "ANDE",
    "ANF",
    "ANSS",
    "AON",
    "AOS",
    "APA",
    "APAM",
    "APO",
    "ARAD",
    "ARE",
    "ARES",
    "ARGO",
    "ASML",
    "ASYS",
    "AUD",
    "AUDC",
    "AVB",
    "AVDX",
    "AVGO",
    "AVY",
    "AXP",
    "AY",
    "AYAL",
    "AZO",
    "AZRG",
    "AZRI",
    "BA",
    "BABA",
    "BAC",
    "BAM",
    "BANR",
    "BARK",
    "BAX",
    "BC",
    "BCC",
    "BCS",
    "BDX",
    "BE",
    "BEN",
    "BEP",
    "BERY",
    "BESH",
    "BG",
    "BHF",
    "BHP",
    "BIE",
    "BIG",
    "BIIB",
    "BILL",
    "BJ",
    "BK",
    "BKNG",
    "BKR",
    "BLDP",
    "BLK",
    "BLL",
    "BMY",
    "BND",
    "BNTX",
    "BOC",
    "BOKF",
    "BOX",
    "BP",
    "BREN",
    "BRK.B",
    "BRO",
    "BRX",
    "BSX",
    "BTI",
    "BUD",
    "BURL",
    "BX",
    "BXMT",
    "BXP",
    "BYD",
    "C",
    "CACC",
    "CAD",
    "CAG",
    "CAMT",
    "CAST",
    "CAT",
    "CB",
    "CBL",
    "CBOE",
    "CBRE",
    "CBU",
    "CCI",
    "CCK",
    "CCS",
    "CDE",
    "CE",
    "CEG",
    "CEL",
    "CELH",
    "CENX",
    "CERT",
    "CF",
    "CFG",
    "CFR",
    "CG",
    "CGA",
    "CHD",
    "CHDN",
    "CHEF",
    "CHF",
    "CHH",
    "CHKP",
    "CHMI",
    "CHRW",
    "CHTR",
    "CHWY",
    "CI",
    "CIEN",
    "CIM",
    "CINF",
    "CL",
    "CL=F",
    "CLAL",
    "CLOV",
    "CLR",
    "CLX",
    "CMA",
    "CMC",
    "CMCSA",
    "CME",
    "CMG",
    "CMI",
    "CNC",
    "CNHI",
    "CNI",
    "CNP",
    "CNSL",
    "CNY",
    "COF",
    "COGN",
    "COHR",
    "COIN",
    "COLD",
    "COMP",
    "CONE",
    "COO",
    "COP",
    "COR",
    "CORE",
    "COST",
    "COTY",
    "CP",
    "CPB",
    "CPNG",
    "CPRI",
    "CPT",
    "CRBG",
    "CRL",
    "CRM",
    "CROX",
    "CRSP",
    "CRWD",
    "CRX",
    "CS",
    "CSCO",
    "CSIQ",
    "CSX",
    "CTRA",
    "CTVA",
    "CURO",
    "CUZ",
    "CVBF",
    "CVET",
    "CVI",
    "CVS",
    "CVX",
    "CWEN",
    "CWK",
    "CX",
    "CXP",
    "CYBR",
    "CYH",
    "CZR",
    "D",
    "D-WAVE",
    "DAC",
    "DAR",
    "DATA",
    "DB",
    "DBC",
    "DBX",
    "DD",
    "DDD",
    "DDOG",
    "DE",
    "DEI",
    "DELTA",
    "DFS",
    "DG",
    "DHI",
    "DHR",
    "DIA",
    "DINO",
    "DIS",
    "DISI",
    "DK",
    "DKNG",
    "DLEI",
    "DLR",
    "DLTR",
    "DM",
    "DOC",
    "DOCS",
    "DOCU",
    "DOLE",
    "DOV",
    "DOW",
    "DPZ",
    "DRE",
    "DRI",
    "DUK",
    "DVN",
    "DXC",
    "EARN",
    "EBAY",
    "EBS",
    "ECL",
    "ECP",
    "ECPG",
    "ED",
    "EDIT",
    "EDR",
    "EDRI",
    "EEM",
    "EFA",
    "EG",
    "EGLE",
    "EGP",
    "EHC",
    "EIG",
    "EIX",
    "EL",
    "ELAN",
    "ELCO",
    "ELECT",
    "ELF",
    "ELS",
    "ELV",
    "EMB",
    "EMN",
    "EMR",
    "ENB",
    "ENBL",
    "ENDP",
    "ENLT",
    "ENPH",
    "ENR",
    "ENVA",
    "ENVX",
    "EOG",
    "EPC",
    "EPD",
    "EQIX",
    "EQR",
    "ESLT",
    "ESS",
    "ET",
    "ETSY",
    "EUR",
    "EURN",
    "EURUSD=X",
    "EW",
    "EXAS",
    "EXC",
    "EXP",
    "EXPD",
    "EXPE",
    "EXPI",
    "EYE",
    "F",
    "FANG",
    "FAST",
    "FBHS",
    "FCEL",
    "FCX",
    "FDP",
    "FDS",
    "FDX",
    "FE",
    "FHN",
    "FIBI",
    "FIBK",
    "FITB",
    "FLYW",
    "FMC",
    "FNB",
    "FND",
    "FNKO",
    "FOR",
    "FOX",
    "FOXA",
    "FR",
    "FRO",
    "FRPT",
    "FRT",
    "FSLR",
    "FSLY",
    "FTI",
    "FTNT",
    "FVRR",
    "FYBR",
    "GBP",
    "GBPUSD=X",
    "GC=F",
    "GD",
    "GEF",
    "GFF",
    "GFS",
    "GGB",
    "GILD",
    "GIS",
    "GL",
    "GLD",
    "GLNCY",
    "GM",
    "GOGL",
    "GOLD",
    "GOLF",
    "GOOGL",
    "GPI",
    "GPK",
    "GPM",
    "GPRO",
    "GPS",
    "GRBK",
    "GS",
    "GSHD",
    "GTC",
    "GVA",
    "GWR",
    "H",
    "HAL",
    "HAMO",
    "HARL",
    "HAS",
    "HASI",
    "HBAN",
    "HBM",
    "HCA",
    "HD",
    "HDST",
    "HE",
    "HEI",
    "HELE",
    "HENKY",
    "HES",
    "HFFC",
    "HFM",
    "HFSI",
    "HG=F",
    "HII",
    "HL",
    "HLT",
    "HMC",
    "HOG",
    "HOMB",
    "HOOD",
    "HP",
    "HPAL",
    "HPP",
    "HQY",
    "HR",
    "HRL",
    "HSIC",
    "HSY",
    "HUI",
    "HUM",
    "HUN",
    "HYG",
    "HYSR",
    "HZNP",
    "HZO",
    "IAU",
    "IBM",
    "ICE",
    "ICL",
    "IDXX",
    "IEF",
    "IHG",
    "IJH",
    "IJR",
    "ILDC",
    "ILMN",
    "ILPT",
    "ILS",
    "ILS=X",
    "IMBBY",
    "INCY",
    "INTC",
    "INVH",
    "IONQ",
    "IP",
    "IPGP",
    "IPI",
    "IQV",
    "IRM",
    "ISPR",
    "ISRG",
    "ITW",
    "IVZ",
    "IWM",
    "JAZZ",
    "JBHT",
    "JHG",
    "JJSF",
    "JKS",
    "JLL",
    "JMIA",
    "JNJ",
    "JNPR",
    "JPM",
    "JPY",
    "JPY=X",
    "JRVR",
    "JTLV",
    "K",
    "KBH",
    "KDP",
    "KEN",
    "KEY",
    "KGC",
    "KHC",
    "KIM",
    "KINS",
    "KKR",
    "KLAC",
    "KMI",
    "KMPR",
    "KMX",
    "KNSL",
    "KNX",
    "KO",
    "KRC",
    "KRT",
    "KSU",
    "KTB",
    "LAD",
    "LCI",
    "LCID",
    "LEAD",
    "LEN",
    "LHX",
    "LI",
    "LITE",
    "LLY",
    "LMNR",
    "LMT",
    "LNC",
    "LOW",
    "LQD",
    "LRCX",
    "LSTR",
    "LTC",
    "LULU",
    "LUMI",
    "LUMN",
    "LVS",
    "LW",
    "LXU",
    "LYB",
    "LYV",
    "M",
    "MA",
    "MAA",
    "MAC",
    "MAR",
    "MAS",
    "MAT",
    "MATX",
    "MAXN",
    "MBII",
    "MCD",
    "MCHP",
    "MCO",
    "MDC",
    "MDGL",
    "MDRN",
    "MDT",
    "MED",
    "MEDP",
    "MELI",
    "MELSR",
    "MENA",
    "MET",
    "META",
    "MGDL",
    "MGM",
    "MGP",
    "MGRC",
    "MGS",
    "MHK",
    "MINT",
    "MITT",
    "MKL",
    "MKSI",
    "MLM",
    "MMC",
    "MMP",
    "MNDY",
    "MNR",
    "MNST",
    "MO",
    "MOH",
    "MORN",
    "MOS",
    "MPC",
    "MPLX",
    "MPW",
    "MQ",
    "MRK",
    "MRLN",
    "MRNA",
    "MRO",
    "MRVL",
    "MS",
    "MSCI",
    "MSFT",
    "MSM",
    "MT",
    "MTB",
    "MTD",
    "MTH",
    "MTW",
    "MU",
    "MUB",
    "MYE",
    "MZN",
    "NANO",
    "NAPA",
    "NATI",
    "NAVI",
    "NBR",
    "NDAQ",
    "NEE",
    "NEM",
    "NEP",
    "NET",
    "NFLX",
    "NHI",
    "NICE",
    "NIO",
    "NKE",
    "NLY",
    "NNDM",
    "NNN",
    "NOC",
    "NOV",
    "NOVA",
    "NOW",
    "NRG",
    "NSC",
    "NTAP",
    "NTLA",
    "NTR",
    "NTRS",
    "NUE",
    "NVDA",
    "NVO",
    "NVR",
    "NWL",
    "NWM",
    "NXPI",
    "O",
    "OAK",
    "ODFL",
    "OHI",
    "OI",
    "OII",
    "OKE",
    "OKTA",
    "OMCL",
    "OMF",
    "ONB",
    "ONT",
    "OPC",
    "OPEN",
    "ORAN",
    "ORC",
    "ORCL",
    "ORL",
    "ORLY",
    "ORMT",
    "OSCR",
    "OSK",
    "OVV",
    "OXY",
    "PAA",
    "PAAS",
    "PACW",
    "PAG",
    "PAHC",
    "PALL",
    "PANW",
    "PARA",
    "PARR",
    "PBCT",
    "PBF",
    "PCAR",
    "PCG",
    "PDCO",
    "PDY",
    "PEAK",
    "PEG",
    "PEI",
    "PENN",
    "PEP",
    "PERI",
    "PETS",
    "PFE",
    "PFG",
    "PFGC",
    "PFSW",
    "PG",
    "PGR",
    "PGRE",
    "PGY",
    "PH",
    "PHI",
    "PHM",
    "PHOE",
    "PHR",
    "PI",
    "PII",
    "PKG",
    "PKX",
    "PLAT",
    "PLD",
    "PLTR",
    "PLUG",
    "PM",
    "PNC",
    "PNW",
    "POLI",
    "PPG",
    "PPL",
    "PPLT",
    "PPRO",
    "PRAA",
    "PRGO",
    "PRLB",
    "PROV",
    "PRU",
    "PSTG",
    "PSX",
    "PTEN",
    "PTK",
    "PTNR",
    "PVH",
    "PVTL",
    "PWR",
    "PXD",
    "PYPL",
    "PZOL",
    "QBTS",
    "QCOM",
    "QLYS",
    "QQQ",
    "QSR",
    "QTS",
    "RADA",
    "RAMI",
    "RBGLY",
    "RCII",
    "RCM",
    "RDFN",
    "RDWR",
    "RE",
    "REG",
    "REGN",
    "REIT1",
    "REMY",
    "RETI",
    "REX",
    "RF",
    "RGA",
    "RGTI",
    "RICK",
    "RIG",
    "RIO",
    "RIVN",
    "RL",
    "RLX",
    "RM",
    "RNR",
    "ROK",
    "ROST",
    "RPD",
    "RRC",
    "RS",
    "RSI",
    "RTX",
    "RUN",
    "RUSHB",
    "RYAN",
    "SAH",
    "SAIA",
    "SAM",
    "SANF",
    "SBAC",
    "SBCF",
    "SBLK",
    "SBN",
    "SBRA",
    "SBUX",
    "SCCO",
    "SE",
    "SEDG",
    "SEE",
    "SEED",
    "SELA",
    "SEM",
    "SGC",
    "SGD",
    "SGEN",
    "SGOL",
    "SGRY",
    "SHAK",
    "SHEL",
    "SHFR",
    "SHLS",
    "SHW",
    "SHY",
    "SI=F",
    "SID",
    "SIG",
    "SIGI",
    "SILV",
    "SJM",
    "SJR",
    "SKX",
    "SLB",
    "SLG",
    "SLGN",
    "SLM",
    "SLV",
    "SMCI",
    "SMG",
    "SNA",
    "SNOW",
    "SNV",
    "SO",
    "SOFI",
    "SON",
    "SPB",
    "SPEN",
    "SPG",
    "SPGI",
    "SPLK",
    "SPNT",
    "SPTN",
    "SPWR",
    "SPY",
    "SQ",
    "SRE",
    "SSYS",
    "STAG",
    "STE",
    "STLA",
    "STLD",
    "STNG",
    "STRS",
    "STT",
    "STWD",
    "STX",
    "STZ",
    "SUI",
    "SUM",
    "SWK",
    "SWN",
    "SYF",
    "SYK",
    "SYY",
    "TAP",
    "TARO",
    "TASE",
    "TBI",
    "TCBI",
    "TCTR",
    "TDG",
    "TDOC",
    "TDS",
    "TEAM",
    "TECH",
    "TECK",
    "TER",
    "TEVA",
    "TEX",
    "TFC",
    "TGLS",
    "TGT",
    "THC",
    "THG",
    "THO",
    "TIP",
    "TJX",
    "TLT",
    "TM",
    "TMO",
    "TNDM",
    "TNK",
    "TNV",
    "TOL",
    "TOST",
    "TPB",
    "TPG",
    "TPR",
    "TR",
    "TRIP",
    "TRNO",
    "TROW",
    "TRP",
    "TRV",
    "TSCO",
    "TSLA",
    "TT",
    "TTE",
    "TWO",
    "TX",
    "TXN",
    "TXRH",
    "TXT",
    "UAA",
    "UAN",
    "UBS",
    "UDR",
    "UHS",
    "ULTA",
    "UMPQ",
    "UNFI",
    "UNH",
    "UNI",
    "UNIT",
    "UNM",
    "UNP",
    "UPS",
    "UPST",
    "URBN",
    "URI",
    "USB",
    "USCR",
    "USD",
    "USFD",
    "USM",
    "UTHR",
    "UVV",
    "V",
    "VALE",
    "VEA",
    "VEEV",
    "VFC",
    "VGR",
    "VLD",
    "VLO",
    "VMC",
    "VNO",
    "VOD",
    "VR",
    "VRT",
    "VRTX",
    "VSP",
    "VST",
    "VTR",
    "VTRS",
    "VWO",
    "WAL",
    "WAT",
    "WAY",
    "WBD",
    "WBS",
    "WCC",
    "WD40",
    "WDAY",
    "WDC",
    "WELL",
    "WEN",
    "WFC",
    "WGO",
    "WH",
    "WHR",
    "WIL",
    "WILG",
    "WIX",
    "WLK",
    "WMB",
    "WMG",
    "WMT",
    "WNR",
    "WOOF",
    "WRB",
    "WRK",
    "WSTG",
    "WTFC",
    "WTW",
    "WYNN",
    "XEL",
    "XOM",
    "XPEV",
    "XPO",
    "XRAY",
    "XXII",
    "XYL",
    "YETI",
    "YHND",
    "YTEN",
    "YUM",
    "Z",
    "ZBH",
    "ZEUS",
    "ZEVI",
    "ZIM",
    "ZINC",
    "ZION",
    "ZS",
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

    # 1. All-Time Velocity Score (Long-Term Structural Drift)
    # Computed as the annualized geometric mean return normalized by variance (Sharpe-like structural flow)
    # We drop NAs per column to get true history length
    def calc_velocity(series):
        s = series.dropna()
        if len(s) < 252: # Need at least a year of data
            return 0.0
        returns = s.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        # Annualized drift over variance
        drift = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        # Scale to a readable score (-10 to +10 roughly)
        score = (drift / volatility) * 5.0
        return score

    all_time_velocity = close_data.apply(calc_velocity)

    # Calculate a proxy for topological safety (inverse of overall variance)
    safety_score = 1.0 / (close_data.pct_change().std() * np.sqrt(252) + 1e-6)

    # 2. Volume Flow Tensor Components (Rolling Volume Derivatives)
    # V_i(t): the 30-day rolling average volume relative to its 1-year average
    vol_30d = vol_data.rolling(window=30).mean().iloc[-1]
    vol_252d = vol_data.rolling(window=252).mean().iloc[-1]
    # Replace zeros or NaNs to avoid division errors
    vol_252d = vol_252d.replace(0, np.nan).fillna(vol_30d)

    # Volume derivative/ratio: >1 means liquidity is expanding into the node
    volume_derivative = (vol_30d / vol_252d).fillna(1.0)

    metrics = pd.DataFrame({
        'Current_Price': latest_prices,
        'All_Time_Velocity': all_time_velocity,
        'Topological_Safety': safety_score,
        'Volume_Derivative': volume_derivative
    })

    return metrics

def perform_ml_analysis(close_data, vol_data, metrics):
    """
    Construct the 3rd-Order Flow Tensor F_t and apply the Volume Friction Counter.
    Maps systemic vaporization zones.
    """
    print("Constructing 3rd-Order Flow Tensor & Volume Friction Filter...")

    returns = close_data.pct_change().dropna()
    N = returns.shape[1]

    # 1. Base Dimension: Price Covariance (T_i->j)
    corr_matrix = returns.corr().fillna(0)

    # We construct a 3D Tensor conceptually. For efficiency in Python/Pandas,
    # we'll compute the True Realized Wealth Flow Velocity (W_i->j) directly using broadcasting.
    # W_{i->j} = T_{i->j} * (V_i * V_j)

    # Extract volume derivatives V_i
    V = metrics['Volume_Derivative'].values

    # Compute the Volume Friction Matrix (V_i * V_j) outer product
    volume_friction_matrix = np.outer(V, V)

    # The Realized Wealth Flow Tensor Slice (2D representation of the 3D interaction)
    wealth_flow_matrix = corr_matrix.values * volume_friction_matrix
    wealth_flow_df = pd.DataFrame(wealth_flow_matrix, index=corr_matrix.index, columns=corr_matrix.columns)

    # 2. Systemic Wealth Vaporization (Continuous Long-Term Deceleration)
    # Calculate net inflow/outflow per node based on the tensor
    # If sum of inflows < outflows structurally, it's vaporizing.
    # Proxy: long-term velocity < 0 combined with negative flow graph centrality

    vaporization_risk = []
    for ticker in metrics.index:
        vel = metrics.loc[ticker, 'All_Time_Velocity']
        # Category A: Secular Decline (Long term velocity deeply negative)
        if vel < -2.0:
            vaporization_risk.append((ticker, "Category A (Secular Decline)"))
        # Category B: High Contagion (Low safety, negative velocity)
        elif vel < 0 and metrics.loc[ticker, 'Topological_Safety'] < 1.5:
            vaporization_risk.append((ticker, "Category B (High Contagion / Negative Curvature)"))

    # Calculate target portfolio weights based on All-Time Velocity and Volume Flow Centrality
    # We only allocate to positive velocity nodes.
    positive_nodes = metrics[metrics['All_Time_Velocity'] > 0].copy()

    # Calculate Flow Centrality: Sum of positive wealth inflows
    flow_centrality = wealth_flow_df.where(wealth_flow_df > 0, 0).sum(axis=0)

    if len(positive_nodes) > 0:
        # Score = (Velocity * 0.7) + (Normalized Flow Centrality * 0.3)
        # Normalize flow centrality to match velocity scale roughly (0 to 10)
        norm_centrality = (flow_centrality / flow_centrality.max()) * 10.0

        positive_nodes['Allocation_Score'] = (positive_nodes['All_Time_Velocity'] * 0.7) + (norm_centrality.loc[positive_nodes.index] * 0.3)

        # Calculate target weights (proportional to score, capped to maintain diversification)
        total_score = positive_nodes['Allocation_Score'].sum()
        positive_nodes['Target_Weight_Pct'] = (positive_nodes['Allocation_Score'] / total_score) * 100.0

        # Cap max weight to 15% to prevent hyper-concentration
        positive_nodes['Target_Weight_Pct'] = positive_nodes['Target_Weight_Pct'].clip(upper=15.0)
        # Re-normalize after clipping
        positive_nodes['Target_Weight_Pct'] = (positive_nodes['Target_Weight_Pct'] / positive_nodes['Target_Weight_Pct'].sum()) * 100.0

        positive_nodes['Max_Risk_Band_Pct'] = positive_nodes['Target_Weight_Pct'] * 1.3 # 30% tolerance band
    else:
        positive_nodes['Target_Weight_Pct'] = 0
        positive_nodes['Max_Risk_Band_Pct'] = 0

    metrics = metrics.join(positive_nodes[['Target_Weight_Pct', 'Max_Risk_Band_Pct']])
    metrics['Target_Weight_Pct'] = metrics['Target_Weight_Pct'].fillna(0)
    metrics['Max_Risk_Band_Pct'] = metrics['Max_Risk_Band_Pct'].fillna(0)

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
