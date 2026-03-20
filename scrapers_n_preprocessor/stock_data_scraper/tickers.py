"""
tickers.py
----------
Central registry of all ticker sets used across the pipeline.
Frozen sets ensure no stage can accidentally mutate the lists.
"""

from dataclasses import dataclass, field

__all__ = ["Tickers"]


@dataclass
class Tickers:
    """Central registry of all ticker sets used across the pipeline.

    Attributes:
        n_samples_per_ticker: Number of trading-day rows kept per ticker (~10 years).
        market_ticker: Benchmark ticker used for relative-return features.
        analysis_tickers: Tickers reserved for EDA / ad-hoc analysis.
        test_tickers: Subset of training_tickers selected via stationarity check.
        training_tickers: Full universe of tickers used for model training.
    """

    n_samples_per_ticker: int = 2765  # 10 years of trading days

    market_ticker: str = "SPY"

    analysis_tickers: frozenset = field(default_factory=lambda: frozenset({
        "NVDA", "META", "INTC", "BA", "TSLA",
    }))

    test_tickers: frozenset = field(default_factory=lambda: frozenset({
        'FNV', 'SKM', 'ARTNA', 'TMO', 'AMGN',
        'LBTYA', 'NOW', 'AEM', 'KT', 'ABT',
        'VZ', 'A', 'DLR', 'TMUS', 'NUE',
        'ODFL', 'AG', 'SBAC', 'STE', 'CF',
        'CMC', 'PFE', 'CDNS', 'HL', 'RIO'
    }))

    synthetic_tickers: frozenset = field(default_factory=lambda: frozenset({
        'SYN01', 'SYN02', 'SYN03', 'SYN04', 'SYN05',
        'SYN06', 'SYN07', 'SYN08', 'SYN09', 'SYN10',
        'SYN11', 'SYN12', 'SYN13', 'SYN14', 'SYN15',
        'SYN16', 'SYN17', 'SYN18', 'SYN19', 'SYN20',
        'SYN21', 'SYN22', 'SYN23', 'SYN24', 'SYN25',
    }))

    training_tickers: frozenset = field(default_factory=lambda: frozenset({
        # ---------------------------------------------------------
        # 1. Communication Services (46)
        # ---------------------------------------------------------
        "EA", "WBD", "OMC", "LYV",
        "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR",
        "TDS", "LUMN", "GOGO", "IRDM", "SBGI", "NXST", "TGNA", "YELP", "TRIP",
        "IAC", "ZG", "MTCH", "LBTYA", "LBTYK", "RCI", "BCE", "TU", "VOD", "TEF",
        "AMX", "KT", "SKM", "NTES", "BIDU", "SOHU", "CCOI", "IDCC", "SATS",
        "AMC", "TTWO", "NYT", "CNK",

        # ---------------------------------------------------------
        # 2. Consumer Discretionary (46)
        # ---------------------------------------------------------
        "LULU", "EXPE",
        "GME", "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX",
        "TGT", "MAR", "HLT", "LVS", "MGM", "WYNN", "GM", "F", "HOG", "TM",
        "HMC", "YUM", "CMG", "DRI", "DPZ", "WEN", "EBAY", "BBY", "KMX", "AZO",
        "ORLY", "ROST", "TSCO", "WSM", "ULTA", "TPR", "RL", "PVH", "VFC", "HAS",
        "MAT", "RCL", "CCL", "NCLH",

        # ---------------------------------------------------------
        # 3. Consumer Staples (46)
        # ---------------------------------------------------------
        "FLO", "JJSF", "BGS", "THS", "SENEA", "PRGO", "WDFC",
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "EL", "CL", "KMB",
        "SYY", "ADM", "KR", "DG", "DLTR", "MNST", "STZ", "TSN", "HRL", "CPB",
        "CAG", "GIS", "MKC", "CLX", "CHD", "TAP", "HSY", "SJM", "BG", "POST",
        "DAR", "INGR",

        # ---------------------------------------------------------
        # 4. Energy (46)
        # ---------------------------------------------------------
        "OKE", "TRGP",
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI",
        "WMB", "HAL", "BKR", "DVN", "FANG", "TRP", "ENB", "SU", "CNQ",
        "IMO", "OVV", "APA", "EQT", "MTDR", "RRC", "AR", "MUR", "CNX", "NFG",
        "PBF", "DK", "CVI", "HP", "PTEN", "NOV", "RES", "OII", "BP", "SHEL",
        "TTE", "EQNR", "PBR", "TPL",

        # ---------------------------------------------------------
        # 5. Financials (46)
        # ---------------------------------------------------------
        "CNO", "UNM", "GL", "RGA", "ERIE", "SIGI", "THG", "AFG", "ORI",
        "RLI", "PRA", "KMPR", "FAF", "STC", "MKL", "WRB",
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA",
        "SCHW", "SPGI", "MCO", "ICE", "CME", "CB", "PGR", "ALL", "TRV", "AIG",
        "MET", "PRU", "AFL", "USB", "PNC", "COF", "RF", "KEY", "HBAN", "FITB",

        # ---------------------------------------------------------
        # 6. Health Care (46)
        # ---------------------------------------------------------
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "CVS", "ELV", "CI", "HUM", "MCK", "CAH", "CNC", "HCA",
        "BDX", "BSX", "EW", "REGN", "VRTX", "ZTS", "ISRG", "SYK", "ZBH", "BAX",
        "A", "MTD", "ILMN", "IDXX", "ALGN", "WST", "RMD", "STE", "TFX", "COO",
        "HOLX", "LH", "BIO", "DGX", "UHS", "BIIB",

        # ---------------------------------------------------------
        # 7. Industrials (46)
        # ---------------------------------------------------------
        "J", "SNA", "NDSN", "WSO", "TNC", "GATX", "TRN", "GBX", "WWD",
        "ATRO", "HEI", "LSTR", "JBHT", "R",
        "GE", "CAT", "DE", "HON", "UNP", "UPS", "RTX", "LMT", "MMM",
        "ETN", "ITW", "WM", "CSX", "NSC", "FDX", "EMR", "PH", "GD", "NOC",
        "AME", "ROP", "TDG", "CMI", "PCAR", "FAST", "CPRT", "ODFL", "GWW",
        "URI", "VRSK", "DAL", "UAL",

        # ---------------------------------------------------------
        # 8. Information Technology (43)
        # ---------------------------------------------------------
        "AAPL", "MSFT", "AVGO", "ORCL", "ADBE", "CRM", "CSCO", "ACN", "AMD",
        "QCOM", "IBM", "TXN", "AMAT", "NOW", "INTU", "MU", "LRCX", "ADI",
        "KLAC", "SNPS", "CDNS", "PANW", "FTNT", "MCHP", "TEL", "GLW", "HPQ",
        "STX", "WDC", "NTAP", "FSLR", "ENPH", "TER", "TRMB", "ZBRA", "TYL",
        "PTC", "OTEX", "FFIV", "ADSK", "ADP", "PAYX", "AKAM", "VRSN", "MSI",

        # ---------------------------------------------------------
        # 9. Materials (46)
        # ---------------------------------------------------------
        "DD",
        "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "PPG", "LYB", "ALB", "FMC",
        "MOS", "ASH", "CF", "VMC", "MLM", "STLD", "NUE", "RIO", "BHP", "VALE",
        "SCCO", "GOLD", "AEM", "WPM", "FNV", "PAAS", "AG", "HL", "CDE", "EXP",
        "IFF", "CE", "EMN", "WLK", "HUN", "OLN", "RPM", "AVY", "SON", "SEE",
        "ATR", "KWR", "CMC", "NEU", "CLF",

        # ---------------------------------------------------------
        # 10. Real Estate (46)
        # ---------------------------------------------------------
        "NHI", "WPC", "ADC", "EPR",
        "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "AVB",
        "EQR", "EXR", "MAA", "UDR", "ARE", "BXP", "CBRE", "CPT", "HST", "KIM",
        "REG", "SBAC", "WY", "IRM", "LAMR", "OHI", "GLPI", "NNN", "FRT", "BRX",
        "KRC", "HIW", "DEI", "SLG", "VNO", "HPP", "ESRT", "MAC", "SKT", "VTR",
        "DOC", "ESS",

        # ---------------------------------------------------------
        # 11. Utilities (46)
        # ---------------------------------------------------------
        "MSEX", "ARTNA", "CWCO", "CPK", "MDU",
        "NEE", "SO", "DUK", "SRE", "AEP", "D", "PEG", "EXC", "XEL", "ED",
        "PCG", "EIX", "WEC", "ES", "DTE", "FE", "PPL", "ETR", "AEE", "CMS",
        "CNP", "ATO", "NI", "EVRG", "LNT", "AWK", "WTRG", "IDA", "OGE", "SR",
        "PNW", "HE", "BKH", "MGEE", "NWE", "OTTR", "UTL", "AVA", "CWT", "AWR"
    }))

    @property
    def all_fetch_tickers(self) -> frozenset:
        """Union of every ticker the scraper needs to download."""
        return (
            self.training_tickers
            | self.test_tickers
            | self.analysis_tickers
            | {self.market_ticker}
        )

    def __repr__(self) -> str:
        return (
            f"Tickers(training={len(self.training_tickers)}, "
            f"test={len(self.test_tickers)}, "
            f"analysis={len(self.analysis_tickers)}, "
            f"fetch={len(self.all_fetch_tickers)})"
        )


if __name__ == '__main__':
    tickers = Tickers()
    print(tickers)
