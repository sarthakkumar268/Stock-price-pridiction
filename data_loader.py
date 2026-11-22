"""data_loader.py - fetches and cleans stock data (Yahoo Finance or CSV)."""

import pandas as pd
import yfinance as yf

FEATURES = ["Open", "High", "Low", "Close", "Volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only OHLCV columns, sort by date, handle missing values."""
    df = df[FEATURES].copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.apply(pd.to_numeric, errors="coerce")
    # Missing values: fill small gaps with the previous day's value,
    # then drop any rows that are still empty (e.g. at the very start).
    df = df.ffill().dropna()
    return df


def load_from_yahoo(ticker: str, start, end) -> pd.DataFrame:
    """Download daily data from Yahoo Finance. Raises ValueError on failure."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Please enter a stock ticker.")
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=False, progress=False)
    except Exception as e:
        raise ValueError(f"Could not download data: {e}")

    if df is None or df.empty:
        raise ValueError(
            f"No data found for '{ticker}'. Check the ticker symbol "
            "(Indian stocks need '.NS', e.g. RELIANCE.NS) and the date range."
        )

    # Newer yfinance versions return MultiIndex columns like ("Close", "AAPL")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return _clean(df)


def load_from_csv(file) -> pd.DataFrame:
    """Load a NIFTY-style CSV (Date, Open, High, Low, Close, Shares Traded, ...)."""
    df = pd.read_csv(file, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Shares Traded": "Volume"})

    missing = [c for c in ["Date"] + FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(),
                                format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return _clean(df)