import os
import json
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

# Paths for data storage
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)
STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")

# Daily granularity CSV used by Method 3 (Survival Analysis)
STOCK_DAILY_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_daily_5y.csv")
MACRO_CSV_PATH = os.path.join(DATA_DIR, "cpi_fedrate_5y.csv")

# 20 Selected Tickers + S&P 500 Benchmark
AI_TECH_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "ORCL", "AMD", "AVGO", "AMZN", "AAPL", "QCOM"]
STAPLES_TICKERS = ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "GIS", "TGT", "SYY"]
ALL_TICKERS = AI_TECH_TICKERS + STAPLES_TICKERS + ["^GSPC"]

START_DATE = "2021-01-01"
END_DATE = "2026-08-31"

def ingest_stock_data():
    """Fetch DAILY stock price data from Yahoo Finance API, then derive a monthly snapshot file.

    - Daily file  -> stock_prices_daily_5y.csv  (used by Method 3: Survival Analysis)
    - Monthly file -> stock_prices_5y.csv       (month-end snapshot, used by Methods 1 & 2)
    """
    print("Downloading fresh DAILY stock data from Yahoo Finance API (20 tickers + S&P 500)...")
    data_frames = []
    for ticker in ALL_TICKERS:
        sector = "AI-Tech" if ticker in AI_TECH_TICKERS else ("Consumer Staples" if ticker in STAPLES_TICKERS else "Benchmark")
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=START_DATE, end=END_DATE, interval="1d")
            if not df.empty and "Close" in df.columns:
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                df["Ticker"] = ticker
                df["Sector"] = sector
                df = df[["Date", "Ticker", "Sector", "Close", "Volume"]]
                data_frames.append(df)
            time.sleep(0.2)  # Polite delay to avoid Yahoo Finance rate limiting
        except Exception as e:
            print(f"Warning downloading {ticker}: {e}")

    if not data_frames:
        raise RuntimeError("No stock data downloaded from Yahoo Finance. Check network / ticker validity.")

    df_daily = pd.concat(data_frames, ignore_index=True)
    df_daily.to_csv(STOCK_DAILY_CSV_PATH, index=False)
    print(f"Successfully saved DAILY stock data to: {STOCK_DAILY_CSV_PATH} ({len(df_daily)} rows)")

    # Derive monthly snapshots (last trading day of each month -> same granularity as old interval="1mo")
    df_daily["Date_dt"] = pd.to_datetime(df_daily["Date"])
    df_daily["MonthKey"] = df_daily["Date_dt"].dt.to_period("M")
    df_monthly = (df_daily.sort_values("Date_dt")
                  .groupby(["Ticker", "Sector", "MonthKey"], as_index=False)
                  .tail(1)
                  .drop(columns=["Date_dt", "MonthKey"])
                  .reset_index(drop=True))
    df_monthly = df_monthly[["Date", "Ticker", "Sector", "Close", "Volume"]]
    df_monthly.to_csv(STOCK_CSV_PATH, index=False)
    print(f"Successfully saved MONTHLY stock data to: {STOCK_CSV_PATH} ({len(df_monthly)} rows)")
    return df_monthly

def ingest_macro_data():
    """Fetch CPI Inflation Rate and Fed Interest Rate directly from FRED public CSV API."""
    print("Fetching fresh macroeconomic indicators (CPI & Fed Rate) from FRED public API...")
    
    # Direct FRED CSV URLs
    cpi_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    fed_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
    
    df_cpi = pd.read_csv(cpi_url)
    df_cpi.columns = [c.strip().lower() for c in df_cpi.columns]
    date_col_cpi = "observation_date" if "observation_date" in df_cpi.columns else ("date" if "date" in df_cpi.columns else df_cpi.columns[0])
    value_col_cpi = "cpiaucsl" if "cpiaucsl" in df_cpi.columns else df_cpi.columns[1]

    df_cpi = df_cpi.rename(columns={date_col_cpi: "Date", value_col_cpi: "CPI_Index"})
    df_cpi["Date"] = pd.to_datetime(df_cpi["Date"]).dt.strftime("%Y-%m-%d")
    df_cpi = df_cpi[(df_cpi["Date"] >= "2020-01-01") & (df_cpi["Date"] <= END_DATE)].sort_values("Date").reset_index(drop=True)
    df_cpi["CPI_Inflation_YoY"] = df_cpi["CPI_Index"].pct_change(12, fill_method=None) * 100
    df_cpi = df_cpi[df_cpi["Date"] >= START_DATE]
    
    df_fed = pd.read_csv(fed_url)
    df_fed.columns = [c.strip().lower() for c in df_fed.columns]
    date_col_fed = "observation_date" if "observation_date" in df_fed.columns else ("date" if "date" in df_fed.columns else df_fed.columns[0])
    value_col_fed = "fedfunds" if "fedfunds" in df_fed.columns else df_fed.columns[1]

    df_fed = df_fed.rename(columns={date_col_fed: "Date", value_col_fed: "Fed_Rate"})
    df_fed["Date"] = pd.to_datetime(df_fed["Date"]).dt.strftime("%Y-%m-%d")
    df_fed = df_fed[(df_fed["Date"] >= START_DATE) & (df_fed["Date"] <= END_DATE)].sort_values("Date").reset_index(drop=True)
    
    df_macro = pd.merge(df_cpi[["Date", "CPI_Inflation_YoY"]], df_fed[["Date", "Fed_Rate"]], on="Date", how="outer")
    df_macro = df_macro.sort_values("Date").reset_index(drop=True)

    df_macro.to_csv(MACRO_CSV_PATH, index=False)
    print(f"Successfully saved fresh macro indicators to: {MACRO_CSV_PATH} ({len(df_macro)} rows)")
    return df_macro

if __name__ == "__main__":
    print("=== Starting Step 1: Raw Data Ingestion ===")
    ingest_stock_data()
    ingest_macro_data()
    print("=== Step 1 Ingestion Complete ===")
