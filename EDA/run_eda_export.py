import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

base_dir = r"d:\University memory\year4 [2569]\Bigdata and Cloud\sp500-analytics-project_GCP\EDA"
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

ai_tech_tickers = ['NVDA', 'MSFT', 'GOOGL', 'AAPL', 'AMZN', 'META', 'AVGO', 'QCOM', 'AMD', 'ORCL']
consumer_staples_tickers = ['PG', 'PEP', 'KO', 'COST', 'WMT', 'MDLZ', 'CL', 'GIS', 'SYY', 'TGT']
all_tickers = ai_tech_tickers + consumer_staples_tickers + ['^GSPC']

print("=== Downloading Market Data ===")
raw_download = yf.download(all_tickers, start='2021-01-01', end='2026-01-01', progress=False)

if isinstance(raw_download.columns, pd.MultiIndex):
    if 'Close' in raw_download.columns.levels[0]:
        raw_close = raw_download['Close'].copy()
    else:
        raw_close = raw_download.xs('Close', axis=1, level=0).copy()
else:
    raw_close = raw_download.copy()

raw_close.columns = [str(c) for c in raw_close.columns]
raw_close.index.name = 'Date'

df_raw = raw_close.reset_index()
df_stocks = df_raw.melt(id_vars=['Date'], var_name='Ticker', value_name='Close').dropna()

def get_sector(t):
    if t in ai_tech_tickers:
        return 'AI-Tech'
    elif t in consumer_staples_tickers:
        return 'Consumer Staples'
    else:
        return 'Benchmark'

df_stocks['Sector'] = df_stocks['Ticker'].apply(get_sector)
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])

# 1. Price Stats Table
print("\n=== PRICE STATS TABLE (20 TICKERS) ===")
price_stats = df_stocks[df_stocks['Sector'] != 'Benchmark'].groupby(['Sector', 'Ticker'])['Close'].describe()
print(price_stats.round(2).to_markdown())

# 2. Return Stats Table
df_stocks['Daily_Return'] = df_stocks.groupby('Ticker')['Close'].pct_change() * 100
df_returns = df_stocks.dropna(subset=['Daily_Return']).copy()

return_ticker_stats = df_returns[df_returns['Sector'] != 'Benchmark'].groupby(['Sector', 'Ticker'])['Daily_Return'].agg(
    Mean='mean',
    Std_Dev='std',
    Min='min',
    Max='max',
    Skewness='skew',
    Kurtosis=lambda x: x.kurt()
)
print("\n=== RETURN STATS TABLE (20 TICKERS) ===")
print(return_ticker_stats.round(4).to_markdown())
