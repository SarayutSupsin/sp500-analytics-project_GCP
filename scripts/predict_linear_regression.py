import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")
MACRO_CSV_PATH = os.path.join(DATA_DIR, "cpi_fedrate_5y.csv")
PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "plot_actual_vs_predicted.png")
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "linear_regression_results.json")

# GCP BigQuery ML Configuration Parameters
def get_gcp_project_id():
    env_id = os.environ.get("GCP_PROJECT_ID")
    if env_id:
        return env_id
    try:
        import google.auth
        _, project = google.auth.default()
        if project:
            return project
    except Exception:
        pass
    return "project-308492-gcp-70-1"

GCP_PROJECT_ID = get_gcp_project_id()
BIGQUERY_DATASET_ID = "sp500_analytics"
ENABLE_BQML = os.environ.get("ENABLE_BQML", "False").lower() == "true"

AI_TECH_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "ORCL", "AMD", "AVGO", "AMZN", "AAPL", "QCOM"]
STAPLES_TICKERS = ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "GIS", "TGT", "SYY"]
ALL_20_TICKERS = AI_TECH_TICKERS + STAPLES_TICKERS

def run_linear_regression():
    """
    [METHOD 2] OLS Linear Regression Analysis (12-Month Rolling Return % YoY)
    Equation: Rolling_12M_Return_t = b0 + b1*Lag1_CPI + b2*Lag1_FedRate + b3*Lag1_SP500_12M + b4*Lag1_Stock_12M
    
    1. Prepares 12-Month Rolling Stock & Market Return (% YoY).
    2. Constructs 1-Month Lagged Features to prevent look-ahead bias.
    3. Fits OLS Regression per ticker.
    4. Calculates R-squared, MAE %, and Beta Coefficients.
    5. Exports time-series prediction predictions per ticker.
    """
    print("=" * 65)
    print("Step 4: OLS Linear Regression Analysis (12-Month Rolling Return % YoY)")
    print("=" * 65)
    
    try:
        from google.cloud import bigquery
        bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        df_stock = bq_client.query(f"SELECT * FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.fact_stock_prices`").to_dataframe()
        df_macro = bq_client.query(f"SELECT * FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_inflation_rates`").to_dataframe()
        if "Fed_Rate" not in df_macro.columns and "Fed_Interest_Rate" in df_macro.columns:
            df_macro.rename(columns={"Fed_Interest_Rate": "Fed_Rate"}, inplace=True)
        if "CPI_Inflation_YoY" not in df_macro.columns and "CPI_Inflation_Rate" in df_macro.columns:
            df_macro.rename(columns={"CPI_Inflation_Rate": "CPI_Inflation_YoY"}, inplace=True)
        if "Date" not in df_macro.columns and "YearMonth" in df_macro.columns:
            df_macro.rename(columns={"YearMonth": "Date"}, inplace=True)
        print(f"[BigQuery Data Warehouse] Successfully queried fact_stock_prices & dim_inflation_rates directly from GCP Cloud Dataset '{BIGQUERY_DATASET_ID}'")
    except Exception as e:
        print(f"[Local Fallback] BigQuery query notice ({e}), loading local CSV files...")
        if not os.path.exists(STOCK_CSV_PATH) or not os.path.exists(MACRO_CSV_PATH):
            raise FileNotFoundError("Input CSV files missing. Please run ingest_raw_data.py first.")
        df_stock = pd.read_csv(STOCK_CSV_PATH)
        df_macro = pd.read_csv(MACRO_CSV_PATH)
    
    df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    df_stock["YearMonth"] = df_stock["Date"].dt.strftime("%Y-%m")
    df_macro["YearMonth"] = pd.to_datetime(df_macro["Date"]).dt.strftime("%Y-%m")
    
    # 1. Compute 12-Month Rolling Return for S&P 500 Market Benchmark (^GSPC)
    sp500 = df_stock[df_stock["Ticker"] == "^GSPC"].sort_values("Date").reset_index(drop=True)
    sp500["SP500_Rolling12M_Return"] = sp500["Close"].pct_change(12) * 100.0
    sp500_map = sp500.set_index("YearMonth")["SP500_Rolling12M_Return"].to_dict()
    
    # 2. Compute 12-Month Rolling Return (% YoY) for 20 tickers
    df_20 = df_stock[df_stock["Ticker"].isin(ALL_20_TICKERS)].copy()
    df_20 = df_20.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df_20["Rolling12M_Return"] = df_20.groupby("Ticker")["Close"].pct_change(12) * 100.0
    
    # 3. Merge with Macro & S&P 500 Benchmark
    df_merged = pd.merge(df_20, df_macro[["YearMonth", "CPI_Inflation_YoY", "Fed_Rate"]], on="YearMonth", how="inner")
    df_merged["SP500_Rolling12M_Return"] = df_merged["YearMonth"].map(sp500_map)
    df_merged = df_merged.sort_values(["Ticker", "YearMonth"]).reset_index(drop=True)
    
    # 4. Construct 1-Month Lagged Features per Ticker
    df_merged["Lag1_CPI"] = df_merged.groupby("Ticker")["CPI_Inflation_YoY"].shift(1)
    df_merged["Lag1_FedRate"] = df_merged.groupby("Ticker")["Fed_Rate"].shift(1)
    df_merged["Lag1_SP500"] = df_merged.groupby("Ticker")["SP500_Rolling12M_Return"].shift(1)
    df_merged["Lag1_Stock"] = df_merged.groupby("Ticker")["Rolling12M_Return"].shift(1)
    
    df_clean = df_merged.dropna(subset=["Rolling12M_Return", "Lag1_CPI", "Lag1_FedRate", "Lag1_SP500", "Lag1_Stock"]).copy()
    
    # 5. Fit OLS Model per Ticker for All 9 Period Regimes
    PERIODS = {
        "ALL": ("2021-01", "2026-12"),
        "2021-2022": ("2021-01", "2022-12"),
        "2023-2024": ("2023-01", "2024-12"),
        "2025-2026": ("2025-01", "2026-12"),
        "2021": ("2021-01", "2021-12"),
        "2022": ("2022-01", "2022-12"),
        "2023": ("2023-01", "2023-12"),
        "2024": ("2024-01", "2024-12"),
        "2025": ("2025-01", "2025-12")
    }

    results = [] # Legacy ALL period list
    series_by_ticker = {} # Legacy ALL period series map
    
    results_by_period = {}
    series_by_period = {}

    sample_ticker_plot = "NVDA"
    y_actual_sample = []
    y_pred_sample = []
    dates_sample = []
    
    features = ["Lag1_CPI", "Lag1_FedRate", "Lag1_SP500", "Lag1_Stock"]
    
    for period_key, (start_ym, end_ym) in PERIODS.items():
        period_results = []
        period_series_map = {}
        
        df_period = df_clean[(df_clean["YearMonth"] >= start_ym) & (df_clean["YearMonth"] <= end_ym)].copy()
        
        for ticker in ALL_20_TICKERS:
            sub = df_period[df_period["Ticker"] == ticker].copy()
            if len(sub) >= 3:
                X = sub[features]
                y = sub["Rolling12M_Return"]
                
                model = LinearRegression()
                model.fit(X, y)
                preds = model.predict(X)
                
                r2 = r2_score(y, preds) if len(sub) > 1 else 0.0
                mae = mean_absolute_error(y, preds)
                sector = "AI-Tech" if ticker in AI_TECH_TICKERS else "Consumer Staples"
                
                metric_entry = {
                    "Ticker": ticker,
                    "Sector": sector,
                    "R2_Score": round(float(r2), 4),
                    "MAE": round(float(mae), 4),
                    "Coeff_Intercept": round(float(model.intercept_), 4),
                    "Coeff_CPI": round(float(model.coef_[0]), 4),
                    "Coeff_FedRate": round(float(model.coef_[1]), 4),
                    "Coeff_SP500": round(float(model.coef_[2]), 4),
                    "Coeff_Lag1_Stock": round(float(model.coef_[3]), 4)
                }
                period_results.append(metric_entry)
                
                series_list = []
                for d_val, act_val, pred_val, cpi_val, fed_val in zip(
                    sub["YearMonth"].values, y.values, preds, sub["CPI_Inflation_YoY"].values, sub["Fed_Rate"].values
                ):
                    series_list.append({
                        "YearMonth": str(d_val),
                        "Actual_Return": round(float(act_val), 2),
                        "Predicted_Return": round(float(pred_val), 2),
                        "CPI": round(float(cpi_val), 2),
                        "Fed_Rate": round(float(fed_val), 2)
                    })
                period_series_map[ticker] = series_list
                
                if period_key == "ALL" and ticker == sample_ticker_plot:
                    y_actual_sample = y.values
                    y_pred_sample = preds
                    dates_sample = sub["YearMonth"].values
                    
        results_by_period[period_key] = period_results
        series_by_period[period_key] = period_series_map
        
    results = results_by_period["ALL"]
    series_by_ticker = series_by_period["ALL"]
                
    # 6. Plot Sample Actual vs Predicted Chart
    plt.figure(figsize=(12, 6))
    plt.plot(dates_sample, y_actual_sample, label=f"Actual 12M Return ({sample_ticker_plot})", marker="o", color="#1f77b4", linewidth=2)
    plt.plot(dates_sample, y_pred_sample, label=f"Predicted 12M Return (OLS Model)", marker="x", color="#d62728", linestyle="--", linewidth=2)
    plt.title(f"OLS Linear Regression Model: 12-Month Rolling Return vs OLS Model ({sample_ticker_plot})\n(Predictors: Lagged CPI, Fed Rate, S&P 500 & Stock Return)", fontsize=13, pad=15)
    plt.xlabel("Year-Month", fontsize=11)
    plt.ylabel("12-Month Rolling Return (%)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close()
    
    print(f"Generated prediction comparison chart: {PLOT_OUTPUT_PATH}")
    
    # 7. Execute BigQuery ML Model Training if ENABLE_BQML is True
    bqml_sql = f"""
    CREATE OR REPLACE MODEL `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.model_ols_rolling_return`
    OPTIONS(model_type='linear_reg', input_label_cols=['Rolling12M_Return']) AS
    WITH monthly_stock AS (
        SELECT 
            Ticker,
            FORMAT_DATE('%Y-%m', Date) AS YearMonth,
            AVG(Close) AS Close
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.fact_stock_prices`
        GROUP BY Ticker, YearMonth
    ),
    stock_returns AS (
        SELECT 
            Ticker,
            YearMonth,
            Close,
            (Close - LAG(Close, 12) OVER (PARTITION BY Ticker ORDER BY YearMonth)) / NULLIF(LAG(Close, 12) OVER (PARTITION BY Ticker ORDER BY YearMonth), 0) * 100 AS Rolling12M_Return
        FROM monthly_stock
    ),
    sp500_returns AS (
        SELECT YearMonth, Rolling12M_Return AS SP500_Return
        FROM stock_returns
        WHERE Ticker = '^GSPC'
    ),
    lagged_features AS (
        SELECT 
            s.Ticker,
            s.YearMonth,
            s.Rolling12M_Return,
            LAG(i.CPI_Inflation_YoY, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_CPI,
            LAG(r.Fed_Interest_Rate, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_FedRate,
            LAG(sp.SP500_Return, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_SP500,
            LAG(s.Rolling12M_Return, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_Stock
        FROM stock_returns s
        LEFT JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_inflation_rates` i ON s.YearMonth = i.YearMonth
        LEFT JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_interest_rates` r ON s.YearMonth = r.YearMonth
        LEFT JOIN sp500_returns sp ON s.YearMonth = sp.YearMonth
    )
    SELECT 
        Rolling12M_Return,
        Lag1_CPI,
        Lag1_FedRate,
        Lag1_SP500,
        Lag1_Stock
    FROM lagged_features
    WHERE Rolling12M_Return IS NOT NULL 
      AND Lag1_CPI IS NOT NULL 
      AND Lag1_FedRate IS NOT NULL 
      AND Lag1_SP500 IS NOT NULL
      AND Lag1_Stock IS NOT NULL
      AND Ticker != '^GSPC'
    """
    
    if ENABLE_BQML:
        print("\nExecuting BigQuery ML Model Training on GCP Cloud...")
        try:
            from google.cloud import bigquery
            bq_client = bigquery.Client(project=GCP_PROJECT_ID)
            query_job = bq_client.query(bqml_sql)
            query_job.result()
            print(f"[BigQuery ML] Successfully trained model 'model_ols_rolling_return' in dataset '{BIGQUERY_DATASET_ID}'")
        except Exception as e:
            print(f"BigQuery ML Execution Warning: {e}")
            
    # 8. Save JSON Summary
    summary = {
        "status": "SUCCESS",
        "ticker_count": len(results),
        "bigquery_ml_sql_sample": bqml_sql.strip(),
        "results": results,
        "series_by_ticker": series_by_ticker,
        "results_by_period": results_by_period,
        "series_by_period": series_by_period
    }
    
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Saved linear regression results JSON: {JSON_OUTPUT_PATH}")
    print("=" * 65)
    return summary

if __name__ == "__main__":
    run_linear_regression()


