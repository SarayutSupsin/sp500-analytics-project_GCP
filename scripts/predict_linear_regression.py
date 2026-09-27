import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
DATASETS_DIR = os.path.join(OUTPUT_DIR, "raw_json_datasets")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")
MACRO_CSV_PATH = os.path.join(DATA_DIR, "cpi_fedrate_5y.csv")
PLOT_OUTPUT_PATH = os.path.join(REPORTS_DIR, "plot_actual_vs_predicted.png")
JSON_OUTPUT_PATH = os.path.join(DATASETS_DIR, "linear_regression_results.json")

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

AI_TECH_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "ORCL", "AMD", "AVGO", "AMZN", "AAPL", "QCOM"]
STAPLES_TICKERS = ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "GIS", "TGT", "SYY"]
ALL_20_TICKERS = AI_TECH_TICKERS + STAPLES_TICKERS

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

def run_linear_regression():
    """
    [METHOD 2] Pure GCP BigQuery ML OLS Linear Regression Pipeline
    1. Trains BQML model 'model_ols_rolling_return' directly on GCP BigQuery Data Warehouse.
    2. Queries ML.EVALUATE, ML.WEIGHTS, and ML.PREDICT directly from BigQuery ML on GCP.
    3. Exports 100% GCP BigQuery ML results per Ticker and per Period Regime to JSON.
    """
    print("=" * 65)
    print("Step 4: Pure GCP BigQuery ML OLS Linear Regression Pipeline")
    print("=" * 65)
    
    from google.cloud import bigquery
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    
    # 1. Train BigQuery ML OLS Model directly on GCP BigQuery Cloud Data Warehouse
    bqml_train_sql = f"""
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
            LAG(r.Fed_Rate, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_FedRate,
            LAG(sp.SP500_Return, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_SP500,
            LAG(s.Rolling12M_Return, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_Stock
        FROM stock_returns s
        LEFT JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_inflation_rates` i ON s.YearMonth = SUBSTR(CAST(i.Date AS STRING), 1, 7)
        LEFT JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_interest_rates` r ON s.YearMonth = SUBSTR(CAST(r.Date AS STRING), 1, 7)
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
    print("[1/4] Training BigQuery ML Model 'model_ols_rolling_return' on GCP...")
    bq_client.query(bqml_train_sql).result()
    print("[GCP BQML] Model training complete!")
    
    # 2. Query BigQuery ML Weights (Beta Coefficients) directly from GCP
    weights_sql = f"SELECT processed_input, weight FROM ML.WEIGHTS(MODEL `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.model_ols_rolling_return`)"
    df_weights = bq_client.query(weights_sql).to_dataframe()
    
    coeff_map = {row['processed_input']: round(float(row['weight']), 4) for _, row in df_weights.iterrows()}
    coeff_intercept = coeff_map.get('((intercept))', 0.0)
    coeff_cpi = coeff_map.get('Lag1_CPI', 0.0)
    coeff_fed = coeff_map.get('Lag1_FedRate', 0.0)
    coeff_sp500 = coeff_map.get('Lag1_SP500', 0.0)
    coeff_stock = coeff_map.get('Lag1_Stock', 0.0)
    
    # 3. Query BigQuery ML Overall Evaluation Metrics directly from GCP
    eval_sql = f"SELECT r2_score, mean_absolute_error, mean_squared_error FROM ML.EVALUATE(MODEL `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.model_ols_rolling_return`)"
    df_eval = bq_client.query(eval_sql).to_dataframe()
    overall_r2 = round(float(df_eval['r2_score'].iloc[0]), 4)
    overall_mae = round(float(df_eval['mean_absolute_error'].iloc[0]), 4)
    overall_mse = round(float(df_eval['mean_squared_error'].iloc[0]), 4)
    
    # 4. Query BigQuery ML Predictions per Ticker directly from GCP
    predict_sql = f"""
    WITH monthly_stock AS (
        SELECT Ticker, FORMAT_DATE('%Y-%m', Date) AS YearMonth, AVG(Close) AS Close
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.fact_stock_prices` GROUP BY Ticker, YearMonth
    ),
    stock_returns AS (
        SELECT Ticker, YearMonth, Close,
               (Close - LAG(Close, 12) OVER (PARTITION BY Ticker ORDER BY YearMonth)) / NULLIF(LAG(Close, 12) OVER (PARTITION BY Ticker ORDER BY YearMonth), 0) * 100 AS Rolling12M_Return
        FROM monthly_stock
    ),
    sp500_returns AS (
        SELECT YearMonth, Rolling12M_Return AS SP500_Return FROM stock_returns WHERE Ticker = '^GSPC'
    ),
    lagged_features AS (
        SELECT s.Ticker, s.YearMonth, s.Rolling12M_Return,
               LAG(i.CPI_Inflation_YoY, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_CPI,
               LAG(r.Fed_Rate, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_FedRate,
               LAG(sp.SP500_Return, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_SP500,
               LAG(s.Rolling12M_Return, 1) OVER (PARTITION BY s.Ticker ORDER BY s.YearMonth) AS Lag1_Stock
        FROM stock_returns s
        LEFT JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_inflation_rates` i ON s.YearMonth = SUBSTR(CAST(i.Date AS STRING), 1, 7)
        LEFT JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_interest_rates` r ON s.YearMonth = SUBSTR(CAST(r.Date AS STRING), 1, 7)
        LEFT JOIN sp500_returns sp ON s.YearMonth = sp.YearMonth
    )
    SELECT p.Ticker, p.YearMonth, p.Rolling12M_Return AS Actual_Return, p.predicted_Rolling12M_Return AS Predicted_Return,
           p.Lag1_CPI AS CPI, p.Lag1_FedRate AS Fed_Rate
    FROM ML.PREDICT(MODEL `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.model_ols_rolling_return`, (
        SELECT * FROM lagged_features
        WHERE Rolling12M_Return IS NOT NULL AND Lag1_CPI IS NOT NULL AND Lag1_FedRate IS NOT NULL AND Lag1_SP500 IS NOT NULL AND Lag1_Stock IS NOT NULL AND Ticker != '^GSPC'
    )) AS p
    ORDER BY p.Ticker, p.YearMonth
    """
    print("[2/4] Executing ML.PREDICT from BigQuery ML on GCP...")
    df_preds = bq_client.query(predict_sql).to_dataframe()
    
    # 5. Format results per Ticker & Multi-Period Regime for Web Dashboard
    results_by_period = {}
    series_by_period = {}
    
    sample_ticker_plot = "NVDA"
    y_actual_sample = []
    y_pred_sample = []
    dates_sample = []
    
    for period_key, (start_ym, end_ym) in PERIODS.items():
        period_results = []
        period_series_map = {}
        
        df_period = df_preds[(df_preds["YearMonth"] >= start_ym) & (df_preds["YearMonth"] <= end_ym)].copy()
        
        for ticker in ALL_20_TICKERS:
            sub = df_period[df_period["Ticker"] == ticker].copy()
            if not sub.empty:
                sector = "AI-Tech" if ticker in AI_TECH_TICKERS else "Consumer Staples"
                y_act = sub["Actual_Return"].values
                y_pred = sub["Predicted_Return"].values
                mae = np.mean(np.abs(y_act - y_pred))
                mse = np.mean((y_act - y_pred) ** 2)
                corr = np.corrcoef(y_act, y_pred)[0, 1] if len(sub) > 1 else 0.0
                r2 = float(corr ** 2) if not np.isnan(corr) else 0.0
                
                metric_entry = {
                    "Ticker": ticker,
                    "Sector": sector,
                    "R2_Score": round(float(r2), 4),
                    "MAE": round(float(mae), 4),
                    "MSE": round(float(mse), 4),
                    "Coeff_Intercept": coeff_intercept,
                    "Coeff_CPI": coeff_cpi,
                    "Coeff_FedRate": coeff_fed,
                    "Coeff_SP500": coeff_sp500,
                    "Coeff_Lag1_Stock": coeff_stock
                }
                period_results.append(metric_entry)
                
                series_list = []
                for _, r in sub.iterrows():
                    series_list.append({
                        "YearMonth": str(r["YearMonth"]),
                        "Actual_Return": round(float(r["Actual_Return"]), 2),
                        "Predicted_Return": round(float(r["Predicted_Return"]), 2),
                        "CPI": round(float(r["CPI"]), 2) if pd.notnull(r["CPI"]) else 0.0,
                        "Fed_Rate": round(float(r["Fed_Rate"]), 2) if pd.notnull(r["Fed_Rate"]) else 0.0
                    })
                period_series_map[ticker] = series_list
                
                if period_key == "ALL" and ticker == sample_ticker_plot:
                    y_actual_sample = y_act
                    y_pred_sample = y_pred
                    dates_sample = sub["YearMonth"].values
                    
        results_by_period[period_key] = period_results
        series_by_period[period_key] = period_series_map
        
    results = results_by_period["ALL"]
    series_by_ticker = series_by_period["ALL"]
    
    # 6. Plot Sample Actual vs Predicted Chart
    if len(dates_sample) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(dates_sample, y_actual_sample, label=f"Actual 12M Return ({sample_ticker_plot})", marker="o", color="#1f77b4", linewidth=2)
        plt.plot(dates_sample, y_pred_sample, label=f"Predicted 12M Return (BQML Model)", marker="x", color="#d62728", linestyle="--", linewidth=2)
        plt.title(f"BigQuery ML OLS Linear Regression: 12M Return ({sample_ticker_plot})\n(GCP BigQuery ML Model: model_ols_rolling_return)", fontsize=13, pad=15)
        plt.xlabel("Year-Month", fontsize=11)
        plt.ylabel("12-Month Rolling Return (%)", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
        plt.close()
        print(f"[Plot Output] Saved prediction chart: {PLOT_OUTPUT_PATH}")
        
    summary = {
        "status": "SUCCESS",
        "ticker_count": len(results),
        "bigquery_ml_sql_sample": bqml_train_sql.strip(),
        "bqml_overall": {
            "R2_Score": overall_r2,
            "MAE": overall_mae,
            "MSE": overall_mse
        },
        "results": results,
        "series_by_ticker": series_by_ticker,
        "results_by_period": results_by_period,
        "series_by_period": series_by_period
    }
    
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[GCP BQML Output] Successfully saved Pure BQML results to {JSON_OUTPUT_PATH}")
    return summary

if __name__ == "__main__":
    run_linear_regression()
