import os
import json
import math
import pandas as pd
import numpy as np

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")
STOCK_DAILY_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_daily_5y.csv")
MACRO_CSV_PATH = os.path.join(DATA_DIR, "cpi_fedrate_5y.csv")
CORR_JSON_PATH = os.path.join(OUTPUT_DIR, "correlation_results.json")
REG_JSON_PATH = os.path.join(OUTPUT_DIR, "linear_regression_results.json")
SURV_JSON_PATH = os.path.join(OUTPUT_DIR, "survival_results.json")
DASHBOARD_JSON_PATH = os.path.join(OUTPUT_DIR, "dashboard_data.json")
DASHBOARD_JS_PATH = os.path.join(OUTPUT_DIR, "dashboard_data.js")

AI_TECH_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "ORCL", "AMD", "AVGO", "AMZN", "AAPL", "QCOM"]
STAPLES_TICKERS = ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "GIS", "TGT", "SYY"]

def clean_nan_values(obj):
    """Recursively converts NaN values to None for valid JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    return obj

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

def export_all_dashboard_data():
    """
    Step 6: Dashboard Data Aggregation & Export
    Aggregates all raw historical data from BigQuery Data Warehouse, correlation, linear regression,
    and survival analysis into a single, consolidated dashboard_data.json and dashboard_data.js for UI rendering.
    """
    print("=" * 65)
    print("Step 6: Exporting Consolidated Dashboard Data (dashboard_data.json & dashboard_data.js)")
    print("=" * 65)
    
    # Verify presence of prior step outputs
    required_files = [CORR_JSON_PATH, REG_JSON_PATH, SURV_JSON_PATH]
    for fpath in required_files:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file missing: {fpath}. Please run steps 1-5 first.")
            
    try:
        from google.cloud import bigquery
        bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        df_stock = bq_client.query(f"SELECT * FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.fact_stock_prices`").to_dataframe()
        df_stock_daily = df_stock.copy()
        df_macro = bq_client.query(f"SELECT * FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_inflation_rates`").to_dataframe()
        print(f"[BigQuery Data Warehouse] Successfully queried fact_stock_prices & dim_inflation_rates directly from GCP Cloud Dataset '{BIGQUERY_DATASET_ID}'")
    except Exception as e:
        print(f"[Local Fallback] BigQuery query notice ({e}), loading local CSV files...")
        if not os.path.exists(STOCK_CSV_PATH) or not os.path.exists(MACRO_CSV_PATH):
            raise FileNotFoundError(f"Required CSV files missing. Please run ingest_raw_data.py first.")
        df_stock = pd.read_csv(STOCK_CSV_PATH)
        df_stock_daily = pd.read_csv(STOCK_DAILY_CSV_PATH) if os.path.exists(STOCK_DAILY_CSV_PATH) else df_stock.copy()
        df_macro = pd.read_csv(MACRO_CSV_PATH)
    
    with open(CORR_JSON_PATH, "r", encoding="utf-8") as f:
        corr_data = json.load(f)
        
    with open(REG_JSON_PATH, "r", encoding="utf-8") as f:
        reg_data = json.load(f)
        
    with open(SURV_JSON_PATH, "r", encoding="utf-8") as f:
        surv_data = json.load(f)
        
    # Format Historical Stock Records (Monthly Snapshot)
    df_stock["Date"] = pd.to_datetime(df_stock["Date"]).dt.strftime("%Y-%m-%d")
    stock_records = df_stock.to_dict(orient="records")

    # Format Daily Stock Records (Daily Granularity for Survival Analysis)
    df_stock_daily["Date"] = pd.to_datetime(df_stock_daily["Date"]).dt.strftime("%Y-%m-%d")
    stock_daily_records = df_stock_daily.to_dict(orient="records")
    
    # Format Macro Records
    macro_records = df_macro.to_dict(orient="records")
    
    consolidated_payload = {
        "metadata": {
            "project_name": "Financial S&P 500 Analytics",
            "stock_count": 20,
            "sectors": ["AI-Tech", "Consumer Staples"],
            "period": "2021-2026 (5 Years)",
            "benchmark": "^GSPC"
        },
        "raw_historical": {
            "stock_prices": stock_records,
            "stock_prices_daily": stock_daily_records,
            "macro_indicators": macro_records
        },
        "analytics": {
            "pearson_correlation": corr_data["results"],
            "linear_regression": reg_data["results"],
            "linear_regression_series": reg_data.get("series_by_ticker", {}),
            "linear_regression_by_period": reg_data.get("results_by_period", {}),
            "linear_regression_series_by_period": reg_data.get("series_by_period", {}),
            "survival_recovery": surv_data
        }
    }
    
    # Clean NaN values for strict valid JSON output
    cleaned_payload = clean_nan_values(consolidated_payload)
    
    with open(DASHBOARD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_payload, f, indent=4)

    with open(DASHBOARD_JS_PATH, "w", encoding="utf-8") as f:
        f.write("window.DASHBOARD_DATA = ")
        json.dump(cleaned_payload, f, indent=4)
        f.write(";\n")
        
    print(f"Successfully created consolidated dashboard dataset JSON: {DASHBOARD_JSON_PATH}")
    print(f"Successfully created consolidated dashboard dataset JS: {DASHBOARD_JS_PATH}")
    print(f"Total historical stock price records (monthly): {len(stock_records)}")
    print(f"Total historical stock price records (daily): {len(stock_daily_records)}")
    print(f"Total macro indicator records: {len(macro_records)}")
    print("=" * 65)
    return cleaned_payload

if __name__ == "__main__":
    export_all_dashboard_data()
