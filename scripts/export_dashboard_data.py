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
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
DATASETS_DIR = os.path.join(OUTPUT_DIR, "raw_json_datasets")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")
STOCK_DAILY_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_daily_5y.csv")
MACRO_CSV_PATH = os.path.join(DATA_DIR, "cpi_fedrate_5y.csv")
CORR_JSON_PATH = os.path.join(DATASETS_DIR, "correlation_results.json")
REG_JSON_PATH = os.path.join(DATASETS_DIR, "linear_regression_results.json")
SURV_JSON_PATH = os.path.join(DATASETS_DIR, "survival_results.json")
DASHBOARD_JSON_PATH = os.path.join(DATASETS_DIR, "dashboard_data.json")

GCS_BUCKET_NAME = "sp500-analytics-bucket"

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

def upload_file_to_gcs(local_path, bucket_name, gcs_blob_name):
    """Uploads a local file to Google Cloud Storage Central Bucket."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_path)
        print(f"Successfully uploaded {os.path.basename(local_path)} to Central Storage: gs://{bucket_name}/{gcs_blob_name}")
    except Exception as e:
        print(f"Notice: GCS upload skipped or fallback applied ({e})")

def export_all_dashboard_data():
    """
    Step 6: Dashboard Data Aggregation & Export
    Aggregates all raw historical data, correlation, linear regression, and survival analysis
    into a single, consolidated dashboard_data.json and dashboard_data.js for UI rendering,
    and automatically uploads them to Central Storage (GCS Bucket).
    """
    print("=" * 65)
    print("Step 6: Exporting Consolidated Dashboard Data (dashboard_data.json & dashboard_data.js)")
    print("=" * 65)
    
    # Verify presence of prior step outputs
    required_files = [STOCK_CSV_PATH, STOCK_DAILY_CSV_PATH, MACRO_CSV_PATH, CORR_JSON_PATH, REG_JSON_PATH, SURV_JSON_PATH]
    for fpath in required_files:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file missing: {fpath}. Please run steps 1-5 first.")
            
    df_stock = pd.read_csv(STOCK_CSV_PATH)
    df_stock_daily = pd.read_csv(STOCK_DAILY_CSV_PATH)
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
        
    print(f"Successfully created consolidated dataset JSON: {DASHBOARD_JSON_PATH}")
    
    # 1. Build Single-File Standalone Deliverable HTML (outputs/sp500_dashboard.html)
    WEB_DIR = os.path.join(BASE_DIR, "web")
    INDEX_HTML_PATH = os.path.join(WEB_DIR, "index.html")
    APP_JS_PATH = os.path.join(WEB_DIR, "app.js")
    STYLES_CSS_PATH = os.path.join(WEB_DIR, "styles.css")
    OUT_SP500_DASHBOARD = os.path.join(OUTPUT_DIR, "sp500_dashboard.html")

    if os.path.exists(INDEX_HTML_PATH) and os.path.exists(APP_JS_PATH) and os.path.exists(STYLES_CSS_PATH):
        try:
            with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
                html_content = f.read()
            with open(STYLES_CSS_PATH, "r", encoding="utf-8") as f:
                css_content = f.read()
            with open(APP_JS_PATH, "r", encoding="utf-8") as f:
                js_content = f.read()

            # Inline CSS
            html_standalone = html_content.replace(
                '<link rel="stylesheet" href="styles.css">',
                f'<style>\n{css_content}\n</style>'
            )

            # Inline Dataset & App JS
            data_js_inline = f"window.DASHBOARD_DATA = {json.dumps(cleaned_payload)};"
            script_block = f"<script>\n{data_js_inline}\n\n{js_content}\n</script>"

            # Replace script loader block at end of HTML
            if "<!-- Consolidated Data Payload JS" in html_standalone:
                split_pos = html_standalone.find("<!-- Consolidated Data Payload JS")
                html_standalone = html_standalone[:split_pos] + script_block + "\n</body>\n</html>"
            else:
                html_standalone = html_standalone.replace('</body>', f'{script_block}\n</body>')

            with open(OUT_SP500_DASHBOARD, "w", encoding="utf-8") as f:
                f.write(html_standalone)
            print(f"Successfully created Single Deliverable HTML: {OUT_SP500_DASHBOARD}")
        except Exception as e:
            print(f"Notice: Standalone HTML build notice ({e})")

    # 2. Upload Complete Package to GCS Bucket
    if os.path.exists(OUT_SP500_DASHBOARD):
        upload_file_to_gcs(OUT_SP500_DASHBOARD, GCS_BUCKET_NAME, "outputs/sp500_dashboard.html")
        
    upload_file_to_gcs(DASHBOARD_JSON_PATH, GCS_BUCKET_NAME, "outputs/raw_json_datasets/dashboard_data.json")
    upload_file_to_gcs(CORR_JSON_PATH, GCS_BUCKET_NAME, "outputs/raw_json_datasets/correlation_results.json")
    upload_file_to_gcs(REG_JSON_PATH, GCS_BUCKET_NAME, "outputs/raw_json_datasets/linear_regression_results.json")
    upload_file_to_gcs(SURV_JSON_PATH, GCS_BUCKET_NAME, "outputs/raw_json_datasets/survival_results.json")

    PLOT_CORR = os.path.join(REPORTS_DIR, "plot_correlation_heatmap.png")
    PLOT_REG = os.path.join(REPORTS_DIR, "plot_actual_vs_predicted.png")
    PLOT_SURV = os.path.join(REPORTS_DIR, "plot_survival_curves.png")

    if os.path.exists(PLOT_CORR):
        upload_file_to_gcs(PLOT_CORR, GCS_BUCKET_NAME, "outputs/reports/plot_correlation_heatmap.png")
    if os.path.exists(PLOT_REG):
        upload_file_to_gcs(PLOT_REG, GCS_BUCKET_NAME, "outputs/reports/plot_actual_vs_predicted.png")
    if os.path.exists(PLOT_SURV):
        upload_file_to_gcs(PLOT_SURV, GCS_BUCKET_NAME, "outputs/reports/plot_survival_curves.png")

    print(f"Total historical stock price records (monthly): {len(stock_records)}")
    print(f"Total historical stock price records (daily): {len(stock_daily_records)}")
    print(f"Total macro indicator records: {len(macro_records)}")
    print("=" * 65)
    return cleaned_payload

if __name__ == "__main__":
    export_all_dashboard_data()



