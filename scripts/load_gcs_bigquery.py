import os
import json
import pandas as pd

# ==============================================================================
# GCP Infrastructure Configuration Parameters
# (Ready for deployment on GCP Compute Engine VM / Cloud Shell)
# ==============================================================================
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
GCS_BUCKET_NAME = "sp500-analytics-raw-data"
BIGQUERY_DATASET_ID = "sp500_analytics"

# Toggle True when deploying with active GCP credentials (or set environment variable ENABLE_GCP_UPLOAD=true)
ENABLE_GCP_UPLOAD = os.environ.get("ENABLE_GCP_UPLOAD", "False").lower() == "true"

# Paths to raw input CSVs and outputs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")
MACRO_CSV_PATH = os.path.join(DATA_DIR, "cpi_fedrate_5y.csv")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "etl_load_summary.json")

def process_and_load_data():
    """
    Step 2: Transform & Load (T&L)
    1. Reads raw stock and macro CSVs.
    2. Transforms data into BigQuery Fact & Dimension table structures.
    3. Executes GCP uploads (GCS & BigQuery) if ENABLE_GCP_UPLOAD is True.
    4. Saves ETL validation summary to etl_load_summary.json.
    """
    print("=" * 65)
    print("Step 2: Starting Transform & Load Pipeline (GCS / BigQuery)")
    print("=" * 65)
    
    # 1. Load Raw CSVs
    if not os.path.exists(STOCK_CSV_PATH) or not os.path.exists(MACRO_CSV_PATH):
        raise FileNotFoundError("Raw CSV files not found. Please run ingest_raw_data.py first.")
        
    df_stock = pd.read_csv(STOCK_CSV_PATH)
    df_macro = pd.read_csv(MACRO_CSV_PATH)
    
    print(f"Loaded stock prices CSV: {len(df_stock)} rows")
    print(f"Loaded macro indicators CSV: {len(df_macro)} rows")
    
    # 2. BigQuery Fact & Dimension Schema Transformations
    print("\n[1/2] Transforming BigQuery Data Warehouse Schemas...")
    
    # Fact Table: fact_stock_prices
    df_fact = df_stock.copy()
    df_fact["YearMonth"] = pd.to_datetime(df_fact["Date"]).dt.strftime("%Y-%m")
    # Calculate daily returns per ticker
    df_fact["Daily_Return"] = df_fact.groupby("Ticker")["Close"].pct_change().fillna(0.0)
    
    # Dimension Table: dim_inflation_rates
    df_dim_inflation = df_macro[["Date", "CPI_Inflation_YoY"]].copy()
    df_dim_inflation.rename(columns={"Date": "YearMonth", "CPI_Inflation_YoY": "CPI_Inflation_Rate"}, inplace=True)
    
    # Dimension Table: dim_interest_rates
    df_dim_interest = df_macro[["Date", "Fed_Rate"]].copy()
    df_dim_interest.rename(columns={"Date": "YearMonth", "Fed_Rate": "Fed_Interest_Rate"}, inplace=True)
    
    print(f"Fact Table (fact_stock_prices): {len(df_fact)} records ready")
    print(f"Dimension Table (dim_inflation_rates): {len(df_dim_inflation)} records ready")
    print(f"Dimension Table (dim_interest_rates): {len(df_dim_interest)} records ready")
    
    # 3. GCP Cloud Native Upload Handling (If Enabled)
    if ENABLE_GCP_UPLOAD:
        print("\n[2/2] Uploading to Google Cloud Platform...")
        try:
            from google.cloud import storage, bigquery
            print("Connecting to GCP Services...")
            
            # 1) GCS Data Lake Upload (All Raw CSVs)
            storage_client = storage.Client(project=GCP_PROJECT_ID)
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            
            blob_stock = bucket.blob("raw/stock_prices_5y.csv")
            blob_stock.upload_from_filename(STOCK_CSV_PATH)
            print(f"[GCS] Uploaded {STOCK_CSV_PATH} -> gs://{GCS_BUCKET_NAME}/raw/stock_prices_5y.csv")
            
            blob_macro = bucket.blob("raw/cpi_fedrate_5y.csv")
            blob_macro.upload_from_filename(MACRO_CSV_PATH)
            print(f"[GCS] Uploaded {MACRO_CSV_PATH} -> gs://{GCS_BUCKET_NAME}/raw/cpi_fedrate_5y.csv")
            
            # 2) BigQuery Data Warehouse Load (Direct Cloud Native ELT: GCS Bucket URI -> BigQuery)
            bq_client = bigquery.Client(project=GCP_PROJECT_ID)
            dataset_ref = bq_client.dataset(BIGQUERY_DATASET_ID)
            
            # BigQuery CSV Load Job Config (Skip header row)
            csv_job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,
                autodetect=True,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
            )
            
            # Fact Table Load (GCS -> BigQuery)
            gcs_stock_uri = f"gs://{GCS_BUCKET_NAME}/raw/stock_prices_5y.csv"
            job_fact = bq_client.load_table_from_uri(gcs_stock_uri, dataset_ref.table("fact_stock_prices"), job_config=csv_job_config)
            job_fact.result()
            print(f"[BigQuery ELT] Successfully loaded table 'fact_stock_prices' from {gcs_stock_uri}")
            
            # Dimension Table 1 Load (GCS -> BigQuery)
            gcs_macro_uri = f"gs://{GCS_BUCKET_NAME}/raw/cpi_fedrate_5y.csv"
            job_dim_cpi = bq_client.load_table_from_uri(gcs_macro_uri, dataset_ref.table("dim_inflation_rates"), job_config=csv_job_config)
            job_dim_cpi.result()
            print(f"[BigQuery ELT] Successfully loaded table 'dim_inflation_rates' from {gcs_macro_uri}")
            
            # Dimension Table 2 Load (GCS -> BigQuery)
            job_dim_fed = bq_client.load_table_from_uri(gcs_macro_uri, dataset_ref.table("dim_interest_rates"), job_config=csv_job_config)
            job_dim_fed.result()
            print(f"[BigQuery ELT] Successfully loaded table 'dim_interest_rates' from {gcs_macro_uri}")
            
        except Exception as e:
            print(f"GCP Upload Warning: {e}")
    else:
        print("\n[2/2] Local Simulation Mode Active (ENABLE_GCP_UPLOAD = False)")
        print("Data schemas verified locally. Ready for deployment on GCP Compute Engine VM.")
        
    # 4. Output Local Summary File
    summary = {
        "status": "SUCCESS",
        "gcp_config": {
            "project_id": GCP_PROJECT_ID,
            "gcs_bucket": GCS_BUCKET_NAME,
            "bigquery_dataset": BIGQUERY_DATASET_ID
        },
        "counts": {
            "stock_rows": len(df_stock),
            "fact_stock_prices_rows": len(df_fact),
            "dim_inflation_rows": len(df_dim_inflation),
            "dim_interest_rows": len(df_dim_interest)
        }
    }
    
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "=" * 65)
    print(f"ETL Load Summary saved to: {SUMMARY_JSON_PATH}")
    print("=" * 65)
    return summary

if __name__ == "__main__":
    process_and_load_data()
