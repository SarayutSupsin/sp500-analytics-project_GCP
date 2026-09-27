# ==============================================================================
# [วิธีคิดที่ 1 / METHOD 1]: Pearson Correlation Analysis (Ticker Level)
# หมวดหมู่: สถิติสำรวจความสัมพันธ์เชิงเส้นเบื้องต้น (Exploratory Data Analysis)
# หน้าที่: คำนวณค่าสัมประสิทธิ์สหสัมพันธ์ Pearson (ค่า r) ระหว่าง % ผลตอบแทนรายเดือน
#        ของหุ้นรายตัวทั้ง 20 หุ้น เทียบกับ อัตราเงินเฟ้อ (CPI % YoY) และอัตราดอกเบี้ย Fed Rate
#        โดยใช้ฟังก์ชัน CORR() ของ BigQuery SQL บน GCP โดยตรง
# ==============================================================================

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path Settings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
DATASETS_DIR = os.path.join(OUTPUT_DIR, "raw_json_datasets")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

PLOT_OUTPUT_PATH = os.path.join(REPORTS_DIR, "plot_correlation_heatmap.png")
JSON_OUTPUT_PATH = os.path.join(DATASETS_DIR, "correlation_results.json")

AI_TECH_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "ORCL", "AMD", "AVGO", "AMZN", "AAPL", "QCOM"]
STAPLES_TICKERS = ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "GIS", "TGT", "SYY"]

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

def run_correlation_analysis():
    """
    [METHOD 1] Pure GCP BigQuery SQL Pearson Correlation Analysis (Ticker Level)
    1. Executes BigQuery SQL CORR() function natively inside BigQuery Data Warehouse.
    2. Calculates Pearson Correlation (r) per ticker against CPI Inflation & Fed Rate.
    3. Saves plot_correlation_heatmap.png and correlation_results.json for Web Dashboard.
    """
    print("=" * 65)
    print("Step 4.4.1: Pure GCP BigQuery SQL Pearson Correlation Pipeline")
    print("=" * 65)
    
    from google.cloud import bigquery
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    
    # BigQuery SQL Query using Native CORR() Aggregate Function
    bq_corr_sql = f"""
    WITH stock_returns AS (
        SELECT 
            Ticker,
            FORMAT_DATE('%Y-%m', Date) AS YearMonth,
            (Close - LAG(Close, 1) OVER (PARTITION BY Ticker ORDER BY Date)) / NULLIF(LAG(Close, 1) OVER (PARTITION BY Ticker ORDER BY Date), 0) * 100 AS Monthly_Return
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.fact_stock_prices`
        WHERE Ticker != '^GSPC'
    ),
    macro_indicators AS (
        SELECT 
            SUBSTR(CAST(Date AS STRING), 1, 7) AS YearMonth,
            CPI_Inflation_YoY,
            Fed_Rate
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_inflation_rates`
    )
    SELECT 
        s.Ticker,
        CORR(s.Monthly_Return, m.CPI_Inflation_YoY) AS Corr_CPI,
        CORR(s.Monthly_Return, m.Fed_Rate) AS Corr_FedRate,
        COUNT(*) AS Sample_Count
    FROM stock_returns s
    JOIN macro_indicators m ON s.YearMonth = m.YearMonth
    WHERE s.Monthly_Return IS NOT NULL 
      AND m.CPI_Inflation_YoY IS NOT NULL 
      AND m.Fed_Rate IS NOT NULL
    GROUP BY s.Ticker
    ORDER BY s.Ticker
    """
    
    print("[1/2] Executing BigQuery SQL CORR() Query on GCP Data Warehouse...")
    df_res = bq_client.query(bq_corr_sql).to_dataframe()
    
    results = []
    for _, r in df_res.iterrows():
        ticker = r["Ticker"]
        sector = "AI-Tech" if ticker in AI_TECH_TICKERS else "Consumer Staples"
        results.append({
            "Ticker": ticker,
            "Sector": sector,
            "Corr_CPI": round(float(r["Corr_CPI"]), 4) if pd.notnull(r["Corr_CPI"]) else 0.0,
            "Corr_FedRate": round(float(r["Corr_FedRate"]), 4) if pd.notnull(r["Corr_FedRate"]) else 0.0,
            "Sample_Count": int(r["Sample_Count"])
        })
        
    df_res_formatted = pd.DataFrame(results)
    
    # Save Heatmap Image
    plt.figure(figsize=(12, 8))
    pivot_df = df_res_formatted.set_index("Ticker")[["Corr_CPI", "Corr_FedRate"]]
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0, linewidths=0.5)
    plt.title("Pearson Correlation Heatmap: Stock Monthly Returns vs Macro Indicators\n(20 Selected S&P 500 Tickers: 2021-2026)", fontsize=13, pad=15)
    plt.ylabel("Ticker (GICS Sector)", fontsize=11)
    plt.xlabel("Macroeconomic Indicators", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close()
    
    print(f"[Plot Output] Saved Correlation Heatmap Image: {PLOT_OUTPUT_PATH}")
    
    # Export JSON
    summary_data = {
        "status": "SUCCESS",
        "ticker_count": len(results),
        "results": results
    }
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"[GCP Output] Successfully saved Pure BigQuery SQL Correlation JSON to {JSON_OUTPUT_PATH}")
    return summary_data

if __name__ == "__main__":
    run_correlation_analysis()
