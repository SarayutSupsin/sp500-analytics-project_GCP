# ==============================================================================
# [วิธีคิดที่ 1 / METHOD 1]: Pearson Correlation Analysis (Ticker Level)
# หมวดหมู่: สถิติสำรวจความสัมพันธ์เชิงเส้นเบื้องต้น (Exploratory Data Analysis)
# หน้าที่: คำนวณค่าสัมประสิทธิ์สหสัมพันธ์ Pearson (ค่า r) ระหว่าง % ผลตอบแทนรายเดือน
#        ของหุ้นรายตัวทั้ง 20 หุ้น เทียบกับ อัตราเงินเฟ้อ (CPI % YoY) และอัตราดอกเบี้ย Fed Rate
# ==============================================================================

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# กำหนดเส้นทางไฟล์ข้อมูลดิบ (Inputs) และไฟล์ผลลัพธ์ (Outputs)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")
MACRO_CSV_PATH = os.path.join(DATA_DIR, "cpi_fedrate_5y.csv")
PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "plot_correlation_heatmap.png")
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
    ขั้นตอนวิเคราะห์วิธีคิดที่ 1: Pearson Correlation Analysis
    1. อ่านข้อมูลดิบราคาปิดรายเดือนและปัจจัยมหภาค จาก GCP BigQuery Data Warehouse
    2. คำนวณ % ผลตอบแทนรายเดือน (Monthly Return %) แยกรายหุ้นทั้ง 20 ตัว
    3. รวมตารางข้อมูลด้วยคีย์ YearMonth (จับคู่รายเดือน)
    4. คำนวณค่า Pearson Correlation (ค่า r) รายหุ้น (Ticker Level)
    5. ส่งออกไฟล์รูปภาพ Heatmap และไฟล์สรุป correlation_results.json
    """
    print("=" * 65)
    print("วิธีคิดที่ 1 [METHOD 1]: Pearson Correlation Analysis (Ticker Level)")
    print("=" * 65)
    
    try:
        from google.cloud import bigquery
        bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        df_stock = bq_client.query(f"SELECT * FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.fact_stock_prices`").to_dataframe()
        df_macro = bq_client.query(f"SELECT * FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.dim_inflation_rates`").to_dataframe()
        print(f"[BigQuery Data Warehouse] Successfully queried fact_stock_prices & dim_inflation_rates directly from GCP Cloud Dataset '{BIGQUERY_DATASET_ID}'")
    except Exception as e:
        print(f"[Local Fallback] BigQuery query notice ({e}), loading local CSV files...")
        if not os.path.exists(STOCK_CSV_PATH) or not os.path.exists(MACRO_CSV_PATH):
            raise FileNotFoundError("ไม่พบไฟล์ข้อมูลดิบ CSV กรุณารัน ingest_raw_data.py ก่อน")
        df_stock = pd.read_csv(STOCK_CSV_PATH)
        df_macro = pd.read_csv(MACRO_CSV_PATH)
    
    # กรองดัชนีอ้างอิงตลาด S&P 500 (^GSPC) ออกเพื่อคำนวณเฉพาะหุ้นรายตัว 20 ตัว
    df_stock = df_stock[df_stock["Ticker"] != "^GSPC"].copy()
    
    # --- สเต็ปที่ 1: คำนวณ % ผลตอบแทนรายเดือน (Monthly Return %) แยกรายหุ้น ---
    df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    df_stock["YearMonth"] = df_stock["Date"].dt.strftime("%Y-%m")
    
    # เรียงลำดับตามวันที่เพื่อคำนวณ % การเปลี่ยนแปลงของราคาปิด (pct_change)
    df_stock = df_stock.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df_stock["Monthly_Return"] = df_stock.groupby("Ticker")["Close"].pct_change() * 100.0
    
    # ลบแถวแรกของแต่ละหุ้นที่เป็น NaN ออก (แถวแรกไม่มีเดือนก่อนหน้าให้เปรียบเทียบ)
    df_stock = df_stock.dropna(subset=["Monthly_Return"])
    
    # --- สเต็ปที่ 2: จับคู่ตารางข้อมูลรายเดือนเข้ากับ CPI YoY และ Fed Rate ---
    df_macro["YearMonth"] = pd.to_datetime(df_macro["Date"]).dt.strftime("%Y-%m")
    df_merged = pd.merge(df_stock, df_macro, on="YearMonth", how="inner", suffixes=("_stock", "_macro"))
    
    # --- สเต็ปที่ 3: คำนวณ Pearson Correlation (ค่า r) แยกราย Ticker ทั้ง 20 ตัว ---
    results = []
    tickers_list = AI_TECH_TICKERS + STAPLES_TICKERS
    
    for ticker in tickers_list:
        sub = df_merged[df_merged["Ticker"] == ticker]
        if len(sub) > 5:
            # คำนวณค่า r ระหว่าง Return % กับ CPI และ Fed Rate จากจุดข้อมูลรายเดือนทั้งหมด (n งวด)
            corr_cpi = sub["Monthly_Return"].corr(sub["CPI_Inflation_YoY"])
            corr_fed = sub["Monthly_Return"].corr(sub["Fed_Rate"])
            sector = "AI-Tech" if ticker in AI_TECH_TICKERS else "Consumer Staples"
            
            results.append({
                "Ticker": ticker,
                "Sector": sector,
                "Corr_CPI": round(float(corr_cpi), 4) if not np.isnan(corr_cpi) else 0.0,
                "Corr_FedRate": round(float(corr_fed), 4) if not np.isnan(corr_fed) else 0.0,
                "Sample_Count": len(sub)
            })
            
    df_res = pd.DataFrame(results)
    
    # --- สเต็ปที่ 4: บันทึกรูปภาพ Correlation Heatmap ---
    plt.figure(figsize=(12, 8))
    pivot_df = df_res.set_index("Ticker")[["Corr_CPI", "Corr_FedRate"]]
    
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0, linewidths=0.5)
    plt.title("Pearson Correlation Heatmap: Stock Monthly Returns vs Macro Indicators\n(20 Selected S&P 500 Tickers: 2021-2026)", fontsize=13, pad=15)
    plt.ylabel("Ticker (GICS Sector)", fontsize=11)
    plt.xlabel("Macroeconomic Indicators", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close()
    
    print(f"บันทึกรูปภาพ Correlation Heatmap สำเร็จ: {PLOT_OUTPUT_PATH}")
    
    # --- สเต็ปที่ 5: ส่งออกไฟล์สรุปผลลัพธ์ JSON สำหรับ Dashboard ---
    summary_data = {
        "status": "SUCCESS",
        "ticker_count": len(df_res),
        "results": results
    }
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"บันทึกไฟล์สรุปผลลัพธ์ JSON สำเร็จ: {JSON_OUTPUT_PATH}")
    print("=" * 65)
    return summary_data

if __name__ == "__main__":
    run_correlation_analysis()

