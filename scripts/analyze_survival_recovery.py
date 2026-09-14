import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
STOCK_CSV_PATH = os.path.join(DATA_DIR, "stock_prices_5y.csv")
PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "plot_survival_curves.png")
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "survival_results.json")

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

def run_survival_analysis():
    """
    Step 5: Kaplan-Meier Survival Analysis & Log-Rank Test
    1. Reads stock prices directly from GCP BigQuery Data Warehouse.
    2. Identifies Drawdown events (>10% drop from peak).
    3. Measures recovery duration in trading days (Duration Days) and censorship (Event=1 if recovered, 0 if not).
    4. Computes Kaplan-Meier survival curves and Log-Rank Test p-value.
    5. Saves plot_survival_curves.png and survival_results.json.
    """
    print("=" * 65)
    print("Step 5: Kaplan-Meier Survival Analysis & Log-Rank Test")
    print("=" * 65)
    
    try:
        from google.cloud import bigquery
        bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        df_stock = bq_client.query(f"SELECT * FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_ID}.fact_stock_prices`").to_dataframe()
        print(f"[BigQuery Data Warehouse] Successfully queried fact_stock_prices directly from GCP Cloud Dataset '{BIGQUERY_DATASET_ID}'")
    except Exception as e:
        print(f"[Local Fallback] BigQuery query notice ({e}), loading local CSV file...")
        if not os.path.exists(STOCK_CSV_PATH):
            raise FileNotFoundError("stock_prices_5y.csv missing. Run ingest_raw_data.py first.")
        df_stock = pd.read_csv(STOCK_CSV_PATH)
    df_stock = df_stock[df_stock["Ticker"] != "^GSPC"].copy()
    df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    df_stock = df_stock.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    
def process_survival_dataset(df_stock_subset):
    events_list = []
    
    for ticker in df_stock_subset["Ticker"].unique():
        sub = df_stock_subset[df_stock_subset["Ticker"] == ticker].copy()
        sub["Date"] = pd.to_datetime(sub["Date"])
        sub = sub.sort_values("Date").reset_index(drop=True)
        
        if len(sub) < 5:
            continue
        sector = "AI-Tech" if ticker in AI_TECH_TICKERS else "Consumer Staples"
        
        peak_price = sub["Close"].iloc[0]
        in_drawdown = False
        start_date = None
        
        for i in range(len(sub)):
            price = sub["Close"].iloc[i]
            cur_date = sub["Date"].iloc[i]
            
            if price >= peak_price:
                if in_drawdown:
                    duration = (cur_date - start_date).days
                    events_list.append({
                        "Ticker": ticker,
                        "Sector": sector,
                        "Duration_Days": max(1, duration),
                        "Event": 1
                    })
                    in_drawdown = False
                peak_price = price
            else:
                dd = (peak_price - price) / peak_price
                if dd >= 0.10 and not in_drawdown:
                    in_drawdown = True
                    start_date = cur_date
                    
        if in_drawdown:
            end_date = sub["Date"].iloc[-1]
            duration = (end_date - start_date).days
            events_list.append({
                "Ticker": ticker,
                "Sector": sector,
                "Duration_Days": max(1, duration),
                "Event": 0
            })
            
    df_events = pd.DataFrame(events_list)
    if len(df_events) == 0:
        return {
            "logrank_p_value": 1.0,
            "statistically_significant": False,
            "tech_median_days": None,
            "staples_median_days": None,
            "tech_curve": {"timeline": [0], "survival_probability": [1.0]},
            "staples_curve": {"timeline": [0], "survival_probability": [1.0]},
            "ticker_level_recovery": []
        }
        
    kmf_tech = KaplanMeierFitter()
    kmf_staples = KaplanMeierFitter()
    
    df_tech = df_events[df_events["Sector"] == "AI-Tech"]
    df_staples = df_events[df_events["Sector"] == "Consumer Staples"]
    
    if len(df_tech) > 0:
        kmf_tech.fit(df_tech["Duration_Days"], event_observed=df_tech["Event"], label="AI-Tech Sector")
    if len(df_staples) > 0:
        kmf_staples.fit(df_staples["Duration_Days"], event_observed=df_staples["Event"], label="Consumer Staples Sector")
        
    p_value = 1.0
    if len(df_tech) > 0 and len(df_staples) > 0:
        lr_res = logrank_test(
            df_tech["Duration_Days"], df_staples["Duration_Days"],
            event_observed_A=df_tech["Event"], event_observed_B=df_staples["Event"]
        )
        p_value = float(lr_res.p_value)
        
    ticker_medians = []
    all_tickers = sorted(df_stock_subset["Ticker"].unique())
    
    for ticker in all_tickers:
        sub_df = df_stock_subset[df_stock_subset["Ticker"] == ticker].copy()
        sub_df["Date"] = pd.to_datetime(sub_df["Date"])
        sub_df = sub_df.sort_values("Date").reset_index(drop=True)
        
        latest_close = sub_df["Close"].iloc[-1]
        all_time_peak = sub_df["Close"].max()
        current_dd = ((latest_close - all_time_peak) / all_time_peak) * 100.0 if all_time_peak > 0 else 0.0
        
        sub_df["CumMax"] = sub_df["Close"].cummax()
        sub_df["DD_Pct"] = ((sub_df["Close"] - sub_df["CumMax"]) / sub_df["CumMax"]) * 100.0
        max_dd_pct = round(float(sub_df["DD_Pct"].min()), 2)
        
        sub_t = df_events[df_events["Ticker"] == ticker] if len(df_events) > 0 else pd.DataFrame()
        sector = "AI-Tech" if ticker in AI_TECH_TICKERS else "Consumer Staples"
        
        if len(sub_t) > 0:
            kmf_t = KaplanMeierFitter()
            kmf_t.fit(sub_t["Duration_Days"], event_observed=sub_t["Event"])
            med = kmf_t.median_survival_time_
            
            t_timeline = [int(x) for x in kmf_t.survival_function_.index]
            t_probs = [round(float(y), 4) for y in kmf_t.survival_function_["KM_estimate"]]
            
            recovered_durations = sub_t[sub_t["Event"] == 1]["Duration_Days"].tolist()
            longest = max(recovered_durations) if len(recovered_durations) > 0 else (max(sub_t["Duration_Days"]) if len(sub_t) > 0 else 0)
            shortest = min(recovered_durations) if len(recovered_durations) > 0 else (min(sub_t["Duration_Days"]) if len(sub_t) > 0 else 0)
            
            ticker_medians.append({
                "Ticker": ticker,
                "Sector": sector,
                "Current_Drawdown_Pct": round(float(current_dd), 2),
                "Max_Historical_Drawdown_Pct": max_dd_pct,
                "Total_Drawdown_Events": len(sub_t),
                "Median_Recovery_Days": float(med) if (med is not None and not np.isinf(med) and not np.isnan(med)) else None,
                "Longest_Recovery_Days": int(longest),
                "Shortest_Recovery_Days": int(shortest),
                "Curve": {
                    "timeline": t_timeline,
                    "survival_probability": t_probs
                }
            })
        else:
            ticker_medians.append({
                "Ticker": ticker,
                "Sector": sector,
                "Current_Drawdown_Pct": round(float(current_dd), 2),
                "Max_Historical_Drawdown_Pct": max_dd_pct,
                "Total_Drawdown_Events": 0,
                "Median_Recovery_Days": None,
                "Longest_Recovery_Days": 0,
                "Shortest_Recovery_Days": 0,
                "Curve": {
                    "timeline": [0],
                    "survival_probability": [1.0]
                }
            })
            
    tech_med = kmf_tech.median_survival_time_ if len(df_tech) > 0 else None
    staples_med = kmf_staples.median_survival_time_ if len(df_staples) > 0 else None
    
    tech_curve = {
        "timeline": [int(x) for x in kmf_tech.survival_function_.index] if len(df_tech) > 0 else [0],
        "survival_probability": [round(float(y), 4) for y in kmf_tech.survival_function_["AI-Tech Sector"]] if len(df_tech) > 0 else [1.0]
    }
    staples_curve = {
        "timeline": [int(x) for x in kmf_staples.survival_function_.index] if len(df_staples) > 0 else [0],
        "survival_probability": [round(float(y), 4) for y in kmf_staples.survival_function_["Consumer Staples Sector"]] if len(df_staples) > 0 else [1.0]
    }

    return {
        "logrank_p_value": round(p_value, 4),
        "statistically_significant": bool(p_value < 0.05),
        "tech_median_days": float(tech_med) if (tech_med is not None and not np.isinf(tech_med) and not np.isnan(tech_med)) else None,
        "staples_median_days": float(staples_med) if (staples_med is not None and not np.isinf(staples_med) and not np.isnan(staples_med)) else None,
        "tech_curve": tech_curve,
        "staples_curve": staples_curve,
        "ticker_level_recovery": ticker_medians
    }

def run_survival_analysis():
    print("=" * 65)
    print("Step 5: Kaplan-Meier Survival Analysis & Log-Rank Test")
    print("=" * 65)
    
    if not os.path.exists(STOCK_CSV_PATH):
        raise FileNotFoundError("stock_prices_5y.csv missing. Run ingest_raw_data.py first.")
        
    df_stock = pd.read_csv(STOCK_CSV_PATH)
    df_stock = df_stock[df_stock["Ticker"] != "^GSPC"].copy()
    df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    df_stock = df_stock.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    
    # Process ALL (5 Years)
    res_all = process_survival_dataset(df_stock)
    
    # Process 2021-2022 (2 Years)
    df_21_22 = df_stock[(df_stock["Date"] >= "2021-01-01") & (df_stock["Date"] <= "2022-12-31")].copy().reset_index(drop=True)
    res_21_22 = process_survival_dataset(df_21_22)
    
    # Process 2023-2024 (2 Years)
    df_23_24 = df_stock[(df_stock["Date"] >= "2023-01-01") & (df_stock["Date"] <= "2024-12-31")].copy().reset_index(drop=True)
    res_23_24 = process_survival_dataset(df_23_24)
    
    # Process 2025-2026 (2 Years)
    df_25_26 = df_stock[df_stock["Date"] >= "2025-01-01"].copy().reset_index(drop=True)
    res_25_26 = process_survival_dataset(df_25_26)
    
    summary = {
        "status": "SUCCESS",
        "logrank_p_value": res_all["logrank_p_value"],
        "statistically_significant": res_all["statistically_significant"],
        "tech_median_days": res_all["tech_median_days"],
        "staples_median_days": res_all["staples_median_days"],
        "tech_curve": res_all["tech_curve"],
        "staples_curve": res_all["staples_curve"],
        "ticker_level_recovery": res_all["ticker_level_recovery"],
        "survival_recovery_by_period": {
            "ALL": res_all,
            "2021-2022": res_21_22,
            "2023-2024": res_23_24,
            "2025-2026": res_25_26
        }
    }
    
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Saved survival results JSON with multi-period data: {JSON_OUTPUT_PATH}")
    print("=" * 65)
    return summary

if __name__ == "__main__":
    run_survival_analysis()
