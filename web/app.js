// Global Dataset Holder
let dashboardData = null;
let currentActiveTab = 1;

// Chart Instance Trackers
let chartOlsPrediction = null;
let chartSurvivalCurve = null;

// Shared helper: sort stock rows so sectors group together (AI-Tech first, then Staples), then ticker A-Z
const SECTOR_ORDER = { "AI-Tech": 0, "Consumer Staples": 1 };
function sortStockRows(list) {
    return (list || []).slice().sort((a, b) => {
        const sa = SECTOR_ORDER[a.Sector] !== undefined ? SECTOR_ORDER[a.Sector] : 99;
        const sb = SECTOR_ORDER[b.Sector] !== undefined ? SECTOR_ORDER[b.Sector] : 99;
        if (sa !== sb) return sa - sb;
        return String(a.Ticker || "").localeCompare(String(b.Ticker || ""));
    });
}

// Initialize Application on Page Load
document.addEventListener("DOMContentLoaded", () => {
    // 1. Immediately restore saved tab UI synchronously (0ms delay, before fetch)
    currentActiveTab = getSavedTab();
    activateTabUI(currentActiveTab);

    // 2. Fetch dataset asynchronously in background
    fetchDashboardData();
});

window.initDashboard = function() {
    fetchDashboardData();
};


// Helper: Determine saved active tab from URL hash or localStorage
function getSavedTab() {
    try {
        const hashMatch = window.location.hash.match(/#tab-([1-3])/);
        if (hashMatch) {
            return parseInt(hashMatch[1]);
        }
        const stored = localStorage.getItem("activeTab");
        if (stored) {
            const parsed = parseInt(stored);
            if (parsed >= 1 && parsed <= 3) return parsed;
        }
    } catch (e) {}
    return 1;
}

// Synchronously activate DOM classes for specified tab
function activateTabUI(tabNumber) {
    currentActiveTab = tabNumber;
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

    const btnEl = document.getElementById(`tab-btn-${tabNumber}`);
    const contentEl = document.getElementById(`tab-content-${tabNumber}`);
    if (btnEl) btnEl.classList.add("active");
    if (contentEl) contentEl.classList.add("active");

    try {
        localStorage.setItem("activeTab", tabNumber);
        if (window.history && window.history.replaceState) {
            window.history.replaceState(null, null, `#tab-${tabNumber}`);
        }
    } catch (e) {}
}

// Fetch Consolidated Data Payload (Hybrid: HTTP Fetch with Offline window.DASHBOARD_DATA Fallback)
async function fetchDashboardData() {
    // 1. Primary Fallback: Check if window.DASHBOARD_DATA is pre-loaded via script tag
    if (window.DASHBOARD_DATA) {
        dashboardData = window.DASHBOARD_DATA;
        onDataReady();
        return;
    }

    // 2. HTTP Server fetch() attempt
    try {
        const response = await fetch("../outputs/dashboard_data.json");
        if (!response.ok) {
            throw new Error(`HTTP Error status: ${response.status}`);
        }
        dashboardData = await response.json();
        onDataReady();
    } catch (error) {
        console.error("Failed to load dashboard dataset:", error);
    }
}

// Data Ready Callback: Populate views and render charts for active tab
function onDataReady() {
    initTab1();
    initTab2Controls();
    initTab3Data();
    
    // Render chart for currently active tab
    if (currentActiveTab === 2) {
        renderTab2Charts();
    } else if (currentActiveTab === 3) {
        const tickerSel = document.getElementById("select-ticker-survival");
        const tickerVal = tickerSel ? tickerSel.value : "ALL";
        renderTab3Charts(null, tickerVal);
    }
}

// Tab Switcher Handler (On user button click)
function switchTab(tabNumber) {
    activateTabUI(tabNumber);

    if (dashboardData) {
        setTimeout(() => {
            if (tabNumber === 2) {
                renderTab2Charts();
            } else if (tabNumber === 3) {
                const tickerSel = document.getElementById("select-ticker-survival");
                const tickerVal = tickerSel ? tickerSel.value : "ALL";
                renderTab3Charts(null, tickerVal);
            }
        }, 50);
    }
}

// ==============================================================================
// [ส่วนกลาง / SHARED INFRASTRUCTURE]: Helper คำนวณสูตร Pearson Correlation เพียวๆ
// หน้าที่: คำนวณค่าสัมประสิทธิ์สหสัมพันธ์ Pearson (ค่า r) จากจุดข้อมูล 2 ชุด (xArr, yArr)
// สูตร: r = Covariance(X, Y) / (StdDev(X) * StdDev(Y))
// ==============================================================================
function computePearsonCorrelation(xArr, yArr) {
    if (!xArr || !yArr || xArr.length < 3 || xArr.length !== yArr.length) return null;
    
    // คัดกรองเอาเฉพาะคู่ข้อมูลที่สมบูรณ์ ไม่เป็น null หรือ NaN
    const validPairs = [];
    for (let i = 0; i < xArr.length; i++) {
        if (xArr[i] !== null && xArr[i] !== undefined && !isNaN(xArr[i]) &&
            yArr[i] !== null && yArr[i] !== undefined && !isNaN(yArr[i])) {
            validPairs.push([xArr[i], yArr[i]]);
        }
    }
    
    if (validPairs.length < 3) return null;
    const n = validPairs.length;
    
    // 1. คำนวณค่าเฉลี่ย (Mean) ของ X และ Y
    const meanX = validPairs.reduce((sum, p) => sum + p[0], 0) / n;
    const meanY = validPairs.reduce((sum, p) => sum + p[1], 0) / n;

    let num = 0;   // Covariance numerator
    let denX = 0;  // Variance X
    let denY = 0;  // Variance Y

    // 2. คำนวณผลรวมความเบี่ยงเบนจากค่าเฉลี่ย
    for (let i = 0; i < n; i++) {
        const dx = validPairs[i][0] - meanX;
        const dy = validPairs[i][1] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }

    if (denX === 0 || denY === 0) return null;
    // 3. คืนค่าสัมประสิทธิ์ Pearson correlation (ทศนิยม 4 ตำแหน่ง)
    return parseFloat((num / Math.sqrt(denX * denY)).toFixed(4));
}

// ==============================================================================
// [วิธีคิดที่ 1 / METHOD 1]: PEARSON CORRELATION ANALYSIS & DYNAMIC FILTERS
// หน้าที่: แสดงผลตาราง Heatmap Correlation และกรองข้อมูลไดนามิกตามช่วงปีที่เลือก
// ==============================================================================
function initTab1() {
    if (!dashboardData) return;
    applyTab1Filters();
}

function applyTab1Filters() {
    if (!dashboardData) return;
    
    const showTech = document.getElementById("filter-sector-tech") ? document.getElementById("filter-sector-tech").checked : true;
    const showStaples = document.getElementById("filter-sector-staples") ? document.getElementById("filter-sector-staples").checked : true;
    const yearRange = document.getElementById("select-year-range") ? document.getElementById("select-year-range").value : "ALL";

    const baseList = dashboardData.analytics.pearson_correlation || [];
    
    // Filter by sector
    let filteredTickers = baseList.filter(item => {
        if (item.Sector === "AI-Tech" && !showTech) return false;
        if (item.Sector === "Consumer Staples" && !showStaples) return false;
        return true;
    });

    // Determine year range boundaries for specific economic regimes, 2-year windows, or single years
    let startYM = "2021-01";
    let endYM = "2026-12";
    if (yearRange === "2021-2022") {
        startYM = "2021-01"; endYM = "2022-12";
    } else if (yearRange === "2023-2024") {
        startYM = "2023-01"; endYM = "2024-12";
    } else if (yearRange === "2025-2026") {
        startYM = "2025-01"; endYM = "2026-12";
    } else if (yearRange === "2021") {
        startYM = "2021-01"; endYM = "2021-12";
    } else if (yearRange === "2022") {
        startYM = "2022-01"; endYM = "2022-12";
    } else if (yearRange === "2023") {
        startYM = "2023-01"; endYM = "2023-12";
    } else if (yearRange === "2024") {
        startYM = "2024-01"; endYM = "2024-12";
    } else if (yearRange === "2025") {
        startYM = "2025-01"; endYM = "2025-12";
    }

    const rawStock = dashboardData.raw_historical.stock_prices || [];
    const rawMacro = dashboardData.raw_historical.macro_indicators || [];

    // Filter macro data by selected period
    const cycleMacro = rawMacro.filter(m => {
        const ym = m.Date.substring(0, 7);
        return ym >= startYM && ym <= endYM;
    });

    // Compute dynamic empirical min, max, and averages from raw macro data
    const validCpi = cycleMacro.map(m => m.CPI_Inflation_YoY).filter(v => v !== null && v !== undefined && !isNaN(v));
    const validFed = cycleMacro.map(m => m.Fed_Rate).filter(v => v !== null && v !== undefined && !isNaN(v));
    
    let cpiRangeStr = "-";
    let fedRangeStr = "-";

    if (validCpi.length > 0) {
        const minCpi = Math.min(...validCpi).toFixed(2);
        const maxCpi = Math.max(...validCpi).toFixed(2);
        const avgCpi = (validCpi.reduce((a, b) => a + b, 0) / validCpi.length).toFixed(2);
        cpiRangeStr = `${minCpi}% - ${maxCpi}% (เฉลี่ย ${avgCpi}%)`;
    }

    if (validFed.length > 0) {
        const minFed = Math.min(...validFed).toFixed(2);
        const maxFed = Math.max(...validFed).toFixed(2);
        const avgFed = (validFed.reduce((a, b) => a + b, 0) / validFed.length).toFixed(2);
        fedRangeStr = `${minFed}% - ${maxFed}% (เฉลี่ย ${avgFed}%)`;
    }

    // Update empirical stat pill badges in UI
    const countEl = document.getElementById("sample-month-count");
    const cpiRangeEl = document.getElementById("sample-cpi-range");
    const fedRangeEl = document.getElementById("sample-fed-range");
    
    if (countEl) countEl.textContent = cycleMacro.length;
    if (cpiRangeEl) cpiRangeEl.textContent = cpiRangeStr;
    if (fedRangeEl) fedRangeEl.textContent = fedRangeStr;

    // If default 5-year period selected, render pre-calculated 5-year JSON results directly
    if (yearRange === "ALL") {
        renderTab1Heatmap(filteredTickers);
        return;
    }

    const cycleMacroMap = {};
    cycleMacro.forEach(m => {
        const ym = m.Date.substring(0, 7);
        cycleMacroMap[ym] = { cpi: m.CPI_Inflation_YoY, fed: m.Fed_Rate };
    });

    const cycleResults = filteredTickers.map(tInfo => {
        const ticker = tInfo.Ticker;
        const sector = tInfo.Sector;
        
        const tStock = rawStock.filter(s => s.Ticker === ticker).sort((a, b) => new Date(a.Date) - new Date(b.Date));
        
        const returnsArr = [];
        const cpiArr = [];
        const fedArr = [];

        for (let i = 1; i < tStock.length; i++) {
            const ym = tStock[i].Date.substring(0, 7);
            if (ym >= startYM && ym <= endYM && cycleMacroMap[ym]) {
                const prevClose = tStock[i - 1].Close;
                const currClose = tStock[i].Close;
                const ret = ((currClose - prevClose) / prevClose) * 100.0;
                
                returnsArr.push(ret);
                cpiArr.push(cycleMacroMap[ym].cpi);
                fedArr.push(cycleMacroMap[ym].fed);
            }
        }

        const corrCpi = computePearsonCorrelation(returnsArr, cpiArr);
        const corrFed = computePearsonCorrelation(returnsArr, fedArr);

        return {
            Ticker: ticker,
            Sector: sector,
            Corr_CPI: corrCpi,
            Corr_FedRate: corrFed
        };
    });

    renderTab1Heatmap(cycleResults);
}

function getHeatmapColorClass(val) {
    if (val === null || val === undefined || isNaN(val)) return "na-cell";
    if (val >= 0.45) return "positive-high";
    if (val >= 0.20) return "positive-med";
    if (val >= 0.05) return "positive-low";
    if (val <= -0.45) return "negative-high";
    if (val <= -0.20) return "negative-med";
    if (val <= -0.05) return "negative-low";
    return "neutral";
}

function renderTab1Heatmap(dataList) {
    const tbody = document.querySelector("#table-heatmap tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    sortStockRows(dataList).forEach(item => {
        const tr = document.createElement("tr");
        const sectorClass = item.Sector === "AI-Tech" ? "tag-tech" : "tag-staples";
        
        const cpiClass = getHeatmapColorClass(item.Corr_CPI);
        const fedClass = getHeatmapColorClass(item.Corr_FedRate);

        const formattedCpi = item.Corr_CPI === null || item.Corr_CPI === undefined ? "N/A" : (item.Corr_CPI >= 0 ? '+' + item.Corr_CPI : item.Corr_CPI);
        const formattedFed = item.Corr_FedRate === null || item.Corr_FedRate === undefined ? "N/A" : (item.Corr_FedRate >= 0 ? '+' + item.Corr_FedRate : item.Corr_FedRate);

        tr.innerHTML = `
            <td><strong>${item.Ticker}</strong></td>
            <td><span class="tag-sector ${sectorClass}">${item.Sector}</span></td>
            <td><div class="heatmap-cell ${cpiClass}">${formattedCpi}</div></td>
            <td><div class="heatmap-cell ${fedClass}">${formattedFed}</div></td>
        `;
        tbody.appendChild(tr);
    });
}

// ==============================================================================
// [วิธีคิดที่ 2 / METHOD 2]: OLS LINEAR REGRESSION MODEL (พยากรณ์ผลตอบแทนรายหุ้น)
// หน้าที่: แสดงสถิติความแม่นยำ R², MAE, ค่าความไว (Coefficients)
//        และวาดกราฟเปรียบเทียบ % ผลตอบแทนจริง vs % ผลพยากรณ์จากโมเดล OLS
// ==============================================================================
let currentTab2TableSector = "ALL";

function filterTab2TableSector(sector) {
    currentTab2TableSector = sector;
    
    const btnAll = document.getElementById("btn-ols-filter-all");
    const btnTech = document.getElementById("btn-ols-filter-tech");
    const btnStaples = document.getElementById("btn-ols-filter-staples");
    
    if (btnAll) btnAll.classList.toggle("active", sector === "ALL");
    if (btnTech) btnTech.classList.toggle("active", sector === "AI-Tech");
    if (btnStaples) btnStaples.classList.toggle("active", sector === "Consumer Staples");

    const periodSelect = document.getElementById("select-ols-year-range");
    const selectedPeriod = periodSelect && periodSelect.value ? periodSelect.value : "ALL";
    const regByPeriodMap = dashboardData ? (dashboardData.analytics.linear_regression_by_period || {}) : {};
    const regList = regByPeriodMap[selectedPeriod] || (dashboardData ? (dashboardData.analytics.linear_regression || []) : []);
    
    const selectEl = document.getElementById("select-ticker-ols");
    const selectedTicker = selectEl && selectEl.value ? selectEl.value : "NVDA";

    renderTab2Table(regList, selectedTicker);
}

function selectTickerFromTable(ticker) {
    const selectEl = document.getElementById("select-ticker-ols");
    if (selectEl) {
        selectEl.value = ticker;
    }
    renderTab2Charts();
}

function renderTab2Data() {
    renderTab2Charts();
}

function initTab2Controls() {
    if (!dashboardData) return;
    
    const select = document.getElementById("select-ticker-ols");
    if (!select) return;
    select.innerHTML = "";

    const regList = dashboardData.analytics.linear_regression || [];
    regList.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item.Ticker;
        opt.textContent = `${item.Ticker} (${item.Sector})`;
        select.appendChild(opt);
    });

    renderTab2Charts();
}

function renderTab2Table(dataList, selectedTicker) {
    const tbody = document.querySelector("#table-ols-metrics tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    let displayList = dataList || [];
    if (currentTab2TableSector && currentTab2TableSector !== "ALL") {
        displayList = displayList.filter(item => item.Sector === currentTab2TableSector);
    }

    sortStockRows(displayList).forEach(item => {
        const tr = document.createElement("tr");
        const sectorClass = item.Sector === "AI-Tech" ? "tag-tech" : "tag-staples";
        if (item.Ticker === selectedTicker) {
            tr.classList.add("row-active");
        }
        tr.onclick = () => selectTickerFromTable(item.Ticker);
        
        const coeffCpi = item.Coeff_CPI !== undefined ? item.Coeff_CPI : item.Coeff_Lag1_CPI;
        const coeffFed = item.Coeff_FedRate !== undefined ? item.Coeff_FedRate : item.Coeff_Lag1_FedRate;
        const mseVal = item.MSE !== undefined && item.MSE !== null ? item.MSE : '-';

        tr.innerHTML = `
            <td><strong>${item.Ticker}</strong></td>
            <td><span class="tag-sector ${sectorClass}">${item.Sector}</span></td>
            <td>${item.R2_Score}</td>
            <td>${item.MAE}%</td>
            <td>${mseVal}</td>
            <td>${coeffCpi >= 0 ? '+' + coeffCpi : coeffCpi}</td>
            <td>${coeffFed >= 0 ? '+' + coeffFed : coeffFed}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderTab2Charts() {
    if (!dashboardData) return;
    
    const selectEl = document.getElementById("select-ticker-ols");
    const selectedTicker = selectEl && selectEl.value ? selectEl.value : "NVDA";

    const periodSelect = document.getElementById("select-ols-year-range");
    const selectedPeriod = periodSelect && periodSelect.value ? periodSelect.value : "ALL";
    
    const regByPeriodMap = dashboardData.analytics.linear_regression_by_period || {};
    const regList = regByPeriodMap[selectedPeriod] || dashboardData.analytics.linear_regression || [];
    
    renderTab2Table(regList, selectedTicker);

    const olsInfo = regList.find(item => item.Ticker === selectedTicker);
    
    // อัปเดตการ์ดสรุปผลสถิติโมเดล OLS สำหรับหุ้นที่เลือก
    const summaryBox = document.getElementById("ols-ticker-summary");
    if (summaryBox && olsInfo) {
        const coeffCpi = olsInfo.Coeff_CPI !== undefined ? olsInfo.Coeff_CPI : olsInfo.Coeff_Lag1_CPI;
        const coeffFed = olsInfo.Coeff_FedRate !== undefined ? olsInfo.Coeff_FedRate : olsInfo.Coeff_Lag1_FedRate;
        const mseVal = olsInfo.MSE !== undefined && olsInfo.MSE !== null ? olsInfo.MSE : '-';
        const periodText = selectedPeriod === "ALL" ? "ภาพรวม 5 ปี (2021 - 2026)" : `ช่วงเวลา: ${selectedPeriod}`;

        summaryBox.innerHTML = `
            <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap; margin-top: 12px; padding: 12px; background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; font-size: 14px;">
                <div><strong>หุ้นที่เลือก:</strong> <span style="color: #38bdf8; font-weight: 600;">${olsInfo.Ticker}</span> (${olsInfo.Sector})</div>
                <div><strong>กรอบเวลาวิเคราะห์:</strong> <span style="color: #fbbf24; font-weight: 600;">${periodText}</span></div>
                <div><strong>R² Score:</strong> <span style="color: #34d399; font-weight: 600;">${olsInfo.R2_Score}</span></div>
                <div><strong>MAE:</strong> <span style="color: #f87171; font-weight: 600;">${olsInfo.MAE}%</span></div>
                <div><strong>MSE:</strong> <span style="color: #c084fc; font-weight: 600;">${mseVal}</span></div>
                <div><strong>สัมประสิทธิ์เงินเฟ้อ (β CPI):</strong> <span style="font-weight: 600;">${coeffCpi >= 0 ? '+' + coeffCpi : coeffCpi}</span></div>
                <div><strong>สัมประสิทธิ์ดอกเบี้ย (β Fed Rate):</strong> <span style="font-weight: 600;">${coeffFed >= 0 ? '+' + coeffFed : coeffFed}</span></div>
            </div>
        `;
    }

    // อ่านข้อมูลอนุกรมเวลา 12-Month Rolling Return ตามช่วงเวลาที่เลือก
    const seriesByPeriodMap = dashboardData.analytics.linear_regression_series_by_period || {};
    const periodSeriesMap = seriesByPeriodMap[selectedPeriod] || dashboardData.analytics.linear_regression_series || {};
    let tickerSeries = periodSeriesMap[selectedTicker] || [];

    const labels = tickerSeries.map(item => item.YearMonth);
    const actualReturns = tickerSeries.map(item => item.Actual_Return);
    const predictedReturns = tickerSeries.map(item => item.Predicted_Return);

    const canvasEl = document.getElementById("chart-ols-prediction");
    if (!canvasEl) return;
    
    // Empty-state guard: no 12M rolling data for the selected period
    const emptyMsgEl = document.getElementById("ols-chart-empty");
    if (labels.length === 0) {
        if (chartOlsPrediction) {
            chartOlsPrediction.destroy();
            chartOlsPrediction = null;
        }
        canvasEl.style.display = "none";
        if (emptyMsgEl) emptyMsgEl.style.display = "block";
        return;
    }
    canvasEl.style.display = "block";
    if (emptyMsgEl) emptyMsgEl.style.display = "none";

    const ctx = canvasEl.getContext("2d");
    
    if (chartOlsPrediction) {
        chartOlsPrediction.destroy();
    }

    chartOlsPrediction = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "ผลตอบแทนจริง (12M YoY %)",
                    data: actualReturns,
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.1)",
                    borderWidth: 2.5,
                    pointRadius: 3,
                    fill: false,
                    tension: 0.2
                },
                {
                    label: "ผลประเมินโมเดล OLS",
                    data: predictedReturns,
                    borderColor: "#f43f5e",
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    pointRadius: 3,
                    fill: false,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 700,
                easing: "easeOutQuart"
            },
            scales: {
                x: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
                y: { 
                    ticks: { color: "#94a3b8" }, 
                    grid: { color: "#334155" }, 
                    title: { display: true, text: "ผลตอบแทนสะสม 12 เดือน (%)", color: "#f8fafc" } 
                }
            },
            plugins: {
                legend: { labels: { color: "#f8fafc" }, position: "top" },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    titleColor: "#f8fafc",
                    bodyColor: "#cbd5e1",
                    borderColor: "#334155",
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        title: function(items) {
                            return items && items[0] ? `เดือน: ${items[0].label}` : "";
                        },
                        label: function(context) {
                            const idx = context.dataIndex;
                            const item = tickerSeries[idx];
                            const val = context.parsed.y;
                            const sign = val >= 0 ? "+" : "";
                            
                            if (context.datasetIndex === 0) {
                                const lines = [`ผลตอบแทนจริง (12M YoY): ${sign}${val.toFixed(2)}%`];
                                if (item && item.CPI !== undefined && item.CPI !== null) {
                                    lines.push(`อัตราเงินเฟ้อ (CPI): ${item.CPI}%`);
                                }
                                if (item && item.Fed_Rate !== undefined && item.Fed_Rate !== null) {
                                    lines.push(`อัตราดอกเบี้ย (Fed Rate): ${item.Fed_Rate}%`);
                                }
                                return lines;
                            } else {
                                return `ประเมินโดย OLS: ${sign}${val.toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        }
    });
}

// ==============================================================================
// [วิธีที่ 3 / TAB 3]: KAPLAN-MEIER SURVIVAL ANALYSIS LOGIC
// ==============================================================================

function initTab3Data() {
    applyTab3Filters();
}

function getPeriodLabel(periodKey) {
    const periodMap = {
        "ALL": "ภาพรวม 5 ปี (2021 - 2026)",
        "2021": "ปี 2021",
        "2022": "ปี 2022",
        "2023": "ปี 2023",
        "2024": "ปี 2024",
        "2025": "ปี 2025",
        "2026": "ปี 2026",
        "2021-2022": "ช่วงปี 2021 - 2022 (2 ปี)",
        "2023-2024": "ช่วงปี 2023 - 2024 (2 ปี)",
        "2025-2026": "ช่วงปี 2025 - 2026 (2 ปี)"
    };
    return periodMap[periodKey] || periodKey;
}

function applyTab3Filters() {
    if (!dashboardData || !dashboardData.analytics || !dashboardData.analytics.survival_recovery) {
        return;
    }
    
    const survDict = dashboardData.analytics.survival_recovery;
    const periodSelect = document.getElementById("select-survival-year-range");
    const selectedPeriod = periodSelect ? periodSelect.value : "ALL";
    
    const tickerSelect = document.getElementById("select-ticker-survival");
    const selectedTicker = tickerSelect ? tickerSelect.value : "ALL";
    
    let periodData = survDict.survival_recovery_by_period ? survDict.survival_recovery_by_period[selectedPeriod] : null;
    if (!periodData) {
        periodData = survDict;
    }
    
    const periodLabel = getPeriodLabel(selectedPeriod);
    const mainTitle = document.getElementById("tab3-main-title");
    if (mainTitle) {
        if (selectedTicker && selectedTicker !== "ALL") {
            mainTitle.textContent = `วิธีที่ 3: Kaplan-Meier Survival Analysis (${periodLabel} | หุ้น ${selectedTicker})`;
        } else {
            mainTitle.textContent = `วิธีที่ 3: Kaplan-Meier Survival Analysis (${periodLabel} | ภาพรวม 2 กลุ่มอุตสาหกรรม)`;
        }
    }
    
    renderTab3KPIs(periodData, selectedTicker, selectedPeriod);
    renderTab3Table(periodData, selectedTicker);
    renderTab3Charts(periodData, selectedTicker, selectedPeriod);
}

function renderTab3KPIs(data, selectedTicker = "ALL", selectedPeriod = "ALL") {
    const techEl = document.getElementById("km-tech-median");
    const staplesEl = document.getElementById("km-staples-median");
    const logrankEl = document.getElementById("km-logrank-p");
    const statusEl = document.getElementById("km-logrank-status");
    const label1 = document.getElementById("km-label-1");
    const label2 = document.getElementById("km-label-2");
    const label3 = document.getElementById("km-label-3");
    const sub1 = document.getElementById("km-sub-1");
    const sub2 = document.getElementById("km-sub-2");
    const card3 = document.getElementById("km-card-3");
    
    if (sub1) sub1.textContent = "";
    if (sub2) sub2.textContent = "";

    if (selectedTicker && selectedTicker !== "ALL" && data.ticker_level_recovery) {
        const tItem = data.ticker_level_recovery.find(x => x.Ticker === selectedTicker);
        if (tItem) {
            const sectorMed = tItem.Sector === "AI-Tech" ? data.tech_median_days : data.staples_median_days;
            const stockMedText = tItem.Median_Recovery_Days !== null && tItem.Median_Recovery_Days !== undefined ? `${tItem.Median_Recovery_Days} วันปฏิทิน` : "อยู่ระหว่างฟื้นตัวในปีนี้";
            const sectorMedText = sectorMed !== null && sectorMed !== undefined ? `${sectorMed} วันปฏิทิน` : "ยังไม่เกิดการฟื้นตัวในปีนี้";
            
            if (label1) label1.textContent = `มัธยฐานเวลาฟื้นตัว (หุ้น ${tItem.Ticker})`;
            if (label2) label2.textContent = `เกณฑ์เฉลี่ยกลุ่มอุตสาหกรรม (${tItem.Sector})`;
            if (label3) label3.textContent = "ผลทดสอบสถิติภาพรวมกลุ่ม (Log-Rank Test)";
            
            if (techEl) techEl.textContent = stockMedText;
            if (staplesEl) staplesEl.textContent = sectorMedText;
            if (logrankEl) logrankEl.textContent = data.logrank_p_value !== undefined ? `p = ${data.logrank_p_value}` : "-";
            if (statusEl) {
                statusEl.textContent = data.statistically_significant ? "มีนัยสำคัญทางสถิติ (p < 0.05)" : "ไม่มีนัยสำคัญทางสถิติ (p >= 0.05)";
                statusEl.style.color = data.statistically_significant ? "#34d399" : "#fbbf24";
            }

            if (card3) card3.style.display = "block";
            return;
        }
    }

    // Default Group Baseline: Show Card 3 for 2-Sector Log-Rank Test
    if (card3) card3.style.display = "block";
    if (label1) label1.textContent = "มัธยฐานเวลาฟื้นตัว (กลุ่ม AI-Tech)";
    if (label2) label2.textContent = "มัธยฐานเวลาฟื้นตัว (กลุ่ม Consumer Staples)";
    if (label3) label3.textContent = "ผลการทดสอบสถิติ (Log-Rank Test)";
    
    if (techEl) {
        techEl.textContent = data.tech_median_days !== null && data.tech_median_days !== undefined ? `${data.tech_median_days} วันปฏิทิน` : "ยังไม่ฟื้นตัวในปีนี้";
    }
    if (staplesEl) {
        staplesEl.textContent = data.staples_median_days !== null && data.staples_median_days !== undefined ? `${data.staples_median_days} วันปฏิทิน` : "ยังไม่ฟื้นตัวในปีนี้";
    }
    if (logrankEl) {
        logrankEl.textContent = data.logrank_p_value !== undefined ? `p = ${data.logrank_p_value}` : "-";
    }
    if (statusEl) {
        statusEl.textContent = data.statistically_significant ? "มีนัยสำคัญทางสถิติ (p < 0.05)" : "ไม่มีนัยสำคัญทางสถิติ (p >= 0.05)";
        statusEl.style.color = data.statistically_significant ? "#34d399" : "#fbbf24";
    }
}

function renderTab3Table(data, selectedTicker = "ALL") {
    const tbody = document.querySelector("#table-survival-tickers tbody");
    if (!tbody) return;
    
    tbody.innerHTML = "";
    let tickerList = data.ticker_level_recovery || [];
    let selectedItem = null;
    let targetTrElement = null;
    
    if (selectedTicker && selectedTicker !== "ALL") {
        selectedItem = tickerList.find(x => x.Ticker === selectedTicker);
        if (selectedItem) {
            tickerList = tickerList.filter(x => x.Sector === selectedItem.Sector);
        }
    }
    
    sortStockRows(tickerList).forEach(item => {
        const tr = document.createElement("tr");
        tr.style.cursor = "pointer";
        tr.id = `tab3-tr-${item.Ticker}`;
        
        if (selectedTicker === item.Ticker) {
            tr.classList.add("selected-row");
            selectedItem = item;
            targetTrElement = tr;
        }
        
        tr.onclick = () => {
            const dropdown = document.getElementById("select-ticker-survival");
            if (tr.classList.contains("selected-row")) {
                if (dropdown) dropdown.value = "ALL";
                applyTab3Filters();
                return;
            }
            if (dropdown) dropdown.value = item.Ticker;
            applyTab3Filters();
        };
        
        const isRecovered = item.Median_Recovery_Days !== null && item.Median_Recovery_Days !== undefined;
        const statusBadgeClass = isRecovered ? "badge-success" : "badge-warning";
        const statusBadgeText = isRecovered ? "ฟื้นตัวสำเร็จ" : "ยังไม่ฟื้นตัวในปีนี้";
        const medText = isRecovered ? `${item.Median_Recovery_Days} วัน` : "-";
        const eventsText = item.Total_Drawdown_Events !== undefined && item.Total_Drawdown_Events !== null ? `${item.Total_Drawdown_Events} ครั้ง` : "-";
        
        tr.innerHTML = `
            <td><strong>${item.Ticker}</strong></td>
            <td><span class="badge ${item.Sector === 'AI-Tech' ? 'badge-tech' : 'badge-staples'}">${item.Sector}</span></td>
            <td><strong>${medText}</strong></td>
            <td><strong>${eventsText}</strong></td>
            <td><span class="badge ${statusBadgeClass}">${statusBadgeText}</span></td>
        `;
        tbody.appendChild(tr);
    });
    
    const badgeBox = document.getElementById("ticker-survival-summary-badge");
    if (badgeBox) {
        badgeBox.style.display = "none";
    }

    if (targetTrElement) {
        setTimeout(() => {
            targetTrElement.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }, 100);
    }
}

function highlightTab3Ticker(item, trElement) {
    document.querySelectorAll("#table-survival-tickers tbody tr").forEach(r => r.classList.remove("selected-row"));
    if (trElement) trElement.classList.add("selected-row");
}

function renderTab3Charts(overrideData, selectedTicker = "ALL", selectedPeriod = "ALL") {
    const canvasEl = document.getElementById("chart-survival-curve");
    if (!canvasEl) return;
    
    let data = overrideData;
    if (!data && dashboardData && dashboardData.analytics && dashboardData.analytics.survival_recovery) {
        const survDict = dashboardData.analytics.survival_recovery;
        const periodSelect = document.getElementById("select-survival-year-range");
        const periodKey = periodSelect ? periodSelect.value : "ALL";
        data = survDict.survival_recovery_by_period ? survDict.survival_recovery_by_period[periodKey] : survDict;
    }
    
    if (!data) return;
    
    const formatCurvePoints = (curveObj) => {
        if (!curveObj || !curveObj.timeline || !curveObj.survival_probability) return [];
        return curveObj.timeline.map((t, idx) => ({
            x: t,
            y: curveObj.survival_probability[idx]
        }));
    };
    
    const techSeries = formatCurvePoints(data.tech_curve);
    const staplesSeries = formatCurvePoints(data.staples_curve);
    
    const isDrill = !!(selectedTicker && selectedTicker !== "ALL");
    const tItem = isDrill && data.ticker_level_recovery ? data.ticker_level_recovery.find(x => x.Ticker === selectedTicker) : null;
    const isTech = tItem ? tItem.Sector === "AI-Tech" : true;

    const datasets = [];
    
    if (!isDrill) {
        // Overview Mode: Show both Sector Benchmarks
        datasets.push({
            label: "กลุ่ม AI-Tech",
            data: techSeries,
            borderColor: "#f43f5e",
            backgroundColor: "rgba(244, 63, 94, 0.1)",
            borderWidth: 2.5,
            pointRadius: 2,
            stepped: true,
            fill: false
        });
        datasets.push({
            label: "กลุ่ม Consumer Staples",
            data: staplesSeries,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 2.5,
            pointRadius: 2,
            stepped: true,
            fill: false
        });
    } else {
        // Single Ticker Mode: Only show parent sector benchmark to avoid chart clutter!
        if (isTech) {
            datasets.push({
                label: "เฉลี่ยกลุ่ม AI-Tech",
                data: techSeries,
                borderColor: "rgba(244, 63, 94, 0.45)",
                backgroundColor: "rgba(244, 63, 94, 0.05)",
                borderWidth: 2,
                borderDash: [6, 4],
                pointRadius: 2,
                stepped: true,
                fill: false
            });
        } else {
            datasets.push({
                label: "เฉลี่ยกลุ่ม Consumer Staples",
                data: staplesSeries,
                borderColor: "rgba(16, 185, 129, 0.45)",
                backgroundColor: "rgba(16, 185, 129, 0.05)",
                borderWidth: 2,
                borderDash: [6, 4],
                pointRadius: 2,
                stepped: true,
                fill: false
            });
        }

        if (tItem && tItem.Curve) {
            const tSeries = formatCurvePoints(tItem.Curve);
            if (tSeries.length > 0) {
                datasets.push({
                    label: `หุ้น ${selectedTicker}`,
                    data: tSeries,
                    backgroundColor: "rgba(56, 189, 248, 0.15)",
                    borderWidth: 3.5,
                    pointRadius: 3,
                    pointBackgroundColor: "#38bdf8",
                    borderColor: "#38bdf8",
                    stepped: true,
                    fill: false
                });
            }
        }
    }
    
    setSurvivalFocusHint(selectedTicker, tItem ? tItem.Sector : null, selectedPeriod);
    
    const ctx = canvasEl.getContext("2d");
    if (chartSurvivalCurve) {
        chartSurvivalCurve.destroy();
    }
    
    chartSurvivalCurve = new Chart(ctx, {
        type: "line",
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 600,
                easing: "easeOutQuart"
            },
            scales: {
                x: {
                    type: "linear",
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#334155" },
                    title: { display: true, text: "ระยะเวลาปฏิทิน (Calendar Days)", color: "#f8fafc" }
                },
                y: {
                    min: 0,
                    max: 1.05,
                    ticks: { color: "#94a3b8", callback: (val) => val <= 1.0 ? val.toFixed(2) : "" },
                    grid: { color: "#334155" },
                    title: { display: true, text: "โอกาสที่ราคายังไม่ฟื้นตัว (%)", color: "#f8fafc" }
                }
            },
            plugins: {
                title: { display: false },
                legend: { 
                    display: true,
                    position: "top", 
                    align: "center",
                    labels: { 
                        color: "#f8fafc",
                        padding: 15,
                        font: { size: 12 }
                    } 
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    titleColor: "#f8fafc",
                    bodyColor: "#cbd5e1",
                    borderColor: "#334155",
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        title: (items) => items && items[0] ? `วันปฏิทินที่: ${items[0].parsed.x} วัน` : "",
                        label: (context) => {
                            const val = context.parsed.y;
                            return `${context.dataset.label}: โอกาสยังไม่ฟื้นตัว ${(val * 100).toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

function setSurvivalFocusHint(selectedTicker, sectorName, selectedPeriod = "ALL") {
    const el = document.getElementById("survival-focus-hint");
    if (!el) return;
    el.style.display = "none";
}

// End of app.js
