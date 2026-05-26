"""
app.py
AI 智慧航班機票搜尋與推薦系統 - Streamlit 主程式
整合 fast-flights 即時航班查詢與 AI 推薦排序 (高質感旅遊風版)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
import datetime

# ── 確保模組路徑正確 ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_sample_data, load_uploaded_data, load_cached_scraped_data,
    get_airport_display_name, get_airport_code, AIRPORT_NAMES_ZH
)
from src.preprocessing import full_preprocessing, filter_flights, prepare_features
from src.recommender import (
    knn_recommend, kmeans_cluster, get_pca_data,
    generate_recommendation_reason, generate_pros_cons,
    PREF_PRICE, PREF_TIME, PREF_BALANCE, PREF_STOPS
)
from src.visualization import (
    plot_price_bar, plot_recommendation_score,
    plot_duration_vs_price_scatter, plot_stops_distribution,
    plot_kmeans_scatter
)
from src.flight_scraper import fetch_scraped_flights

# ── Streamlit 頁面設定 (隱藏側邊欄) ──────────────────────────────────
st.set_page_config(
    page_title="AI 智慧航班機票搜尋與推薦系統",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── 自訂 CSS 美化 (極致旅遊風) ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800;900&family=Noto+Sans+TC:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', 'Noto Sans TC', sans-serif !important;
    background: linear-gradient(160deg, #f0f7ff 0%, #e8f4f8 40%, #f5f0ff 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* ─── 隱藏預設 UI ─── */
[data-testid="collapsedControl"] { display: none; }
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stAppDeployButton { display: none !important; }

/* ─── 頂部 Hero Banner ─── */
.header-banner {
    background:
        linear-gradient(135deg, rgba(10, 18, 40, 0.88) 0%, rgba(20, 40, 120, 0.70) 60%, rgba(120, 60, 0, 0.50) 100%),
        url('https://images.unsplash.com/photo-1527631746610-bca00a040d60?q=90&w=2400&auto=format&fit=crop');
    background-size: cover;
    background-position: center 55%;
    padding: 80px 20px 70px 20px;
    text-align: center;
    color: white;
    margin-top: -60px;
    margin-bottom: 45px;
    margin-left: -40px;
    margin-right: -40px;
    border-bottom: 3px solid #f59e0b;
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #f59e0b, #e879f9, #38bdf8, #34d399, #f59e0b);
    background-size: 300% 100%;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 0% 0%; }
    100% { background-position: 300% 0%; }
}
.header-tag {
    display: inline-block;
    background: rgba(245, 158, 11, 0.25);
    border: 1px solid rgba(245, 158, 11, 0.5);
    color: #fcd34d;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 50px;
    margin-bottom: 20px;
}
.header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.6rem;
    font-weight: 900;
    margin-bottom: 15px;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #ffffff 0%, #bfdbfe 50%, #fde68a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.header-subtitle {
    font-size: 1.15rem;
    font-weight: 400;
    color: rgba(226, 232, 240, 0.9);
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.7;
    letter-spacing: 0.5px;
}
.header-stats {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 30px;
}
.header-stat {
    text-align: center;
}
.header-stat-value {
    font-size: 1.6rem;
    font-weight: 900;
    font-family: 'Outfit', sans-serif;
    color: #fcd34d;
}
.header-stat-label {
    font-size: 0.8rem;
    color: rgba(226,232,240,0.75);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ─── 玻璃卡片 ─── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 22px !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    background: rgba(255, 255, 255, 0.80) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(30, 58, 138, 0.07), 0 1px 0 rgba(255,255,255,0.9) inset;
    padding: 28px 32px !important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 20px 45px rgba(30, 58, 138, 0.1), 0 1px 0 rgba(255,255,255,0.9) inset;
    transform: translateY(-2px);
}

/* ─── 輸入元件標籤 ─── */
.stSelectbox label, .stDateInput label, .stNumberInput label, .stTextInput label {
    font-weight: 700 !important;
    color: #334155 !important;
    font-size: 0.92rem !important;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}

/* ─── 搜尋按鈕 ─── */
.stButton > button {
    background: linear-gradient(135deg, #f59e0b, #ea580c, #dc2626) !important;
    background-size: 200% auto !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    padding: 0.85rem 2.2rem !important;
    box-shadow: 0 8px 25px rgba(234, 88, 12, 0.35) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.stButton > button:hover {
    background-position: right center !important;
    transform: translateY(-4px) scale(1.03) !important;
    box-shadow: 0 15px 30px rgba(234, 88, 12, 0.50) !important;
}

/* ─── Metric Cards ─── */
.metric-container {
    background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
    border-radius: 18px;
    padding: 28px 20px;
    box-shadow: 0 4px 20px rgba(30,58,138,0.05), 0 1px 0 rgba(255,255,255,0.8) inset;
    text-align: center;
    border: 1px solid rgba(226,232,240,0.8);
    border-bottom: 4px solid transparent;
    margin-bottom: 25px;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}
.metric-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-container:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 35px rgba(30,58,138,0.10);
    border-bottom: 4px solid #3b82f6;
}
.metric-container:hover::before { opacity: 1; }
.metric-label {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.metric-value {
    color: #0f172a;
    font-size: 2.4rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    line-height: 1;
}
.metric-highlight {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    line-height: 1;
}

/* ─── Top VIP Flight Card ─── */
.top-flight-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(251,191,36,0.3);
    border-left: 5px solid #fbbf24;
    border-radius: 24px;
    padding: 40px 45px;
    margin: 18px 0 45px 0;
    box-shadow: 0 25px 50px rgba(0,0,0,0.25), 0 0 0 1px rgba(251,191,36,0.1) inset;
    position: relative;
    overflow: hidden;
    color: #f8fafc;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
}
.top-flight-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(251,191,36,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.top-flight-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 35px 60px rgba(0,0,0,0.3);
}
.top-flight-card::after {
    content: '👑 AI TOP PICK';
    position: absolute;
    top: 32px;
    right: -42px;
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    color: #000;
    padding: 8px 50px;
    font-weight: 900;
    font-size: 0.82rem;
    letter-spacing: 2px;
    transform: rotate(45deg);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.top-flight-title {
    color: #fbbf24;
    font-size: 1.9rem;
    font-weight: 900;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.5px;
}
.top-flight-card b, .top-flight-card strong { color: #f1f5f9; }

/* ─── VIP 分隔線 ─── */
.vip-divider {
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.08);
}

/* ─── 比價按鈕區塊 ─── */
.price-source-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}
.compare-btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
}
.compare-btn {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 10px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.92rem;
    text-decoration: none;
    transition: all 0.25s ease;
    border: 1.5px solid;
    letter-spacing: 0.3px;
}
.btn-gf {
    background: rgba(66,133,244,0.1);
    border-color: rgba(66,133,244,0.4);
    color: #93c5fd;
}
.btn-gf:hover {
    background: #4285f4;
    color: white;
    box-shadow: 0 5px 20px rgba(66,133,244,0.4);
    transform: translateY(-2px);
}
.btn-sky {
    background: rgba(0,148,216,0.1);
    border-color: rgba(0,148,216,0.4);
    color: #67e8f9;
}
.btn-sky:hover {
    background: #0094d8;
    color: white;
    box-shadow: 0 5px 20px rgba(0,148,216,0.4);
    transform: translateY(-2px);
}
.btn-trip {
    background: rgba(0,164,109,0.1);
    border-color: rgba(0,164,109,0.4);
    color: #6ee7b7;
}
.btn-trip:hover {
    background: #00a46d;
    color: white;
    box-shadow: 0 5px 20px rgba(0,164,109,0.4);
    transform: translateY(-2px);
}

/* ─── AI Info Card ─── */
.ai-info-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
    border: 1px solid #dbeafe;
    border-left: 5px solid #3b82f6;
    border-radius: 20px;
    padding: 28px 35px;
    margin-bottom: 40px;
    box-shadow: 0 8px 25px rgba(59,130,246,0.06);
    transition: all 0.3s ease;
}
.ai-info-card:hover {
    box-shadow: 0 15px 35px rgba(59,130,246,0.1);
    transform: translateY(-3px);
}
.ai-info-title {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
}
.ai-info-icon {
    font-size: 2rem;
    background: linear-gradient(135deg, #dbeafe, #ede9fe);
    padding: 12px;
    border-radius: 14px;
}
.ai-info-heading {
    margin: 0;
    color: #0f172a;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.3px;
    font-size: 1.2rem;
}
.ai-info-section {
    padding: 14px 18px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.ai-info-section.algo {
    background: rgba(59,130,246,0.05);
    border-left: 3px solid #3b82f6;
}
.ai-info-section.price {
    background: rgba(16,185,129,0.05);
    border-left: 3px solid #10b981;
}

/* ─── 候補選擇卡片 ─── */
.alt-flight-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    transition: all 0.3s ease;
}
.alt-flight-card:hover {
    box-shadow: 0 12px 28px rgba(30,58,138,0.08);
    border-color: #c7d2fe;
    transform: translateY(-3px);
}

/* ─── 頁籤美化 ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(241,245,249,0.8);
    border-radius: 14px;
    padding: 6px;
    backdrop-filter: blur(10px);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 700;
    font-size: 0.92rem;
    color: #64748b;
    background: transparent;
    border: none;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1e293b !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08) !important;
}

/* ─── 無航班提示 ─── */
.no-flight-alert {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fde68a;
    border-left: 5px solid #f59e0b;
    padding: 30px;
    border-radius: 16px;
    margin: 20px 0;
    text-align: center;
}
.no-flight-alert h3 { color: #b45309; font-weight: 800; margin-bottom: 10px; }
.no-flight-alert p { color: #92400e; font-size: 1.05rem; }

/* ─── 搜尋條件標題 ─── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: 0.3px;
}

</style>

""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 頂部 Hero Header
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <div class="header-tag">✨ AI-Powered Flight Recommender</div>
    <div class="header-title">啟程，探索世界</div>
    <div class="header-subtitle">結合 KNN 機器學習與即時爬蟲，幫您在海量航班中精準找出最符合旅遊偏好的完美機票組合。</div>
    <div class="header-stats">
        <div class="header-stat">
            <div class="header-stat-value">100+</div>
            <div class="header-stat-label">Supported Routes</div>
        </div>
        <div class="header-stat">
            <div class="header-stat-value">Real-Time</div>
            <div class="header-stat-label">Price Data</div>
        </div>
        <div class="header-stat">
            <div class="header-stat-value">AI KNN</div>
            <div class="header-stat-label">Smart Ranking</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 查詢條件過濾卡片 (Filter Card)
# ═══════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    st.markdown("<div class='section-heading'>🔍 規劃您的旅程</div>", unsafe_allow_html=True)

    # ── 第一排：基本設定 ──
    c1, c2, c3 = st.columns(3)
    data_source = "使用 Google Flights 爬蟲查詢"
    with c1:
        flight_type = st.selectbox("航班類型", options=["one-way", "round-trip"], format_func=lambda x: "單程" if x == "one-way" else "來回")
    with c2:
        all_airports = [get_airport_display_name(c) for c in AIRPORT_NAMES_ZH.keys()]
        idx_tpe = all_airports.index(get_airport_display_name("TPE")) if get_airport_display_name("TPE") in all_airports else 0
        origin_display = st.selectbox("🛫 出發地", options=all_airports, index=idx_tpe)
        origin_code = get_airport_code(origin_display)
    with c3:
        idx_nrt = all_airports.index(get_airport_display_name("NRT")) if get_airport_display_name("NRT") in all_airports else 1
        dest_display = st.selectbox("🛬 目的地", options=all_airports, index=idx_nrt)
        destination_code = get_airport_code(dest_display)

    # ── 第二排：日期與細節設定 ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    
    target_dates_input = (datetime.date.today() + datetime.timedelta(days=14), datetime.date.today() + datetime.timedelta(days=20))
    uploaded_file = None

    with r2_c1:
        target_dates = st.date_input("📅 查詢日期區間 (建議 7 天內)", value=target_dates_input)
        
        # 解析日期成 list
        if isinstance(target_dates, tuple) and len(target_dates) == 2:
            start_date, end_date = target_dates
            delta = end_date - start_date
            # 限制最多查詢 14 天避免 API 被鎖
            max_days = min(delta.days, 13)
            date_list = [(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(max_days + 1)]
            display_date_str = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        elif isinstance(target_dates, tuple) and len(target_dates) == 1:
            date_list = [target_dates[0].strftime("%Y-%m-%d")]
            display_date_str = target_dates[0].strftime("%Y-%m-%d")
        else:
            date_list = [target_dates.strftime("%Y-%m-%d")]
            display_date_str = target_dates.strftime("%Y-%m-%d")
        
        target_date = date_list[0] # 保留變數相容性
            
    with r2_c2:
        adults = st.number_input("👥 旅客人數", min_value=1, max_value=9, value=1)
        
    with r2_c3:
        currency = st.selectbox("💱 顯示幣別", options=["TWD", "USD", "JPY"], index=0)
        
    with r2_c4:
        max_stops = st.selectbox("🔄 最大轉機次數", [0, 1, 2, 3], index=1, format_func=lambda x: "直飛 (推薦)" if x==0 else f"{x} 次轉機")

    # ── 第三排：進階服務與偏好 ──
    st.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    r3_c1, r3_c2 = st.columns([1.5, 1])
    with r3_c1:
        preference = st.selectbox("🎯 AI 推薦優先順序", [PREF_PRICE, PREF_TIME, PREF_STOPS, PREF_BALANCE], index=3, 
                                  help="系統將根據您的偏好，為每一班機票進行智能打分與排序")
        
        # 開發者除錯開關
        show_debug = st.checkbox("🛠️ 顯示爬蟲原始資料除錯資訊")
        
    with r3_c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Spacer
        analyze_btn = st.button("✈️ 啟程搜尋航班", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 輔助函式：繪製航段明細卡片
# ═══════════════════════════════════════════════════════════════════════════
def build_compare_links(orig: str, dest: str, date: str, adults: int = 1) -> str:
    """Build comparison deep-link buttons for Google Flights, Skyscanner, and Trip.com"""
    d = date.replace('-', '')  # e.g. 20260608
    d_sky = date[2:].replace('-', '')  # e.g. 260608
    n_adults = max(1, int(adults))
    
    gf_url  = f"https://www.google.com/travel/flights?q=Flights%20to%20{dest}%20from%20{orig}%20on%20{date}&adults={n_adults}"
    sky_url = f"https://www.skyscanner.com.tw/transport/flights/{orig.lower()}/{dest.lower()}/{d_sky}/?adults={n_adults}&cabinclass=economy&ref=home"
    trip_url = f"https://tw.trip.com/flights/{orig.lower()}-to-{dest.lower()}/tickets-{orig.lower()}-{dest.lower()}?flightWay=S&departureDate={date}&adult={n_adults}"
    
    return f"""
<div class="compare-btn-row">
    <a href="{gf_url}" target="_blank" class="compare-btn btn-gf">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/></svg>
        Google Flights 查看價格
    </a>
    <a href="{sky_url}" target="_blank" class="compare-btn btn-sky">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M21 3L3 10.53v.98l6.84 2.65L12.48 21h.98L21 3z"/></svg>
        Skyscanner 比價
    </a>
    <a href="{trip_url}" target="_blank" class="compare-btn btn-trip">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
        Trip.com 查詢
    </a>
</div>
"""

def fmt_time(raw: str) -> str:
    """將 fast-flights 回傳的英文時間格式轉換為繁體中文格式
    例如: '1:35 AM on Sat, May 30' → '週六 5/30 01:35'
    """
    import re
    if not raw or not isinstance(raw, str):
        return raw
    raw = raw.strip()

    # 星期對照
    week_zh = {
        'Mon': '週一', 'Tue': '週二', 'Wed': '週三',
        'Thu': '週四', 'Fri': '週五', 'Sat': '週六', 'Sun': '週日'
    }
    # 月份對照
    month_zh = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    # 匹配 "1:35 AM on Sat, May 30" 或 "1:35 AM"
    m = re.match(
        r'(\d{1,2}):(\d{2})\s*(AM|PM)(?:\s+on\s+(\w+),?\s+(\w+)\s+(\d{1,2}))?',
        raw, re.IGNORECASE
    )
    if not m:
        return raw

    hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3).upper()
    weekday_en = m.group(4)
    month_en   = m.group(5)
    day        = m.group(6)

    # 12h → 24h
    if ampm == 'AM':
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12

    time_str = f"{hour:02d}:{minute:02d}"

    if weekday_en and month_en and day:
        wk  = week_zh.get(weekday_en[:3], weekday_en)
        mon = month_zh.get(month_en[:3], month_en)
        return f"{wk} {mon}/{day} {time_str}"
    return time_str


def render_segment_card(airline, flight_no, segment_str):
    """將以 | 分隔的航段字串轉成精美的 HTML 登機證卡片"""
    parts = str(segment_str).split('|')
    if len(parts) == 6:
        orig, dep_time, dest, arr_time, dur, real_flight_no = parts
        
        orig_name = get_airport_display_name(orig)
        dest_name = get_airport_display_name(dest)
        
        # 若為未知航班，則隱藏或顯示次要文字
        flight_no_html = f'<span class="flight-number">{real_flight_no}</span>' if real_flight_no != "未知航班" else ''
        
        html = f"""
                    <div class="airport-name">{dest_name}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info(segment_str)

# ═══════════════════════════════════════════════════════════════════════════
# 主流程與結果展示
# ═══════════════════════════════════════════════════════════════════════════
# 如果按下按鈕，標記已搜尋，並要求重新計算
if analyze_btn:
    st.session_state['has_searched'] = True
    st.session_state['needs_compute'] = True

if not st.session_state.get('has_searched', False):
    st.markdown("<div style='text-align: center; color: #64748b; margin-top: 50px;'><h4>💡 請設定上方的旅程條件後，點擊「✈️ 啟程搜尋航班」開啟您的旅程。</h4></div>", unsafe_allow_html=True)
    st.stop()

if st.session_state.get('needs_compute', True):
    with st.spinner("🤖 AI 正在為您搜羅全球航班並進行智能分析，請稍候..."):
        df_raw = None
        fallback_triggered = False
    
        # 1. 資料載入邏輯與備援機制
        if data_source == "使用 Google Flights 爬蟲查詢":
            df_raw, msg = fetch_scraped_flights(
                origin=origin_code,
                destination=destination_code,
                dates=date_list,
                flight_type=flight_type,
                adults=adults,
                currency=currency
            )
        
            if df_raw is not None:
                st.success(f"✅ 成功為您掃描並找到 {len(df_raw)} 個機票組合！")
            else:
                # 判斷是否為明確的「查無航班」
                if "找不到" in msg:
                    st.markdown(f"""
                    <div class="no-flight-alert">
                        <h3>📭 查無航班資料</h3>
                        <p>非常抱歉，在 <b>{display_date_str}</b> 區間內，我們找不到從 <b>{origin_display}</b> 飛往 <b>{dest_display}</b> 的符合航班。</p>
                        <p style='font-size: 0.95rem; margin-top: 5px; color: #d97706;'>👉 建議您：嘗試更改出發日期、選擇不同的目的地，或是放寬最大轉機次數限制。</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                else:
                    fallback_triggered = True
                    st.warning(f"⚠️ 即時連線發生問題 ({msg})，系統為您切換至離線快取資料。")
                
                    # Fallback
                    df_raw, cache_msg = load_cached_scraped_data()
                    if df_raw is None:
                        df_raw, sample_msg = load_sample_data()
                        if df_raw is None:
                            st.error("目前沒有任何可用的航班備援資料。")
                            st.stop()

        elif data_source == "使用已快取爬蟲資料":
            df_raw, msg = load_cached_scraped_data()
            if df_raw is None:
                st.error(msg)
                df_raw, _ = load_sample_data()
                st.info("已為您切換至內建範例航班。")
            else:
                st.success("✅ " + msg)
            
        elif data_source == "上傳 CSV":
            if uploaded_file:
                df_raw, msg = load_uploaded_data(uploaded_file)
                if df_raw is None:
                    st.error(msg)
                    st.stop()
                st.success("✅ " + msg)
            else:
                st.error("❌ 請先上傳檔案。")
                st.stop()
            
        else:
            df_raw, msg = load_sample_data()
            if df_raw is None:
                st.error(msg)
                st.stop()
            st.success("✅ " + msg)

        # 確保備援資料有 query_date
        if "query_date" not in df_raw.columns:
            df_raw["query_date"] = "未知日期"

        # 2. 資料預處理
        df_clean = full_preprocessing(df_raw)
    
        # 3. 過濾條件
        df_filtered, filter_msg = filter_flights(df_clean, origin_code, destination_code, max_stops)
        if df_filtered.empty:
            st.markdown(f"""
            <div class="no-flight-alert">
                <h3>📭 條件過於嚴苛</h3>
                <p>找不到符合您「<b>目前過濾條件</b>」的航班。</p>
                <p style='font-size: 0.95rem; margin-top: 5px; color: #d97706;'>👉 {filter_msg}</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
    
        # 4. AI KNN 推薦
        df_ml = prepare_features(df_filtered)
        df_recommended, scaler, X_scaled = knn_recommend(df_ml, preference)
    
        # Debug 資訊顯示
        if show_debug:
            st.markdown("### 🛠️ 開發者除錯資訊")
            st.write("解析後的 DataFrame (前 10 筆)：")
            debug_cols = ["airline", "flight_no", "price", "total_price", "adults", "departure_time", "arrival_time", "data_source"]
            st.dataframe(df_recommended[[c for c in debug_cols if c in df_recommended.columns]].head(10))
        
            if "_raw_fast_flights" in df_recommended.columns:
                st.write("fast-flights 原始回傳結果 (第一筆)：")
                st.code(df_recommended.iloc[0]["_raw_fast_flights"], language="text")

        # 5. AI KMeans 分群
        n_clusters = min(4, max(2, len(df_recommended) - 1))
        if len(df_recommended) > 2:
            df_clustered, cluster_msg = kmeans_cluster(df_recommended, n_clusters=n_clusters)
            df_result = df_clustered.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
            pca_df = get_pca_data(df_result, scaler)
        else:
            df_result = df_recommended.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
            df_result["cluster_label"] = "資料過少未分類"
            pca_df = None
            cluster_msg = "航班資料數量不足，已略過分群分析。"

        df_result["排名"] = df_result.index + 1
    
    st.session_state['df_result'] = df_result
    st.session_state['pca_df'] = pca_df
    st.session_state['cluster_msg'] = cluster_msg
    st.session_state['needs_compute'] = False

else:
    # 直接讀取快取，避免拉動 Slider 時重新爬蟲
    df_result = st.session_state['df_result']
    pca_df = st.session_state['pca_df']
    cluster_msg = st.session_state['cluster_msg']

# ═══════════════════════════════════════════════════════════════════════════
# 分數閾值控制列
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;
            background:rgba(255,255,255,0.75); backdrop-filter:blur(12px);
            border:1px solid rgba(226,232,240,0.9); border-radius:16px;
            padding:16px 24px; margin-bottom:28px;
            box-shadow:0 4px 15px rgba(30,58,138,0.05);">
    <span style="font-size:1.3rem;">&#x1F3AF;</span>
    <span style="font-weight:800; color:#1e293b; font-size:1rem; white-space:nowrap;">
        AI 分數閾值篩選
    </span>
</div>
""", unsafe_allow_html=True)

_thr_col, _btn_col = st.columns([3, 1])
with _thr_col:
    score_threshold = st.slider(
        "🧠 最低 AI 推薦分數閾值（低於此分數的航班不列出）",
        min_value=0, max_value=100, value=85, step=5,
        format="%d 分",
        help="調整此數則即可篩選高分航班。預設為 85 分以上。"
    )
with _btn_col:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    show_all = st.button("📋 展開全部航班", use_container_width=True)

if show_all:
    score_threshold = 0

max_score = df_result["recommendation_score"].max()
force_show_all = False
original_threshold = score_threshold
if max_score < score_threshold:
    force_show_all = True
    score_threshold = 0

# 根據閾值篩選（第一名不受閾值限制）
df_display_result = df_result[
    (df_result["recommendation_score"] >= score_threshold) | (df_result["排名"] == 1)
].copy()
hidden_count = len(df_result) - len(df_display_result)

if force_show_all:
    st.markdown(
        f"<div style='background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); "
        f"border-radius:10px; padding:10px 18px; margin-bottom:18px; font-size:0.95rem; color:#991b1b;'>"
        f"⚠️ <b>注意：</b>本次搜尋所有航班的 AI 評分皆低於您設定的閾值 ({original_threshold} 分)。已為您自動展開所有航班。</div>",
        unsafe_allow_html=True
    )
elif hidden_count > 0:
    st.markdown(
        f"<div style='background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3); "
        f"border-radius:10px; padding:10px 18px; margin-bottom:18px; font-size:0.95rem; color:#92400e;'>"
        f"🔍 目前展示 <b>AI 分數 ≥ {score_threshold} 分</b> 的航班共 <b>{len(df_display_result)} 班</b>，"
        f"另有 <b>{hidden_count} 班</b>因分數不足閾值而隱藏。可調低閾值或點擊「展開全部航班」查看。</div>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════
# 結果展示區
# ═══════════════════════════════════════════════════════════════════════════

# ── 指標卡片 (根據篩選後的 df_display_result) ──
c1, c2, c3, c4 = st.columns(4)
top_flight = df_result.iloc[0]  # 首選始終是全部中最高分

# 安全取得總票價與單人票價
total_price_val = top_flight.get('total_price', top_flight['price'])
adults_val = top_flight.get('adults', adults)

with c1:
    st.markdown(f"""<div class="metric-container"><div class="metric-label">📊 精選航班數</div><div class="metric-value">{len(df_display_result)} <span style="font-size:1rem;font-weight:600;color:#64748b;">班</span></div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-container"><div class="metric-label">🏷️ 最低總票價</div><div class="metric-highlight"><span style="font-size:1.2rem;">{currency}</span> {df_result['total_price'].min() if 'total_price' in df_result else df_result['price'].min():,.0f}</div></div>""", unsafe_allow_html=True)
with c3:
    min_h, min_m = divmod(int(df_result['duration_minutes'].min()), 60)
    st.markdown(f"""<div class="metric-container"><div class="metric-label">⚡ 極速抵達時間</div><div class="metric-value">{min_h}h {min_m}m</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-container"><div class="metric-label">🧠 AI 智能評分</div><div class="metric-highlight">{top_flight['recommendation_score']:.1f} <span style="font-size:1rem;font-weight:600;color:#64748b;">/ 100</span></div></div>""", unsafe_allow_html=True)

# 判斷是否為非即時資料，顯示提示
ds = top_flight.get('data_source', '')
ds_warning = ""
if "快取" in ds or "範例" in ds or "舊版" in ds:
    ds_warning = f"<div style='margin-top: 15px; background: #fffbeb; color: #b45309; padding: 8px 12px; border-radius: 6px; font-size: 0.9rem;'>⚠️ <b>注意：</b>目前顯示的是 {ds}，可能不是即時查詢結果。</div>"
else:
    ds_warning = f"<div style='margin-top: 15px; color: #10b981; font-size: 0.95rem; font-weight:600;'>🟢 資料來源：{ds}</div>"

# 產生 Google Flights 訂票網址
gf_url = f"https://www.google.com/travel/flights?q=Flights%20to%20{destination_code}%20from%20{origin_code}%20on%20{top_flight.get('flight_date', target_date)}"
book_btn_html = f"""
<a href="{gf_url}" target="_blank" style="text-decoration: none; display: inline-block; margin-top: 15px;">
    <div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; padding: 10px 24px; border-radius: 50px; font-weight: 700; font-size: 1rem; box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3); transition: all 0.3s ease; text-align: center;">
        ✈️ 前往 Google Flights 查看與訂票
    </div>
</a>
"""

# ── 最推薦卡片 ──
f_date_top = top_flight.get('flight_date', target_date)
top_h, top_m = divmod(int(top_flight['duration_minutes']), 60)
stops_text = "直飛航班" if top_flight['stops'] == 0 else f"轉機 {top_flight['stops']} 次"

flight_no_display = top_flight.get('flight_no', '')
title_flight_no = f" {flight_no_display}" if flight_no_display and flight_no_display != "未知航班" else ""
compare_links = build_compare_links(origin_code, destination_code, f_date_top, adults_val)

html_content = f"""
<div class="top-flight-card">
<div class="top-flight-title">🏆 首選推薦：{top_flight['airline']}{title_flight_no}</div>
<div style="font-size: 1.12rem; line-height: 2.0; color: #cbd5e1;">
<div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
    <span style="font-size:1.35rem; font-weight:900; color:#f8fafc;">🛫 {origin_display}</span>
    <span style="color:#fbbf24; font-size:1.5rem;">&#x2794;</span>
    <span style="font-size:1.35rem; font-weight:900; color:#f8fafc;">🛬 {dest_display}</span>
</div>
📅 航班日期：<span style="font-weight:700; color:white;">{f_date_top}</span><br>
🕒 <span style="font-weight:700; color:white;">{fmt_time(top_flight['departure_time'])}</span> 起飛 ➔ <span style="font-weight:700; color:white;">{fmt_time(top_flight['arrival_time'])}</span> 抵達 
<span style="background:rgba(255,255,255,0.08); padding:2px 10px; border-radius:6px; font-size:0.9rem; color:#94a3b8;">{top_h}h {top_m}m</span><br>
🔄 轉機資訊：<span style="background:rgba(255,255,255,0.12); padding:3px 12px; border-radius:20px; font-size:0.9rem; font-weight:700; color:#fbbf24; border:1px solid rgba(251,191,36,0.3);">{stops_text}</span>
</div>
<div class="vip-divider">
<div style="margin-bottom: 10px;">
    <span class="price-source-badge">&#x1F4E1; 價格來源：Google Flights 即時爬蟲 (含稅總價)</span>
</div>
<div style="display:flex; align-items:flex-end; gap:20px; flex-wrap:wrap;">
    <div>
        <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:2px; text-transform:uppercase; letter-spacing:1px;">單人票價</div>
        <div style="color:#e2e8f0; font-size:1.25rem; font-weight:700;">{currency} {top_flight['price']:,.0f}</div>
    </div>
    <div>
        <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:2px; text-transform:uppercase; letter-spacing:1px;">人數</div>
        <div style="color:#e2e8f0; font-size:1.25rem; font-weight:700;">{int(adults_val)} 位成人</div>
    </div>
    <div>
        <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:2px; text-transform:uppercase; letter-spacing:1px;">總計票價</div>
        <div style="font-size:2.2rem; font-weight:900; font-family:'Outfit',sans-serif; color:#f59e0b; line-height:1;">{currency} {total_price_val:,.0f}</div>
    </div>
    <div>
        <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:2px; text-transform:uppercase; letter-spacing:1px;">AI 評分</div>
        <div style="font-size:2.2rem; font-weight:900; font-family:'Outfit',sans-serif; color:#38bdf8; line-height:1;">{top_flight['recommendation_score']:.1f}</div>
    </div>
</div>
<div style="margin-top: 16px; background: rgba(59,130,246,0.08); padding: 12px 18px; border-radius: 12px; border-left: 3px solid #38bdf8;">
✨ <b style="color:#93c5fd;">AI 推薦原因</b>：<span style="color:#e2e8f0;">{generate_recommendation_reason(top_flight, preference, 1)}</span>
</div>
{compare_links}
</div>
{ds_warning}
</div>
"""
st.markdown(html_content, unsafe_allow_html=True)

# ── AI 系統聲明與演算法說明 ──
ai_info_html = f"""
<div class="ai-info-card">
    <div class="ai-info-title">
        <div class="ai-info-icon">🤖</div>
        <h4 class="ai-info-heading">AI 推薦演算法與價格資料來源說明</h4>
    </div>
    <div class="ai-info-section algo">
        <b style="color:#1e3a8a; font-size:1.05rem;">📌 計算機票排名的依據為何？</b><br>
        <span style="color:#475569; line-height:1.8;">本系統採用 <b>K-Nearest Neighbors (KNN) 機器學習演算法</b>。系統將所有航班的「票價、飛行時間、轉機次數」特徵經 MinMaxScaler 標準化後，映射至多維度特徵空間。接著根據您設定的「AI 推薦優先順序」（目前策略：<b style="color:#2563eb;">{preference}</b>）賦予不同特徵權重，計算各航班與「完美理想航班（最低價、最短時、零轉機）」的歐氏距離。<b>距離越短 = AI 評分越高</b>。</span>
    </div>
    <div class="ai-info-section price">
        <b style="color:#065f46; font-size:1.05rem;">💰 價格是哪裡的價格資訊？如何進行比價？</b><br>
        <span style="color:#475569; line-height:1.8;">系統顯示的所有票價均由後台即時爬蟲 <b>Google Flights</b> 動態抓取的最新<b>「含稅總價」</b>，並自動換算為 {currency} 顯示。航班編號則與 <b>Flightradar24</b> 跨平台交叉比對。若您希望對比其他平台價格，可點擊上方「比價按鈕」前往 Google Flights、Skyscanner 或 Trip.com 進行即時比較。</span>
    </div>
</div>
"""
st.markdown(ai_info_html, unsafe_allow_html=True)

# -- 頁籤分區 --
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 推薦結果總覽", "✈️ 航班機票列表", "🧠 AI 分群分析", "📊 圖表視覺化", "📥 原始資料下載"
])

with tab1:
    total_shown = len(df_display_result) - 1  # 除首選
    if total_shown <= 0:
        st.markdown(
            f"<div class='no-flight-alert'><h3>💭 目前僅有首選航班符合栀準</h3>"
            f"<p>在 AI 分數 ≥ <b>{score_threshold} 分</b> 的條件下，除首選推薦外沒有其他候補航班。可展開全部航班或調低分數閾值。</p></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h4 style='margin: 10px 0 22px 0; color:#1e293b; font-weight:800;'>"
            f"📝 候補選擇 — AI 分數 ≥ {score_threshold} 分 ({total_shown} 班)</h4>",
            unsafe_allow_html=True
        )
    for idx, row in df_display_result.iterrows():
        rank = int(row["排名"])
        if rank == 1:
            continue # 第一名已顯示於上方
            
        icon = "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        
        f_no = row.get('flight_no', '')
        exp_flight_no = f" {f_no}" if f_no and f_no != "未知航班" else ""
        
        expander_title = f"{icon} {row['airline']}{exp_flight_no} ｜ 總價: {currency} {row.get('total_price', row['price']):,.0f} ｜ 出發: {row.get('flight_date', target_date)} ｜ 🤖 分數: {row['recommendation_score']:.1f}"
        
        with st.expander(expander_title, expanded=(rank==2)):
            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.markdown("<h5 style='font-weight:800; color:#0f172a; margin-bottom:15px;'>🗺️ 航段資訊</h5>", unsafe_allow_html=True)
                
                # 候補選擇改用與首選卡片相同的純文字精簡顯示，保持旅遊質感
                cand_h, cand_m = divmod(int(row['duration_minutes']), 60)
                cand_stops = "直飛航班" if row['stops'] == 0 else f"轉機 {row['stops']} 次"
                
                # 新增票價明細與訂票按鈕於候補卡片中
                total_p = row.get('total_price', row['price'])
                adults_c = row.get('adults', adults)
                f_date = row.get('flight_date', target_date)
                gf_url_cand = f"https://www.google.com/travel/flights?q=Flights%20to%20{destination_code}%20from%20{origin_code}%20on%20{f_date}"
                
                price_html = f"""
<div style="font-size: 1.15rem; line-height: 1.9; color: #334155; margin-bottom: 20px;">
    <span style="font-weight:800; color:#0f172a;">🛫 {origin_display} ➔ 🛬 {dest_display}</span><br>
    🕒 <span style="font-weight:700;">{fmt_time(row['departure_time'])}</span> 起飛 ➔ <span style="font-weight:700;">{fmt_time(row['arrival_time'])}</span> 抵達 ({cand_h}小時 {cand_m}分鐘)<br>
    📅 航班日期：<span style="font-weight:700;">{f_date}</span><br>
    🔄 轉機資訊：<span style="background:#e2e8f0; padding:2px 8px; border-radius:4px; font-size:1rem;">{cand_stops}</span><br>
</div>
<div style="margin-top: 15px; padding: 15px; border-radius: 12px; background: #f8fafc; border: 1px solid #e2e8f0;">
<div style="font-size: 0.95rem; color: #475569; margin-bottom: 5px;">💰 <b>單人票價</b>：{currency} {row['price']:,.0f} &nbsp;|&nbsp; 👥 <b>人數</b>：{int(adults_c)}</div>
<div style="font-size: 1.1rem; color: #0f172a;">💳 <b>總計票價</b>：<b style="color: #ea580c; font-size: 1.3rem;">{currency} {total_p:,.0f}</b></div>
<a href="{gf_url_cand}" target="_blank" style="text-decoration: none; display: inline-block; margin-top: 12px; width: 100%;">
<div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 8px 15px; border-radius: 8px; font-weight: 700; font-size: 0.95rem; text-align: center; transition: all 0.2s;">
✈️ 前往 Google Flights 查看與訂票
</div>
</a>
</div>
"""
                st.markdown(price_html, unsafe_allow_html=True)
                
            with col2:
                pros, cons = generate_pros_cons(row, df_result)
                st.markdown("<h5 style='font-weight:800; color:#16a34a; margin-bottom:10px;'>✅ 航班優點</h5>", unsafe_allow_html=True)
                for p in pros: st.markdown(f"<div style='margin-bottom:8px; color:#334155;'>• {p}</div>", unsafe_allow_html=True)
                
                st.markdown("<h5 style='font-weight:800; color:#ea580c; margin-top:20px; margin-bottom:10px;'>⚠️ 注意事項</h5>", unsafe_allow_html=True)
                for c in cons: st.markdown(f"<div style='margin-bottom:8px; color:#334155;'>• {c}</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h4 style='margin-bottom: 20px; color:#1e293b; font-weight:800;'>完整航班機票數據表</h4>", unsafe_allow_html=True)
    
    display_cols = {
        "排名": "排名", "airline": "航空公司", "flight_no": "航班編號", 
        "origin": "出發地", "destination": "目的地", "route_detail": "航線",
        "departure_time": "出發時間", "arrival_time": "抵達時間", "duration_minutes": "飛行時間（分鐘）",
        "stops": "轉機次數", "price": "單人票價", "adults": "成人數", "total_price": "總票價", "currency": "幣別",
        "recommendation_score": "推薦分數", "cluster_label": "航班類型", "data_source": "資料來源"
    }
    
    df_display = df_result[[c for c in display_cols.keys() if c in df_result.columns]].copy()
    df_display.rename(columns=display_cols, inplace=True)
    
    if "單人票價" in df_display.columns:
        df_display["單人票價"] = df_display["單人票價"].apply(lambda x: f"{currency} {x:,.0f}")
    if "總票價" in df_display.columns:
        df_display["總票價"] = df_display["總票價"].apply(lambda x: f"{currency} {x:,.0f}")
            
    df_display["推薦分數"] = df_display["推薦分數"].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("<h4 style='margin-bottom: 20px; color:#1e293b; font-weight:800;'>機器學習航班分群分析</h4>", unsafe_allow_html=True)
    if pca_df is not None:
        st.plotly_chart(plot_kmeans_scatter(pca_df), use_container_width=True)
    else:
        st.info("💡 " + cluster_msg)

with tab4:
    st.markdown("<h4 style='margin-bottom: 20px; color:#1e293b; font-weight:800;'>視覺化數據分佈</h4>", unsafe_allow_html=True)
    st.plotly_chart(plot_price_bar(df_result), use_container_width=True)
    st.plotly_chart(plot_recommendation_score(df_result), use_container_width=True)
    st.plotly_chart(plot_duration_vs_price_scatter(df_result), use_container_width=True)
    st.plotly_chart(plot_stops_distribution(df_result), use_container_width=True)

with tab5:
    st.markdown("<h4 style='margin-bottom: 20px; color:#1e293b; font-weight:800;'>原始資料下載匯出</h4>", unsafe_allow_html=True)
    csv_buffer = io.StringIO()
    df_display.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 點擊下載機票清單 (CSV)",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name=f"Trip_Search_Results_{origin_code}_{destination_code}.csv",
        mime="text/csv",
        type="primary"
    )
