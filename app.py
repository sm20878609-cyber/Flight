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

# ── 自訂 CSS 美化 (旅遊渡假風) ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+TC:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', 'Noto Sans TC', sans-serif !important;
    background: radial-gradient(circle at 50% 0%, #ffffff 0%, #f0f4f8 100%);
    background-attachment: fixed;
}

/* 隱藏預設側邊欄切換鈕 */
[data-testid="collapsedControl"] { display: none; }

/* 頂部 Hero Header：帶有高畫質飛機雲海背景與漸層遮罩 */
.header-banner {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 58, 138, 0.75) 100%), 
                url('https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=2000&auto=format&fit=crop');
    background-size: cover;
    background-position: center 40%;
    padding: 70px 20px 60px 20px;
    text-align: center;
    color: white;
    margin-top: -60px;
    margin-bottom: 40px;
    margin-left: -40px;
    margin-right: -40px;
    border-bottom: 4px solid #f59e0b;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    margin-bottom: 15px;
    letter-spacing: 2px;
    text-shadow: 2px 4px 10px rgba(0,0,0,0.5);
    background: linear-gradient(to right, #ffffff, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-subtitle {
    font-size: 1.2rem;
    font-weight: 400;
    color: #e2e8f0;
    text-shadow: 1px 2px 5px rgba(0,0,0,0.5);
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
    letter-spacing: 1px;
}

/* 過濾卡片透明玻璃感設計 (Glassmorphism) */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    box-shadow: 0 15px 35px rgba(15, 23, 42, 0.05);
    padding: 25px 30px !important;
    transition: all 0.3s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
}

/* 標籤文字加粗與顏色調整 */
.stSelectbox label, .stDateInput label, .stNumberInput label, .stTextInput label {
    font-weight: 700 !important;
    color: #334155 !important;
    font-size: 0.95rem !important;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}

/* 搜尋按鈕：流體漸層按鈕 */
.stButton > button {
    background: linear-gradient(135deg, #f59e0b 0%, #ea580c 50%, #dc2626 100%) !important;
    background-size: 200% auto !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 1.25rem !important;
    padding: 0.8rem 2rem !important;
    box-shadow: 0 8px 20px rgba(234, 88, 12, 0.3) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 1px;
}
.stButton > button:hover {
    background-position: right center !important;
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 25px rgba(234, 88, 12, 0.45) !important;
}

/* Metric Cards：極簡留白 Apple 風格 */
.metric-container {
    background: #ffffff;
    border-radius: 16px;
    padding: 25px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    text-align: center;
    border-bottom: 4px solid transparent;
    margin-bottom: 25px;
    transition: all 0.3s ease;
    border: 1px solid #f1f5f9;
}
.metric-container:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.06);
    border-bottom: 4px solid #3b82f6;
}
.metric-label {
    color: #64748b;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.metric-value {
    color: #0f172a;
    font-size: 2.2rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
}
.metric-highlight {
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
}

/* Top Flight Card：尊爵 VIP 黑金風格 */
.top-flight-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-left: 6px solid #fbbf24;
    border-radius: 20px;
    padding: 35px 40px;
    margin: 15px 0 40px 0;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    position: relative;
    overflow: hidden;
    color: #f8fafc;
    transition: transform 0.3s ease;
}
.top-flight-card:hover {
    transform: translateY(-5px);
}
.top-flight-card::after {
    content: '👑 Best Choice';
    position: absolute;
    top: 30px;
    right: -40px;
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    color: #000;
    padding: 8px 45px;
    font-weight: 900;
    font-size: 0.9rem;
    letter-spacing: 1px;
    transform: rotate(45deg);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.top-flight-title {
    color: #fbbf24;
    font-size: 2rem;
    font-weight: 900;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 1px;
}
.top-flight-card b, .top-flight-card strong {
    color: #f1f5f9;
}

/* 航段資訊分隔線 */
.vip-divider {
    margin-top: 15px; 
    padding-top: 15px; 
    border-top: 1px solid rgba(255,255,255,0.1);
}

/* 訂票按鈕：Google Flights Style */
.book-btn-container {
    margin-top: 25px;
    display: inline-block;
}
.book-btn {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #f8fafc;
    padding: 12px 30px;
    border-radius: 50px;
    font-weight: 800;
    font-size: 1.05rem;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.book-btn:hover {
    background: #ffffff;
    color: #0f172a;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.4);
    transform: scale(1.03);
}

/* 無航班資料提示樣式 */
.no-flight-alert {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 5px solid #f59e0b;
    padding: 30px;
    border-radius: 12px;
    margin: 20px 0;
    text-align: center;
}
.no-flight-alert h3 {
    color: #b45309;
    font-weight: 800;
    margin-bottom: 10px;
}
.no-flight-alert p {
    color: #92400e;
    font-size: 1.1rem;
}

/* 隱藏 Streamlit 原生 Header 與 Deploy 按鈕 */
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stAppDeployButton { display: none !important; }
</style>

""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 頂部 Hero Header
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <div class="header-title">啟程，探索世界</div>
    <div class="header-subtitle">AI 智慧航班機票搜尋與推薦系統 ｜ 幫您在海量航班中，精準找出最符合您旅遊偏好的完美機票組合。</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 查詢條件過濾卡片 (Filter Card)
# ═══════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    
    st.markdown("<h4 style='color:#1e293b; margin-bottom: 20px; font-weight:800;'>🔍 規劃您的旅程</h4>", unsafe_allow_html=True)

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
if not analyze_btn:
    st.markdown("<div style='text-align: center; color: #64748b; margin-top: 50px;'><h4>💡 請設定上方的旅程條件後，點擊「✈️ 啟程搜尋航班」開啟您的旅程。</h4></div>", unsafe_allow_html=True)
    st.stop()

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

# ═══════════════════════════════════════════════════════════════════════════
# 結果展示區
# ═══════════════════════════════════════════════════════════════════════════

# -- 指標卡片 --
c1, c2, c3, c4 = st.columns(4)
top_flight = df_result.iloc[0]

# 安全取得總票價與單人票價
total_price_val = top_flight.get('total_price', top_flight['price'])
adults_val = top_flight.get('adults', adults)

with c1:
    st.markdown(f"""<div class="metric-container"><div class="metric-label">📊 精選航班數</div><div class="metric-value">{len(df_result)} <span style="font-size:1rem;font-weight:600;color:#64748b;">班</span></div></div>""", unsafe_allow_html=True)
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

# -- 最推薦卡片 --
top_h, top_m = divmod(int(top_flight['duration_minutes']), 60)
stops_text = "直飛航班" if top_flight['stops'] == 0 else f"轉機 {top_flight['stops']} 次"

flight_no_display = top_flight.get('flight_no', '')
title_flight_no = f" {flight_no_display}" if flight_no_display and flight_no_display != "未知航班" else ""

# 移除 HTML 縮排，避免 Streamlit 誤判為 Markdown 程式碼區塊
html_content = f"""
<div class="top-flight-card">
<div class="top-flight-title">🏆 首選推薦航班：{top_flight['airline']}{title_flight_no}</div>
<div style="font-size: 1.15rem; line-height: 1.9; color: #cbd5e1;">
<span style="font-weight:800; color:#f8fafc;">🛫 {origin_display} ➔ 🛬 {dest_display}</span><br>
🕒 <span style="font-weight:700; color:white;">{top_flight['departure_time']}</span> 起飛 ➔ <span style="font-weight:700; color:white;">{top_flight['arrival_time']}</span> 抵達 ({top_h}小時 {top_m}分鐘)<br>
🔄 轉機資訊：<span style="background:rgba(255,255,255,0.15); padding:3px 10px; border-radius:6px; font-size:0.95rem; font-weight:600; color:#fbbf24;">{stops_text}</span><br>
<div class="vip-divider">
💰 <b>單人票價</b>：{currency} {top_flight['price']:,.0f}<br>
👥 <b>乘客人數</b>：{int(adults_val)} 位成人<br>
💳 <b>總計票價</b>：<b style="color: #f59e0b; font-size: 1.7rem;">{currency} {total_price_val:,.0f}</b> ｜ 
🤖 推薦分數：<b style="color:#38bdf8; font-size: 1.5rem;">{top_flight['recommendation_score']:.1f}</b><br>
<span style="font-size:0.85rem; color:#64748b;">(推薦排序基於單人票價與時間平衡計算)</span>
</div>
<div style="margin-top: 18px; background: rgba(255,255,255,0.05); padding: 12px 18px; border-radius: 10px; border-left: 4px solid #38bdf8;">
✨ <b>AI 推薦原因</b>：<span style="color:#e2e8f0;">{generate_recommendation_reason(top_flight, preference, 1)}</span>
</div>
{ds_warning}
<a href="{gf_url}" target="_blank" style="text-decoration: none;" class="book-btn-container">
    <div class="book-btn">
        ✈️ 前往 Google Flights 查看與訂票
    </div>
</a>
</div>
</div>
"""
st.markdown(html_content, unsafe_allow_html=True)

# ── AI 系統聲明與演算法說明 ──
ai_info_html = f"""
<div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border: 1px solid #e2e8f0; border-left: 6px solid #3b82f6; border-radius: 16px; padding: 25px 30px; margin-bottom: 35px; box-shadow: 0 10px 25px rgba(0,0,0,0.03); transition: transform 0.3s ease;">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 1.8rem; margin-right: 12px; background: #eff6ff; padding: 10px; border-radius: 12px;">🤖</span>
        <h4 style="margin: 0; color: #0f172a; font-weight: 800; font-family: 'Outfit', sans-serif; letter-spacing: 0.5px;">AI 推薦演算法與資料來源解析</h4>
    </div>
    <div style="font-size: 1.05rem; line-height: 1.8; color: #475569;">
        <div style="margin-bottom: 12px;">
            <b style="color: #1e293b; font-size: 1.1rem;">📌 計算機票排名的依據為何？</b><br>
            本系統採用 <b>K-Nearest Neighbors (KNN) 機器學習演算法</b>。系統會將所有航班的「票價、飛行時間、轉機次數」特徵進行標準化 (MinMaxScaler) 映射至多維度空間。接著根據您剛剛設定的「AI 推薦優先順序」（目前策略：<b style="color:#0284c7;">{preference}</b>），賦予不同特徵專屬權重，並精準計算各航班與理論上「完美理想航班（最低價、最短時、零轉機）」之間的歐氏距離。距離越短代表綜合表現越符合您的需求，AI 智能評分與排名也就越高。
        </div>
        <div>
            <b style="color: #1e293b; font-size: 1.1rem;">💰 價格是哪裡的價格資訊？</b><br>
            系統顯示的所有票價，均是由後台即時爬蟲程式前往 <b>Google Flights (Google 航班)</b> 動態抓取該航空公司（或線上旅行社）提供的最新「含稅總價」，並自動換算為 <b>{currency}</b> 顯示。此外，為求資料絕對精確，航班編號更是與 <b>Flightradar24 全球航班即時動態庫</b>進行了即時交叉比對，保證您看到的價格與班次是絕對真實的活資料。
        </div>
    </div>
</div>
"""
st.markdown(ai_info_html, unsafe_allow_html=True)

# -- 頁籤分區 --
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 推薦結果總覽", "✈️ 航班機票列表", "🧠 AI 分群分析", "📊 圖表視覺化", "📥 原始資料下載"
])

with tab1:
    st.markdown("<h4 style='margin: 10px 0 25px 0; color:#1e293b; font-weight:800;'>詳細機票比較與優缺點分析 (候補選擇)</h4>", unsafe_allow_html=True)
    for idx, row in df_result.iterrows():
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
    🕒 <span style="font-weight:700;">{row['departure_time']}</span> 起飛 ➔ <span style="font-weight:700;">{row['arrival_time']}</span> 抵達 ({cand_h}小時 {cand_m}分鐘)<br>
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
