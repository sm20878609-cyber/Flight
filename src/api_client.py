"""
src/api_client.py
即時航班 API 客戶端
負責安全管理 API 金鑰，呼叫 SearchAPI Google Flights 端點，
支援多日期批次查詢，並將結果轉換為系統標準的 DataFrame 格式。
"""

import requests
import pandas as pd
import random
import os
import calendar
import datetime
import streamlit as st
from typing import Tuple, Optional, List

# 低成本航空 (LCC) 關鍵字列表
LCC_KEYWORDS = [
    "台灣虎航", "Tigerair", "樂桃航空", "Peach", "酷航", "Scoot", 
    "捷星亞洲", "Jetstar", "亞洲航空", "亞航", "AirAsia", "越捷航空", "VietJet",
    "宿霧太平洋", "Cebu Pacific", "香港快運", "HK Express", 
    "真航空", "Jin Air", "易斯達航空", "Eastar Jet", "濟州航空", "Jeju Air", 
    "泰國獅子航空", "Thai Lion Air", "春秋航空", "Spring Airlines"
]

def get_api_key(user_api_key: Optional[str] = None) -> Optional[str]:
    """安全獲取 API Key"""
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()
    try:
        if "SEARCHAPI_KEY" in st.secrets:
            return st.secrets["SEARCHAPI_KEY"]
    except Exception:
        pass
    env_key = os.getenv("SEARCHAPI_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return None

def get_representative_dates(year: int, month: int) -> List[str]:
    """
    取得月份代表日：
    若該月天數 >= 25，回傳 5 號、15 號、25 號
    若不足 25 天，回傳 5 號、月中、月底
    """
    _, days_in_month = calendar.monthrange(year, month)
    
    if days_in_month >= 25:
        days = [5, 15, 25]
    else:
        mid = days_in_month // 2
        days = [5, mid, days_in_month]
        
    dates = []
    for d in days:
        date_str = f"{year}-{month:02d}-{d:02d}"
        dates.append(date_str)
    return dates

def get_all_dates_in_month(year: int, month: int) -> List[str]:
    """取得該月份的所有日期"""
    _, days_in_month = calendar.monthrange(year, month)
    dates = []
    for d in range(1, days_in_month + 1):
        date_str = f"{year}-{month:02d}-{d:02d}"
        dates.append(date_str)
    return dates

def fetch_live_flights(
    api_key: str, 
    departure_id: str, 
    arrival_id: str, 
    outbound_date: str,
    flight_type: str = "one_way",
    return_date: Optional[str] = None,
    adults: int = 1,
    currency: str = "TWD"
) -> Tuple[Optional[pd.DataFrame], str]:
    """呼叫 SearchAPI Google Flights API 取得單日航班資料"""
    if not api_key:
        return None, "未提供 API Key"
        
    base_url = os.getenv("SEARCHAPI_BASE_URL", "https://www.searchapi.io/api/v1/search")
    
    params = {
        "engine": "google_flights",
        "api_key": api_key,
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "flight_type": flight_type,
        "adults": adults,
        "currency": currency,
        "hl": "zh-tw",
        "gl": "tw"
    }
    
    if flight_type == "round_trip" and return_date:
        params["return_date"] = return_date
        
    try:
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code != 200:
            return None, f"API 請求失敗 (HTTP {response.status_code})"
            
        data = response.json()
        
        all_flights = []
        if "best_flights" in data:
            all_flights.extend(data["best_flights"])
        if "other_flights" in data:
            all_flights.extend(data["other_flights"])
            
        if not all_flights:
            return None, "找不到符合條件的航班。"
            
        rows = []
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        for i, f in enumerate(all_flights):
            legs = f.get("flights", [])
            if not legs:
                continue
                
            first_leg = legs[0]
            last_leg = legs[-1]
            
            flight_id = f"API_{outbound_date.replace('-','')}_{i+1:03d}"
            airline = first_leg.get("airline", "Unknown")
            flight_no = first_leg.get("flight_number", "Unknown")
            dep_time = first_leg.get("departure_airport", {}).get("time", "00:00")
            arr_time = last_leg.get("arrival_airport", {}).get("time", "00:00")
            duration_mins = f.get("total_duration", 0)
            stops = len(legs) - 1
            base_price = f.get("price", 0)
            
            if base_price == 0:
                continue
                
            # 判斷是否為低成本航空
            is_lcc = any(lcc.lower() in airline.lower() for lcc in LCC_KEYWORDS)
            
            # 使用內建估計表補上附加費用
            if is_lcc:
                baggage_fee = 1000
                seat_selection_fee = 250
                meal_fee = 300
                payment_fee = 150
                base_comfort = 3.0
                base_ontime = 0.80
            else:
                baggage_fee = 0
                seat_selection_fee = 150
                meal_fee = 0
                payment_fee = 0
                base_comfort = 4.5
                base_ontime = 0.92
                
            legroom_text = first_leg.get("detected_extensions", {}).get("seat_type", "")
            if "Below Average" in legroom_text:
                base_comfort -= 0.5
            elif "Above Average" in legroom_text:
                base_comfort += 0.5
                
            if first_leg.get("is_often_delayed", False):
                base_ontime -= 0.15
                
            comfort_score = base_comfort + round(random.uniform(-0.2, 0.2), 1)
            on_time_rate = base_ontime + round(random.uniform(-0.03, 0.03), 2)
            
            comfort_score = max(1.0, min(5.0, comfort_score))
            on_time_rate = max(0.5, min(1.0, on_time_rate))
            
            rows.append({
                "flight_id": flight_id,
                "airline": airline,
                "flight_no": flight_no,
                "origin": departure_id,
                "destination": arrival_id,
                "departure_time": dep_time,
                "arrival_time": arr_time,
                "duration_minutes": duration_mins,
                "stops": stops,
                "base_price": base_price,
                "baggage_fee": baggage_fee,
                "seat_selection_fee": seat_selection_fee,
                "meal_fee": meal_fee,
                "payment_fee": payment_fee,
                "comfort_score": round(comfort_score, 1),
                "on_time_rate": round(on_time_rate, 2),
                "query_date": today_str,
                "outbound_date": outbound_date
            })
            
        df = pd.DataFrame(rows)
        
        if df.empty:
            return None, "搜尋成功，但無法解析航班票價資訊。"
            
        return df, "成功"
        
    except Exception as e:
        return None, f"API 查詢發生錯誤: {str(e)}"

def search_flights_multiple_dates(
    departure_id: str,
    arrival_id: str,
    dates: List[str],
    adults: int,
    currency: str,
    flight_type: str,
    api_key: str
) -> Tuple[Optional[pd.DataFrame], str]:
    """多日期批次查詢，並在 GUI 顯示進度與去重"""
    if not api_key:
        return None, "未提供 API Key"
        
    all_dfs = []
    success_count = 0
    fail_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, date_str in enumerate(dates):
        status_text.write(f"正在查詢 {date_str} 的航班資料... ({i+1}/{len(dates)})")
        
        df, msg = fetch_live_flights(
            api_key=api_key,
            departure_id=departure_id,
            arrival_id=arrival_id,
            outbound_date=date_str,
            flight_type=flight_type,
            adults=adults,
            currency=currency
        )
        
        if df is not None:
            all_dfs.append(df)
            success_count += 1
        else:
            st.warning(f"⚠️ {date_str} 查詢失敗: {msg}")
            fail_count += 1
            
        progress_bar.progress((i + 1) / len(dates))
        
    status_text.empty()
    progress_bar.empty()
    
    if not all_dfs:
        return None, f"所有 {len(dates)} 天的查詢皆失敗，無資料可分析。"
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # 去除重複航班 (保留第一筆)
    # 根據: airline, flight_no, departure_time, arrival_time, base_price 去重
    before_len = len(final_df)
    final_df.drop_duplicates(
        subset=["airline", "flight_no", "departure_time", "arrival_time", "base_price"],
        keep="first",
        inplace=True
    )
    after_len = len(final_df)
    
    # 成功後寫入快取檔案
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "cached_api_flights.csv")
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        final_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass
        
    msg = f"查詢完成！成功 {success_count} 天，失敗 {fail_count} 天。共取得 {before_len} 筆資料，去重後剩餘 {after_len} 筆。"
    return final_df, msg
