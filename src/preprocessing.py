"""
src/preprocessing.py
資料預處理模組：負責過濾不合法資料、特徵轉換與去重。
"""

import pandas as pd
import numpy as np
from typing import Tuple

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    # 將需要數值的欄位強制轉為數字
    numeric_cols = ["price", "duration_minutes", "stops"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            
    # 移除缺失值與不合法資料
    df_clean.dropna(subset=numeric_cols, inplace=True)
    df_clean = df_clean[df_clean["price"] > 0]
    df_clean = df_clean[df_clean["duration_minutes"] > 0]
    df_clean = df_clean[df_clean["stops"] >= 0]
    
    # 萃取起飛小時 (departure_hour) 作為 ML 特徵
    if "departure_time" in df_clean.columns:
        # 提取出發小時 (用於判斷時段)
        df_clean["departure_hour"] = pd.to_datetime(df_clean["departure_time"], format='mixed', errors="coerce").dt.hour
        # 如果無法解析出時間，填補預設 12 (中午)
        df_clean["departure_hour"] = df_clean["departure_hour"].fillna(12).astype(int)
    else:
        df_clean["departure_hour"] = 12
        
    # 去重：根據 airline, flight_no, departure_time, arrival_time, price
    dup_cols = ["airline", "flight_no", "departure_time", "arrival_time", "price"]
    exist_dup_cols = [c for c in dup_cols if c in df_clean.columns]
    df_clean.drop_duplicates(subset=exist_dup_cols, keep="first", inplace=True)
    
    return df_clean

def filter_flights(df: pd.DataFrame, origin: str, destination: str, max_stops: int) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        return df, "無可用航班"
        
    df_filtered = df[
        (df["origin"].str.upper() == origin.upper()) &
        (df["destination"].str.upper() == destination.upper())
    ].copy()
    
    df_filtered = df_filtered[df_filtered["stops"] <= max_stops]
    
    if df_filtered.empty:
        return df_filtered, "找不到符合您條件的航班（可能已過濾掉轉機次數過多的航班）。"
        
    return df_filtered, "過濾成功"

def full_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """準備後續機器學習所需的特徵"""
    return df.copy()
