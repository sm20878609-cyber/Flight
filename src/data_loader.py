"""
src/data_loader.py
資料載入模組：負責載入內建範例資料、快取爬蟲資料，或使用者上傳的 CSV 檔案。
包含機場中英文名稱對照表。
"""

import pandas as pd
import os
from typing import Tuple, Optional

REQUIRED_COLUMNS = [
    "flight_id", "airline", "flight_no", "origin", "destination",
    "departure_time", "arrival_time", "duration_minutes", "stops",
    "price", "total_price", "adults", "currency", "route_detail", "segment_detail", "query_date", "data_source"
]

COLUMN_NAMES_ZH = {
    "flight_id": "航班 ID",
    "airline": "航空公司",
    "flight_no": "航班編號",
    "origin": "出發地",
    "destination": "目的地",
    "departure_time": "出發時間",
    "arrival_time": "抵達時間",
    "duration_minutes": "飛行時間（分鐘）",
    "stops": "轉機次數",
    "price": "機票價格",
    "currency": "幣別",
    "route_detail": "完整航線",
    "segment_detail": "航段明細",
    "query_date": "查詢日期",
    "data_source": "資料來源"
}

AIRPORT_NAMES_ZH = {
    "TPE": "台北桃園 (TPE)", "TSA": "台北松山 (TSA)", "RMQ": "台中清泉崗 (RMQ)", "KHH": "高雄小港 (KHH)",
    "NRT": "東京成田 (NRT)", "HND": "東京羽田 (HND)", "KIX": "大阪關西 (KIX)", "ITM": "大阪伊丹 (ITM)",
    "NGO": "名古屋中部 (NGO)", "CTS": "札幌新千歲 (CTS)", "OKA": "沖繩那霸 (OKA)", "FUK": "福岡 (FUK)",
    "ICN": "首爾仁川 (ICN)", "GMP": "首爾金浦 (GMP)", "PUS": "釜山金海 (PUS)",
    "PVG": "上海浦東 (PVG)", "SHA": "上海虹橋 (SHA)", "PEK": "北京首都 (PEK)", "PKX": "北京大興 (PKX)",
    "CAN": "廣州白雲 (CAN)", "SZX": "深圳寶安 (SZX)", "CTU": "成都天府 (CTU)",
    "HKG": "香港赤鱲角 (HKG)", "MFM": "澳門 (MFM)",
    "SIN": "新加坡樟宜 (SIN)", "BKK": "曼谷素萬那普 (BKK)", "DMK": "曼谷廊曼 (DMK)",
    "KUL": "吉隆坡國際 (KUL)", "MNL": "馬尼拉尼諾依 (MNL)",
    "SGN": "胡志明市 (SGN)", "HAN": "河內內排 (HAN)", "DPS": "峇里島 (DPS)",
    "JKT": "雅加達 (JKT)", "CGK": "雅加達蘇加諾 (CGK)",
    "DEL": "新德里英迪拉甘地 (DEL)", "DXB": "杜拜 (DXB)", "DOH": "多哈哈馬德 (DOH)",
    "SYD": "雪梨 (SYD)", "MEL": "墨爾本 (MEL)", "AKL": "奧克蘭 (AKL)",
    "LHR": "倫敦希斯洛 (LHR)", "CDG": "巴黎戴高樂 (CDG)", "FRA": "法蘭克福 (FRA)",
    "AMS": "阿姆斯特丹史基浦 (AMS)", "ZRH": "蘇黎世 (ZRH)",
    "LAX": "洛杉磯 (LAX)", "SFO": "舊金山 (SFO)", "JFK": "紐約甘迺迪 (JFK)",
    "SEA": "西雅圖 (SEA)", "ORD": "芝加哥奧海爾 (ORD)", "YVR": "溫哥華 (YVR)",
}

def get_airport_display_name(code: str) -> str:
    return AIRPORT_NAMES_ZH.get(code.upper(), code.upper())

def get_airport_code(display_name: str) -> str:
    stripped = display_name.strip()
    if len(stripped) == 3 and stripped.isalpha():
        return stripped.upper()
    import re
    match = re.search(r'\(([A-Z]{3})\)', stripped)
    if match:
        return match.group(1)
    return stripped

def validate_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_zh = [COLUMN_NAMES_ZH.get(col, col) for col in missing]
        return False, "、".join(missing_zh)
    return True, ""

def load_data_from_path(filepath: str, fallback_msg: str) -> Tuple[Optional[pd.DataFrame], str]:
    if not os.path.exists(filepath):
        return None, fallback_msg
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        # 為了相容性，若缺少新欄位則自動補齊
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                if col == "price":
                    df["price"] = df.get("base_price", 0)
                elif col == "total_price":
                    df["total_price"] = df.get("price", 0)
                elif col == "adults":
                    df["adults"] = 1
                elif col == "data_source":
                    df["data_source"] = "內建範例 / 舊版快取"
                else:
                    df[col] = "未知"
        
        ok, msg = validate_columns(df)
        if ok:
            return df, ""
        return None, f"欄位不完整：{msg}"
    except Exception as e:
        return None, f"讀取失敗：{str(e)}"

def load_cached_scraped_data() -> Tuple[Optional[pd.DataFrame], str]:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "cached_scraped_flights.csv")
    df, err = load_data_from_path(path, "找不到快取的爬蟲資料")
    if df is not None:
        df["data_source"] = "快取爬蟲資料"
        return df, f"成功載入快取的爬蟲資料（共 {len(df)} 筆）。"
    return None, err

def load_sample_data() -> Tuple[Optional[pd.DataFrame], str]:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_flights.csv")
    df, err = load_data_from_path(path, "找不到內建範例資料檔案")
    if df is not None:
        df["data_source"] = "內建範例資料"
        return df, f"成功載入內建範例資料（共 {len(df)} 筆）。"
    return None, err

def load_uploaded_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="big5")

        if df.empty:
            return None, "上傳的 CSV 檔案是空的。"

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                if col == "price": df["price"] = df.get("base_price", 0)
                elif col == "total_price": df["total_price"] = df.get("price", 0)
                elif col == "adults": df["adults"] = 1
                elif col == "data_source": df["data_source"] = "使用者上傳 CSV"
                else: df[col] = "未知"

        ok, msg = validate_columns(df)
        if not ok:
            return None, f"上傳的 CSV 欄位不完整。缺少欄位：{msg}"

        return df, f"已載入您上傳的資料（共 {len(df)} 筆）。"
    except Exception as e:
        return None, f"讀取上傳檔案時發生錯誤：{str(e)}"
