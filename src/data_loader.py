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
    # ── 台灣 ──────────────────────────────────────────────────────────────────
    "TPE": "台北桃園 (TPE)", "TSA": "台北松山 (TSA)",
    "RMQ": "台中清泉崗 (RMQ)", "KHH": "高雄小港 (KHH)", "TNN": "台南 (TNN)",

    # ── 日本 ──────────────────────────────────────────────────────────────────
    "NRT": "東京成田 (NRT)", "HND": "東京羽田 (HND)",
    "KIX": "大阪關西 (KIX)", "ITM": "大阪伊丹 (ITM)",
    "NGO": "名古屋中部 (NGO)", "CTS": "札幌新千歲 (CTS)",
    "OKA": "沖繩那霸 (OKA)", "FUK": "福岡 (FUK)",
    "HIJ": "廣島 (HIJ)", "SDJ": "仙台 (SDJ)", "KMJ": "熊本 (KMJ)",
    "KOJ": "鹿兒島 (KOJ)", "OIT": "大分 (OIT)", "NGS": "長崎 (NGS)",
    "UKB": "神戶 (UKB)", "TKS": "德島 (TKS)", "MYJ": "松山 (MYJ)",
    "TAK": "高松 (TAK)", "KCZ": "高知 (KCZ)", "AOJ": "青森 (AOJ)",
    "HKD": "函館 (HKD)", "OKJ": "岡山 (OKJ)", "KMQ": "小松 (KMQ)",

    # ── 韓國 ──────────────────────────────────────────────────────────────────
    "ICN": "首爾仁川 (ICN)", "GMP": "首爾金浦 (GMP)",
    "PUS": "釜山金海 (PUS)", "CJU": "濟州 (CJU)",
    "TAE": "大邱 (TAE)", "CJJ": "清州 (CJJ)", "RSU": "麗水順天 (RSU)",

    # ── 中國大陸 ────────────────────────────────────────────────────────────
    "PVG": "上海浦東 (PVG)", "SHA": "上海虹橋 (SHA)",
    "PEK": "北京首都 (PEK)", "PKX": "北京大興 (PKX)",
    "CAN": "廣州白雲 (CAN)", "SZX": "深圳寶安 (SZX)",
    "CTU": "成都天府 (CTU)", "WUH": "武漢天河 (WUH)",
    "CKG": "重慶江北 (CKG)", "XMN": "廈門高崎 (XMN)",
    "HGH": "杭州蕭山 (HGH)", "NKG": "南京祿口 (NKG)",
    "CGO": "鄭州新鄭 (CGO)", "CSX": "長沙黃花 (CSX)",
    "KMG": "昆明長水 (KMG)", "TAO": "青島膠東 (TAO)",
    "TNA": "濟南遙牆 (TNA)", "HAK": "海口美蘭 (HAK)",
    "SYX": "三亞鳳凰 (SYX)", "URC": "烏魯木齊地窩堡 (URC)",
    "DLC": "大連周水子 (DLC)", "SHE": "瀋陽桃仙 (SHE)",
    "HRB": "哈爾濱太平 (HRB)", "TYN": "太原武宿 (TYN)",
    "KWL": "桂林兩江 (KWL)", "NNG": "南寧吳圩 (NNG)",
    "WNZ": "溫州龍灣 (WNZ)", "FOC": "福州長樂 (FOC)",

    # ── 香港 / 澳門 ────────────────────────────────────────────────────────
    "HKG": "香港赤鱲角 (HKG)", "MFM": "澳門 (MFM)",

    # ── 東南亞：新加坡、泰國、馬來西亞 ────────────────────────────────────
    "SIN": "新加坡樟宜 (SIN)",
    "BKK": "曼谷素萬那普 (BKK)", "DMK": "曼谷廊曼 (DMK)",
    "HKT": "普吉島 (HKT)", "CNX": "清邁 (CNX)", "USM": "蘇梅島 (USM)",
    "KUL": "吉隆坡國際 (KUL)", "SZB": "吉隆坡蘇邦 (SZB)",
    "BKI": "亞庇 (BKI)", "KCH": "古晉 (KCH)", "PEN": "檳城 (PEN)",
    "LGK": "浮羅交怡 (LGK)", "JHB": "新山 (JHB)",

    # ── 東南亞：越南、菲律賓、印尼 ────────────────────────────────────────
    "SGN": "胡志明市 (SGN)", "HAN": "河內內排 (HAN)",
    "DAD": "峴港 (DAD)", "CXR": "金蘭灣 (CXR)", "PQC": "富國島 (PQC)",
    "MNL": "馬尼拉尼諾依 (MNL)", "CEB": "宿霧麥克坦 (CEB)",
    "DVO": "達沃 (DVO)", "ILO": "伊洛伊洛 (ILO)",
    "CGK": "雅加達蘇加諾 (CGK)", "DPS": "峇里島 (DPS)",
    "SUB": "泗水朱安達 (SUB)", "UPG": "望加錫 (UPG)",
    "KNO": "棉蘭 (KNO)", "BPN": "巴里巴板 (BPN)",

    # ── 東南亞：緬甸、柬埔寨、寮國 ────────────────────────────────────────
    "RGN": "仰光 (RGN)", "MDL": "曼德勒 (MDL)",
    "PNH": "金邊 (PNH)", "REP": "暹粒吳哥 (REP)",
    "VTE": "永珍 (VTE)", "LPQ": "龍坡邦 (LPQ)",

    # ── 南亞：印度、斯里蘭卡、尼泊爾、孟加拉 ────────────────────────────
    "DEL": "新德里英迪拉甘地 (DEL)", "BOM": "孟買查特拉帕蒂 (BOM)",
    "BLR": "班加羅爾 (BLR)", "MAA": "清奈 (MAA)",
    "CCU": "加爾各答 (CCU)", "HYD": "海德拉巴 (HYD)",
    "AMD": "艾哈邁達巴德 (AMD)", "COK": "科欽 (COK)",
    "GOI": "果阿 (GOI)", "JAI": "齋浦爾 (JAI)",
    "CMB": "可倫坡班達拉奈克 (CMB)",
    "KTM": "加德滿都特里布萬 (KTM)",
    "DAC": "達卡沙賈拉勒 (DAC)",

    # ── 中東 ──────────────────────────────────────────────────────────────────
    "DXB": "杜拜國際 (DXB)", "DOH": "多哈哈馬德 (DOH)",
    "AUH": "阿布達比 (AUH)", "RUH": "利雅德阿卜杜勒阿濟茲 (RUH)",
    "JED": "吉達阿卜杜勒阿濟茲 (JED)", "MED": "麥地那 (MED)",
    "MCT": "馬斯喀特 (MCT)", "BAH": "巴林 (BAH)",
    "KWI": "科威特 (KWI)", "AMM": "安曼皇后阿利亞 (AMM)",
    "BEY": "貝魯特 (BEY)", "TLV": "台拉維夫本古里安 (TLV)",
    "IST": "伊斯坦堡 (IST)", "SAW": "伊斯坦堡薩比哈 (SAW)",
    "AYT": "安塔利亞 (AYT)", "ESB": "安卡拉埃森博阿 (ESB)",

    # ── 中亞 / 南高加索 ────────────────────────────────────────────────────
    "ALA": "阿拉木圖 (ALA)", "NQZ": "努爾蘇丹 (NQZ)",
    "GYD": "巴庫海達爾阿利耶夫 (GYD)", "EVN": "葉里溫斯瓦納茨 (EVN)",

    # ── 澳大利亞 / 大洋洲 ─────────────────────────────────────────────────
    "SYD": "雪梨 (SYD)", "MEL": "墨爾本 (MEL)",
    "BNE": "布里斯本 (BNE)", "PER": "伯斯 (PER)",
    "ADL": "阿德雷德 (ADL)", "OOL": "黃金海岸 (OOL)",
    "CNS": "凱恩斯 (CNS)", "DRW": "達爾文 (DRW)",
    "CBR": "坎培拉 (CBR)", "HBA": "荷伯特 (HBA)",
    "AKL": "奧克蘭 (AKL)", "CHC": "基督城 (CHC)",
    "WLG": "威靈頓 (WLG)", "ZQN": "皇后鎮 (ZQN)",
    "NAN": "斐濟楠迪 (NAN)", "PPT": "大溪地法阿 (PPT)",

    # ── 英國 / 愛爾蘭 ──────────────────────────────────────────────────────
    "LHR": "倫敦希斯洛 (LHR)", "LGW": "倫敦蓋特威克 (LGW)",
    "STN": "倫敦史坦斯特 (STN)", "LTN": "倫敦盧頓 (LTN)",
    "LCY": "倫敦城市 (LCY)", "MAN": "曼徹斯特 (MAN)",
    "BHX": "伯明罕 (BHX)", "EDI": "愛丁堡 (EDI)",
    "GLA": "格拉斯哥 (GLA)", "DUB": "都柏林 (DUB)",

    # ── 西歐 ──────────────────────────────────────────────────────────────────
    "CDG": "巴黎戴高樂 (CDG)", "ORY": "巴黎奧利 (ORY)",
    "FRA": "法蘭克福 (FRA)", "MUC": "慕尼黑 (MUC)",
    "DUS": "杜塞道夫 (DUS)", "HAM": "漢堡 (HAM)",
    "TXL": "柏林布蘭登堡 (BER)", "BER": "柏林布蘭登堡 (BER)",
    "AMS": "阿姆斯特丹史基浦 (AMS)", "BRU": "布魯塞爾 (BRU)",
    "ZRH": "蘇黎世 (ZRH)", "GVA": "日內瓦 (GVA)",
    "VIE": "維也納 (VIE)", "MAD": "馬德里巴拉哈斯 (MAD)",
    "BCN": "巴塞隆納 (BCN)", "LIS": "里斯本 (LIS)",
    "FCO": "羅馬費烏米奇諾 (FCO)", "MXP": "米蘭馬爾彭薩 (MXP)",
    "NAP": "拿坡里 (NAP)", "VCE": "威尼斯馬可波羅 (VCE)",
    "ATH": "雅典埃萊夫塞里斯 (ATH)", "HER": "克里特島 (HER)",
    "SKG": "塞薩洛尼基 (SKG)", "OPO": "波爾圖 (OPO)",
    "SVQ": "塞維亞 (SVQ)", "AGP": "馬拉加 (AGP)",
    "PMI": "帕爾馬馬約卡 (PMI)", "LPA": "大加那利島 (LPA)",
    "TFS": "特內里費南 (TFS)",

    # ── 北歐 ──────────────────────────────────────────────────────────────────
    "ARN": "斯德哥爾摩阿蘭達 (ARN)", "CPH": "哥本哈根 (CPH)",
    "OSL": "奧斯陸加勒穆恩 (OSL)", "HEL": "赫爾辛基萬塔 (HEL)",
    "REK": "雷克雅維克 (KEF)",

    # ── 東歐 / 俄羅斯 ─────────────────────────────────────────────────────
    "WAW": "華沙肖邦 (WAW)", "PRG": "布拉格瓦茨拉夫 (PRG)",
    "BUD": "布達佩斯 (BUD)", "BUH": "布加勒斯特奧托佩尼 (OTP)",
    "OTP": "布加勒斯特奧托佩尼 (OTP)", "SOF": "索非亞 (SOF)",
    "ZAG": "薩格勒布 (ZAG)", "LJU": "盧布爾雅那 (LJU)",
    "BEG": "貝爾格勒 (BEG)", "KBP": "基輔鮑里斯皮爾 (KBP)",
    "SVO": "莫斯科謝列梅季耶沃 (SVO)", "LED": "聖彼得堡普爾科沃 (LED)",
    "VKO": "莫斯科伏努科沃 (VKO)",

    # ── 非洲 ──────────────────────────────────────────────────────────────────
    "JNB": "約翰尼斯堡奧坦博 (JNB)", "CPT": "開普敦 (CPT)",
    "DUR": "德班金王科德 (DUR)", "CAI": "開羅 (CAI)",
    "CMN": "卡薩布蘭卡穆罕默德五世 (CMN)", "ALG": "阿爾及爾 (ALG)",
    "TUN": "突尼斯卡塔基 (TUN)", "LOS": "拉各斯穆爾塔拉 (LOS)",
    "ADD": "阿迪斯阿貝巴博萊 (ADD)", "NBO": "奈洛比喬莫肯雅塔 (NBO)",
    "DAR": "三蘭港朱利葉斯尼雷爾 (DAR)", "ACC": "阿克拉科托卡 (ACC)",
    "ABJ": "阿比讓 (ABJ)", "TNR": "安塔那那利佛 (TNR)",
    "RUN": "留尼旺羅蘭加羅斯 (RUN)", "MRU": "模里西斯 (MRU)",

    # ── 北美：美國 ────────────────────────────────────────────────────────
    "LAX": "洛杉磯 (LAX)", "SFO": "舊金山 (SFO)",
    "JFK": "紐約甘迺迪 (JFK)", "EWR": "紐約紐瓦克 (EWR)",
    "LGA": "紐約拉瓜迪亞 (LGA)", "ORD": "芝加哥奧海爾 (ORD)",
    "MDW": "芝加哥中途島 (MDW)", "ATL": "亞特蘭大哈茨菲爾德 (ATL)",
    "DFW": "達拉斯沃斯堡 (DFW)", "MIA": "邁阿密 (MIA)",
    "MCO": "奧蘭多 (MCO)", "SEA": "西雅圖塔科馬 (SEA)",
    "LAS": "拉斯維加斯麥卡倫 (LAS)", "DEN": "丹佛 (DEN)",
    "PHX": "鳳凰城天港 (PHX)", "IAD": "華盛頓杜勒斯 (IAD)",
    "DCA": "華盛頓雷根 (DCA)", "BOS": "波士頓洛根 (BOS)",
    "MSP": "明尼阿波利斯 (MSP)", "DTW": "底特律韋恩郡 (DTW)",
    "PHL": "費城 (PHL)", "CLT": "夏洛特道格拉斯 (CLT)",
    "SAN": "聖地牙哥 (SAN)", "TPA": "坦帕 (TPA)",
    "PDX": "波特蘭 (PDX)", "HNL": "火奴魯魯 (HNL)",
    "ANC": "安克拉治 (ANC)", "IAH": "休士頓喬治布希 (IAH)",
    "HOU": "休士頓霍比 (HOU)", "SLC": "鹽湖城 (SLC)",
    "BNA": "納許維爾 (BNA)", "MEM": "孟菲斯 (MEM)",
    "STL": "聖路易斯 (STL)", "BWI": "巴爾的摩 (BWI)",

    # ── 北美：加拿大 ──────────────────────────────────────────────────────
    "YVR": "溫哥華 (YVR)", "YYZ": "多倫多皮爾遜 (YYZ)",
    "YUL": "蒙特婁特魯多 (YUL)", "YYC": "卡加利 (YYC)",
    "YEG": "埃德蒙頓 (YEG)", "YOW": "渥太華 (YOW)",

    # ── 中美洲 / 加勒比海 ────────────────────────────────────────────────
    "MEX": "墨西哥城貝尼托胡亞雷斯 (MEX)", "GDL": "瓜達拉哈拉 (GDL)",
    "CUN": "坎昆 (CUN)", "SJO": "聖荷西胡安聖馬丁 (SJO)",
    "GUA": "瓜地馬拉城 (GUA)", "BOG": "波哥大埃爾多拉多 (BOG)",
    "MGA": "馬拿瓜奧古斯托 (MGA)", "SAL": "聖薩爾瓦多 (SAL)",

    # ── 南美洲 ────────────────────────────────────────────────────────────
    "GRU": "聖保羅瓜盧柳斯 (GRU)", "GIG": "里約熱內盧加利昂 (GIG)",
    "SDU": "里約熱內盧桑托斯杜蒙 (SDU)", "BSB": "巴西利亞 (BSB)",
    "CNF": "貝洛奧里藏特 (CNF)", "FOR": "福塔萊薩 (FOR)",
    "SSA": "薩爾瓦多 (SSA)", "EZE": "布宜諾斯艾利斯埃塞薩 (EZE)",
    "AEP": "布宜諾斯艾利斯豪爾赫紐伯里 (AEP)",
    "SCL": "聖地牙哥科莫多羅 (SCL)", "LIM": "利馬豪爾赫夏維茲 (LIM)",
    "GYE": "瓜亞基爾 (GYE)", "UIO": "基多馬里斯卡爾蘇克雷 (UIO)",
    "MVD": "蒙得維的亞 (MVD)", "ASU": "亞松森 (ASU)",
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
