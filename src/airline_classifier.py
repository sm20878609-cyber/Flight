"""
src/airline_classifier.py
航空公司分類模組：定義廉航（LCC）與全服務航空（FSC）清單
資料來源：IATA、各航空公司官網、業界公認分類
"""

# ── 廉航（LCC / Low-Cost Carrier）清單 ──────────────────────────────────────
# 以下航空公司以「低成本、使用者付費、精簡服務」為商業模式
LCC_AIRLINES = {
    # 台灣
    "Tigerair Taiwan",          # 台灣虎航 (IT)
    # 日本
    "Peach Aviation",           # 樂桃航空 (MM)
    "Jetstar Japan",            # 捷星日本 (GK)
    "Spring Japan",             # 春秋航空日本 (IJ)
    "ZIP Air",                  # 捷星旗下 (ZG)
    # 新加坡/大洋洲
    "Scoot",                    # 酷航 (TR) - 新加坡航空旗下LCC
    "Jetstar",                  # 捷星航空 (JQ) - 澳洲/大洋洲
    "Jetstar Asia",             # 捷星亞洲 (3K)
    # 香港
    "Hong Kong Express",        # 香港快運 (UO) - 國泰旗下LCC
    # 韓國
    "Jeju Air",                 # 濟州航空 (7C)
    "Jin Air",                  # 真航空 (LJ)
    "Eastar Jet",               # 易斯達航空 (ZE)
    "T'way Air",                # 德威航空 (TW)
    "Air Busan",                # 釜山航空 (BX)
    "Air Seoul",                # 首爾航空 (RS)
    # 馬來西亞/東南亞
    "AirAsia",                  # 亞洲航空 (AK)
    "AirAsia X",                # 亞洲航空X (D7) - 長程LCC
    "Batik Air",                # 峇迪航空 (OD) - 前馬印航空
    "Malindo Air",              # 馬林多航空
    # 越南
    "VietJet Air",              # 越捷航空 (VJ)
    "Bamboo Airways",           # 竹子航空 (QH) - 混合模式
    # 菲律賓
    "Cebu Pacific",             # 宿霧太平洋 (5J)
    "Philippine AirAsia",       # 菲律賓亞洲航空
    # 泰國
    "Thai Lion Air",            # 泰國獅航 (SL)
    "Thai AirAsia",             # 泰國亞洲航空 (FD)
    "NokAir",                   # 鳥航 (DD)
    # 印尼
    "Lion Air",                 # 獅航 (JT)
    "Indonesia AirAsia",
    "Wings Air",
    # 中國
    "Spring Airlines",          # 春秋航空 (9C)
    "Lucky Air",                # 祥鵬航空 (8L)
    "West Air",                 # 重慶航空 (PN)
    "Ruili Airlines",           # 瑞麗航空
    # 印度
    "IndiGo",                   # 靛藍航空 (6E)
    "SpiceJet",                 # 香料航空 (SG)
    "GoAir",                    # 印度廉航
    # 歐洲
    "Ryanair",                  # 瑞安航空 (FR)
    "EasyJet",                  # 易捷航空 (U2)
    "Wizz Air",                 # 威茲航空 (W6)
    "Norwegian",                # 挪威航空 (DY)
    "Vueling",                  # 伏林航空 (VY)
    "Transavia",                # 全歐洲航空
    # 美洲
    "Southwest Airlines",       # 西南航空 (WN)
    "Spirit Airlines",          # 精神航空 (NK)
    "Frontier Airlines",        # 邊疆航空 (F9)
    "Allegiant Air",            # 快樂航空 (G4)
    "Sun Country Airlines",
    "Volaris",                  # 墨西哥廉航
    "VivaAerobus",
    # 中東/非洲
    "FlyDubai",                 # 迪拜廉航 (FZ)
    "Air Arabia",               # 阿拉伯航空 (G9)
    "Flydeal",
}

# ── 全服務航空（FSC / Full-Service Carrier）清單 ──────────────────────────────
FSC_AIRLINES = {
    # 台灣
    "EVA Air",                  # 長榮航空 (BR)
    "China Airlines",           # 中華航空 (CI)
    "Starlux Airlines",         # 星宇航空 (JX) - 精品全服務
    # 日本
    "ANA",                      # 全日空 (NH)
    "JAL",                      # 日本航空 (JL)
    # 韓國
    "Korean Air",               # 大韓航空 (KE)
    "Asiana Airlines",          # 韓亞航空 (OZ)
    # 香港/中國
    "Cathay Pacific",           # 國泰航空 (CX)
    "Hong Kong Airlines",       # 香港航空 (HX)
    "Air Macau",                # 澳門航空 (NX)
    "China Eastern",            # 中國東方航空 (MU)
    "China Southern",           # 中國南方航空 (CZ)
    "Air China",                # 中國國際航空 (CA)
    "XiamenAir",                # 廈門航空 (MF)
    "Hainan Airlines",          # 海南航空 (HU)
    # 東南亞
    "Singapore Airlines",       # 新加坡航空 (SQ)
    "Thai Airways",             # 泰國國際航空 (TG)
    "Malaysia Airlines",        # 馬來西亞航空 (MH)
    "Garuda Indonesia",         # 印尼鷹航 (GA)
    "Vietnam Airlines",         # 越南航空 (VN)
    "Philippine Airlines",      # 菲律賓航空 (PR)
    "Royal Brunei Airlines",    # 汶萊皇家航空 (BI)
    # 中東
    "Emirates",                 # 阿聯酋航空 (EK)
    "Qatar Airways",            # 卡達航空 (QR)
    "Etihad Airways",           # 阿提哈德航空 (EY)
    "Gulf Air",                 # 海灣航空 (GF)
    # 南亞
    "Air India",                # 印度航空 (AI)
    "SriLankan Airlines",       # 斯里蘭卡航空 (UL)
    # 歐洲
    "Lufthansa",                # 漢莎航空 (LH)
    "Air France",               # 法國航空 (AF)
    "British Airways",          # 英國航空 (BA)
    "KLM",                      # 荷蘭皇家航空 (KL)
    "Swiss International Air Lines",
    "Austrian Airlines",
    "Finnair",                  # 芬蘭航空 (AY)
    "Iberia",                   # 伊比利亞航空 (IB)
    "Alitalia",                 # 義大利航空
    "Turkish Airlines",         # 土耳其航空 (TK)
    "Aeroflot",                 # 俄羅斯航空 (SU)
    # 美洲
    "United Airlines",          # 聯合航空 (UA)
    "Delta Air Lines",          # 達美航空 (DL)
    "American Airlines",        # 美國航空 (AA)
    "Air Canada",               # 加拿大航空 (AC)
    "LATAM Airlines",           # 拉丁美洲航空
    "Avianca",
    # 大洋洲
    "Qantas",                   # 澳洲航空 (QF)
    "Air New Zealand",          # 紐西蘭航空 (NZ)
    "Fiji Airways",
}


def classify_airline(airline_name: str) -> str:
    """
    分類航空公司類型
    返回: '廉航 (LCC)' | '全服務航空 (FSC)' | '未知'
    """
    if not airline_name:
        return "未知"
    primary = airline_name.split(",")[0].strip()

    # 精確匹配
    if primary in LCC_AIRLINES:
        return "廉航 (LCC)"
    if primary in FSC_AIRLINES:
        return "全服務航空 (FSC)"

    # 模糊匹配（部分名稱）
    for name in LCC_AIRLINES:
        if name.lower() in primary.lower() or primary.lower() in name.lower():
            return "廉航 (LCC)"
    for name in FSC_AIRLINES:
        if name.lower() in primary.lower() or primary.lower() in name.lower():
            return "全服務航空 (FSC)"

    return "未知"
