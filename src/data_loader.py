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
    "TPE": "【台灣】台北桃園 (TPE)", "TSA": "【台灣】台北松山 (TSA)",
    "RMQ": "【台灣】台中清泉崗 (RMQ)", "KHH": "【台灣】高雄小港 (KHH)", "TNN": "【台灣】台南 (TNN)",

    # ── 日本 ──────────────────────────────────────────────────────────────────
    "NRT": "【東北亞】東京成田 (NRT)", "HND": "【東北亞】東京羽田 (HND)",
    "KIX": "【東北亞】大阪關西 (KIX)", "ITM": "【東北亞】大阪伊丹 (ITM)",
    "NGO": "【東北亞】名古屋中部 (NGO)", "CTS": "【東北亞】札幌新千歲 (CTS)",
    "OKA": "【東北亞】沖繩那霸 (OKA)", "FUK": "【東北亞】福岡 (FUK)",
    "HIJ": "【東北亞】廣島 (HIJ)", "SDJ": "【東北亞】仙台 (SDJ)", "KMJ": "【東北亞】熊本 (KMJ)",
    "KOJ": "【東北亞】鹿兒島 (KOJ)", "OIT": "【東北亞】大分 (OIT)", "NGS": "【東北亞】長崎 (NGS)",
    "UKB": "【東北亞】神戶 (UKB)", "TKS": "【東北亞】德島 (TKS)", "MYJ": "【東北亞】松山 (MYJ)",
    "TAK": "【東北亞】高松 (TAK)", "KCZ": "【東北亞】高知 (KCZ)", "AOJ": "【東北亞】青森 (AOJ)",
    "HKD": "【東北亞】函館 (HKD)", "OKJ": "【東北亞】岡山 (OKJ)", "KMQ": "【東北亞】小松 (KMQ)",

    # ── 韓國 ──────────────────────────────────────────────────────────────────
    "ICN": "【東北亞】首爾仁川 (ICN)", "GMP": "【東北亞】首爾金浦 (GMP)",
    "PUS": "【東北亞】釜山金海 (PUS)", "CJU": "【東北亞】濟州 (CJU)",
    "TAE": "【東北亞】大邱 (TAE)", "CJJ": "【東北亞】清州 (CJJ)", "RSU": "【東北亞】麗水順天 (RSU)",

    # ── 中國大陸 ────────────────────────────────────────────────────────────
    "PVG": "【中港澳】上海浦東 (PVG)", "SHA": "【中港澳】上海虹橋 (SHA)",
    "PEK": "【中港澳】北京首都 (PEK)", "PKX": "【中港澳】北京大興 (PKX)",
    "CAN": "【中港澳】廣州白雲 (CAN)", "SZX": "【中港澳】深圳寶安 (SZX)",
    "CTU": "【中港澳】成都天府 (CTU)", "WUH": "【中港澳】武漢天河 (WUH)",
    "CKG": "【中港澳】重慶江北 (CKG)", "XMN": "【中港澳】廈門高崎 (XMN)",
    "HGH": "【中港澳】杭州蕭山 (HGH)", "NKG": "【中港澳】南京祿口 (NKG)",
    "CGO": "【中港澳】鄭州新鄭 (CGO)", "CSX": "【中港澳】長沙黃花 (CSX)",
    "KMG": "【中港澳】昆明長水 (KMG)", "TAO": "【中港澳】青島膠東 (TAO)",
    "TNA": "【中港澳】濟南遙牆 (TNA)", "HAK": "【中港澳】海口美蘭 (HAK)",
    "SYX": "【中港澳】三亞鳳凰 (SYX)", "URC": "【中港澳】烏魯木齊地窩堡 (URC)",
    "DLC": "【中港澳】大連周水子 (DLC)", "SHE": "【中港澳】瀋陽桃仙 (SHE)",
    "HRB": "【中港澳】哈爾濱太平 (HRB)", "TYN": "【中港澳】太原武宿 (TYN)",
    "KWL": "【中港澳】桂林兩江 (KWL)", "NNG": "【中港澳】南寧吳圩 (NNG)",
    "WNZ": "【中港澳】溫州龍灣 (WNZ)", "FOC": "【中港澳】福州長樂 (FOC)",

    # ── 香港 / 澳門 ────────────────────────────────────────────────────────
    "HKG": "【中港澳】香港赤鱲角 (HKG)", "MFM": "【中港澳】澳門 (MFM)",

    # ── 東南亞：新加坡、泰國、馬來西亞 ────────────────────────────────────
    "SIN": "【東南亞】新加坡樟宜 (SIN)",
    "BKK": "【東南亞】曼谷素萬那普 (BKK)", "DMK": "【東南亞】曼谷廊曼 (DMK)",
    "HKT": "【東南亞】普吉島 (HKT)", "CNX": "【東南亞】清邁 (CNX)", "USM": "【東南亞】蘇梅島 (USM)",
    "KUL": "【東南亞】吉隆坡國際 (KUL)", "SZB": "【東南亞】吉隆坡蘇邦 (SZB)",
    "BKI": "【東南亞】亞庇 (BKI)", "KCH": "【東南亞】古晉 (KCH)", "PEN": "【東南亞】檳城 (PEN)",
    "LGK": "【東南亞】浮羅交怡 (LGK)", "JHB": "【東南亞】新山 (JHB)",

    # ── 東南亞：越南、菲律賓、印尼 ────────────────────────────────────────
    "SGN": "【東南亞】胡志明市 (SGN)", "HAN": "【東南亞】河內內排 (HAN)",
    "DAD": "【東南亞】峴港 (DAD)", "CXR": "【東南亞】金蘭灣 (CXR)", "PQC": "【東南亞】富國島 (PQC)",
    "MNL": "【東南亞】馬尼拉尼諾依 (MNL)", "CEB": "【東南亞】宿霧麥克坦 (CEB)",
    "DVO": "【東南亞】達沃 (DVO)", "ILO": "【東南亞】伊洛伊洛 (ILO)",
    "CGK": "【東南亞】雅加達蘇加諾 (CGK)", "DPS": "【東南亞】峇里島 (DPS)",
    "SUB": "【東南亞】泗水朱安達 (SUB)", "UPG": "【東南亞】望加錫 (UPG)",
    "KNO": "【東南亞】棉蘭 (KNO)", "BPN": "【東南亞】巴里巴板 (BPN)",

    # ── 東南亞：緬甸、柬埔寨、寮國 ────────────────────────────────────────
    "RGN": "【東南亞】仰光 (RGN)", "MDL": "【東南亞】曼德勒 (MDL)",
    "PNH": "【東南亞】金邊 (PNH)", "REP": "【東南亞】暹粒吳哥 (REP)",
    "VTE": "【東南亞】永珍 (VTE)", "LPQ": "【東南亞】龍坡邦 (LPQ)",

    # ── 南亞：印度、斯里蘭卡、尼泊爾、孟加拉 ────────────────────────────
    "DEL": "【南亞】新德里英迪拉甘地 (DEL)", "BOM": "【南亞】孟買查特拉帕蒂 (BOM)",
    "BLR": "【南亞】班加羅爾 (BLR)", "MAA": "【南亞】清奈 (MAA)",
    "CCU": "【南亞】加爾各答 (CCU)", "HYD": "【南亞】海德拉巴 (HYD)",
    "AMD": "【南亞】艾哈邁達巴德 (AMD)", "COK": "【南亞】科欽 (COK)",
    "GOI": "【南亞】果阿 (GOI)", "JAI": "【南亞】齋浦爾 (JAI)",
    "CMB": "【南亞】可倫坡班達拉奈克 (CMB)",
    "KTM": "【南亞】加德滿都特里布萬 (KTM)",
    "DAC": "【南亞】達卡沙賈拉勒 (DAC)",

    # ── 中東 ──────────────────────────────────────────────────────────────────
    "DXB": "【中東】杜拜國際 (DXB)", "DOH": "【中東】多哈哈馬德 (DOH)",
    "AUH": "【中東】阿布達比 (AUH)", "RUH": "【中東】利雅德阿卜杜勒阿濟茲 (RUH)",
    "JED": "【中東】吉達阿卜杜勒阿濟茲 (JED)", "MED": "【中東】麥地那 (MED)",
    "MCT": "【中東】馬斯喀特 (MCT)", "BAH": "【中東】巴林 (BAH)",
    "KWI": "【中東】科威特 (KWI)", "AMM": "【中東】安曼皇后阿利亞 (AMM)",
    "BEY": "【中東】貝魯特 (BEY)", "TLV": "【中東】台拉維夫本古里安 (TLV)",
    "IST": "【中東】伊斯坦堡 (IST)", "SAW": "【中東】伊斯坦堡薩比哈 (SAW)",
    "AYT": "【中東】安塔利亞 (AYT)", "ESB": "【中東】安卡拉埃森博阿 (ESB)",

    # ── 中亞 / 南高加索 ────────────────────────────────────────────────────
    "ALA": "【中亞】阿拉木圖 (ALA)", "NQZ": "【中亞】努爾蘇丹 (NQZ)",
    "GYD": "【中亞】巴庫海達爾阿利耶夫 (GYD)", "EVN": "【中亞】葉里溫斯瓦納茨 (EVN)",

    # ── 澳大利亞 / 大洋洲 ─────────────────────────────────────────────────
    "SYD": "【大洋洲】雪梨 (SYD)", "MEL": "【大洋洲】墨爾本 (MEL)",
    "BNE": "【大洋洲】布里斯本 (BNE)", "PER": "【大洋洲】伯斯 (PER)",
    "ADL": "【大洋洲】阿德雷德 (ADL)", "OOL": "【大洋洲】黃金海岸 (OOL)",
    "CNS": "【大洋洲】凱恩斯 (CNS)", "DRW": "【大洋洲】達爾文 (DRW)",
    "CBR": "【大洋洲】坎培拉 (CBR)", "HBA": "【大洋洲】荷伯特 (HBA)",
    "AKL": "【大洋洲】奧克蘭 (AKL)", "CHC": "【大洋洲】基督城 (CHC)",
    "WLG": "【大洋洲】威靈頓 (WLG)", "ZQN": "【大洋洲】皇后鎮 (ZQN)",
    "NAN": "【大洋洲】斐濟楠迪 (NAN)", "PPT": "【大洋洲】大溪地法阿 (PPT)",

    # ── 英國 / 愛爾蘭 ──────────────────────────────────────────────────────
    "LHR": "【西歐】倫敦希斯洛 (LHR)", "LGW": "【西歐】倫敦蓋特威克 (LGW)",
    "STN": "【西歐】倫敦史坦斯特 (STN)", "LTN": "【西歐】倫敦盧頓 (LTN)",
    "LCY": "【西歐】倫敦城市 (LCY)", "MAN": "【西歐】曼徹斯特 (MAN)",
    "BHX": "【西歐】伯明罕 (BHX)", "EDI": "【西歐】愛丁堡 (EDI)",
    "GLA": "【西歐】格拉斯哥 (GLA)", "DUB": "【西歐】都柏林 (DUB)",

    # ── 西歐 ──────────────────────────────────────────────────────────────────
    "CDG": "【西歐】巴黎戴高樂 (CDG)", "ORY": "【西歐】巴黎奧利 (ORY)",
    "FRA": "【西歐】法蘭克福 (FRA)", "MUC": "【西歐】慕尼黑 (MUC)",
    "DUS": "【西歐】杜塞道夫 (DUS)", "HAM": "【西歐】漢堡 (HAM)",
    "TXL": "【西歐】柏林布蘭登堡 (BER)", "BER": "【西歐】柏林布蘭登堡 (BER)",
    "AMS": "【西歐】阿姆斯特丹史基浦 (AMS)", "BRU": "【西歐】布魯塞爾 (BRU)",
    "ZRH": "【西歐】蘇黎世 (ZRH)", "GVA": "【西歐】日內瓦 (GVA)",
    "VIE": "【中歐】維也納 (VIE)", "MAD": "【南歐】馬德里巴拉哈斯 (MAD)",
    "BCN": "【南歐】巴塞隆納 (BCN)", "LIS": "【南歐】里斯本 (LIS)",
    "FCO": "【南歐】羅馬費烏米奇諾 (FCO)", "MXP": "【南歐】米蘭馬爾彭薩 (MXP)",
    "NAP": "【南歐】拿坡里 (NAP)", "VCE": "【南歐】威尼斯馬可波羅 (VCE)",
    "ATH": "【南歐】雅典埃萊夫塞里斯 (ATH)", "HER": "【南歐】克里特島 (HER)",
    "SKG": "【南歐】塞薩洛尼基 (SKG)", "OPO": "【南歐】波爾圖 (OPO)",
    "SVQ": "【南歐】塞維亞 (SVQ)", "AGP": "【南歐】馬拉加 (AGP)",
    "PMI": "【南歐】帕爾馬馬約卡 (PMI)", "LPA": "【南歐】大加那利島 (LPA)",
    "TFS": "【南歐】特內里費南 (TFS)",

    # ── 北歐 ──────────────────────────────────────────────────────────────────
    "ARN": "【北歐】斯德哥爾摩阿蘭達 (ARN)", "CPH": "【北歐】哥本哈根 (CPH)",
    "OSL": "【北歐】奧斯陸加勒穆恩 (OSL)", "HEL": "【北歐】赫爾辛基萬塔 (HEL)",
    "REK": "【北歐】雷克雅維克 (KEF)",

    # ── 東歐 / 俄羅斯 ─────────────────────────────────────────────────────
    "WAW": "【東歐】華沙肖邦 (WAW)", "PRG": "【東歐】布拉格瓦茨拉夫 (PRG)",
    "BUD": "【東歐】布達佩斯 (BUD)", "BUH": "【東歐】布加勒斯特奧托佩尼 (OTP)",
    "OTP": "【東歐】布加勒斯特奧托佩尼 (OTP)", "SOF": "【東歐】索非亞 (SOF)",
    "ZAG": "【東歐】薩格勒布 (ZAG)", "LJU": "【東歐】盧布爾雅那 (LJU)",
    "BEG": "【東歐】貝爾格勒 (BEG)", "KBP": "【東歐】基輔鮑里斯皮爾 (KBP)",
    "SVO": "【東歐】莫斯科謝列梅季耶沃 (SVO)", "LED": "【東歐】聖彼得堡普爾科沃 (LED)",
    "VKO": "【東歐】莫斯科伏努科沃 (VKO)",

    # ── 非洲 ──────────────────────────────────────────────────────────────────
    "JNB": "【非洲】約翰尼斯堡奧坦博 (JNB)", "CPT": "【非洲】開普敦 (CPT)",
    "DUR": "【非洲】德班金王科德 (DUR)", "CAI": "【非洲】開羅 (CAI)",
    "CMN": "【非洲】卡薩布蘭卡穆罕默德五世 (CMN)", "ALG": "【非洲】阿爾及爾 (ALG)",
    "TUN": "【非洲】突尼斯卡塔基 (TUN)", "LOS": "【非洲】拉各斯穆爾塔拉 (LOS)",
    "ADD": "【非洲】阿迪斯阿貝巴博萊 (ADD)", "NBO": "【非洲】奈洛比喬莫肯雅塔 (NBO)",
    "DAR": "【非洲】三蘭港朱利葉斯尼雷爾 (DAR)", "ACC": "【非洲】阿克拉科托卡 (ACC)",
    "ABJ": "【非洲】阿比讓 (ABJ)", "TNR": "【非洲】安塔那那利佛 (TNR)",
    "RUN": "【非洲】留尼旺羅蘭加羅斯 (RUN)", "MRU": "【非洲】模里西斯 (MRU)",

    # ── 北美：美國 ────────────────────────────────────────────────────────
    "LAX": "【北美】洛杉磯 (LAX)", "SFO": "【北美】舊金山 (SFO)",
    "JFK": "【北美】紐約甘迺迪 (JFK)", "EWR": "【北美】紐瓦克 (EWR)",
    "LGA": "【北美】紐約拉瓜迪亞 (LGA)", "ORD": "【北美】芝加哥奧海爾 (ORD)",
    "MDW": "【北美】芝加哥中途島 (MDW)", "ATL": "【北美】亞特蘭大哈茨菲爾德 (ATL)",
    "DFW": "【北美】達拉斯沃斯堡 (DFW)", "MIA": "【北美】邁阿密 (MIA)",
    "MCO": "【北美】奧蘭多 (MCO)", "SEA": "【北美】西雅圖塔科馬 (SEA)",
    "LAS": "【北美】拉斯維加斯麥卡倫 (LAS)", "DEN": "【北美】丹佛 (DEN)",
    "PHX": "【北美】鳳凰城天港 (PHX)", "IAD": "【北美】華盛頓杜勒斯 (IAD)",
    "DCA": "【北美】華盛頓雷根 (DCA)", "BOS": "【北美】波士頓洛根 (BOS)",
    "MSP": "【北美】明尼阿波利斯 (MSP)", "DTW": "【北美】底特律韋恩郡 (DTW)",
    "PHL": "【北美】費城 (PHL)", "CLT": "【北美】夏洛特道格拉斯 (CLT)",
    "SAN": "【北美】聖地牙哥 (SAN)", "TPA": "【北美】坦帕 (TPA)",
    "PDX": "【北美】波特蘭 (PDX)", "HNL": "【北美】檀香山 (HNL)",
    "ANC": "【北美】安克拉治 (ANC)", "IAH": "【北美】休士頓喬治布希 (IAH)",
    "HOU": "【北美】休士頓霍比 (HOU)", "SLC": "【北美】鹽湖城 (SLC)",
    "BNA": "【北美】納許維爾 (BNA)", "MEM": "【北美】孟菲斯 (MEM)",
    "STL": "【北美】聖路易斯 (STL)", "BWI": "【北美】巴爾的摩 (BWI)",

    # ── 北美：加拿大 ──────────────────────────────────────────────────────
    "YVR": "【北美】溫哥華 (YVR)", "YYZ": "【北美】多倫多皮爾遜 (YYZ)",
    "YUL": "【北美】蒙特婁特魯多 (YUL)", "YYC": "【北美】卡加利 (YYC)",
    "YEG": "【北美】埃德蒙頓 (YEG)", "YOW": "【北美】渥太華 (YOW)",

    # ── 中美洲 / 加勒比海 ────────────────────────────────────────────────
    "MEX": "【中美洲】墨西哥城 (MEX)", "GDL": "【中美洲】瓜達拉哈拉 (GDL)",
    "CUN": "【中美洲】坎昆 (CUN)", "SJO": "【中美洲】聖荷西 (SJO)",
    "GUA": "【中美洲】瓜地馬拉城 (GUA)",
    "MGA": "【中美洲】馬拿瓜奧古斯托 (MGA)", "SAL": "【中美洲】聖薩爾瓦多 (SAL)",

    # ── 南美洲 ────────────────────────────────────────────────────────────
    "BOG": "【南美洲】波哥大埃爾多拉多 (BOG)",
    "GRU": "【南美洲】聖保羅瓜盧柳斯 (GRU)", "GIG": "【南美洲】里約熱內盧加利昂 (GIG)",
    "SDU": "【南美洲】里約熱內盧桑托斯杜蒙 (SDU)", "BSB": "【南美洲】巴西利亞 (BSB)",
    "CNF": "【南美洲】貝洛奧里藏特 (CNF)", "FOR": "【南美洲】福塔萊薩 (FOR)",
    "SSA": "【南美洲】薩爾瓦多 (SSA)", "EZE": "【南美洲】布宜諾斯艾利斯埃塞薩 (EZE)",
    "AEP": "【南美洲】布宜諾斯艾利斯豪爾赫紐伯里 (AEP)",
    "SCL": "【南美洲】聖地牙哥科莫多羅 (SCL)", "LIM": "【南美洲】利馬豪爾赫夏維茲 (LIM)",
    "GYE": "【南美洲】瓜亞基爾 (GYE)", "UIO": "【南美洲】基多馬里斯卡爾蘇克雷 (UIO)",
    "MVD": "【南美洲】蒙得維的亞 (MVD)", "ASU": "【南美洲】亞松森 (ASU)",
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

def get_categorized_airports() -> list:
    """回傳分類後的機場選單列表，包含分類標題與縮排的機場名稱。"""
    options = []
    current_category = ""
    for code, display in AIRPORT_NAMES_ZH.items():
        if display.startswith("【"):
            end_idx = display.find("】")
            if end_idx != -1:
                category = display[1:end_idx]
                airport_name = display[end_idx+1:]
                
                if category != current_category:
                    options.append(f"【 🌍 {category} 】")
                    current_category = category
                # 使用全形空白縮排
                options.append(f"　{airport_name.strip()}")
                continue
        options.append(display)
    return options

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
