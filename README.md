# ✈️ AI 智慧航班機票搜尋與推薦系統 (AI-Powered Flight Recommender)

![Premium UI](https://img.shields.io/badge/UI-Premium_Glassmorphism-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)

本專案為 Machine Learning in Engineering Science 課程的期中專題實作系統。
我們打造了一個極致高質感的現代化機票推薦平台，結合 **Google Flights 即時爬蟲**、**Flightradar24 航班驗證**，並利用 **KNN (K-Nearest Neighbors)** 機器學習演算法，為使用者推薦最完美的機票組合！

---

## ✨ 核心特色亮點

- 💎 **尊爵版高質感 UI (Premium Glassmorphism)**：比照國際頂級 OTA（如 Airbnb、Expedia）的視覺設計，結合毛玻璃特效、Apple 風格數據卡片與流體漸層按鈕。
- 🤖 **AI 智慧評分演算法 (KNN)**：打破傳統單一價格排序！系統將「票價、飛行時間、轉機次數」特徵標準化，計算出與「完美理想航班」的歐氏距離，提供精準的綜合推薦分數（0~100 分）。
- 📡 **雙引擎即時資料驗證**：
  - **價格與航線**：背景自動執行 `fast-flights` 爬取 Google Flights 即時含稅票價。
  - **航班編號 (Flight No.)**：結合 `curl_cffi` 無頭爬蟲，**即時連線 Flightradar24 全球資料庫**，進行航線與航空公司的交叉比對，保證您看到的航班編號 100% 真實運作中！
- 🧠 **機器學習分群 (KMeans)**：利用 KMeans 搭配 PCA 降維，將搜尋到的航班自動分為「低價小資」、「快速直飛」、「高端舒適」等群體，並以 Plotly 進行 3D/2D 視覺化。

---

## 🚀 別人要使用時該怎麼做？(安裝與執行指南)

任何人只要將此專案 Clone 到電腦上，遵循以下三個簡單步驟，即可在自己的電腦上運行這個強大的系統！

### 步驟 1：下載專案 (Clone Repository)
請先確保電腦已安裝 [Git](https://git-scm.com/) 以及 [Python (3.9 以上版本)](https://www.python.org/)，然後打開終端機 (Terminal / 命令提示字元) 執行：
```bash
git clone https://github.com/sm20878609-cyber/Flight.git
cd Flight
```

### 步驟 2：安裝所需套件 (Install Dependencies)
強烈建議使用虛擬環境 (Virtual Environment) 避免套件衝突。
```bash
# 安裝 requirements.txt 內的所有套件
pip install -r requirements.txt
```
*(系統主要依賴：streamlit, pandas, scikit-learn, fast-flights, curl_cffi, plotly 等)*

### 步驟 3：啟動系統 (Run the App)
執行以下指令啟動 Streamlit 伺服器：
```bash
streamlit run app.py
```
執行後，瀏覽器會自動開啟網頁 `http://localhost:8501`。
接著，您只要在畫面上設定**出發地、目的地、日期區間**，點擊**「✈️ 啟程搜尋航班」**，系統就會開始為您執行爬蟲與 AI 運算！

---

## 📂 專案架構
```text
Flight/
├── app.py                   # Streamlit 前端與主程式入口
├── requirements.txt         # 系統所需套件清單
├── README.md                # 專案說明文件
├── data/                    # 暫存快取檔與備援範例資料
└── src/                     # 核心引擎模組
    ├── flight_scraper.py    # Google Flights 爬蟲與 Flightradar24 動態驗證引擎
    ├── recommender.py       # KNN 推薦與 KMeans 分群機器學習演算法
    ├── preprocessing.py     # 資料清洗與特徵標準化 (MinMaxScaler)
    ├── visualization.py     # Plotly 互動式圖表繪製
    └── data_loader.py       # 容錯備援機制與資料載入
```

## ⚠️ 注意事項
1. **爬蟲限制**：即時爬蟲依賴於外部網站 (Google / FR24)，若短時間內查詢頻率過高（例如狂按搜尋），可能會被對方伺服器暫時封鎖 IP。若發生此狀況，系統會自動切換為讀取內建的「備援範例資料」。
2. **最終票價**：機票價格瞬息萬變，本系統之票價供學術研究與比價參考，實際訂票請點擊「前往 Google Flights 查看與訂票」以官網最後結帳為準。
