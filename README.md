# AI 智慧航班機票搜尋與推薦系統
*(AI-Powered Flight Ticket Search and Recommendation System)*

本專案為 Machine Learning in Engineering Science 課程的期中專題實作系統 (Topic 3)。
系統透過開源的 `fast-flights` 爬蟲套件即時抓取 Google Flights 的單日航班資料，並透過機器學習方法 (KNN, KMeans) 為使用者進行票價與時間的平衡分析，推薦最適合的航班組合。

## ✨ 系統特色
- 🆓 **完全免費**：不需申請任何付費 API Key，透過 `fast-flights` 進行即時爬蟲。
- 🤖 **AI 推薦系統**：基於 KNN 概念，根據使用者偏好（最便宜、最快抵達、最少轉機、價格與時間平衡），為每一班航班打出 0~100 的推薦分數。
- 🧠 **自動分群**：利用 KMeans 演算法與 PCA 降維，將搜尋到的航班自動分為「低價」、「快速」、「高價」等群組。
- 🛡️ **高可用備援機制**：即使遇到爬蟲阻擋或網路中斷，系統也會自動 fallback 到內建的 CSV 資料檔，確保應用程式不會 Crash。
- 🎨 **現代化流暢 UI**：全繁體中文介面，採用頂部過濾卡片設計與 Plotly 互動式視覺化圖表。

## 🛠️ 安裝與執行
1. 安裝所需套件：
   ```bash
   pip install -r requirements.txt
   ```
2. 啟動 Streamlit 系統：
   ```bash
   streamlit run app.py
   ```

## 📂 專案結構
- `app.py`: Streamlit 系統主程式
- `src/flight_scraper.py`: `fast-flights` 爬蟲模組
- `src/data_loader.py`: 資料匯入與備援機制模組
- `src/preprocessing.py`: 資料清洗與特徵轉換
- `src/recommender.py`: KNN 推薦與 KMeans 分群演算法
- `src/visualization.py`: Plotly 圖表繪製模組
- `data/`: 存放內建範例資料與爬蟲快取 CSV 檔

## ⚠️ 系統限制
1. 爬蟲資料可能受來源網站 (Google Flights) 變動影響，頻繁查詢可能導致 IP 暫時被限制。
2. 票價隨時變動，實際價格與開票結果請以航空公司或訂票平台最後顯示為準。
3. 本系統專注於「航班機票比較與 AI 推薦」，不處理最終的訂票與金流付款功能。
