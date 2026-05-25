"""
src/recommender.py
AI 推薦與分群模組
使用 KNN-based 概念計算推薦分數，以及使用 KMeans 進行航班分群。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Tuple, List

PREF_PRICE = "最便宜"
PREF_TIME = "最快抵達"
PREF_BALANCE = "價格與時間平衡"
PREF_STOPS = "最少轉機"

def knn_recommend(df: pd.DataFrame, preference: str) -> Tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    """
    基於 KNN 概念的推薦排序：
    1. 將特徵標準化
    2. 根據偏好建立 Ideal Vector
    3. 計算各航班與 Ideal Vector 的距離 (轉換為 0-100 分)
    """
    df_rec = df.copy()
    features = ["price", "duration_minutes", "stops", "departure_hour"]
    
    X = df_rec[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 找出各特徵標準化後的極值，用以建構 Ideal Vector
    min_price_idx = 0
    min_dur_idx = 1
    min_stops_idx = 2
    
    # Ideal Vector: 最佳的狀態是價格最低、時間最短、轉機最少
    ideal_vector = np.array([
        np.min(X_scaled[:, min_price_idx]),
        np.min(X_scaled[:, min_dur_idx]),
        np.min(X_scaled[:, min_stops_idx]),
        0 # departure_hour 先給 0，稍後依偏好動態權重
    ])
    
    # 根據偏好調整權重
    weights = np.ones(4)
    if preference == PREF_PRICE:
        weights = np.array([3.0, 1.0, 1.0, 0.5])
    elif preference == PREF_TIME:
        weights = np.array([1.0, 3.0, 1.0, 0.5])
    elif preference == PREF_STOPS:
        weights = np.array([1.0, 1.0, 3.0, 0.5])
    else: # 價格與時間平衡
        weights = np.array([2.0, 2.0, 1.0, 0.5])
        
    weighted_X = X_scaled * weights
    weighted_ideal = ideal_vector * weights
    
    # 計算歐氏距離
    distances = np.linalg.norm(weighted_X - weighted_ideal, axis=1)
    
    # 將距離轉換為 0-100 的分數，距離越短分數越高
    if np.max(distances) == np.min(distances):
        scores = np.ones(len(distances)) * 100
    else:
        scores = 100 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances)) * 100
        
    df_rec["recommendation_score"] = scores
    df_rec = df_rec.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
    
    return df_rec, scaler, X_scaled

def kmeans_cluster(df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, str]:
    """將航班分群以找出不同類型的票價結構"""
    df_clust = df.copy()
    features = ["price", "duration_minutes", "stops"]
    
    if len(df_clust) < n_clusters:
        n_clusters = len(df_clust)
        if n_clusters < 2:
            df_clust["cluster_label"] = "資料過少未分類"
            return df_clust, "航班資料數量不足，已略過分群分析。"
            
    X = df_clust[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clust["cluster"] = clusters
    
    # 依群集特徵給予標籤
    cluster_labels = {}
    for c in range(n_clusters):
        c_data = df_clust[df_clust["cluster"] == c]
        avg_price = c_data["price"].mean()
        avg_dur = c_data["duration_minutes"].mean()
        
        if avg_price <= df_clust["price"].quantile(0.33):
            label = "低價航班"
        elif avg_dur <= df_clust["duration_minutes"].quantile(0.33):
            label = "快速航班"
        elif avg_price >= df_clust["price"].quantile(0.66) and avg_dur >= df_clust["duration_minutes"].quantile(0.66):
            label = "高價/長時航班"
        else:
            label = "平衡型航班"
            
        cluster_labels[c] = f"{label} (群組 {c+1})"
        
    df_clust["cluster_label"] = df_clust["cluster"].map(cluster_labels)
    return df_clust, "分群成功"

def get_pca_data(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    features = ["price", "duration_minutes", "stops"]
    X = df[features].values
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["cluster_label"] = df["cluster_label"]
    pca_df["airline"] = df["airline"]
    pca_df["flight_no"] = df["flight_no"]
    pca_df["price"] = df["price"]
    
    return pca_df

def generate_recommendation_reason(row: pd.Series, preference: str, rank: int) -> str:
    if rank == 1:
        if preference == PREF_PRICE:
            return "此航班在搜尋結果中價格極具競爭力，非常適合重視預算的旅客。"
        elif preference == PREF_TIME:
            return "此航班飛行時間短，能最快抵達目的地，適合重視效率的商務或旅遊客。"
        elif preference == PREF_STOPS:
            return "此航班轉機次數最少，免去轉機奔波的勞累，適合帶小孩或長輩同行的旅客。"
        else:
            return "此航班在票價與飛行時間之間取得極佳的平衡，是整體 C/P 值最高的首選。"
    elif rank <= 3:
        return "表現優異的替代方案，提供極佳的綜合價值。"
    return "一般航班選擇。"

def generate_pros_cons(row: pd.Series, df_all: pd.DataFrame) -> Tuple[List[str], List[str]]:
    pros = []
    cons = []
    
    med_price = df_all["price"].median()
    med_dur = df_all["duration_minutes"].median()
    
    if row["price"] <= med_price * 0.8:
        pros.append("機票價格顯著低於平均水準")
    elif row["price"] > med_price * 1.2:
        cons.append("機票價格較高")
        
    if row["duration_minutes"] <= med_dur * 0.8:
        pros.append("飛行時間非常短")
    elif row["duration_minutes"] > med_dur * 1.2:
        cons.append("飛行時間偏長")
        
    if row["stops"] == 0:
        pros.append("直飛航班，省時省力")
    elif row["stops"] >= 2:
        cons.append("轉機次數較多，可能較為勞累")
        
    if not pros:
        pros.append("各項指標表現平均穩定")
    if not cons:
        cons.append("無明顯缺點")
        
    return pros, cons
