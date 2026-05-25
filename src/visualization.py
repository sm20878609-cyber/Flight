"""
src/visualization.py
資料視覺化模組：負責使用 Plotly 繪製各種互動式圖表。
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def apply_default_layout(fig, title: str):
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Noto Sans TC, sans-serif")),
        font=dict(family="Noto Sans TC, sans-serif"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(font_family="Noto Sans TC, sans-serif")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    return fig

def plot_price_bar(df: pd.DataFrame) -> go.Figure:
    df_plot = df.head(10).copy()
    df_plot['航班顯示'] = df_plot['airline'] + ' ' + df_plot['flight_no']
    
    fig = px.bar(
        df_plot, 
        x='航班顯示', 
        y='price',
        title="Top 10 最推薦航班 - 機票價格比較",
        labels={'price': '機票價格 (NT$)', '航班顯示': '航班'},
        color='price',
        color_continuous_scale=px.colors.sequential.Blues_r,
        text='price'
    )
    fig.update_traces(texttemplate='NT$ %{text:,.0f}', textposition='outside')
    return apply_default_layout(fig, "Top 10 最推薦航班 - 機票價格比較")

def plot_recommendation_score(df: pd.DataFrame) -> go.Figure:
    df_plot = df.head(10).copy()
    df_plot['航班顯示'] = df_plot['airline'] + ' ' + df_plot['flight_no']
    
    fig = px.bar(
        df_plot,
        x='recommendation_score',
        y='航班顯示',
        orientation='h',
        title="Top 10 最推薦航班 - AI 推薦分數",
        labels={'recommendation_score': '推薦分數', '航班顯示': '航班'},
        color='recommendation_score',
        color_continuous_scale=px.colors.sequential.Teal,
        text='recommendation_score'
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='inside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return apply_default_layout(fig, "Top 10 最推薦航班 - AI 推薦分數")

def plot_duration_vs_price_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x='duration_minutes',
        y='price',
        color='stops',
        size='recommendation_score',
        hover_name='airline',
        hover_data=['flight_no', 'departure_time', 'arrival_time'],
        title="飛行時間 vs 機票價格散佈圖",
        labels={'duration_minutes': '飛行時間 (分鐘)', 'price': '機票價格 (NT$)', 'stops': '轉機次數'},
        color_continuous_scale=px.colors.sequential.Plasma
    )
    return apply_default_layout(fig, "飛行時間 vs 機票價格散佈圖 (泡泡大小:推薦分數)")

def plot_stops_distribution(df: pd.DataFrame) -> go.Figure:
    stops_counts = df['stops'].value_counts().reset_index()
    stops_counts.columns = ['轉機次數', '航班數量']
    stops_counts['轉機次數'] = stops_counts['轉機次數'].apply(lambda x: '直飛' if x == 0 else f"{x}次轉機")
    
    fig = px.pie(
        stops_counts,
        values='航班數量',
        names='轉機次數',
        title="航班轉機次數分布",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return apply_default_layout(fig, "航班轉機次數分布")

def plot_kmeans_scatter(pca_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="cluster_label",
        hover_data=["airline", "flight_no", "price"],
        title="航班特徵 KMeans 分群散佈圖 (PCA 降維)",
        labels={"cluster_label": "所屬群集"}
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    return apply_default_layout(fig, "航班特徵 KMeans 分群散佈圖 (PCA 降維)")
