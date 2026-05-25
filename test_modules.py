import sys, os
sys.path.insert(0, '.')
from src.data_loader import load_sample_data, get_unique_origins, get_unique_destinations
from src.preprocessing import full_preprocessing, filter_flights, prepare_features
from src.cost_calculator import calculate_all_costs
from src.recommender import knn_recommend, kmeans_cluster, get_pca_data
from src.visualization import plot_true_cost_bar

print("=== Module import OK ===")

df, msg = load_sample_data()
print(f"Data load: {msg}")
print(f"Columns: {list(df.columns)}")
print(f"Rows: {len(df)}")

df2 = full_preprocessing(df)
origins = get_unique_origins(df2)
destinations = get_unique_destinations(df2)
print(f"Origins: {origins}")
print(f"Destinations: {destinations}")

df3, fmsg = filter_flights(df2, "TPE", "NRT", 1)
print(f"Filter: {fmsg}")

df4 = calculate_all_costs(df3, True, 1, False, False, 300)
print(f"Cost range: {df4['true_total_cost'].min():.0f} - {df4['true_total_cost'].max():.0f}")

df5 = prepare_features(df4)
df6, sc, xs = knn_recommend(df5, "price_balance")

# try all preferences
from src.recommender import PREF_PRICE, PREF_TIME, PREF_BALANCE, PREF_COMFORT
for pref in [PREF_PRICE, PREF_TIME, PREF_BALANCE, PREF_COMFORT]:
    dfp, _, _ = knn_recommend(df5, pref)
    scores = dfp["recommendation_score"].round(1).tolist()
    print(f"Pref '{pref}': scores={scores}")

df7, cmsg = kmeans_cluster(df6, n_clusters=3)
print(f"KMeans: {cmsg}")
print(f"Cluster labels: {df7['cluster_label'].unique().tolist()}")

pca_df = get_pca_data(df7, sc)
print(f"PCA OK: {pca_df is not None}")

print("=== ALL TESTS PASSED ===")
