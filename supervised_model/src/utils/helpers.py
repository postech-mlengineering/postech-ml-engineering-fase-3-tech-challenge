import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator


def get_best_k(scaled_data, max_k=10):
    k_range = range(1, max_k + 1)
    sse = []

    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)
    
    kl = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
    return kl.elbow


def check_missing_values(df):
    df_empty = pd.DataFrame({
        'EMPTIES': df.isnull().sum(),
        'EMPTIES (%)': (df.isnull().sum() / len(df) * 100).round(2)
    }).reset_index().rename(columns={'index': 'COLUMNS'})
    df_empty = df_empty[df_empty['EMPTIES'] > 0].sort_values(by='EMPTIES (%)', ascending=False)
    if df_empty.empty:
        print('Nenhum valor nulo encontrado no DataFrame.')
        return None
    return df_empty