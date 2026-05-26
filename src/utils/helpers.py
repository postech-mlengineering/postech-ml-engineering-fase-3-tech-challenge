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