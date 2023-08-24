import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate sample data for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Find the optimal number of clusters using the Elbow Method
inertia_values = []
silhouette_scores = []
possible_k_values = range(1, 11)

for k in possible_k_values:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(X)
    inertia_values.append(model.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X, model.labels_))

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(possible_k_values, inertia_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(possible_k_values[1:], silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Optimal k")
plt.show()
