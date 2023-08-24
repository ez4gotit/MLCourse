import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Generate sample data for dimensionality reduction
X, _ = make_blobs(n_samples=300, centers=3, n_features=5, cluster_std=0.60, random_state=0)

# Apply PCA to reduce dimensionality to 2 principal components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the original data and the reduced data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Feature 1')
plt.scatter(X[:, 2], X[:, 3], c='green', label='Feature 2')
plt.scatter(X[:, 4], X[:, 0], c='red', label='Feature 3')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Original Data")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='blue')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Reduced Data using PCA")
plt.tight_layout()
plt.show()
