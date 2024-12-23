import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

# Load Yelp review dataset (update with actual file path)
data = pd.read_json("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json", lines=True)

# Check the structure of the dataset (for debugging)
print(data.head())
print(data.columns)

# Preprocess the data
# For simplicity, let's assume 'text' contains the review text for clustering

# Vectorize the reviews using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['text'])

# Compute cosine similarity between the reviews
similarity_matrix = cosine_similarity(X)

# Perform clustering using different algorithms

# KMeans Clustering with 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42)
kmeans_2_labels = kmeans_2.fit_predict(similarity_matrix)

# KMeans Clustering with 5 clusters
kmeans_5 = KMeans(n_clusters=5, random_state=42)
kmeans_5_labels = kmeans_5.fit_predict(similarity_matrix)

# Agglomerative Clustering with 2 clusters
agg_2 = AgglomerativeClustering(n_clusters=2)
agg_2_labels = agg_2.fit_predict(similarity_matrix)

# Agglomerative Clustering with 5 clusters
agg_5 = AgglomerativeClustering(n_clusters=5)
agg_5_labels = agg_5.fit_predict(similarity_matrix)

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
similarity_pca = pca.fit_transform(similarity_matrix)

# Visualize KMeans (2 clusters)
plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=kmeans_2_labels, cmap='viridis')
plt.title("K-Means Clustering (2 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("kmeans_2_clusters.png", bbox_inches='tight')
plt.show()

# Visualize KMeans (5 clusters)
plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=kmeans_5_labels, cmap='viridis')
plt.title("K-Means Clustering (5 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("kmeans_5_clusters.png", bbox_inches='tight')
plt.show()

# Visualize Agglomerative Clustering (2 clusters)
plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=agg_2_labels, cmap='viridis')
plt.title("Agglomerative Clustering (2 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("agg_2_clusters.png", bbox_inches='tight')
plt.show()

# Visualize Agglomerative Clustering (5 clusters)
plt.figure(figsize=(8, 6))
plt.scatter(similarity_pca[:, 0], similarity_pca[:, 1], c=agg_5_labels, cmap='viridis')
plt.title("Agglomerative Clustering (5 clusters)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("agg_5_clusters.png", bbox_inches='tight')
plt.show()
