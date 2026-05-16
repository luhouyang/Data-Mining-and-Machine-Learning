import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from matplotlib.patches import Ellipse
import warnings

# Suppress minor sklearn warnings for clean output
warnings.filterwarnings('ignore')

# --- 1. Metric Calculation Function ---
def calc_distances(X, labels):
    unique_labels = set(labels) - {-1} 
    
    if len(unique_labels) < 2:
        if len(unique_labels) == 1:
            mask = labels == list(unique_labels)[0]
            pts = X[mask]
            intra = np.mean(pdist(pts)) if len(pts) > 1 else 0
        else:
            intra = 0
        return intra, float('nan')

    intra_dists = []
    centroids = []
    
    for k in unique_labels:
        mask = labels == k
        pts = X[mask]
        centroids.append(pts.mean(axis=0))
        if len(pts) > 1:
            intra_dists.append(np.mean(pdist(pts)))
            
    intra = np.mean(intra_dists) if intra_dists else 0
    inter = np.mean(pdist(centroids))
    return intra, inter

# --- 2. Helper to Draw GMM Ellipses ---
def draw_ellipse(ax, mean, covar, color):
    vals, vecs = np.linalg.eigh(covar)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * 2 * np.sqrt(np.maximum(vals, 1e-10)) 
    ell = Ellipse(xy=mean, width=w, height=h, angle=theta, color=color, alpha=0.25, zorder=1)
    ax.add_patch(ell)

# --- 3. Generate and Scale Datasets ---
n_samples = 300

# 1. Spherical Blobs (Perfect for K-Means)
X_spherical, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.5, random_state=10)
X_spherical = StandardScaler().fit_transform(X_spherical)

# 2. Double Circle (Perfect for DBSCAN)
X_circ, _ = make_circles(n_samples=n_samples, factor=0.4, noise=0.05, random_state=42)
X_circ = StandardScaler().fit_transform(X_circ)

# 3. Anisotropic Blobs (Perfect for GMM)
np.random.seed(42)
c1 = np.random.randn(150, 2) @ [[2.0, -1.0], [-1.0, 2.0]] + [3, 3]
c2 = np.random.randn(150, 2) @ [[1.0, 1.5], [1.5, 1.0]] + [-3, -3]
c3 = np.random.randn(150, 2) @ [[2.5, 0.5], [0.5, 1.0]] + [4, -4]
X_blob = np.vstack([c1, c2, c3])
X_blob = StandardScaler().fit_transform(X_blob)

# Store datasets with target number of clusters (k) and DBSCAN epsilon (eps)
datasets = [
    ("Spherical Blobs\n(K-Means Ideal)", X_spherical, 3, 0.25),
    ("Double Circle\n(Non-linear)", X_circ, 2, 0.35),
    ("Anisotropic Blobs\n(Varying Variance)", X_blob, 3, 0.35)
]

algorithms = ["K-Means", "DBSCAN", "GMM"]

# --- 4. Run Clustering and Plot ---
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row, (dataset_name, X, k, eps) in enumerate(datasets):
    
    # Initialize algorithms
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    gmm = GaussianMixture(n_components=k, random_state=42)
    
    # Fit and Predict
    labels_km = kmeans.fit_predict(X)
    labels_db = dbscan.fit_predict(X)
    labels_gmm = gmm.fit_predict(X)
    
    predictions = [labels_km, labels_db, labels_gmm]
    
    for col, (algo_name, labels) in enumerate(zip(algorithms, predictions)):
        ax = axes[row, col]
        
        # Color mapping: noise (-1) gets RGBA Grey
        colors = np.array([(0.7, 0.7, 0.7, 1.0) if l == -1 else plt.cm.tab10(l % 10) for l in labels])
        
        ax.scatter(X[:, 0], X[:, 1], color=colors, s=15, alpha=0.8, zorder=2)
        
        # Add Ellipses ONLY for the GMM column
        if algo_name == "GMM":
            for i in range(gmm.n_components):
                draw_ellipse(ax, gmm.means_[i], gmm.covariances_[i], plt.cm.tab10(i % 10))
        
        # Calculate Metrics
        intra, inter = calc_distances(X, labels)
        
        # Formatting
        if row == 0:
            ax.set_title(algo_name, fontsize=14, fontweight='bold', pad=15)
        if col == 0:
            ax.set_ylabel(dataset_name, fontsize=12, fontweight='bold', labelpad=15)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add metrics below the image
        inter_text = f"{inter:.2f}" if not np.isnan(inter) else "N/A"
        ax.set_xlabel(f"Intra: {intra:.2f}  |  Inter: {inter_text}", fontsize=11, color='#333333')

plt.tight_layout()
plt.show()