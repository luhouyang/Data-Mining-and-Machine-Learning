import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
from scipy.spatial import distance
from IPython.display import HTML
import matplotlib

# Increase memory limit
matplotlib.rcParams['animation.embed_limit'] = 50.0

# 1. Setup Data
X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
EPS = 0.2

# 2. RUN REAL DBSCAN FIRST
real_dbscan = DBSCAN(eps=EPS, min_samples=5).fit(X)
final_labels = real_dbscan.labels_
cluster_ids = np.unique(final_labels[final_labels >= 0])

# 3. Create Path with a "Pause" at the end
history = []
revealed_labels = np.full(X.shape[0], -1)
visited = np.zeros(X.shape[0], dtype=bool)

for c_id in cluster_ids:
    cluster_indices = np.where(final_labels == c_id)[0].tolist()
    current_idx = cluster_indices[0]

    while cluster_indices:
        visited[current_idx] = True
        revealed_labels[current_idx] = c_id
        history.append({
            'current': current_idx,
            'labels': revealed_labels.copy(),
            'show_circle': True
        })

        if current_idx in cluster_indices:
            cluster_indices.remove(current_idx)

        if not cluster_indices:
            break

        # Find next closest point within the same cluster
        dists = distance.cdist([X[current_idx]],
                               X[cluster_indices],
                               'euclidean')[0]
        current_idx = cluster_indices[np.argmin(dists)]

# --- THE FIX: Append 50 identical frames at the end to "pause" ---
final_state = history[-1].copy()
final_state['show_circle'] = False  # Hide the red circle during the pause
for _ in range(50):
    history.append(final_state)

# 4. Animation Setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('white')
scatter = ax.scatter(X[:,
                       0],
                     X[:,
                       1],
                     c='white',
                     edgecolors='#ddd',
                     s=40,
                     lw=0.5,
                     zorder=2)
search_circle = plt.Circle((0,
                            0),
                           EPS,
                           color='red',
                           fill=True,
                           alpha=0.15,
                           zorder=3)
ax.add_patch(search_circle)

# Colors for exactly 2 clusters
cluster_colors = {0: '#1f77b4', 1: '#ff7f0e'}


def update(frame):
    step = history[frame]
    curr_idx = step['current']

    # Hide circle during the final pause
    if step['show_circle']:
        search_circle.set_visible(True)
        search_circle.center = (X[curr_idx, 0], X[curr_idx, 1])
    else:
        search_circle.set_visible(False)

    colors = []
    for i, l in enumerate(step['labels']):
        if l == -1: colors.append('#f0f0f0')
        elif l == 0: colors.append(cluster_colors[0])
        elif l == 1: colors.append(cluster_colors[1])
        else: colors.append('#333333')

    scatter.set_facecolors(colors)

    # Only highlight the leading edge if the circle is visible
    if step['show_circle']:
        scatter.get_facecolors()[curr_idx] = [1, 0, 0, 1]

    return scatter, search_circle


ani = FuncAnimation(fig, update, frames=len(history), interval=30, blit=True)

# 4. Save as GIF
# Note: Requires 'pillow' library (pip install pillow)
writer = PillowWriter(fps=30)
ani.save("dbscan_progression.gif", writer=writer)

plt.show()
