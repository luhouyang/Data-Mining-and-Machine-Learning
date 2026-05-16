import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from IPython.display import HTML
import matplotlib

# Increase memory limit
matplotlib.rcParams['animation.embed_limit'] = 50.0

# 1. Data Generation
np.random.seed(42)
c1 = np.random.randn(150, 2) @ [[1.5, 0], [0, 0.4]] + [4, 4]
c2 = np.random.randn(150, 2) @ [[0.3, 0], [0, 1.8]] + [-4, 0]
c3 = np.random.randn(150, 2) @ [[0.6, 0], [0, 0.6]] + [0, -5]
X = np.vstack([c1, c2, c3])

n_components = 3

# 2. Explicit Random Initialization (Starting points)
np.random.seed(15)
random_idx = np.random.choice(len(X), n_components, replace=False)
initial_means = X[random_idx]

initial_covs = np.array([np.eye(2) * 1.5 for _ in range(n_components)])
initial_precisions = np.linalg.inv(initial_covs)
initial_weights = np.ones(n_components) / n_components

gmm = GaussianMixture(n_components=n_components,
                      max_iter=1,
                      warm_start=True,
                      means_init=initial_means,
                      precisions_init=initial_precisions,
                      weights_init=initial_weights)

history = []

# --- NEW TIMING SETTINGS ---
FPS = 10  # Standard safe framerate for all platforms
FRAMES_PER_STEP = 4  # 10 frames @ 10fps = 1 full second per EM step
PAUSE_FRAMES = 12  # 30 frames @ 10fps = 3 full seconds pause at the end

# Capture the explicitly initialized state (Step 0)
gmm.fit(initial_means)
gmm.means_ = initial_means.copy()
gmm.covariances_ = initial_covs.copy()

step_0_state = {
    'means': gmm.means_.copy(),
    'covars': gmm.covariances_.copy(),
    'labels': gmm.predict(X),
    'em_step': 0  # Track step number explicitly
}

# Duplicate Step 0
for _ in range(FRAMES_PER_STEP):
    history.append(step_0_state)

# Run the actual EM iterations
for i in range(1, 31):
    gmm.fit(X)

    current_state = {
        'means': gmm.means_.copy(),
        'covars': gmm.covariances_.copy(),
        'labels': gmm.predict(X),
        'em_step': i
    }

    # Duplicate the current step to create the delay
    for _ in range(FRAMES_PER_STEP):
        history.append(current_state)

    if gmm.converged_:
        break

# Add the final pause at the end
for _ in range(PAUSE_FRAMES):
    history.append(history[-1])


# 3. Helper for Ellipses
def draw_ellipse(ax, mean, covar, color):
    vals, vecs = np.linalg.eigh(covar)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * 2 * np.sqrt(np.maximum(vals, 1e-10))
    ell = Ellipse(xy=mean,
                  width=w,
                  height=h,
                  angle=theta,
                  color=color,
                  alpha=0.35,
                  zorder=3)
    ax.add_patch(ell)


# 4. Animation Setup
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#FF5733', '#33FF57', '#3357FF']


def update(frame):
    ax.clear()
    step = history[frame]
    ax.scatter(X[:,
                 0],
               X[:,
                 1],
               c=[colors[l] for l in step['labels']],
               s=20,
               alpha=0.3)

    for i in range(n_components):
        draw_ellipse(ax, step['means'][i], step['covars'][i], colors[i])
        ax.scatter(step['means'][i][0],
                   step['means'][i][1],
                   c=colors[i],
                   s=50,
                   edgecolors='black',
                   zorder=4)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Read the step number directly from the state dictionary
    ax.set_title(f"GMM Migration (Raw EM Step {step['em_step']})")
    ax.axis('off')
    return ax,


# Interval doesn't matter for the GIF, but 100ms keeps the Colab preview at 10fps
ani = FuncAnimation(fig, update, frames=len(history), interval=100, blit=False)

# 5. Save as GIF
# Explicitly set fps=10. Because we repeated frames 10 times, each step lasts 1 second.
writer = PillowWriter(fps=FPS)
ani.save("gmm_progression.gif", writer=writer)

plt.show()
