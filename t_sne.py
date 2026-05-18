import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. t-SNE Model (2D Visualization ke liye)
# Perplexity batati hai ki kitne neighbors ko dekhna hai
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 3. Visualization
plt.figure(figsize=(8, 5))
for i, name in enumerate(iris.target_names):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=name)

plt.title("t-SNE: Finding Clusters (Non-Linear)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend()
plt.show()