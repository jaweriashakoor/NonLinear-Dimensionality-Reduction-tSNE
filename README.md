<div align="center">

# <span style="color:#f43f5e">Non-Linear Manifold Learning (t-SNE Clustering)</span>

![Python](https://img.shields.io/badge/Python-3.10%2B-1e293b?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Computation-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-059669?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-f43f5e?style=for-the-badge)

<br/>

<h3>
  <em>"Preserving High-Dimensional Local Neighborhood Topologies via Probabilistic Similarity Mappings and Manifold Embeddings"</em><br/>
</h3>

<br/>

[📌 Overview](#-overview) • [✨ Features](#-features) • [🚀 Pipeline Flow](#-pipeline-flow) • [⚙️ Installation](#%EF%B8%8F-installation) • [⚙️ Hyperparameter Breakdown](#-hyperparameter-tuning-breakdown) • [🧠 Mathematical Foundations](#-mathematical-foundations) • [🆚 Dimensionality Reduction Matrix](#-dimensionality-reduction-matrix-pca-vs-autoencoder-vs-t-sne) • [🤝 Contributing](#-contributing)

</div>

---

## 📌 Overview

**Non-Linear Manifold Learning** is an advanced unsupervised machine learning repository that implements **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to maps intricate, high-dimensional dataset structural spaces down to a highly clear 2D projection canvas. Processing the 4-dimensional attributes of the classic *Iris Dataset*, this pipeline transforms data distances into conditional probabilities that capture local similarities.

Unlike linear projection workflows, t-SNE prioritizes retaining small neighborhood clusters, effectively separating complex, non-linear botanical boundaries on a clean, low-dimensional scatter dashboard.

---

## ✨ Features

### 🎯 Core Capabilities
| Feature | Description |
|---|---|
| 🌐 **Probabilistic Proximity Engine** | Converts geometric distances into Gaussian joint probabilities in the high-dimensional space. |
| 🔄 **Student-t Distribution Space** | Employs a heavy-tailed Student-t distribution in the 2D space to solve crowding problems. |
| 🎛️ **Neighborhood Balance Control** | Implements parameterized perplexity loops (`perplexity=30`) to balance local vs. global target views. |
| 📉 **Non-Linear Manifold Mapping** | Learns complex, twisted underlying structures that standard linear coordinate changes drop. |
| 🎨 **High-Contrast Topology Plots** | Generates an optimized visual scatter layout showing clean cluster isolations for all target species. |

### 🌟 Design Highlights
- 🧠 **Topology-Preserving Summary** — Isolates structural sub-classes without using classification labels during data fitting.
- 🧪 **Crowding-Problem Mitigation** — Dynamically repels distant clusters in the low-dimensional embedding grid to prevent visualization overlaps.
- ⚡ **Gradient-Optimized Embeddings** — Leverages Kullback-Leibler divergence minimization to map points smoothly into 2D space.

---

## 🚀 Pipeline Flow

┌─────────────────────────────────────────────────────────┐│                    IRIS DATASET INGESTION               ││          (4-Dimensional High-Dimensional Data Matrix)   │└─────────────────────────────────────────────────────────┘│▼[ Convert High-D Euclidean Distances to Probs ]│▼┌───────────────────────────────────────────────────────┐│               NEIGHBORHOOD PERPLEXITY LOOKUP          ││  Calibrate Local Search Radius vs Scale (Perplexity=30)│└───────────────────────────┬───────────────────────────┘│▼[ Map Low-D Joint Similarities via Student-t Graph ]│▼┌───────────────────────────────────────────────────────┐│                KL-DIVERGENCE MINIMIZATION             ││   Iterate Gradient Updates to Align Spatial Structures │└───────────────────────────┬───────────────────────────┘│▼┌─────────────────────────────────────────────────────────┐│                 NON-LINEAR EMBEDDING CANVAS             ││                                                         ││   Render: Matplotlib 2D t-SNE Local Coordinate Scatter  ││   Axis 1: Dimension 1 isolating micro-neighborhoods     ││   Axis 2: Dimension 2 optimizing structural layouts     │└─────────────────────────────────────────────────────────┘
---

## ⚙️ Installation

### Prerequisites

Make sure your local computer has these foundational software components set up:
- Python 3.10+
- Git

---

### 🔧 Step-by-Step Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/jaweriashakoor/NonLinear-Dimensionality-Reduction-tSNE.git](https://github.com/jaweriashakoor/NonLinear-Dimensionality-Reduction-tSNE.git)
cd NonLinear-Dimensionality-Reduction-tSNE
2. Create a Virtual EnvironmentBashpython -m venv myenv

# Windows (PowerShell)
.\myenv\Scripts\Activate.ps1

# Linux / macOS
source myenv/bin/activate
3. Install DependenciesBashpip install numpy scikit-learn matplotlib
4. Execute the t-SNE Pipeline ScriptBashpython tsne_reduction.py
⚙️ Hyperparameter Tuning Breakdownt-SNE's structural map grouping changes significantly based on your neighborhood tuning weights:Parameter KeyCode ConfigurationFunctional PurposeOperational ImpactPerplexityperplexity=30Gauges the effective number of local neighbors to look at during mapping.Lower values focus strictly on micro-details; higher values integrate global structural layouts.Random Staterandom_state=42Stabilizes the random initialization configurations of the low-dimensional map.Since t-SNE is probabilistic, tracking with a static seed guarantees identical visual shapes on every execution run.🧠 Mathematical FoundationsHigh-Dimensional Proximity Probabilitiest-SNE begins by calculating conditional probabilities that represent similarities between high-dimensional points $x_i$ and $x_j$. The probability that $x_i$ chooses $x_j$ as its neighbor under a Gaussian distribution is defined as:$$p_{j\mid i} = \frac{\exp(-\left\|x_i - x_j\right\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\left\|x_i - x_k\right\|^2 / 2\sigma_i^2)}$$Solving the Crowding ProblemWhen mapping high-dimensional spaces to 2D, the available volume drops dramatically, which often causes points to crowd together in the center. To fix this, t-SNE uses a heavy-tailed Student-t Distribution (with one degree of freedom) to compute similarities $q_{ij}$ in the low-dimensional space:$$q_{ij} = \frac{(1 + \left\|y_i - y_j\right\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \left\|y_l - y_k\right\|^2)^{-1}}$$The Optimization GoalThe algorithm optimizes the positions of the 2D points $y_i$ by minimizing the Kullback-Leibler (KL) Divergence between the two probability distributions using gradient descent:$$KL(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$This error-checking step penalizes the model heavily if it maps near neighbors in high-dimensional space far apart in 2D space, ensuring your final clusters stay tightly packed.🆚 Dimensionality Reduction Matrix: PCA vs Autoencoder vs t-SNETo contextualize where t-SNE fits into your growing machine learning portfolio, look at this architectural paradigm comparison:Feature DimensionPrincipal Component Analysis (PCA)Autoencoder Neural Networkst-SNE Manifold LearningMathematical EngineLinear Coordinate RotationNon-Linear Structural CompactionProbabilistic Proximity MappingStructural GoalMaximizes overall statistical varianceMinimizes entry reconstruction lossPreserves local neighbor topologyOptimization BaseCovariance EigenvaluesMean Squared Error ($MSE$)Kullback-Leibler ($KL$) DivergencePrimary Use CaseFast data compression & preprocessingScalable latent embedding generationHigh-fidelity data visualization🤝 ContributingContributions help keep our open-source tools robust, optimized, and ready for development!Bash# 1. Fork the Project Repository
# 2. Setup your feature branch
git checkout -b feature/perplexity-tuning-grid

# 3. Commit functional updates
git commit -m "Add: Implement multi-panel subplots analyzing different perplexity values"

# 4. Push updates to origin branch
git push origin feature/perplexity-tuning-grid

# 5. Open a Pull Request
📄 LicenseThis analysis framework is distributed as open-source software under the terms of the MIT License.
