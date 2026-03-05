# Roll-call Voting and the Dynamics of Polarization in the Federal Chamber

This repository contains the methodological framework and the statistical engine used to analyze political polarization in the Brazilian Chamber of Deputies (2014–2025). The project processes massive roll-call voting datasets to map the ideological spectrum and statistically validate the impact of tactical maneuvers (such as obstruction) on the perceived political distance between legislators.

## 🚀 Key Features

The main script (`analyzer.py`) executes a complete Political Data Science pipeline, structured into the following pillars:

### 1. Categorical Divergence Metrics

Unlike traditional models that apply continuous distances to nominal data, this code implements metrics tailored to the Brazilian multiparty system:

* **Strong Divergence ($Div^{\text{strong}}$):** Treats obstruction as a sign of strategic opposition, focusing on ideological purity.
* **Weak Divergence ($Div^{\text{weak}}$):** Normalizes divergence by the universe of active engagement, capturing tactical evasion.
* **MDS (Multidimensional Scaling):** Projects deputies into a two-dimensional space to visualize ideological clusters.

### 2. Statistical Stability (Bootstrap)

To ensure that results are structural rather than the product of sampling noise, the code performs a **Bootstrap routine with 50 independent resamplings** per time window. This generates 95% confidence intervals for the polarization metrics.

### 3. Variance Decomposition (Between vs. Within)

The model separates global polarization into two fundamental components:

* **Inter-Centroid Distance ($Between$):** Measures the actual distance between the centers of gravity of the Government and Opposition blocs.
* **Intra-Cluster Dispersion ($Within$):** Measures the internal cohesion of each bloc, identifying the level of party discipline.

### 4. Dynamic Cluster Optimization ($K$-Ideal)

Using the **Silhouette Score** algorithm, the code dynamically tests the optimal number of clusters ($K$ from 2 to 6) for each period. This allows for the automatic detection of when the system moves away from a bipolar structure toward fragmentation (such as the emergence of the "Centrão").

### 5. Econometric Validation

The project includes formal tests to validate the premises of the metrics:

* **PanelOLS (Fixed Effects):** Isolates intra-party behavior to prove that obstruction is a strategic tool for polarization.
* **Aggregate OLS Regression:** Quantifies how much parliamentary tactics explain the difference between the proposed metrics ($R^2 \approx 0.70$).

## ⏱️ Quick Start

Follow these steps to set up the environment and reproduce the analysis.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/polarization-brazil-chamber.git
cd polarization-brazil-chamber

```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn statsmodels linearmodels matplotlib seaborn

```

### 4. Run the Analysis

Ensure `features_v2.csv` is in `data/interim/`, then run:

```bash
python source/vote_plots.py

```

## 🛠 `PoliticalPolarizationAnalyzer` Class Structure

| Method | Technical Description |
| --- | --- |
| `load_data()` | Data cleaning and handling of mergers (e.g., the creation of UNIÃO Brasil). |
| `run_analysis()` | Runs the MDS and K-Means engine for each 12-month rolling window. |
| `plot_polarization_combined()` | Generates the main plot with confidence bands via Bootstrap. |
| `run_formal_strategy_test_panel()` | Estimates the fixed effects regression model (Party-Time Fixed Effects). |
| `plot_quadrants()` | Classifies deputies into: *Root Supporters*, *Coordinators*, and *Centrists*. |

## 📂 File Organization

```text
├── data/
│   ├── interim/    # Processed voting data (features_v2.csv)
│   └── processed/  # Statistical results and /plots/ directory
├── source/
│   └── analyzer.py # Main analysis script
├── main.tex        # Full academic paper
└── ref.bib         # Bibliographic database

```

---

**How to Cite:**

> Albuquerque, P. C., & Sousa, R. A. (2026). *Roll-call voting and the Dynamics of Polarization in the Federal Chamber*. University of Brasília (UnB) / Central Bank of Brazil.
