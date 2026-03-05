# Roll-call Voting and the Dynamics of Polarization in the Federal Chamber

This repository contains the methodological framework and statistical engine used to analyze political polarization in the Brazilian Chamber of Deputies (2014–2025). The project processes massive roll-call voting datasets to map the ideological spectrum and validate the impact of tactical maneuvers (such as obstruction) on political distance.

## 🚀 Key Features

* **Categorical Divergence Metrics:** Implements $Div^{\text{strong}}$ and $Div^{\text{weak}}$ to handle the ambiguity of abstentions and strategic obstruction in a multiparty system.
* **Structural Decomposition:** Separates polarization into Inter-Centroid Distance (antagonism) and Intra-Cluster Dispersion (cohesion).
* **Bootstrap Robustness:** Uses 50 independent resamplings per window to generate 95% confidence intervals (GMED).
* **Dynamic Clustering:** Employs Silhouette Score optimization to detect the optimal number of legislative blocs ($K$) over time.

## 📂 Project Structure

* `source/`: Contains the core analysis scripts, including:
    * `votes_plots_v2.py`: Main engine for MDS mapping and polarization metrics.
    * `votes_plots_euclidean_v2.py`: Benchmark metrics for comparative analysis.
    * `votes_plots_divergencia_forte_v2.py`: Implementation of Strong Divergence.
* `requirements.txt`: List of necessary Python libraries (scikit-learn, statsmodels, linearmodels, etc.).
* `.gitignore`: Configured to exclude heavy data files and local environment caches.

## ⏱️ Quick Start

1. **Install Dependencies:**
   `pip install -r requirements.txt`
2. **Data Setup:**
   Place your voting data (`features_v2.csv`) in `data/interim/`.
3. **Run Analysis:**
   Execute the scripts within the `source/` directory to generate plots and regression tables.

---
**How to Cite:**
> Albuquerque, P. C., & Sousa, R. A. (2026). *Roll-call voting and the Dynamics of Polarization in the Federal Chamber*. University of Brasília (UnB).
