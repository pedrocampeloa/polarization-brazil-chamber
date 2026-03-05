import pandas as pd
import numpy as np
import os
import logging
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

import warnings

warnings.filterwarnings('ignore')

# Force interactive mode for PyCharm's Scientific Tool window
plt.ion()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class MetricType(Enum):
    STANDARD = "standard"
    STRONG = "strong"
    WEAK = "weak"
    BENCHMARK = "benchmark"


class PoliticalPolarizationAnalyzer:
    def __init__(self, start_date='2014-01-01', end_date='2024-07-01', base_path=None):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        src_path_ = base_path if base_path else os.path.abspath(os.getcwd())
        self.src_path = src_path_ if 'source' in src_path_ else os.path.join(src_path_, 'source')
        self.root_path = os.path.dirname(self.src_path)
        self.interim_path = os.path.join(self.root_path, 'data', 'interim')
        self.processed_path = os.path.join(self.root_path, 'data', 'processed')

        # Folder is inside data/processed/plots/ based on your init
        self.plots_path = os.path.join(self.processed_path, 'plots')
        os.makedirs(self.plots_path, exist_ok=True)

        self.transicao_legislatura = pd.to_datetime(['2014-12-31', '2018-12-31', '2022-12-31'])
        self.party_colors = {
            'MDB': '#1f77b4', 'PDT': '#ff7f0e', 'PL': '#2ca02c', 'PP': '#e377c2',
            'PSB': '#9467bd', 'PSD': '#8c564b', 'PT': '#d62728', 'REPUBLICANOS': '#7f7f7f',
            'UNIÃO': '#bcbd22', 'PSL': '#17becf', 'PSDB': '#e7969c', 'DEM': '#ffbb78'
        }

    def _save_and_show(self, filename):
        """Helper to save plot and handle display."""
        path = os.path.join(self.plots_path, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to: {path}")
        plt.show()
        plt.close('all')  # Prevents plots from stacking in memory

    def load_data(self):
        logging.info("Carregando dados...")
        cols = ['idDeputado', 'nome', 'siglaPartido', 'data', 'idVotacao', 'voto']
        self.df_votos = pd.read_csv(os.path.join(self.interim_path, 'features_v2.csv'), sep=';', usecols=cols)
        self.df_votos['data'] = pd.to_datetime(self.df_votos['data'], dayfirst=True, errors='coerce')
        self.df_votos['siglaPartido'] = self.df_votos['siglaPartido'].replace({'PMDB': 'MDB', 'UNIAO': 'UNIÃO'})

        marco_2021 = pd.Timestamp('2021-03-01')
        mask_fusao = (self.df_votos['siglaPartido'].isin(['PSL', 'DEM'])) & (self.df_votos['data'] >= marco_2021)
        self.df_votos.loc[mask_fusao, 'siglaPartido'] = 'UNIÃO'

    def run_analysis(self, metric_type_str='standard', vote_pct=0.7, vote_dep=0.8, bootstrap_runs=50):
        metric = MetricType(metric_type_str)
        logging.info(f"Processando Métrica: {metric.value} com {bootstrap_runs} Bootstrap runs")

        results, deputy_results = [], []
        periods = self._generate_periods()

        for start, end in periods:
            df_p_full = self.df_votos[(self.df_votos['data'] >= start) &
                                      (self.df_votos['data'] <= end)].copy()
            if df_p_full.empty: continue

            df_party = (df_p_full.sort_values(['idDeputado', 'data'])
                        [['idDeputado', 'nome', 'siglaPartido']]
                        .drop_duplicates(subset='idDeputado', keep='last'))
            df_p_full = df_p_full.drop(['nome', 'siglaPartido'], axis=1).merge(df_party, on='idDeputado', how='left')

            v_counts = df_p_full['idVotacao'].value_counts() / 513
            valid_v_ids = v_counts[v_counts > vote_pct].index.tolist()
            if len(valid_v_ids) < 5: continue

            bootstrap_gmeds = []
            for _ in range(bootstrap_runs):
                sample_v_ids = resample(valid_v_ids, n_samples=int(len(valid_v_ids) * 0.8), replace=False)
                df_sample = df_p_full[df_p_full['idVotacao'].isin(sample_v_ids)]

                d_counts_s = df_sample['idDeputado'].value_counts() / len(sample_v_ids)
                df_sample = df_sample[df_sample['idDeputado'].isin(d_counts_s[d_counts_s > vote_dep].index)]

                if df_sample.empty or len(df_sample['idDeputado'].unique()) < 2: continue

                mapping_s = {'Não': -1, 'Sim': 1, 'Artigo 17': np.nan, 'Abstenção': np.nan}
                mapping_s['Obstrução'] = -1 if metric == MetricType.STRONG else (
                    np.nan if metric == MetricType.WEAK else 0.0)
                df_sample['vote_val'] = df_sample['voto'].replace(mapping_s)
                pivot_s = df_sample.pivot_table(index='idDeputado', columns='idVotacao', values='vote_val').fillna(0.0)

                if metric in [MetricType.STANDARD, MetricType.BENCHMARK]:
                    pos_s = MDS(n_components=2, dissimilarity='euclidean', random_state=None).fit_transform(
                        pivot_s.to_numpy())
                else:
                    dist_matrix_s = self._get_dist_matrix(pivot_s.to_numpy(), metric)
                    pos_s = MDS(n_components=2, dissimilarity='precomputed', random_state=None).fit_transform(
                        dist_matrix_s)
                bootstrap_gmeds.append(pdist(pos_s).mean())

            df_p = df_p_full[df_p_full['idVotacao'].isin(valid_v_ids)]
            d_counts = df_p['idDeputado'].value_counts() / len(valid_v_ids)
            df_p = df_p[df_p['idDeputado'].isin(d_counts[d_counts > vote_dep].index)]
            if df_p.empty: continue

            mapping = {'Não': -1, 'Sim': 1, 'Artigo 17': np.nan, 'Abstenção': np.nan}
            mapping['Obstrução'] = -1 if metric == MetricType.STRONG else (np.nan if metric == MetricType.WEAK else 0.0)
            df_p['vote_val'] = df_p['voto'].replace(mapping)
            pivot = df_p.pivot_table(index=['idDeputado', 'nome', 'siglaPartido'], columns='idVotacao',
                                     values='vote_val').fillna(0.0)

            if metric in [MetricType.STANDARD, MetricType.BENCHMARK]:
                mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
                pos = mds.fit_transform(pivot.to_numpy())
            else:
                dist_matrix = self._get_dist_matrix(pivot.to_numpy(), metric)
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                pos = mds.fit_transform(dist_matrix)

            best_k = 2
            max_sil = -1
            for k_t in range(2, 7):
                if pos.shape[0] > k_t:
                    km_t = KMeans(n_clusters=k_t, random_state=42, n_init=10).fit(pos)
                    score = silhouette_score(pos, km_t.labels_)
                    if score > max_sil: max_sil = score; best_k = k_t

            km_2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(pos)
            centers_2 = km_2.cluster_centers_

            inter_cluster_dist = pairwise_distances(centers_2[0].reshape(1, -1), centers_2[1].reshape(1, -1))[0][0]
            all_dists_to_own_center = []
            for cluster_id in [0, 1]:
                mask = km_2.labels_ == cluster_id
                pts = pos[mask]
                if len(pts) > 0:
                    dists = np.linalg.norm(pts - centers_2[cluster_id], axis=1)
                    all_dists_to_own_center.extend(dists)
            avg_intra_cluster_dispersion = np.mean(all_dists_to_own_center)

            df_dep = pivot.reset_index()[['idDeputado', 'nome', 'siglaPartido']]
            df_dep['period_start'] = start.strftime('%Y-%m')
            df_dep['mds_dim1'], df_dep['mds_dim2'] = pos[:, 0], pos[:, 1]
            df_dep['kmeans_2'] = km_2.labels_

            d0 = pairwise_distances(pos, centers_2[0].reshape(1, -1)).flatten()
            d1 = pairwise_distances(pos, centers_2[1].reshape(1, -1)).flatten()
            df_dep['dist_own'] = np.where(km_2.labels_ == 0, d0, d1)
            df_dep['dist_other'] = np.where(km_2.labels_ == 0, d1, d0)

            for cid in [0, 1]:
                mask = df_dep['kmeans_2'] == cid
                if mask.sum() > 1:
                    df_dep.loc[mask, 'NormDistCentrProprioCluster'] = (df_dep.loc[mask, 'dist_own'] - df_dep.loc[
                        mask, 'dist_own'].mean()) / df_dep.loc[mask, 'dist_own'].std()
                    df_dep.loc[mask, 'NormDistCentrOutroCluster'] = (df_dep.loc[mask, 'dist_other'] - df_dep.loc[
                        mask, 'dist_other'].mean()) / df_dep.loc[mask, 'dist_other'].std()

            results.append({
                'period_start': start,
                'Euclidiana_MDS': np.mean(bootstrap_gmeds) if bootstrap_gmeds else pdist(pos).mean(),
                'gmed_low': np.percentile(bootstrap_gmeds, 2.5) if bootstrap_gmeds else np.nan,
                'gmed_high': np.percentile(bootstrap_gmeds, 97.5) if bootstrap_gmeds else np.nan,
                'Inter_Cluster_Dist': inter_cluster_dist,
                'Intra_Cluster_Disp': avg_intra_cluster_dispersion,
                'k_ideal': best_k,
                'silhouette_score': max_sil
            })

            dm = squareform(pdist(pos))
            np.fill_diagonal(dm, np.nan)
            avg_dist = np.nanmean(dm, axis=1)
            df_dep['normalized_avg_distance'] = (avg_dist - np.nanmean(avg_dist)) / np.nanstd(avg_dist)
            deputy_results.append(df_dep)

        return pd.DataFrame(results), pd.concat(deputy_results)

    def _get_dist_matrix(self, matrix, metric):
        n = matrix.shape[0]
        dist_mtx = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if metric == MetricType.STRONG:
                    mask = (matrix[i] != 0) & (matrix[j] != 0)
                    diffs = np.sum(matrix[i, mask] != matrix[j, mask])
                else:
                    mask = ~((matrix[i] == 0) & (matrix[j] == 0))
                    diffs = np.sum(
                        (matrix[i, mask] != matrix[j, mask]) & (matrix[i, mask] != 0) & (matrix[j, mask] != 0))
                total = np.sum(mask)
                dist_mtx[i, j] = dist_mtx[j, i] = (diffs / total) if total > 0 else 0
        return dist_mtx

    def _generate_periods(self):
        periods, curr = [], self.start_date
        while curr <= self.end_date:
            p_end = curr + pd.DateOffset(months=11) + pd.offsets.MonthEnd(0)
            if not any((self.transicao_legislatura > curr) & (self.transicao_legislatura < p_end)):
                periods.append((curr, p_end))
            curr += pd.DateOffset(months=1)
        return periods

    def run_formal_strategy_test_aggregated(self, df_wek_res, df_str_res, vote_pct=0.7):
        import statsmodels.api as sm
        logging.info("Executando Ponto 1.1: Regressão Agregada...")
        df_reg = pd.merge(df_wek_res[['period_start', 'Euclidiana_MDS']],
                          df_str_res[['period_start', 'Euclidiana_MDS']],
                          on='period_start', suffixes=('_weak', '_strong'))
        df_reg['Divergencia_Diff'] = df_reg['Euclidiana_MDS_weak'] - df_reg['Euclidiana_MDS_strong']
        rates = []
        for start, end in self._generate_periods():
            df_p = self.df_votos[(self.df_votos['data'] >= start) & (self.df_votos['data'] <= end)]
            v_counts = df_p['idVotacao'].value_counts() / 513
            df_p = df_p[df_p['idVotacao'].isin(v_counts[v_counts > vote_pct].index)]
            if df_p.empty: continue
            rates.append({'period_start': start,
                          'TaxaObstrucao': (df_p['voto'] == 'Obstrução').mean(),
                          'TaxaAbstencao': (df_p['voto'] == 'Abstenção').mean()})
        df_final = pd.merge(df_reg, pd.DataFrame(rates), on='period_start')
        X = sm.add_constant(df_final[['TaxaObstrucao', 'TaxaAbstencao']])
        model = sm.OLS(df_final['Divergencia_Diff'], X).fit()
        print(model.summary())
        return model

    def run_formal_strategy_test_panel(self, df_wek_dep, df_str_dep):
        from linearmodels.panel import PanelOLS
        logging.info("Executando Ponto 1.2: Modelo de Painel por Partido...")

        def get_stats(df):
            df['is_obstrucao'] = (df['voto'] == 'Obstrução').astype(
                int) if 'voto' in df.columns else 0  # Ajuste caso voto não esteja no dep_df
            return df.groupby(['period_start', 'siglaPartido'])['normalized_avg_distance'].mean().reset_index()

        # Nota: Para a taxa de obstrução correta, precisamos cruzar com df_votos original ou ter a coluna no dep_df
        # Aqui assumimos o cálculo das distâncias por partido/janela
        stats_wek = get_stats(df_wek_dep).rename(columns={'normalized_avg_distance': 'dist_weak'})
        stats_str = get_stats(df_str_dep).rename(columns={'normalized_avg_distance': 'dist_strong'})

        df_panel = pd.merge(stats_wek, stats_str, on=['period_start', 'siglaPartido'])
        df_panel['diff_div'] = df_panel['dist_weak'] - df_panel['dist_strong']

        # Adicionando a taxa de obstrução vinda da base original para o painel
        obs_rates = self.df_votos.groupby([self.df_votos['data'].dt.to_period('M'), 'siglaPartido'])['voto'].apply(
            lambda x: (x == 'Obstrução').mean()).reset_index()
        obs_rates.columns = ['period_start', 'siglaPartido', 'TaxaObstrucao']
        obs_rates['period_start'] = obs_rates['period_start'].dt.to_timestamp()

        df_panel['period_start'] = pd.to_datetime(df_panel['period_start'])
        df_panel = pd.merge(df_panel, obs_rates, on=['period_start', 'siglaPartido'])

        df_panel = df_panel.set_index(['siglaPartido', 'period_start'])
        res = PanelOLS(df_panel.diff_div, sm.add_constant(df_panel.TaxaObstrucao), entity_effects=True,
                       time_effects=True).fit()

        print("\n--- RESULTADOS PONTO 1.2 (PAINEL) ---")
        print(res.summary)
        return res

    def plot_polarization_combined(self, df_euc, df_str, df_wek):
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()

        ax1.plot(df_euc['period_start'], df_euc['Euclidiana_MDS'],
                 label='Euclidiana MDS (Mean)', color='#1f77b4', linewidth=1.5)
        ax1.fill_between(df_euc['period_start'], df_euc['gmed_low'], df_euc['gmed_high'],
                         color='#1f77b4', alpha=0.15, label='95% IC (Bootstrap)')
        ax1.set_ylabel('Average Distance')

        ax2 = ax1.twinx()
        ax2.plot(df_wek['period_start'], df_wek['Euclidiana_MDS'], label='Weak Divergence', color='#ff7f0e', ls='--')
        ax2.plot(df_str['period_start'], df_str['Euclidiana_MDS'], label='Strong Divergence', color='#d62728', ls='--')
        ax2.fill_between(df_wek['period_start'], df_wek['gmed_low'], df_wek['gmed_high'],
                         color='#ff7f0e', alpha=0.1, label='Weak 95% IC')
        ax2.set_ylabel('Divergence Index')

        self._add_political_context(ax1, df_euc['period_start'].min(), df_euc['period_start'].max())

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize='small', frameon=True)

        plt.title('Polarization Robustness: MDS Distance with Bootstrap Confidence Intervals')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        self._save_and_show("polarization_combined.png")

    def plot_party_trajectories(self, deputy_df, selected_parties=None):
        logging.info("Generating Party Trajectories Plot with legends...")
        if selected_parties is None:
            selected_parties = ['PT', 'PL', 'MDB', 'PP', 'PSDB', 'PSD', 'PSB', 'PDT', 'UNIÃO']

        df_party = deputy_df.groupby(['siglaPartido', 'period_start'])['normalized_avg_distance'].mean().reset_index()
        df_party['period_start'] = pd.to_datetime(df_party['period_start'])

        plt.figure(figsize=(14, 8))
        for party in selected_parties:
            data = df_party[df_party['siglaPartido'] == party].sort_values('period_start')
            if not data.empty:
                plt.plot(data['period_start'], data['normalized_avg_distance'], label=party,
                         color=self.party_colors.get(party, 'gray'), marker='o', markersize=5, linewidth=1.8, alpha=0.8)

        self._add_political_context(plt.gca(), df_party['period_start'].min(), df_party['period_start'].max())
        plt.title(r'Evolution of Average Normalized Distances by Party ($MED_{P,t}^{Z}$)', fontsize=15)
        plt.ylabel('Normalized Mean Euclidean Distance (Z-score)')
        plt.xlabel('Legislative Period')
        plt.legend(title='Political Parties', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small',
                   frameon=True, shadow=True)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save_and_show("party_trajectories_final.png")

    def plot_strategy_correlation(self, df_wek_res, df_str_res):
        logging.info("Executando Ponto 1.3: Gráfico de Correlação...")
        df_corr = pd.merge(df_wek_res[['period_start', 'Euclidiana_MDS']],
                           df_str_res[['period_start', 'Euclidiana_MDS']],
                           on='period_start', suffixes=('_weak', '_strong'))
        df_corr['Diff'] = df_corr['Euclidiana_MDS_weak'] - df_corr['Euclidiana_MDS_strong']
        rates = []
        for start, _ in self._generate_periods():
            df_p = self.df_votos[self.df_votos['data'].dt.to_period('M') == pd.Period(start, freq='M')]
            if not df_p.empty:
                rates.append({'period_start': start, 'TaxaObstrucao': (df_p['voto'] == 'Obstrução').mean()})

        df_plot = pd.merge(df_corr, pd.DataFrame(rates), on='period_start')
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df_plot, x='TaxaObstrucao', y='Diff', scatter_kws={'s': 50, 'alpha': 0.6},
                    line_kws={'color': 'red'})
        plt.title("Validação: Correlação entre Obstrução e Divergência das Métricas")
        plt.grid(True, alpha=0.3)
        self._save_and_show("strategy_correlation.png")

    def plot_mds_grid(self, deputy_df, years):
        max_abs_x = deputy_df['mds_dim1'].abs().max()
        max_abs_y = deputy_df['mds_dim2'].abs().max()
        global_limit = max(max_abs_x, max_abs_y) * 1.15

        num_years = len(years)
        cols = 3
        rows = (num_years + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        fig.suptitle('Political Spectrum Evolution (Deputies Distance)', fontsize=20, y=1.02)
        axes_flat = axes.flatten() if rows > 1 else axes

        for i, year in enumerate(years):
            ax = axes_flat[i]
            p_str = f"{year}-01"
            df_p = deputy_df[deputy_df['period_start'] == p_str]
            if not df_p.empty:
                ax.scatter(df_p['mds_dim1'], df_p['mds_dim2'], c=df_p['kmeans_2'], cmap='viridis', s=50, alpha=0.7,
                           edgecolors='white', linewidths=0.2)
                ax.set_title(f"Year: {year}", fontsize=15, fontweight='bold')
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.set_xlim(-global_limit, global_limit)
                ax.set_ylim(-global_limit, global_limit)
            else:
                ax.set_title(f"Year: {year} (No data)", color='red')
                ax.axis('off')

        for j in range(i + 1, len(axes_flat)): fig.delaxes(axes_flat[j])
        plt.tight_layout()
        self._save_and_show("mds_evolution_grid_full.png")

    def plot_quadrants(self, deputy_df, period, cluster_name="Government"):
        cid = 0 if cluster_name == "Government" else 1
        df_p = deputy_df[(deputy_df['period_start'] == period) & (deputy_df['kmeans_2'] == cid)]
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        for party, color in self.party_colors.items():
            m = df_p['siglaPartido'] == party
            if m.any():
                ax.scatter(df_p.loc[m, 'NormDistCentrProprioCluster'], df_p.loc[m, 'NormDistCentrOutroCluster'],
                           label=party, color=color, alpha=0.7)
        ax.axhline(0, color='gray', ls='--');
        ax.axvline(0, color='gray', ls='--')
        ax.text(-1, 2, f"Root\n{cluster_name}", ha='left');
        ax.text(1, 2, f"Independent\n{cluster_name}", ha='right')
        ax.text(-1, -2, f"{cluster_name}\nCoordinator", ha='left');
        ax.text(1, -2, "Centrist", ha='right')
        plt.title(f"Deputies Distance to the {cluster_name} Cluster ({period})")
        plt.xlabel("Normalized Distance to Own Centroid");
        plt.ylabel("Normalized Distance to Other Centroid")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
        plt.tight_layout()
        self._save_and_show(f"quadrants_{cluster_name}_{period}.png")

    def plot_polarization_decomposition(self, df_res, metric_name="standard"):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.plot(df_res['period_start'], df_res['Inter_Cluster_Dist'], label='Polarization (Between-Cluster Distance)',
                color='#d62728', marker='s', linewidth=2)
        ax.plot(df_res['period_start'], df_res['Intra_Cluster_Disp'],
                label='Fragmentation (Average Within-Cluster Dispersion)', color='#1f77b4', linestyle='--', marker='o')
        self._add_political_context(ax, df_res['period_start'].min(), df_res['period_start'].max())
        plt.title(f'Decomposition of Polarization: {metric_name.capitalize()} Metric', fontsize=14)
        plt.ylabel('Distance Units')
        plt.legend(loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        self._save_and_show(f"decomposition_{metric_name}.png")

    def plot_optimal_k_trend(self, df_res, metric_name="weak"):
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        ax.step(df_res['period_start'], df_res['k_ideal'], where='post', color='#2c3e50', linewidth=2,
                label='Optimal K (Silhouette)')
        ax.fill_between(df_res['period_start'], 2, df_res['k_ideal'], where=(df_res['k_ideal'] > 2), color='#f1c40f',
                        alpha=0.3, label='Multi-block structure')
        self._add_political_context(ax, df_res['period_start'].min(), df_res['period_start'].max())
        ax.set_yticks([2, 3, 4, 5, 6])
        plt.title(f'Evolution of Optimal Number of Clusters ({metric_name.capitalize()})', fontsize=14)
        plt.ylabel('Number of Clusters (K)')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.legend(loc='upper left')
        self._save_and_show(f"k_ideal_trend_{metric_name}.png")

    def plot_party_contribution(self, df_dep, period):
        df_p = df_dep[df_dep['period_start'] == period].copy()
        party_contrib = df_p.groupby('siglaPartido').agg(
            {'normalized_avg_distance': 'mean', 'idDeputado': 'count'}).rename(columns={'idDeputado': 'size'})
        party_contrib['contribution_score'] = party_contrib['normalized_avg_distance'] * (
                    party_contrib['size'] / party_contrib['size'].sum())
        party_contrib = party_contrib.sort_values('contribution_score', ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(party_contrib.index, party_contrib['contribution_score'],
                color=[self.party_colors.get(p, 'gray') for p in party_contrib.index])
        plt.axhline(0, color='black', lw=1)
        plt.title(f"Party Contribution to Global Polarization ({period})")
        plt.ylabel("Contribution Score (Dist * Weight)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_and_show(f"party_contribution_{period}.png")

    def plot_tactical_voting_profile(self, vote_pct=0.7):
        v_counts = self.df_votos['idVotacao'].value_counts() / 513
        df_f = self.df_votos[self.df_votos['idVotacao'].isin(v_counts[v_counts > vote_pct].index.tolist())].copy()
        df_f['mes'] = df_f['data'].dt.to_period('M')
        df_tactics = df_f[df_f['voto'].isin(['Obstrução', 'Abstenção'])].groupby(['mes', 'voto']).size().unstack(
            fill_value=0)
        df_pct = df_tactics.divide(df_f.groupby('mes').size(), axis=0) * 100
        plt.figure(figsize=(15, 6))
        df_pct.plot(kind='area', stacked=True, ax=plt.gca(), alpha=0.7, color=['#f39c12', '#2c3e50'])
        plt.title("Evolução de Tácticas Parlamentares: Obstrução e Abstenção", fontsize=14)
        plt.ylabel("% do Total de Registos de Voto")
        plt.xlabel("Período")
        plt.axvspan('2021-01', '2021-12', color='red', alpha=0.1, label='Pico Polarização')
        plt.tight_layout()
        self._save_and_show("tactical_voting_profile.png")

    def _add_political_context(self, ax, start_date, end_date):
        milestones = [('2014-01-01', 'Dilma II (L55)'), ('2016-08-31', 'Temer (L55)'),
                      ('2019-01-01', 'Bolsonaro (L56)'), ('2023-01-01', 'Lula III (L57)'), ('2025-02-01', '')]
        y_min, y_max = ax.get_ylim()
        for i in range(len(milestones) - 1):
            date_curr, label = pd.to_datetime(milestones[i][0]), milestones[i][1]
            if start_date <= date_curr <= end_date: ax.axvline(x=date_curr, color='gray', ls='--', lw=0.8, alpha=0.4)
            actual_start, actual_end = max(date_curr, start_date), min(pd.to_datetime(milestones[i + 1][0]), end_date)
            if actual_start < actual_end: ax.text(actual_start + (actual_end - actual_start) / 2,
                                                  y_min + (y_max - y_min) * 0.02, label, fontsize=8, color='#444444',
                                                  ha='center', va='bottom', fontweight='bold')

if __name__ == "__main_1_":
    analyzer = PoliticalPolarizationAnalyzer()
    analyzer.load_data()

    # 1. Rodar pipeline
    df_euc_res, df_euc_dep = analyzer.run_analysis('standard')
    analyzer.plot_optimal_k_trend(df_euc_res, "Euclidean")

    df_str_res, df_str_dep = analyzer.run_analysis('strong')
    analyzer.plot_optimal_k_trend(df_str_res, "strong")

    df_wek_res, df_wek_dep = analyzer.run_analysis('weak')
    analyzer.plot_optimal_k_trend(df_wek_res, "weak")

    analyzer.run_formal_strategy_test_aggregated(df_wek_res, df_str_res)  # 1.1
    analyzer.run_formal_strategy_test_panel(df_wek_dep, df_str_dep)  # 1.2
    analyzer.plot_strategy_correlation(df_wek_res, df_str_res)  # 1.3

    year_list = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    # 2. Gerar Plots
    analyzer.plot_polarization_combined(df_euc_res, df_str_res, df_wek_res)
    analyzer.plot_party_trajectories(df_wek_dep)

    analyzer.plot_mds_grid(df_wek_dep, year_list)

    analyzer.plot_quadrants(df_wek_dep, '2024-05', "Government")
    analyzer.plot_quadrants(df_wek_dep, '2024-05', "Opposition")

    # NOVO: Gerar o gráfico de decomposição para a métrica Euclidiana (Benchmark)
    # Isso mostrará se o "esfriamento" no governo Temer foi coesão ou aproximação
    analyzer.plot_polarization_decomposition(df_euc_res, "Euclidean")
    analyzer.plot_polarization_decomposition(df_wek_res, "Weak Divergence")

    # 2. Chamar a contribuição para períodos-chave
    analyzer.plot_party_contribution(df_wek_dep, '2021-05') #O auge da polarização no Governo Bolsonaro
    analyzer.plot_party_contribution(df_wek_dep, '2024-05') #reorganização no Governo Lula III

    analyzer.plot_tactical_voting_profile()