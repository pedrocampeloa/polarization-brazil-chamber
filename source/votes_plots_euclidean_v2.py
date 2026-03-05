import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import os
import warnings

warnings.filterwarnings('ignore')

# Diretórios
src_path = os.path.abspath(os.getcwd())
root_path = os.path.dirname(src_path)
data_path = os.path.join(root_path, 'data')
interim_path = os.path.join(data_path, 'interim')
processed_path = os.path.join(data_path, 'processed')

# Carregar dados
df_bloco = pd.read_csv(os.path.join(interim_path,'blocos_partidarios.csv'), sep=';')
df_bloco.loc[df_bloco['siglaPartido'] == 'PMDB', 'siglaPartido'] = 'MDB'

cols = ['idDeputado', 'nome', 'idPartido', 'siglaPartido', 'data', 'idLegislatura', 'idVotacao', 'voto']
df_votos = pd.read_csv(os.path.join(interim_path, 'features_v2.csv'), sep=';', usecols=cols)
df_votos = df_votos.drop_duplicates(subset=['idDeputado', 'idVotacao'])
df_votos['year'] = pd.to_datetime(df_votos['data']).dt.year
df_votos.loc[df_votos['siglaPartido'] == 'PMDB', 'siglaPartido'] = 'MDB'

# Correção para fusão PSL e DEM a partir de março de 2021
df_votos['data'] = pd.to_datetime(df_votos['data'], dayfirst=True, errors='coerce')  # Assegurar formato datetime para comparação
marco_2021 = pd.Timestamp('2021-03-01')
mask_fusao = (df_votos['siglaPartido'].isin(['PSL', 'DEM'])) & (df_votos['data'] >= marco_2021)
df_votos.loc[mask_fusao, 'siglaPartido'] = 'UNIÃO'

# Checar e ajustar possíveis problemas com o caractere 'Â'
df_votos['siglaPartido'] = df_votos['siglaPartido'].replace('UNIAO', 'UNIÃO')


# Parâmetros
pct_votes = 0.6
pct_dep = 0.6

# Função para criar períodos
def assign_voting_periods(df, start_month, end_month, transicao_legislatura):
    periods = []
    current_period_start = start_month

    while current_period_start <= end_month:
        current_period_end = current_period_start + pd.DateOffset(months=11)
        current_period_end = current_period_end + pd.offsets.MonthEnd(0)

        if not any((transicao_legislatura > current_period_start) & (transicao_legislatura < current_period_end)):
            periods.append((current_period_start, current_period_end))

        current_period_start += pd.DateOffset(months=1)

    return periods

# Função para calcular a distância média Euclidiana
def calculate_average_euclidean_distance(points):
    distances = pdist(points, metric='euclidean')
    return np.mean(distances)

# Função para calcular a distância média do Cosseno
def calculate_average_sqeuclidean_distance(points):
    distances = pdist(points, metric='sqeuclidean')
    return np.mean(distances)

# Função para calcular o stress de Kruskal normalizado
def calculate_normalized_stress(original_distances, mds_distances):
    numerator = np.sqrt(np.sum((original_distances - mds_distances) ** 2))
    denominator = np.sqrt(np.sum(original_distances ** 2))
    stress_normalized = numerator / denominator
    return stress_normalized

# Datas de transição de legislatura
transicao_legislatura = pd.to_datetime(['2014-12-31', '2018-12-31', '2022-12-31'])

# Definir período
start_month = pd.Timestamp('2014-01-01')
end_month = pd.Timestamp('2024-07-01')

# Gerar períodos
periods = assign_voting_periods(df_votos, start_month, end_month, transicao_legislatura)

# Lista para armazenar resultados
results = []

# Lista para armazenar distâncias de deputados
deputy_distances_results = []

# Lista para armazenar resultados de stress
stress_results = []

# Loop para cada período
for period_start, period_end in periods:
    print(f"Período: {period_start.strftime('%Y-%m')} a {period_end.strftime('%Y-%m')}")

    # Filtrar dados do período
    df_votos_f = df_votos[(df_votos['data'] >= period_start) & (df_votos['data'] <= period_end)]
    df_party  = df_votos_f.sort_values(['idDeputado','data'])[['nome', 'siglaPartido']].drop_duplicates().groupby('nome', as_index=False).last()
    df_votos_f = df_votos_f.drop('siglaPartido',axis=1).merge(df_party,how='left', on = 'nome')

    # Verificar se há dados suficientes
    if len(df_votos_f) < 2:
        continue

    # Remover votações com poucos votantes
    df_vot_size = df_votos_f['idVotacao'].value_counts().reset_index()
    df_vot_size.columns = ['idVotacao', 'size']
    df_vot_size['pct'] = df_vot_size['size'] / 513
    list_vote_id = df_vot_size[df_vot_size['pct'] > pct_votes]['idVotacao']

    # Remover deputados com baixa frequência
    df_dep_size = df_votos_f[df_votos_f['idVotacao'].isin(list_vote_id)][['idDeputado', 'nome']].value_counts().reset_index().rename(columns={'count': 'size'})
    df_dep_size['pct'] = df_dep_size['size'] / len(list_vote_id)
    list_dep_id = df_dep_size[df_dep_size['pct'] > pct_dep]['idDeputado']

    df_votos_f = df_votos_f[(df_votos_f['idVotacao'].isin(list_vote_id)) & (df_votos_f['idDeputado'].isin(list_dep_id))]

    # Mapear votos para valores numéricos
    mapping_votes = {'Não': -1, 'Sim': 1, 'Artigo 17': np.nan, 'Abstenção': np.nan, 'Obstrução': 0.0}
    df_votos_f['vote'] = df_votos_f['voto'].replace(mapping_votes)

    id_cols = ['idDeputado', 'nome', 'siglaPartido']
    df_pivot = df_votos_f.pivot_table(index=id_cols, columns='idVotacao', values='vote')

    # Preencher NaNs com 0.5
    df_filled = df_pivot.fillna(0.0)


    # Verificar se há dados suficientes após o pivot
    if df_filled.shape[0] < 2:
        continue


    # Calcular as distâncias originais (matriz de distâncias no espaço original)
    original_distances = pairwise_distances(df_filled.to_numpy(), metric='euclidean')
    
    # Dicionário para armazenar o resultado do stress para cada dimensão
    stress_per_period = {'period_start': period_start.strftime('%Y-%m'), 'period_end': period_end.strftime('%Y-%m')}

    # Testar diferentes números de dimensões (de 1 até 10)
    for n_dims in range(1, 2): #mudar para mais se quiser fazer análise
        mds = MDS(n_components=n_dims, dissimilarity='precomputed', random_state=42)
        mds_distances = mds.fit_transform(original_distances)
        # Calcular a matriz de distâncias no espaço MDS
        mds_pairwise_distances = pairwise_distances(mds_distances, metric='euclidean')
        # Calcular o stress de Kruskal normalizado para este número de dimensões
        stress_normalized = calculate_normalized_stress(original_distances, mds_pairwise_distances)
        # Adicionar o resultado para a dimensão corrente ao dicionário
        stress_per_period[f'Dimensão {n_dims}'] = stress_normalized

    # Adicionar os resultados deste período à lista geral
    stress_results.append(stress_per_period)
    
    # Aplicar MDS
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
    positions = mds.fit_transform(original_distances)
    
    # Guardar o valor do stress de Kruskal para o período
    #stress_value = mds.stress_
    
    # K-means clustering com k=2 e k=3
    kmeans_2 = KMeans(n_clusters=2, random_state=42)
    kmeans_2.fit(positions)
    clusters_k2 = kmeans_2.fit_predict(positions) 
    
    kmeans_3 = KMeans(n_clusters=3, random_state=42)
    kmeans_3.fit(positions)
    clusters_k3 = kmeans_3.fit_predict(positions)
    
    # Calcular os centróides de cada cluster
    centroides_k2 = kmeans_2.cluster_centers_
    
    # Calcular a distância entre os dois centróides (k=2)
    distancia_entre_centroides_k2 = pairwise_distances(centroides_k2)[0,1]
    
    # Calcular distâncias médias entre os pontos MDS
    avg_euclidean_distance = calculate_average_euclidean_distance(positions)
    avg_sqeuclidean_distance = calculate_average_sqeuclidean_distance(positions)
    
    #Calcular as distância conforme algoritmo de Espaço-Produto de Hausmann e Hidalgo
    df_reset = df_filled.reset_index()
    df_reduced = df_reset.drop(columns=['nome', 'siglaPartido'])
    mcp = df_reduced.drop(columns=['idDeputado']).to_numpy()
    mcp_t = mcp.transpose()
    coocorrencia = mcp.dot(mcp_t)
    ubiquidade_diagonal = np.diag(np.diag(coocorrencia))
    ubiquidade_diagonal_inversa = np.linalg.inv(ubiquidade_diagonal)
    prob_cond = np.matmul(ubiquidade_diagonal_inversa, coocorrencia)
    prob_cond_t = prob_cond.transpose()
    distancias_EP = np.minimum(prob_cond, prob_cond_t) #essa linha aplica a mínima probabilidade condicional entre x e y
    distancias_EP = 1 - distancias_EP # Transformar em distâncias
    np.fill_diagonal(distancias_EP, 0) #desnecessário dado o passo anterior
    distancias_EP_df=pd.DataFrame(distancias_EP)
    distancias_EP_df.to_csv(os.path.join(processed_path, f"distanciasEP_{period_start.strftime('%Y-%m')}.csv"))
    mask_diagonal = np.eye(distancias_EP.shape[0], dtype=bool)
    distancia_media_EP = np.mean(distancias_EP[~mask_diagonal])
    distancias_EP_aoquadrado = np.square(distancias_EP)
    mask_diagonal2 = np.eye(distancias_EP_aoquadrado.shape[0], dtype=bool)
    media_valores_ao_quadrado = np.mean(distancias_EP_aoquadrado[~mask_diagonal2])
    

    
    #Calcular a distância média e variância de cada deputado em relação aos outros usando operações vetorizadas
    distance_matrix = squareform(pdist(positions, metric='euclidean'))
    # Obter as distâncias de cada deputado para os outros, excluindo a si próprio
    np.fill_diagonal(distance_matrix, np.nan)  # Preencher a diagonal com NaN para excluir a si próprio
    max_distancia_deputados = np.nanmax(distance_matrix)
    
    # Armazenar resultados gerais
    result = {
        'period_start': period_start,
        'period_end': period_end,
        'Euclidiana_MDS': avg_euclidean_distance,
        'SqEuclidiana_MDS': avg_sqeuclidean_distance,
        'Distancia_EP': distancia_media_EP,
        'SQDistancia_EP': media_valores_ao_quadrado,
        'DistanciaEntreCentroidesK2': distancia_entre_centroides_k2,
        'MaxDistanciaDeputados': max_distancia_deputados
    }
    results.append(result)
    
    # Calcular a distância média e a variância de maneira vetorizada
    avg_distances = np.nanmean(distance_matrix, axis=1)  # Distância média para cada deputado
    variances = np.nanvar(distance_matrix, axis=1)  # Variância das distâncias para cada deputado
    stds = np.nanstd(distance_matrix, axis=1)
    medians = np.nanmedian(distance_matrix, axis=1)

    # Adicionar colunas de distância média e variância diretamente no DataFrame `df_filled`
    df_filled['avg_distance'] = avg_distances
    df_filled['variance_distance'] = variances
    df_filled['std_distance'] = stds
    df_filled['median_distance'] = medians
    
    # Adição: Calcular a média das distâncias médias de todos os deputados neste período
    mean_avg_distance = df_filled['avg_distance'].mean()
    std_avg_distance = df_filled['avg_distance'].std()


    df_filled['normalized_avg_distance'] = (df_filled['avg_distance'] - mean_avg_distance) / std_avg_distance

    # Adição: Normalizar a distância média de cada deputado pela média das distâncias médias do período
   # df_filled['normalized_avg_distance'] = df_filled['avg_distance'] / mean_avg_distance
    
    # Calcular a distância média e a variância de maneira vetorizada para o Espaço-Produto
    avg_distances_EP = np.nanmean(distancias_EP, axis=1)  # Distância média para cada deputado
    variances_EP = np.nanvar(distancias_EP, axis=1)  # Variância das distâncias para cada deputado
    stds_EP = np.nanstd(distancias_EP, axis=1)
    medians_EP = np.nanmedian(distancias_EP, axis=1)
    df_filled['avg_distance_EP'] = avg_distances_EP
    df_filled['variance_distance_EP'] = variances_EP
    df_filled['std_distance_EP'] = stds_EP
    df_filled['median_distance_EP'] = medians_EP
    mean_avg_distance_EP = df_filled['avg_distance_EP'].mean()
    std_avg_distance_EP = df_filled['avg_distance_EP'].std()
    
    df_filled['normalized_avg_distance_EP'] = (df_filled['avg_distance_EP'] - mean_avg_distance_EP) / std_avg_distance_EP
    #df_filled['normalized_avg_distance_EP'] = df_filled['avg_distance_EP'] / mean_avg_distance_EP
    
    
    # Adicionar a coluna do período
    df_filled['period_start'] = period_start.strftime('%Y-%m')
    
    # Adicionar os clusters ao DataFrame
    df_filled['kmeans_2'] = kmeans_2.labels_
    df_filled['kmeans_3'] = kmeans_3.labels_
    
    # Adicionar as posições do MDS ao DataFrame
    df_filled['mds_dim1'] = positions[:, 0]
    df_filled['mds_dim2'] = positions[:, 1]
    
    # Resetar o índice para transformar `idDeputado` e `nome` em colunas
    df_filled_reset = df_filled.reset_index()
    

    # Distância de cada deputado para o centróide do próprio cluster
    df_filled_reset['DistCentrProprioCluster'] = np.where(
        clusters_k2 == 0, 
        pairwise_distances(positions, centroides_k2[0].reshape(1, -1)).flatten(), 
        pairwise_distances(positions, centroides_k2[1].reshape(1, -1)).flatten()
    )
    
    # Distância de cada deputado para o centróide do outro cluster
    df_filled_reset['DistCentrOutroCluster'] = np.where(
        clusters_k2 == 0, 
        pairwise_distances(positions, centroides_k2[1].reshape(1, -1)).flatten(), 
        pairwise_distances(positions, centroides_k2[0].reshape(1, -1)).flatten()
    )
    
    # Calcular a média e o desvio padrão das distâncias para o centróide do próprio cluster
    media_dist_proprio_cluster_0 = df_filled_reset[df_filled_reset['kmeans_2'] == 0]['DistCentrProprioCluster'].mean()
    std_dist_proprio_cluster_0 = df_filled_reset[df_filled_reset['kmeans_2'] == 0]['DistCentrProprioCluster'].std()
    
    media_dist_proprio_cluster_1 = df_filled_reset[df_filled_reset['kmeans_2'] == 1]['DistCentrProprioCluster'].mean()
    std_dist_proprio_cluster_1 = df_filled_reset[df_filled_reset['kmeans_2'] == 1]['DistCentrProprioCluster'].std()
    
    # Calcular a média e o desvio padrão das distâncias para o centróide do outro cluster
    media_dist_outro_cluster_0 = df_filled_reset[df_filled_reset['kmeans_2'] == 0]['DistCentrOutroCluster'].mean()
    std_dist_outro_cluster_0 = df_filled_reset[df_filled_reset['kmeans_2'] == 0]['DistCentrOutroCluster'].std()
    
    media_dist_outro_cluster_1 = df_filled_reset[df_filled_reset['kmeans_2'] == 1]['DistCentrOutroCluster'].mean()
    std_dist_outro_cluster_1 = df_filled_reset[df_filled_reset['kmeans_2'] == 1]['DistCentrOutroCluster'].std()
    
    # Normalizar (z-score) as distâncias para o centróide do próprio cluster
    df_filled_reset['NormDistCentrProprioCluster'] = np.where(
        df_filled_reset['kmeans_2'] == 0,
        (df_filled_reset['DistCentrProprioCluster'] - media_dist_proprio_cluster_0) / std_dist_proprio_cluster_0,
        (df_filled_reset['DistCentrProprioCluster'] - media_dist_proprio_cluster_1) / std_dist_proprio_cluster_1
    )
    
    # Normalizar (z-score) as distâncias para o centróide do outro cluster
    df_filled_reset['NormDistCentrOutroCluster'] = np.where(
        df_filled_reset['kmeans_2'] == 0,
        (df_filled_reset['DistCentrOutroCluster'] - media_dist_outro_cluster_1) / std_dist_outro_cluster_1,  # Média e std do outro cluster (1)
        (df_filled_reset['DistCentrOutroCluster'] - media_dist_outro_cluster_0) / std_dist_outro_cluster_0   # Média e std do outro cluster (0)
    )

    
    # Selecionar as colunas desejadas
    df_deputy_distances_period = df_filled_reset[['period_start', 'idDeputado', 'nome', 'siglaPartido', 'normalized_avg_distance', 
                                                  'avg_distance', 'median_distance', 'std_distance', 'variance_distance',
                                                  'normalized_avg_distance_EP', 'avg_distance_EP', 'median_distance_EP', 
                                                  'std_distance_EP', 'variance_distance_EP', 'kmeans_2', 'kmeans_3',
                                                  'mds_dim1', 'mds_dim2', 'DistCentrProprioCluster', 'DistCentrOutroCluster',
                                                  'NormDistCentrProprioCluster', 'NormDistCentrOutroCluster',]]
    
    # Adicionar ao DataFrame final
    deputy_distances_results.append(df_deputy_distances_period)
    

# Concatenar todos os resultados
df_deputy_distances = pd.concat(deputy_distances_results)

# Salvar distâncias de cada deputado em CSV
df_deputy_distances.to_csv(os.path.join(processed_path, f"deputy_distances_euclidean.csv"), index=False)

# Converter resultados gerais em DataFrame
df_results = pd.DataFrame(results)

# Salvar resultados gerais em CSV
df_results.to_csv(os.path.join(processed_path, f"average_mds_distances_euclidean.csv"), index=False)

# Converter os resultados de stress em um DataFrame
df_stress = pd.DataFrame(stress_results)

# Salvar o DataFrame de stress em um arquivo CSV
#df_stress.to_csv('C:/Users/renantes/Desktop/Profissional/Discurso Politico/data/stress_analysis_mds.csv', index=False)

# Plotar os resultados gerais com o eixo X exibindo semestres
plt.figure(figsize=(12, 6))
plt.plot(df_results['period_start'], df_results['Euclidiana_MDS'], marker='o', label='Euclidiana MDS')

# Customizações
plt.title('Polarization in the Federal Chamber')
plt.xlabel('Semester/Year')
plt.ylabel('Deputies in Roll Call Votes Average Distance')
plt.legend()

# Configuração do eixo X para mostrar datas sem sobreposição, com rotação de 90 graus
plt.xticks(rotation=90)  # Alinhamento perpendicular do texto
plt.yticks()

plt.tight_layout()
plt.show()

# Geração automática de cores para os partidos (usando cores do Matplotlib)
partidos_cores = {}
available_colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)  # Tabela de cores disponíveis no Matplotlib
partidos_unicos = df_deputy_distances['siglaPartido'].unique()

for i, partido in enumerate(partidos_unicos):
    partidos_cores[partido] = available_colors[i % len(available_colors)]  # Gera cores automaticamente para cada partido

# Parâmetro ajustável: número mínimo de deputados por partido em cada período
min_deputados_por_partido = 8  # Você pode ajustar este valor

# DataFrame para armazenar as médias das distâncias normalizadas por partido
partido_distancias_results = []

# Loop para calcular a média das distâncias normalizadas por partido em cada período
for period_start, period_end in periods:
    # Filtrar dados do período atual
    df_period = df_deputy_distances[df_deputy_distances['period_start'] == period_start.strftime('%Y-%m')]

    # Agrupar por partido e calcular a média das distâncias normalizadas
    partido_group = df_period.groupby('siglaPartido').agg(
        avg_normalized_distance=('normalized_avg_distance', 'mean'),
        num_deputados=('idDeputado', 'size')
    ).reset_index()

    # Filtrar partidos que têm pelo menos 'min_deputados_por_partido' deputados no período
    partido_group = partido_group[partido_group['num_deputados'] >= min_deputados_por_partido]

    # Adicionar a coluna de período
    partido_group['period_start'] = period_start.strftime('%Y-%m')

    # Armazenar os resultados
    partido_distancias_results.append(partido_group)

# Concatenar todos os resultados em um DataFrame
df_partido_distancias = pd.concat(partido_distancias_results)

# Salvar o CSV com as médias das distâncias normalizadas por partido


df_partido_distancias.to_csv(os.path.join(processed_path, 'partido_distancias_euclidean.csv'), index=False)

# Lista de partidos a serem incluídos no plot
partidos_selecionados = ['MDB', 'PDT', 'PL', 'PP', 'PSB', 'PSD', 'PT', 'REPUBLICANOS', 'UNIÃO', 'PSL', 'PSDB', 'DEM']

# Cores manualmente ajustadas para cada partido, garantindo contraste
partidos_cores = {
    'MDB': '#1f77b4',        # Azul
    'PDT': '#ff7f0e',        # Laranja
    'PL': '#2ca02c',         # Verde
    'PP': '#e377c2',        # Rosa
    'PSB': '#9467bd',        # Roxo
    'PSD': '#8c564b',        # Marrom
    'PT': '#d62728',         # Vermelho
    'REPUBLICANOS': '#7f7f7f', # Cinza
    'UNIÃO': '#bcbd22',      # Verde amarelado
    'PSL': '#17becf',        # Ciano
    'PSDB': '#e7969c',       # Rosa claro
    'DEM': '#ffbb78'         # Pêssego
}

# Plotar a evolução dos partidos ao longo do tempo (somente partidos com pelo menos min_deputados_por_partido deputados)
plt.figure(figsize=(12, 6))

# Plotar a evolução das médias das distâncias normalizadas para partidos selecionados
for partido in partidos_selecionados:
    # Filtrar o DataFrame apenas para o partido atual
    df_partido = df_partido_distancias[df_partido_distancias['siglaPartido'] == partido]
    
    # Verificar se há dados para o partido
    if df_partido.empty:
        continue
    
    # Plotar os dados do partido com a cor específica
    plt.plot(pd.to_datetime(df_partido['period_start'], format='%Y-%m'), df_partido['avg_normalized_distance'], 
             label=partido, color=partidos_cores.get(partido, 'gray'), marker='o')

# Customizações do gráfico
plt.title('Distances for Selected Parties')
# plt.xlabel('Período')
plt.ylabel('Average Normalized Distances')
plt.grid(False)

# Configurações do eixo X (formatar as datas)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Ticks a cada 6 meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato Ano-Mês
plt.xticks(rotation=90)

# Adicionar legenda e posicioná-la fora do gráfico
plt.legend(title='Parties', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

# Exibir o gráfico
plt.tight_layout()
plt.show()


# Períodos selecionados divididos para dois plots
selected_periods_1 = ['2014-01', '2015-01', '2016-01', '2017-01', '2018-01']
selected_periods_2 = ['2019-01', '2020-01', '2021-01', '2022-01', '2023-01']

# Primeiro conjunto de períodos - 5 slots em um plot
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, period in enumerate(selected_periods_1):
    ax = axes[i]  # Posição do gráfico no eixo de subplots
    
    # Filtrar dados para o período específico
    df_period = df_deputy_distances[df_deputy_distances['period_start'] == period]
    
    # Verificar se há dados para o período
    if len(df_period) == 0:
        continue

    # Obter as posições do MDS e os clusters de k=2
    positions_period = df_period[['mds_dim1', 'mds_dim2']].to_numpy()
    clusters_period = df_period['kmeans_2'].to_numpy()

    # Plotar resultados do MDS com cores baseadas nos clusters de k=2
    scatter = ax.scatter(positions_period[:, 0], positions_period[:, 1], c=clusters_period, cmap='viridis', s=50)
    
    # Definir limites dos eixos X e Y
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    # Customizar o gráfico
    ax.set_title(f'Período: {period}')
    plt.grid(False)

# Ajustar o layout e exibir o primeiro gráfico
plt.tight_layout()
plt.show()

# Segundo conjunto de períodos - outros 5 slots em outro plot
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, period in enumerate(selected_periods_2):
    ax = axes[i]  # Posição do gráfico no eixo de subplots
    
    # Filtrar dados para o período específico
    df_period = df_deputy_distances[df_deputy_distances['period_start'] == period]
    
    # Verificar se há dados para o período
    if len(df_period) == 0:
        continue

    # Obter as posições do MDS e os clusters de k=2
    positions_period = df_period[['mds_dim1', 'mds_dim2']].to_numpy()
    clusters_period = df_period['kmeans_2'].to_numpy()

    # Plotar resultados do MDS com cores baseadas nos clusters de k=2
    scatter = ax.scatter(positions_period[:, 0], positions_period[:, 1], c=clusters_period, cmap='viridis', s=50)
    
    # Definir limites dos eixos X e Y
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    # Customizar o gráfico
    ax.set_title(f'Período: {period}')
    plt.grid(False)

# Ajustar o layout e exibir o segundo gráfico
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 3, figsize=(15, 12))

selected_periods = ['2014-01', '2015-01', '2016-01', '2017-01', '2018-01',
                    '2019-01', '2020-01', '2021-01', '2022-01', '2023-01']

# Definir as datas para os plots
plot_dates = ['2021-01', '2023-01']

# Geração automática de cores para todos os partidos, mantendo cores já atribuídas aos partidos selecionados
all_partidos = df_deputy_distances['siglaPartido'].unique()
available_colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
for i, partido in enumerate(all_partidos):
    if partido not in partidos_cores:  # Manter cores para partidos já selecionados, adicionar novas para os restantes
        partidos_cores[partido] = available_colors[i % len(available_colors)]
        
clusters = [0, 1]  # 0 para Governista, 1 para Oposicionista

# Definindo títulos para cada cenário
titles = {
    (plot_dates[0], 0): "Deputies Distance to the Government Bloc - 2021",
    (plot_dates[0], 1): "Deputies Distance to the Opposition Bloc - 2021",
    (plot_dates[1], 0): "Deputies Distance to the Government Bloc - 2023",
    (plot_dates[1], 1): "Deputies Distance to the Opposition Bloc - 2023",
}

# Loop pelos períodos e clusters para gerar os quatro gráficos
for period in plot_dates:
    for k in clusters:
        # Filtrar dados para o período específico
        df_period = df_deputy_distances[df_deputy_distances['period_start'] == period]
        df_cluster = df_period[df_period['kmeans_2'] == k]  # Filtrar dados para o cluster específico

        # Criar uma nova figura
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.suptitle(titles[(period, k)], fontsize=16)

        # Plotar scatter plot com as cores do partido
        partido_handles = {}
        for _, row in df_cluster.iterrows():
            partido = row['siglaPartido']
            color = partidos_cores.get(partido, 'gray')  # Definir a cor conforme o partido
            scatter = ax.scatter(row['NormDistCentrProprioCluster'], row['NormDistCentrOutroCluster'], color=color, label=partido, s=50, alpha=0.7)
            if partido not in partido_handles:  # Armazena apenas uma entrada para cada partido
                partido_handles[partido] = scatter

        # Configurações do subplot
        ax.set_xlabel('Normalized Distance to Own Centroid')
        ax.set_ylabel('Normalized Distance to Other Centroid')
        
        # Linhas no zero para os eixos X e Y
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Linha horizontal no zero
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)  # Linha vertical no zero
        ax.grid(False)  # Desativa o grid principal

        # Adicionar legenda fora da área de plotagem
        fig.legend(partido_handles.values(), partido_handles.keys(), loc='center left', bbox_to_anchor=(1, 0.5), title='Partidos', fontsize='small')

        # Ajustar layout e exibir o gráfico
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Espaço extra à direita para a legenda e para o título
        plt.show()

# Plotar os espectros políticos anuais em um só plot
selected_periods = ['2014-01', '2015-01', '2016-01', '2017-01', '2018-01', 
                    '2019-01', '2020-01', '2021-01', '2022-01', '2023-01']

# Configurar o layout de subplots com 5 linhas e 2 colunas
fig, axes = plt.subplots(5, 2, figsize=(5, 20))
fig.suptitle('Deputies Distance', fontsize=16)

# Loop sobre os períodos e os eixos para cada subplot
for i, period in enumerate(selected_periods):
    row = i // 2  # Define a linha (0 a 4)
    col = i % 2   # Define a coluna (0 ou 1)
    ax = axes[row, col]  # Acessa o subplot correto

    # Filtrar dados para o período específico
    df_period = df_deputy_distances[df_deputy_distances['period_start'] == period]
    
    # Verificar se há dados para o período
    if len(df_period) == 0:
        continue

    # Obter as posições do MDS e os clusters de k=2
    positions_period = df_period[['mds_dim1', 'mds_dim2']].to_numpy()
    clusters_period = df_period['kmeans_2'].to_numpy()

    # Plotar resultados do MDS com cores baseadas nos clusters de k=2
    scatter = ax.scatter(positions_period[:, 0], positions_period[:, 1], c=clusters_period, cmap='viridis', s=50)
    
    # Definir limites dos eixos X e Y
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    # Configurar o título de cada subplot
    ax.set_title(f'{period}')

    # Adicionar linhas no zero para os eixos X e Y
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Desativar grid
    ax.grid(False)

# Ajustar o layout para evitar sobreposição e exibir o gráfico
#plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar para não sobrepor o título principal
plt.show()
