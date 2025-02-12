import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from typing import List
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
warnings.filterwarnings("ignore")


def plot_reduced_dim(features, labels):

    plot_pca(features, labels, n_components=2)
    plot_pca(features, labels, n_components=3)
    plot_tsne(features, labels, n_components=2)
    plot_tsne(features, labels, n_components=3)


def icc_select_reproducible(features: pd.DataFrame,
                            labels: pd.DataFrame,
                            threshold: float,
                            bin_widths: List[int]) -> pd.Series:
    """
    Selects reproducible features based on Intraclass Correlation Coefficient (ICC).
    
    Args:
        features (pd.DataFrame): DataFrame containing feature data.
        labels (pd.DataFrame): DataFrame containing class labels.
        threshold (float): ICC threshold to consider a feature as redundant (from 0. to 1.).
        comparison_type (str): Type of comparison to perform. Must be one of 'colon', 'node', 'fat', or 'all'.
        bin_widths (List[int]): List of integer bin widths to consider for ICC calculation.
        
    Returns:
        pd.Series: Series containing names of redundant features with high ICC values.
        
    Raises:
        ValueError: If an invalid comparison type is provided.
        
    Notes:
        The function filters features based on their reproducibility across different bin widths.
        Features with ICC values greater than the specified threshold (e.g. 0.75) are considered reproducible.
    """

    assert len(bin_widths) == 2, "Only two bin widths are supported yet."
    
    data = features.copy()
    print("Data shape:", data.shape)
    print("Data shape for comparison:", data.shape)
    feature_names = list(set(col.split('_binWidth')[0] for col in data.columns if 'binWidth' in col))
    data['subject'] = range(len(data))
    # Function to calculate ICC for each feature across bin widths
    icc_results = []
    for feature in feature_names:
        feature_data = data[['subject'] + [f"{feature}_binWidth{bin_width}" for bin_width in bin_widths].copy()]
        feature_data = feature_data.melt(id_vars=['subject'], var_name='bin_width', value_name='value')
        icc = pg.intraclass_corr(data=feature_data, targets='subject', raters='bin_width', ratings='value')
        icc_value = icc[icc['Type'] == 'ICC2']['ICC'].values[0]
        # reproducible features does not differ significantly across bin widths
        # so the first bin width is selected by default, as other bin width is redundant
        # Note it works only for two bin widths
        icc_results.append((f"{feature}_binWidth{bin_widths[0]}", icc_value))
    icc_df = pd.DataFrame(icc_results, columns=['Feature', 'ICC'])
    reproducible_features = icc_df[icc_df['ICC'] > threshold]['Feature']
    print(f"Number of reproducible features (high ICC): {len(reproducible_features.tolist())}")

    return reproducible_features


def select_best_from_clusters(features: pd.DataFrame,
                              n_clusters: int,
                              num_features: int,
                              random_state: int) -> pd.Series:
    # Apply clustering (e.g., KMeans)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(features)
    # Select most representative features for each cluster using SelectKBest
    # Use clusters as target labels for feature selection
    selector = SelectKBest(score_func=f_classif, k=num_features)  # Select top k features
    selector.fit(features, clusters)
    # Get mask of selected features
    selected_features_mask = selector.get_support()
    selected_features = features.columns[selected_features_mask]
    # Reduce dimensions by keeping only selected features
    reduced_df = features[selected_features]
    return reduced_df

def plot_tsne(features, labels, n_components=2):
    # 2D t-SNE
    tsne = TSNE(n_components=n_components, perplexity=30, learning_rate=200, random_state=42)
    tsne_results = tsne.fit_transform(features)

    tsne_df = pd.DataFrame(tsne_results, columns=[f"TSNE-{i}" for i in range(1, n_components + 1)])
    tsne_df['Class Label'] = labels

    fig = plt.figure(figsize=(10, 7))
    if n_components == 2:
        sns.scatterplot(x='TSNE-1', y='TSNE-2', hue='Class Label', data=tsne_df, alpha=0.8)
        plt.title('t-SNE Visualization of Radiomic Features by Class Label', fontsize=16, weight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(loc='best', title='Class Label')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(tsne_df['TSNE-1'], tsne_df['TSNE-2'], tsne_df['TSNE-3'],
                             c=tsne_df['Class Label'], cmap='viridis', alpha=0.8)
        legend1 = ax.legend(*scatter.legend_elements(), title="Class Label", loc='upper right')
        ax.add_artist(legend1)
        plt.title('3D t-SNE Visualization of Radiomic Features by Class Label', fontsize=16, weight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_zlabel('t-SNE Dimension 3', fontsize=12)
        plt.tight_layout()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return


def plot_pca(features, labels, n_components=2):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)

    pca_df = pd.DataFrame(data=pca_features, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    pca_df['class_label'] = labels

    fig = plt.figure(figsize=(10, 7))
    if n_components == 2:
        ax = fig.add_subplot(111)
        sns.scatterplot(ax=ax, data=pca_df, x='PC1', y='PC2', hue='class_label', s=100)
        plt.title('2D PCA of Radiomic Features by Class Label', fontsize=16, weight='bold')
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(title='Class Label')

    elif n_components == 3:  
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
                             c=pca_df['class_label'], cmap='viridis', alpha=0.8)
        legend = ax.legend(*scatter.legend_elements(), title="Class Label", loc='upper left')
        ax.add_artist(legend)
        plt.title('3D PCA of Radiomic Features by Class Label', fontsize=16, weight='bold')
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.set_zlabel('Principal Component 3', fontsize=12)
        ax.set_box_aspect(None, zoom=0.85)
    plt.tight_layout()
    plt.show()
    
    # Explained Variance by Principal Components
    pca = PCA()
    pca.fit(features)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    components = range(1, len(explained_variance) + 1)
    pca_df = pd.DataFrame({
        'Principal Component': components,
        'Explained Variance (%)': explained_variance * 100,
        'Cumulative Variance (%)': cumulative_variance * 100,
    })

    # Bar plot for Explained Variance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Principal Component', y='Explained Variance (%)', data=pca_df)
    plt.title('Explained Variance by Principal Components', fontsize=16, weight='bold')
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Explained Variance (%)', fontsize=12)
    plt.xticks(ticks=components[::10], labels=components[::10])  # Show every 10th component
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Line plot for Cumulative Variance
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Principal Component', y='Cumulative Variance (%)', data=pca_df, marker='o', color='b')
    plt.title('Cumulative Variance by Principal Components', fontsize=16, weight='bold')
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Cumulative Variance (%)', fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    return