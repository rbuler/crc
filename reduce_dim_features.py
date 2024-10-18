import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduce_dim(df, comparison_type):

    comparison_pairs = {
        'colon': [1, 4],
        'node': [2, 5],
        'fat': [3, 6], 
        'all': [1, 2, 3, 4, 5, 6]
    }
    
    if comparison_type not in comparison_pairs:
        raise ValueError("Invalid comparison type. Choose from 'fat', 'node', 'colon' and 'all'.")
    
    # Filter the dataset based on the comparison type
    class_labels = comparison_pairs[comparison_type]
    df = df[df['class_label'].isin(class_labels)]
    
    features = df.drop(columns=['class_label', 'patient_id'])
    labels = df['class_label']
    patient_ids = df['patient_id']

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 2D PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features_scaled)

    pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
    pca_df['class_label'] = labels
    pca_df['patient_id'] = patient_ids

    # 2D PCA Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class_label', palette='viridis', s=100)
    plt.title('PCA of Radiomic Features by Class Label', fontsize=16, weight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Class Label')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3D PCA
    pca_3d = PCA(n_components=3)
    pca_features_3d = pca_3d.fit_transform(features_scaled)

    pca_df_3d = pd.DataFrame(data=pca_features_3d, columns=['PC1', 'PC2', 'PC3'])
    pca_df_3d['class_label'] = labels
    pca_df_3d['patient_id'] = patient_ids

    # 3D PCA Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_df_3d['PC1'], pca_df_3d['PC2'], pca_df_3d['PC3'],
                         c=pca_df_3d['class_label'], cmap='viridis', alpha=0.8)

    legend1 = ax.legend(*scatter.legend_elements(), title="Class Label", loc='upper left')
    ax.add_artist(legend1)
    plt.title('3D PCA of Radiomic Features by Class Label', fontsize=16, weight='bold')
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)
    ax.set_box_aspect(None, zoom=0.85)
    plt.tight_layout()
    plt.show()

    # Explained Variance by Principal Components
    pca = PCA()
    pca.fit(features_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    components = range(1, len(explained_variance) + 1)
    pca_df = pd.DataFrame({
        'Principal Component': components,
        'Explained Variance (%)': explained_variance * 100,
        'Cumulative Variance (%)': cumulative_variance * 100,
    })

    # Print the first 20 features
    print(pca_df.head(20))

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

    # 2D t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_results = tsne.fit_transform(pca_features)

    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE-1', 'TSNE-2'])
    tsne_df['Class Label'] = labels

    # 2D t-SNE Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='TSNE-1', y='TSNE-2', hue='Class Label', palette='viridis', data=tsne_df, alpha=0.8)
    plt.title('t-SNE Visualization of Radiomic Features by Class Label', fontsize=16, weight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(loc='best', title='Class Label')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 3D t-SNE
    tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, random_state=42)
    tsne_results = tsne.fit_transform(features_scaled)

    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE-1', 'TSNE-2', 'TSNE-3'])
    tsne_df['Class Label'] = labels

    # 3D t-SNE Plot
    fig = plt.figure(figsize=(10, 7))
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
    plt.show()
