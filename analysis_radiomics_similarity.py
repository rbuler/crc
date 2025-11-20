# %%
import os
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def align_columns(dfs):
    # take intersection of columns
    cols = set(dfs[0].columns)
    for d in dfs[1:]:
        cols = cols.intersection(set(d.columns))
    cols = sorted(cols)
    return [d[cols].copy() for d in dfs], cols


def remove_low_variance(X_df, threshold=1e-5):
    sel = VarianceThreshold(threshold)
    sel.fit(X_df.values)
    mask = sel.get_support()
    return X_df.iloc[:, mask], mask


def fit_transform_pipeline(unhealthy_gt_df, unhealthy_pred_df, healthy_df, variance_threshold=1e-3, pca_components=None):
    # align columns
    dfs_aligned, cols = align_columns([unhealthy_gt_df, unhealthy_pred_df, healthy_df])
    ug, up, hp = dfs_aligned

    # optionally subsample if many samples (user may choose to do debug outside)

    # remove low variance features across combined set
    combined = pd.concat([ug, up, hp], axis=0)
    combined_reduced, mask = remove_low_variance(combined, threshold=variance_threshold)
    selected_cols = combined_reduced.columns

    ug_sel = ug[selected_cols].values
    up_sel = up[selected_cols].values
    hp_sel = hp[selected_cols].values

    # scale
    scaler = StandardScaler()
    scaler.fit(np.vstack([ug_sel, up_sel, hp_sel]))
    ug_scaled = scaler.transform(ug_sel)
    up_scaled = scaler.transform(up_sel)
    hp_scaled = scaler.transform(hp_sel)

    # PCA (optional) before UMAP to speed up
    if pca_components is not None and pca_components > 0 and pca_components < ug_scaled.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        pca.fit(np.vstack([ug_scaled, up_scaled, hp_scaled]))
        ug_proj = pca.transform(ug_scaled)
        up_proj = pca.transform(up_scaled)
        hp_proj = pca.transform(hp_scaled)
    else:
        ug_proj, up_proj, hp_proj = ug_scaled, up_scaled, hp_scaled

    return ug_proj, up_proj, hp_proj, selected_cols


def plot_umap(ug_proj, up_proj, hp_proj, outpath, title=None, random_state=42, n_neighbors=15, min_dist=0.05, n_components=2):
    print(ug_proj.shape, up_proj.shape, hp_proj.shape)
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, metric='cosine', n_components=n_components)
    all_data = np.vstack([ug_proj, up_proj, hp_proj])
    emb = reducer.fit_transform(all_data)
    n1 = ug_proj.shape[0]
    n2 = up_proj.shape[0]
    n3 = hp_proj.shape[0]
    print(f'UMAP embedding shape: {emb.shape}, splits: {n1}, {n2}, {n3}')
    emb_ug = emb[:n1]
    emb_up = emb[n1:n1+n2]
    emb_hp = emb[n1+n2:]

    if n_components == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=emb_hp[:, 0], y=emb_hp[:, 1], label='healthy_pred', alpha=0.8, s=50,
                        color='#2f3451', edgecolor='k', linewidth=0.4, marker='X')
        sns.scatterplot(x=emb_up[:, 0], y=emb_up[:, 1], label='unhealthy_pred', alpha=0.9, s=60,
                        color='#df9da0', edgecolor='k', linewidth=0.4, marker='o')
        sns.scatterplot(x=emb_ug[:, 0], y=emb_ug[:, 1], label='unhealthy_gt', alpha=0.9, s=60,
                        color='#6aa5dd', edgecolor='k', linewidth=0.4, marker='s')
        plt.title(title or 'UMAP projection')
        plt.xlabel('Embedding 1')
        plt.ylabel('Embedding 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(emb_hp[:, 0], emb_hp[:, 1], emb_hp[:, 2], label='healthy_pred', alpha=0.8, s=30,
                   c='#2f3451', marker='X', edgecolors='k', linewidths=0.4)
        ax.scatter(emb_up[:, 0], emb_up[:, 1], emb_up[:, 2], label='unhealthy_pred', alpha=0.9, s=30,
                   c='#df9da0', marker='o', edgecolors='k', linewidths=0.4)
        ax.scatter(emb_ug[:, 0], emb_ug[:, 1], emb_ug[:, 2], label='unhealthy_gt', alpha=0.9, s=30,
                   c="#6aa5dd", marker='s', edgecolors='k', linewidths=0.4)
        ax.set_title(title or 'UMAP projection (3D)')
        ax.set_xlabel('Embedding 1')
        ax.set_ylabel('Embedding 2')
        ax.set_zlabel('Embedding 3')
        ax.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()



# %%
df_path = 'radiomics_umap_output/radiomics_features_extracted_11-19-02-35-16.csv'  # bin with 0.01
# df_path = 'radiomics_umap_output/radiomics_features_extracted_11-19-14-08-57.csv' # bin with 0.1
df = pd.read_csv(df_path)

outdir = 'radiomics_umap_output/radiomics_analysis'
variance_threshold = 1e-5
pca_components = None
random_seed = 42
os.makedirs(outdir, exist_ok=True)

df = df.dropna(subset=['region', 'fold']).reset_index(drop=True)
df = df[df['extraction_status'] == 'ok']

val_gt = df[df['region']=='val_gt']
val_pred = df[df['region']=='val_pred']
test_pred = df[df['region']=='test_fp']
test_pred_folds = {
    f: test_pred[test_pred['fold'] == f].reset_index(drop=True)
    for f in sorted(test_pred['fold'].unique())
}
val_gt = val_gt.dropna(axis=1).reset_index(drop=True)
val_pred = val_pred.dropna(axis=1).reset_index(drop=True)
for f in test_pred_folds:
    test_pred_folds[f] = test_pred_folds[f].dropna(axis=1).reset_index(drop=True)

val_gt = val_gt.select_dtypes(include=['float64'])
val_pred = val_pred.select_dtypes(include=['float64'])
for f in test_pred_folds:
    test_pred_folds[f] = test_pred_folds[f].select_dtypes(include=['float64'])

for i, hf in enumerate(test_pred_folds, start=1):
    print(f'Processing healthy file {i}/{len(test_pred_folds)}: {hf}')
    hp_df = test_pred_folds[hf]

    try:
        ug_proj, up_proj, hp_proj, selected_cols = fit_transform_pipeline(val_gt, val_pred, hp_df, variance_threshold=variance_threshold, pca_components=pca_components)
    except Exception as e:
        print('Error in pipeline for fold', i, e)
        continue
    umap_path = os.path.join(outdir, f'umap_fold_{i}.png')
    plot_umap(ug_proj, up_proj, hp_proj, umap_path, title=f'Fold {i} UMAP', n_neighbors=5, min_dist=0.99, n_components=2)

# %%