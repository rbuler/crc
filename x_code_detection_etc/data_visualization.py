# %%
import yaml
import utils
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import CRCDataset
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from x_code_detection_etc.reduce_dim_features import icc_select_reproducible, plot_reduced_dim, select_best_from_clusters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

np.random.seed(config['seed'])

if __name__ == '__main__':
    
    dataset = CRCDataset(root_dir=config['dir']['root'],
                         clinical_data_dir=config['dir']['clinical_data'],
                         nii_dir=config['dir']['nii_images'],
                         dcm_dir=config['dir']['dcm_images'],
                         config=config,
                         transform=None,
                         save_new_masks=False)
    # ------------------------
    
    selected_classes = ['lymph_node_positive', 'lymph_node_negative']
    df = dataset.radiomic_features[dataset.radiomic_features['class_name'].isin(selected_classes)]
    df['patient_id'] = df['patient_id'].astype(int)
    df = df.sort_values(by='patient_id').reset_index(drop=True)
    
    counts_per_patient = df.groupby('patient_id')['class_name'].value_counts()

    # Iterate through the grouped data and print the results
    for patient_id, class_counts in counts_per_patient.groupby(level=0):
        counts_str = ", ".join([f"{class_name}: {count}" for class_name, count in class_counts.items()])
        for class_name, count in class_counts.items():
            dataset.clinical_data.loc[dataset.clinical_data['patient_id'] == patient_id, class_name[1]] = count
    dataset.clinical_data[
        ['lymph_node_positive', 'lymph_node_negative']] = dataset.clinical_data[
            ['lymph_node_positive', 'lymph_node_negative']].fillna(0).astype(int)
    dataset.update_clinical_data()

    columns_to_select = ["patient_id", "wmN", "pN", "lymph_node_positive", "lymph_node_negative", "wmNlymph_node_positive_overnoding", "pNlymph_node_positive_overnoding",
                         "Liczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane",
                         "wmNLiczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane_overnoding"]
    subset = dataset.clinical_data[columns_to_select]

    # select only patients that have already have images
    ids = []
    for i in range(len(dataset)):
        ids.append(int(dataset.get_patient_id(i)))
    subset = subset[subset['patient_id'].isin(ids)]

    #TODO drop patients with no target (rad+clinical)
    # ------------------------
    # %%
    comparison_pairs = {
        'colon': [1, 4],
        'node': [2, 5],
        'fat': [3, 6], 
        'all': [1, 2, 3, 4, 5, 6]
    }
    
    radiomics = dataset.radiomic_features
    dataset.radiomic_features['class_name'].value_counts().plot(kind='bar', title='Class Distribution', )
    plt.xticks(rotation=45)
    plt.show()
    features = dataset.radiomic_features[dataset.radiomic_features.columns[4:]]
    labels = dataset.radiomic_features[dataset.radiomic_features.columns[:4]]
    comparison_type = 'node'
    features_for_comparison = features[labels['class_label'].isin(comparison_pairs[comparison_type])]
    labels_for_comparison = labels[labels['class_label'].isin(comparison_pairs[comparison_type])].reset_index(drop=True)
    
    # Step 1: Exclusion of nonreproducible features (in case of multiple bin widths)
    
    if config['radiomics']['multiple_binWidth']['if_multi']:
        bin_widths = config['radiomics']['multiple_binWidth']['binWidths']
        selected_features_icc = icc_select_reproducible(features=features,
                                                        labels=labels,
                                                        threshold=0.75,
                                                        bin_widths=bin_widths)
        reproducible_features = features_for_comparison[selected_features_icc]
    else:
        reproducible_features = features_for_comparison
    logger.info(f"{reproducible_features.shape=}")
    
    # Step 2: Selection of the most relevant variables for the respective task
    
    # some columns might be dropped due to having uninformative values
    features_prior_to_selection = reproducible_features.loc[:, reproducible_features.nunique() > 1]
    logger.info(f"Removing columns with uninformative values (single unique value):\n\t{features_prior_to_selection.shape=}")
    # some columns might be duplicates and should be removed to reduce dim and multicollinearity
    features_prior_to_selection = features_prior_to_selection.T.drop_duplicates().T
    logger.info(f"Removing duplicate columns:\n\t{features_prior_to_selection.shape=}")
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_prior_to_selection)
    scaled_df = pd.DataFrame(features_scaled, columns=features_prior_to_selection.columns)

    # LassoCV for feature selection
    lasso_cv = linear_model.LassoCV(cv=5,
                                    random_state=config['seed'])
    lasso_cv.fit(features_scaled, labels_for_comparison['class_label'])
    best_alpha = lasso_cv.alpha_
    logger.info(f"Best alpha: {best_alpha}")
    lasso_coefficients_cv = lasso_cv.coef_
    selected_features_cv_names = features_prior_to_selection.columns[lasso_coefficients_cv != 0]
    selected_features_df = scaled_df[selected_features_cv_names]

    # Step 3: Building correlation clusters

    corr = selected_features_df.corr()
    ax = sns.clustermap(corr, linewidths=.5, figsize=(13,13))
    _ = plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

    # Step 4: Data visualization - once the data dim has been reduced.
    plot_reduced_dim(selected_features_df, labels_for_comparison['class_label'])
    
    
    # Step 5: Selection of most representative features for each cluster
    df = select_best_from_clusters(features=selected_features_df,
                                   n_clusters=10,
                                   num_features=10,
                                   random_state=config['seed'])
    plot_reduced_dim(df, labels_for_comparison['class_label'])

    # Step 5.5: Add clinical features to the selected features
    # TODO
    
    # Step 6: Model fitting with remaining features (usually 3-10 fts)
    # TODO

# %%