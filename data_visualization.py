import yaml
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import CRCDataset
from reduce_dim_features import *
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# %%
if __name__ == '__main__':
    
    root = config['dir']['root']
    dataset = CRCDataset(root, transform=None,
                         save_new_masks=False)

    radiomics = dataset.radiomic_features
    dataset.radiomic_features['class_name'].value_counts().plot(kind='bar', title='Class Distribution', )
    plt.xticks(rotation=45)
    plt.show()
    features = dataset.radiomic_features[dataset.radiomic_features.columns[4:]]
    labels = dataset.radiomic_features[dataset.radiomic_features.columns[:4]]
    selected_features_icc = icc_select_reproducible(features=features,
                                                    labels=labels,
                                                    comparison_type='node',
                                                    bin_widths=config['binWidths'])

#%%
    # TO DO: steps from README.md
#     print(f"Features shape: {features.shape}")
#     features = features.T.drop_duplicates().T
#     print(f"Features shape: {features.shape}")
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
#     lasso_cv = linear_model.LassoCV(cv=5,
#                                     # max_iter=10_000,
#                                     # n_jobs=-1,
#                                     random_state=config['seed'])
#     lasso_cv.fit(features_scaled, labels['class_label'])
#     best_alpha = lasso_cv.alpha_
#     print(f"Best alpha: {best_alpha}")
#     lasso_coefficients_cv = lasso_cv.coef_
#     selected_features_cv_names = features.columns[lasso_coefficients_cv != 0]

# #%%
#     selected_features_cv = features[selected_features_cv_names]
#     print("Selected Features by LassoCV:", selected_features_cv_names)
#     corr = selected_features_cv.corr()
#     ax = sns.clustermap(corr, linewidths=.5, figsize=(13,13))
#     _ = plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
#     plt.show()

# %%

# tbc
# reduce_dim(selected_features_cv, comparison_type='all') # 'colon', 'node', 'fat', 'all'
