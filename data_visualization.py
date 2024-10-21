import yaml
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import CRCDataset
from reduce_dim_features import reduce_dim
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
    col = dataset.radiomic_features.pop("class_label")
    dataset.radiomic_features.insert(0, col.name, col)
    dataset.radiomic_features['class_label'].value_counts().plot(kind='bar', title='Class Distribution')

    features = dataset.radiomic_features.drop(columns=['patient_id', 'class_label', 'class_name', 'instance_label'])

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    # LassoCV automatically selects the best alpha
    lasso_cv = linear_model.LassoCV(cv=5, max_iter=100_000)

    lasso_cv.fit(features_scaled, dataset.radiomic_features['class_label'])

    # Get the optimal alpha
    best_alpha = lasso_cv.alpha_
    print(f"Best alpha: {best_alpha}")

    # Get the selected features
    lasso_coefficients_cv = lasso_cv.coef_
    selected_features_cv_names = features.columns[lasso_coefficients_cv != 0]
    selected_features_cv = features[selected_features_cv_names]
    for col in ['instance_label', 'class_label', 'class_name', 'patient_id']:
        selected_features_cv.insert(0, col, dataset.radiomic_features[col])

    print("Selected Features by LassoCV:", selected_features_cv_names)

    corr = selected_features_cv.corr()
    # ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    ax = sns.clustermap(corr, linewidths=.5, figsize=(13,13))
    _ = plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()


    # reduce_dim(radiomics, comparison_type='fat') # 'colon', 'node', 'fat', 'all'

# %%


reduce_dim(selected_features_cv, comparison_type='all') # 'colon', 'node', 'fat', 'all'
