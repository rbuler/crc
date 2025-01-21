import torch
import numpy as np
import yaml
import logging
import utils
from radiomics import setVerbosity
from dataset import CRCDataset
from reduce_dim_features import plot_reduced_dim
# from classifiers import Classifier
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# from utils import view_slices

logger = logging.getLogger(__name__)
logger_radiomics = logging.getLogger("radiomics")
setVerbosity(30)

logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# %%
if __name__ == '__main__':
    
    root = config['dir']['root']
    dataset = CRCDataset(root, clinical_data=config['dir']['clinical_data'], transform=None,
                         save_new_masks=False)


    dataset.clinical_data = dataset.clinical_data[config['clinical_data_attributes'].keys()]
    dataset.clinical_data.dropna(subset=['Nr pacjenta'], inplace=True)

    for column, dtype in config['clinical_data_attributes'].items():
        dataset.clinical_data[column] = dataset.clinical_data[column].astype(dtype)
    
    dataset.clinical_data = dataset.clinical_data.reset_index(drop=True)


    
    ## x = reduce_dim  TODO add return value

    comparison_pairs = {
        'colon': [1, 4],
        'node': [2, 5],
        'fat': [3, 6], 
        'all': [1, 2, 3, 4, 5, 6]
    }
    comparison_type = 'node'

    radiomic_features = dataset.radiomic_features[dataset.radiomic_features.columns[4:]]
    labels = dataset.radiomic_features[dataset.radiomic_features.columns[:4]]
    labels_for_comparison = labels[labels['class_label'].isin(comparison_pairs[comparison_type])].reset_index(drop=True)
    
    plot_reduced_dim(radiomic_features, labels_for_comparison['class_name'])
    
    ## Example classification using XGBoost
    # data = radiomic_features
    # filtered_data = data[data['class_name'].isin(['lymph_node_positive', 'lymph_node_negative'])]
    # filtered_data['class_name'] = filtered_data['class_name'].map({'lymph_node_positive': 1, 'lymph_node_negative': 0})
    # y = filtered_data['class_name']
    # X = filtered_data.drop(columns=['class_label', 'patient_id', 'instance_label', 'class_name'])
    # patient_ids = filtered_data['patient_id']
    # X = X.values
    # y = y.values
    # classifier = Classifier(X, y, patient_ids.values, classifier_name='XGBoost')
    # classifier.train_classifier()
    
    # try:
    #     view_slices(img, mask, title='3D Mask Slices')
    # except Exception as e:
    #     print(e)
# %%
