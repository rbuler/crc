import yaml
import utils
import torch
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from radiomics import setVerbosity
from dataset import CRCDataset
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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
    clinical_data = config['dir']['clinical_data']
    dataset = CRCDataset(root,
                         clinical_data=clinical_data,
                         config=config,
                         transform=None,
                         save_new_masks=False)
    
    selected_classes = ['lymph_node_positive', 'lymph_node_negative']
    
    df = dataset.radiomic_features[dataset.radiomic_features['class_name'].isin(selected_classes)]
    df['patient_id'] = df['patient_id'].astype(int)
    df = df.sort_values(by='patient_id').reset_index(drop=True)
    
    counts_per_patient = df.groupby('patient_id')['class_name'].value_counts()
    
    for patient_id, class_counts in counts_per_patient.groupby(level=0):
        counts_str = ", ".join([f"{class_name}: {count}" for class_name, count in class_counts.items()])
        for class_name, count in class_counts.items():
            dataset.clinical_data.loc[dataset.clinical_data['Nr pacjenta'] == patient_id, class_name[1]] = count
    
    dataset.clinical_data[
        ['lymph_node_positive', 'lymph_node_negative']] = dataset.clinical_data[
            ['lymph_node_positive', 'lymph_node_negative']].fillna(0).astype(int)
    dataset.update_clinical_data()

    # columns_to_select = ["Nr pacjenta", "wmN", "pN", "lymph_node_positive", "lymph_node_negative", "wmNlymph_node_positive_overnoding", "pNlymph_node_positive_overnoding",
    #                      "Liczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane",
    #                      "wmNLiczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane_overnoding"]
    columns_to_select = ["Nr pacjenta", "wmN"]
    subset = dataset.clinical_data[columns_to_select]
    subset.rename(columns={"Nr pacjenta": "patient_id"}, inplace=True)

    # select only patients that have already have images
    ids = []
    for i in range(len(dataset)):
        ids.append(int(dataset.get_patient_id(i)))
    subset = subset[subset['patient_id'].isin(ids)]

    # bad quality/invalid annotations
    temp_to_drop = [140, 139, 138, 136, 132, 129, 128, 123, 120, 115, 113, 110, 108, 107, 102, 101, 99, 98,
                    96, 88, 86, 75, 70, 61, 49, 48, 37, 26, 21, 8, 3, 2]
    subset = subset[~subset['patient_id'].isin(temp_to_drop)]




    # sorted by patient_id
    new_df = subset.merge(df, how='inner', on='patient_id')
    
    binary_labels = new_df[new_df.columns[3]]
    multi_labels = new_df[new_df.columns[1]]
    features = new_df[new_df.columns[5:]]
    # split using the same seed
    
    def generate_mil_bags(df, patient_col='patient_id',
                          feature_cols=None,
                          instance_label_col='class_label',
                          bag_label_col='bag_label'):
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in {patient_col, instance_label_col, bag_label_col}]
        
        bags = defaultdict(lambda: {'instances': [], 'instance_labels': [], 'bag_label': None})
        
        for _, row in df.iterrows():
            patient_id = row[patient_col]
            feature_vector = row[feature_cols].tolist()
            instance_label = row[instance_label_col]
            bag_label = row[bag_label_col]
            
            bags[patient_id]['instances'].append(feature_vector)
            bags[patient_id]['instance_labels'].append(instance_label)
            bags[patient_id]['bag_label'] = bag_label  # Ensures consistent bag label for each patient
        
        return dict(bags)

    bags = generate_mil_bags(new_df, patient_col='patient_id', feature_cols=features.columns, instance_label_col='class_name', bag_label_col='wmN')

# %%
