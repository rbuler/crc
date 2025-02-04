import yaml
import utils
import torch
import torch.nn as nn
import logging
import numpy as np
from dataset import CRCDataset
from utils import generate_mil_bags, summarize_bags
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger_radiomics = logging.getLogger("radiomics")
logging.basicConfig(level=logging.ERROR)

class MILNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MILNetwork, self).__init__()
        
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        
        self.aggregation = lambda x: torch.max(x, dim=1)[0]
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, bag):
        b, i, f = bag.size()
        bag = bag.view(b * i, f)
        x = self.instance_encoder(bag)
        x = x.view(b, i, -1)
        x = self.aggregation(x)
        x = self.classifier(x)
        return x.squeeze(1)


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
    # bad quality/invalid annotations
    temp_to_drop = [140, 139, 138, 136, 132, 129, 128, 123, 120, 115, 113, 110, 108, 107, 102, 101, 99, 98,
                    96, 88, 86, 75, 70, 61, 49, 48, 37, 26, 21, 8, 3, 2]
    
    ids = list(set(ids) - set(temp_to_drop))
    subset = subset[subset['patient_id'].isin(ids)]

    # sorted by patient_id
    new_df = subset.merge(df, how='inner', on='patient_id')
    
    np.random.seed(1)
    ids = np.unique(new_df['patient_id'])
    train_ids = np.random.choice(ids, int(0.7 * len(ids)), replace=False)
    remaining_ids = np.setdiff1d(ids, train_ids)
    valid_ids = np.random.choice(remaining_ids, int(0.5 * len(remaining_ids)), replace=False)
    test_ids = np.setdiff1d(remaining_ids, valid_ids)


    binary_labels = new_df[new_df.columns[3]]
    multi_labels = new_df[new_df.columns[1]]
    features = new_df[new_df.columns[5:]]
    features = torch.tensor(features.values, dtype=torch.float32)
    # normalize features
    # TODO 
    # normalize feature using only training set statistics
    features = (features - features.mean()) / features.std()
    #
    #

    bags = generate_mil_bags(new_df, patient_col='patient_id', features=features, instance_label_col='class_name', bag_label_col='wmN')

    train_bags = {k: v for k, v in bags.items() if k in train_ids}
    valid_bags = {k: v for k, v in bags.items() if k in valid_ids}
    test_bags = {k: v for k, v in bags.items() if k in test_ids}  

    train_positive, train_negative = summarize_bags(train_bags)
    valid_positive, valid_negative = summarize_bags(valid_bags)
    test_positive, test_negative = summarize_bags(test_bags)
    
    def collate_fn(batch):
        return batch

    train_loader = DataLoader(list(train_bags.values()), batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(list(valid_bags.values()), batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(list(test_bags.values()), batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"Training set: {train_positive} positive bags, {train_negative} negative bags")
    print(f"Validation set: {valid_positive} positive bags, {valid_negative} negative bags")
    print(f"Test set: {test_positive} positive bags, {test_negative} negative bags")

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(valid_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    model = MILNetwork(input_dim=features.size(1), hidden_dim=64)
    for batch in train_loader:
        for bag in batch:
            instances = torch.stack(bag['instances']).unsqueeze(0)  # Add batch dimension
            bag_label = bag['bag_label']
            output = model(instances)
            print(output)
# %%
