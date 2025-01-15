import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from connected_components import create_instance_level_mask
from extract_radiomics import get_radiomics

class CRCDataset(Dataset):
    def __init__(self, root, clinical_data, config, transform=None, save_new_masks=True):
        self.root = root
        self.images_path = []
        self.masks_path = []
        self.instance_masks_path = []
        self.mapping_path = []

        if save_new_masks:
            for root, dirs, files in os.walk(self.root, topdown=False):
                for name in files:
                    f = os.path.join(root, name)
                    if 'labels.nii.gz' in f:
                        create_instance_level_mask(f, save_dir=f, verbose=False)
    
        for root, dirs, files in os.walk(self.root, topdown=False):
            for name in files:
                f = os.path.join(root, name)
                if 'labels.nii.gz' in f:
                    self.masks_path.append(f)
                elif 'instance_mask.nii.gz' in f:
                    self.instance_masks_path.append(f)
                elif 'nii.gz' in f:
                    self.images_path.append(f)
                elif 'mapping.pkl' in f:
                    self.mapping_path.append(f)

        self.radiomic_features = get_radiomics(self.images_path,
                                                   self.masks_path,
                                                   self.instance_masks_path,
                                                   self.mapping_path)
        default_missing = pd._libs.parsers.STR_NA_VALUES
        self.clinical_data = pd.read_csv(
            clinical_data,
            delimiter=';',
            encoding='utf-8',
            index_col=False,
            na_filter=True,
            na_values=default_missing)
        

        self.clinical_data.columns = self.clinical_data.columns.str.strip()
        self.clinical_data = self.clinical_data[config['clinical_data_attributes'].keys()]
        self.clinical_data.dropna(subset=['Nr pacjenta'], inplace=True)


        for column, dtype in config['clinical_data_attributes'].items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)
        self.clinical_data = self.clinical_data.reset_index(drop=True)


        self._clean_tnm_data()
        category_order_N = {'0': 0, '1a': 1, '1b': 2, '2a': 3, '2b': 4, '1c': np.nan, 'nan': np.nan}
        category_order_T = {'0': 0, '1': 1, '2': 2, '3': 3, '4a': 4, '4b': 5, 'nan': np.nan}
        self._compute_overstaging_tnm('wmT', 'pT', category_order_T, 'overstaging_T')
        self._compute_overstaging_tnm('wmN', 'pN', category_order_N, 'overstaging_N')


        




        mapping = {"background": 0,
            "colon_positive": 1,
            "lymph_node_positive": 2,
            "suspicious_fat": 3,
            "colon_negative": 4,
            "lymph_node_negative": 5,
            "unsuspicious_fat": 6}
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        if isinstance(self.radiomic_features, pd.DataFrame):
            self.radiomic_features.insert(1, 'class_name', self.radiomic_features['class_label'].map(reverse_mapping))


    def __len__(self):
        # todo
        return len(self.images_path)
    

    def get_patient_id(self, idx):
        patient_id = os.path.basename(self.images_path[idx]).split('_')[0].split(' ')[0]
        return ''.join(filter(str.isdigit, patient_id))


    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        instance_mask_path = self.instance_masks_path[idx]
        radiomic_features = self.radiomic_features[self.radiomic_features['patient_id'] == self.get_patient_id(idx)]
        clinical_data = self.clinical_data[self.clinical_data['Nr pacjenta'] == int(self.get_patient_id(idx))]

        img = np.asarray(nib.load(image_path).dataobj)
        img = torch.from_numpy(img)

        mask = np.asarray(nib.load(mask_path).dataobj)
        mask = torch.from_numpy(mask)
        
        instance_mask = np.asarray(nib.load(instance_mask_path).dataobj)
        instance_mask = torch.from_numpy(instance_mask)
    
        if len(img.shape) == 3:
            _, _, _ = img.shape
        elif len(img.shape) == 4:
            img = img[:, :, 0, :]
            mask = mask[:, :, 0, :]
            instance_mask = instance_mask[:, :, 0, :]
        img = img.permute(2, 1, 0)  # slice, height, width
        mask = mask.permute(2, 1, 0)
        instance_mask = instance_mask.permute(2, 1, 0)

        img = (img - img.min()) / (img.max() - img.min())
        
        return img, mask, instance_mask, radiomic_features, clinical_data


    def _clean_tnm_data(self):

        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].notna() & (self.clinical_data['TNM wg mnie'] != '')]
        
        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.startswith('T')]
        
        self.clinical_data.dropna(how='all', axis=0, inplace=True)
        self.clinical_data.dropna(subset=['TNM wg mnie'], inplace=True)

        self.clinical_data = self.clinical_data[
            ~self.clinical_data['TNM wg mnie'].str.contains('/')
        ]

        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.contains(r'T', regex=True) &
            self.clinical_data['TNM wg mnie'].str.contains(r'N', regex=True)
        ]


        def make_lower_case(tnm_string):
            return ''.join([char.lower() if char in ['A', 'B', 'C', 'X'] else char for char in tnm_string])

        self.clinical_data['TNM wg mnie'] = self.clinical_data['TNM wg mnie'].apply(make_lower_case)
    
        self.clinical_data['wmT'] = self.clinical_data['TNM wg mnie'].str.extract(r'T(\d+[a-bA-B]?)')
        self.clinical_data['wmN'] = self.clinical_data['TNM wg mnie'].str.extract(r'N(\d+[a-cA-C]?)')
    

    def _compute_overstaging_tnm(self, wm_column: str, p_column: str, category_order: dict, result_column: str):
        wm_numeric = self.clinical_data[wm_column].map(category_order)
        p_numeric = self.clinical_data[p_column].map(category_order)
        comparison = wm_numeric - p_numeric
        result = comparison.apply(lambda x: 
            -1 if x < 0 else 
            (1 if x > 0 else 
            (0 if pd.notna(x) else np.nan)))
        self.clinical_data[result_column] = result


    def fill_num_nodes(
        self,
        col: str,
        node_count_N: dict,
        result_column: str
    ):
        node_count = self.clinical_data[col].map(node_count_N)
        
        self.clinical_data[result_column] = node_count

        def calculate_overnoding(row):
            range = row[result_column]
            positive = row['lymph_node_positive']

            _min, _max = (range if isinstance(range, list) else [range, range])


            if positive < _min:
                return -1
            elif _min <= positive <= _max:
                return 0
            else:
                return 1
        name = col + '_overnoding'
        self.clinical_data[name] = self.clinical_data.apply(calculate_overnoding, axis=1)


    def update_clinical_data(self):
        node_count_N = {'0': 0, '1a': 1, '1b': [2, 3], '1c': 0, '2a': [4, 6], '2b': [7, 99], 'nan': np.nan}
        self.fill_num_nodes('wmN', node_count_N, 'wmN_node_count')
        self.fill_num_nodes('pN', node_count_N, 'pN_node_count')
        

