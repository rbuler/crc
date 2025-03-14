import os
import re
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict
from torch.utils.data import Dataset
from connected_components import create_instance_level_mask
from utils import get_3d_bounding_boxes
from extract_radiomics import get_radiomics
import logging
logger = logging.getLogger(__name__)

class CRCDataset(Dataset):
    def __init__(self, root_dir: os.PathLike,
                 clinical_data_dir: os.PathLike,
                 nii_dir: os.PathLike,
                 dcm_dir: os.PathLike,
                 config, transform=None, save_new_masks=False):
        
        self.root = root_dir
        self.clinical_data = clinical_data_dir
        self.nii_dir = nii_dir
        self.dcm_dir = dcm_dir

        self.nii_images_path = []
        # self.dcm_images_path = []
        self.masks_path = []
        self.instance_masks_path = []
        self.mapping_path = []

        nii_pth = os.path.join(self.root, self.nii_dir)
        if save_new_masks:
            for root, dirs, files in os.walk(nii_pth, topdown=False):
                for name in files:
                    f = os.path.join(root, name)
                    if 'labels.nii.gz' in f:
                        create_instance_level_mask(f, save_dir=f, verbose=True)

        pattern = re.compile(r'^\d+a$') # take only those ##a
        for root, dirs, files in os.walk(nii_pth, topdown=False):
            for name in files:
                f = os.path.join(root, name)
                folder_name = f.split('/')[-2]
                if not pattern.match(folder_name):
                    logger.info(f"Skipping {folder_name}. Pattern does not match. {pattern}")
                    continue
                if 'labels.nii.gz' in f:
                    self.masks_path.append(f)
                elif 'instance_mask.nii.gz' in f:
                    self.instance_masks_path.append(f)
                elif 'nii.gz' in f:
                    self.nii_images_path.append(f)
                elif 'mapping.pkl' in f:
                    self.mapping_path.append(f)

        dcm_pth = os.path.join(self.root, self.dcm_dir)
        self.dcm_images_path = self._group_dicom_folders(dcm_pth)
        # TODO: Add support for dicom images
        # 3 phases: arterial, portal venous, non-contrast


        self.radiomic_features = get_radiomics(self.nii_images_path,
                                                   self.masks_path,
                                                   self.instance_masks_path,
                                                   self.mapping_path)
        default_missing = pd._libs.parsers.STR_NA_VALUES
        clinical_data_dir = os.path.join(self.root, self.clinical_data)
        self.clinical_data = pd.read_csv(
            clinical_data_dir,
            delimiter=';',
            encoding='utf-8',
            index_col=False,
            na_filter=True,
            na_values=default_missing)
        

        self.clinical_data.columns = self.clinical_data.columns.str.strip()
        self.clinical_data = self.clinical_data[config['clinical_data_attributes'].keys()]
        
        self.clinical_data.dropna(subset=['Nr pacjenta'], inplace=True)
        # self.clinical_data.dropna(subset=['Liczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane'], inplace=True)
        
        for column, dtype in config['clinical_data_attributes'].items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)
        self.clinical_data = self.clinical_data.reset_index(drop=True)
        
        self.clinical_data.rename(columns={'Nr pacjenta': 'patient_id'}, inplace=True)
        
        self._clean_tnm_clinical_data()

        category_order_N = {'0': 0, '1a': 1, '1b': 2, '2a': 3, '2b': 4, '1c': np.nan, 'nan': np.nan}
        self._compute_overstaging_tnm('wmN', 'pN', category_order_N, 'N_wm_p_overstaging')
        
        
        ### TODO - think of 'x' and 'is' in the future
        #         
        # category_order_T = {'0': 0, '1': 1, '2': 2, '3': 3, '4a': 4, '4b': 5, 'nan': np.nan}
        # self._compute_overstaging_tnm('wmT', 'pT', category_order_T, 'overstaging_T')
        #
        #
        
        
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
        return len(self.nii_images_path)
    

    def get_patient_id(self, idx):
        patient_id = os.path.basename(self.nii_images_path[idx]).split('_')[0].split(' ')[0]
        return ''.join(filter(str.isdigit, patient_id))


    def __getitem__(self, idx):
        image_path = self.nii_images_path[idx]
        mask_path = self.masks_path[idx]
        mapping_path = self.mapping_path[idx]
        instance_mask_path = self.instance_masks_path[idx]
        radiomic_features = self.radiomic_features[self.radiomic_features['patient_id'] == self.get_patient_id(idx)]
        clinical_data = self.clinical_data[self.clinical_data['patient_id'] == int(self.get_patient_id(idx))]

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
        
        bboxes = get_3d_bounding_boxes(instance_mask, mapping_path)

        return img, bboxes, mask, instance_mask, radiomic_features, clinical_data


    def _clean_tnm_clinical_data(self):

        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].notna() & (self.clinical_data['TNM wg mnie'] != '')]
        
        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.startswith('T')]
        
        self.clinical_data.dropna(how='all', axis=0, inplace=True)
        self.clinical_data.dropna(subset=['TNM wg mnie'], inplace=True)



        # 
        # ONLY PART OF TNM BEFORE '/' IS USED
        # CONSIDER CHANGING THIS IN THE FUTURE 
        #        
        self.clinical_data['TNM wg mnie'] = self.clinical_data[
            'TNM wg mnie'].str.replace(r'^.*/', '', regex=True)

        # self.clinical_data = self.clinical_data[
        #     ~self.clinical_data['TNM wg mnie'].str.contains('/')
        # ]



        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.contains(r'T', regex=True) &
            self.clinical_data['TNM wg mnie'].str.contains(r'N', regex=True)
        ]


        def make_lower_case(tnm_string):
            return ''.join([char.lower() if char in ['A', 'B', 'C', 'X'] else char for char in tnm_string])

        self.clinical_data['TNM wg mnie'] = self.clinical_data['TNM wg mnie'].apply(make_lower_case)
    
        self.clinical_data['wmT'] = self.clinical_data['TNM wg mnie'].str.extract(r'T([0-4]+[a-b]?|x|is)?')
        self.clinical_data['wmN'] = self.clinical_data['TNM wg mnie'].str.extract(r'N([0-3]+[a-c]?)?')
        self.clinical_data['wmM'] = self.clinical_data['TNM wg mnie'].str.extract(r'M([0-1]+)?')


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
        range_column: str,
        value_column: str
    ):
        node_count = self.clinical_data[col].map(node_count_N)
        
        self.clinical_data[range_column] = node_count

        def calculate_overnoding(row):
            range = row[range_column]
            positive = row[value_column]

            if isinstance(positive, str):
                numbers = re.findall(r'\d+', positive)
                positive = sum(map(int, numbers))

            elif not isinstance(positive, int):
                positive = int(positive)

            _min, _max = (range if isinstance(range, list) else [range, range])

            if positive < _min:
                return -1
            elif _min <= positive <= _max:
                return 0
            else:
                return 1
        name = col + value_column + '_overnoding'

        self.clinical_data[name] = self.clinical_data.apply(calculate_overnoding, axis=1)


    def update_clinical_data(self):
        node_count_N = {'0': 0, '1a': 1, '1b': [2, 3], '1c': 0, '2a': [4, 6], '2b': [7, 99], 'nan': np.nan}
        self.fill_num_nodes('wmN', node_count_N, 'wmN_node_count', 'lymph_node_positive')
        self.fill_num_nodes('pN', node_count_N, 'pN_node_count', 'lymph_node_positive')
        
        #
        # TODO - change column name so it is shorter
        #
        #

        self.fill_num_nodes('wmN', node_count_N, 'wmN_node_count', 'Liczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane')
        
        
    def _group_dicom_folders(self, dicom_folders_path):
        grouped = defaultdict(dict)
        pattern = re.compile(r'(\d+)([bc]?)')

        for folder in dicom_folders_path:
            match = pattern.fullmatch(folder)
            if match:
                base_num = int(match.group(1))
                suffix = match.group(2)

                if suffix == 'b':
                    grouped[base_num]['b'] = folder
                elif suffix == 'c':
                    grouped[base_num]['c'] = folder
                else:
                    grouped[base_num]['a'] = folder
        return [dict(a=group.get('a', ''), b=group.get('b', ''), c=group.get('c', '')) for group in grouped.values()]
