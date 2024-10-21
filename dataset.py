import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from connected_components import create_instance_level_mask
from extract_radiomics import get_radiomics

class CRCDataset(Dataset):
    def __init__(self, root, transform=None, save_new_masks=True):
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
        mapping = {"background": 0,
                   "colon_positive": 1,
                   "lymph_node_positive": 2,
                   "suspicious_fat": 3,
                   "colon_negative": 4,
                   "lymph_node_negative": 5,
                   "unsuspicious_fat": 6}
        reverse_mapping = {v: k for k, v in mapping.items()}
        self.radiomic_features['class_name'] = self.radiomic_features['class_label'].map(reverse_mapping)


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
        
        return img, mask, instance_mask, radiomic_features
