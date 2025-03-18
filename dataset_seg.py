import os
import re
import torch
import random
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import logging


logger = logging.getLogger(__name__)


class CRCDataset_seg(Dataset):
    def __init__(self, root_dir: os.PathLike,
                 nii_dir: os.PathLike,
                 clinical_data_dir: os.PathLike,
                 config,
                 transforms=None,
                 patch_size: tuple = (64, 64, 64),
                 stride: int = 32,
                 num_patches_per_sample: int = 50):  
                         
        self.root = root_dir
        self.nii_dir = nii_dir
        self.clinical_data = clinical_data_dir

        self.patch_size = patch_size
        self.stride = stride
        self.num_patches_per_sample = num_patches_per_sample
        self.images_path = []
        self.masks_path = []
        self.instance_masks_path = []
        self.mapping_path = []
        
        self.cut_images_path = []
        self.cut_mask_path = []

        self.transforms = transforms
        self.train_mode = False

        nii_pth = os.path.join(self.root, self.nii_dir)

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
                elif 'labels_cut.nii.gz' in f:
                    self.cut_mask_path.append(f)
                elif '_cut.nii.gz' in f:
                    self.cut_images_path.append(f)
                elif 'instance_mask.nii.gz' in f:
                    self.instance_masks_path.append(f)
                elif 'nii.gz' in f:
                    self.images_path.append(f)
                elif 'mapping.pkl' in f:
                    self.mapping_path.append(f)


        clinical_data_dir = os.path.join(self.root, self.clinical_data)
        default_missing = pd._libs.parsers.STR_NA_VALUES
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

        for column, dtype in config['clinical_data_attributes'].items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        self.clinical_data = self.clinical_data.reset_index(drop=True)        
        self.clinical_data.rename(columns={'Nr pacjenta': 'patient_id'}, inplace=True)
        self._clean_tnm_clinical_data()


    def __getitem__(self, idx):
        cut = True
        if not cut:
            image_path = self.images_path[idx]
            mask_path = self.masks_path[idx]
        else:
            image_path = self.cut_images_path[idx]
            mask_path = self.cut_mask_path[idx]

        instance_mask_path = self.instance_masks_path[idx]

        image = np.asarray(nib.load(image_path).dataobj)
        image = torch.from_numpy(image)

        mask = np.asarray(nib.load(mask_path).dataobj)
        mask = torch.from_numpy(mask)

        instance_mask = np.asarray(nib.load(instance_mask_path).dataobj)
        instance_mask = torch.from_numpy(instance_mask)

        if len(image.shape) == 4:
            image = image[:, :, 0, :]
        if len(mask.shape) == 4:
            mask = mask[:, :, 0, :]
        if len(instance_mask.shape) == 4:
            instance_mask = instance_mask[:, :, 0, :]
        # instance_mask = instance_mask.permute(1, 0, 2) # need to permute to get correct bounding boxes
        window_center = 45
        window_width = 400
        image = self.window_and_normalize_ct(image,
                                             window_center=window_center,
                                             window_width=window_width)

        # # assign 0 values to mask > 1
        mask[mask == 2] = 0
        mask[mask == 3] = 0
        mask[mask == 4] = 0
        mask[mask == 5] = 0
        mask[mask == 6] = 0


        # temporary cuz model requires patch of shape (64 128 128)
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        patches = self.extract_patches(image, mask)
        num_to_select = min(8, len(patches))

        selected_patches = random.sample(patches, num_to_select)
        img_patch = torch.stack([p[0] for p in selected_patches])
        mask_patch = torch.stack([p[1] for p in selected_patches])

        
        if (self.transforms is not None) and self.train_mode:
            data_to_transform = {"image": img_patch, "mask": mask_patch}
            transformed_patches = self.transforms[0](data_to_transform)  # train_transforms
            img_patch, mask_patch = transformed_patches["image"], transformed_patches["mask"]
        elif (self.transforms is not None) and not self.train_mode:
            data_to_transform = {"image": img_patch, "mask": mask_patch}
            transformed_patches = self.transforms[1](data_to_transform)  # val_transforms
            img_patch, mask_patch = transformed_patches["image"], transformed_patches["mask"]

        return img_patch, mask_patch, image, mask


    def __len__(self):
    # todo
        return len(self.images_path)
    
    def set_mode(self, train_mode):
        self.train_mode = train_mode


    def get_patient_id(self, idx):
        patient_id = os.path.basename(self.images_path[idx]).split('_')[0].split(' ')[0]
        return ''.join(filter(str.isdigit, patient_id))
    
    
    def window_and_normalize_ct(self, ct_image, window_center=45, window_width=400):

        lower_bound = window_center - window_width / 2
        upper_bound = window_center + window_width / 2
        ct_windowed = np.clip(ct_image, lower_bound, upper_bound)
        normalized = (ct_windowed - lower_bound) / (upper_bound - lower_bound)
        
        return normalized


    def _clean_tnm_clinical_data(self):

        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].notna() & (self.clinical_data['TNM wg mnie'] != '')]
        
        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.startswith('T')]
        
        self.clinical_data.dropna(how='all', axis=0, inplace=True)
        self.clinical_data.dropna(subset=['TNM wg mnie'], inplace=True)

        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.contains(r'T', regex=True) &
            self.clinical_data['TNM wg mnie'].str.contains(r'N', regex=True)
        ]


        def make_lower_case(tnm_string):
            return ''.join([char.lower() if char in ['A', 'B', 'C', 'X'] else char for char in tnm_string])

        self.clinical_data['TNM wg mnie'] = self.clinical_data['TNM wg mnie'].apply(make_lower_case)
    
        self.clinical_data['wmT'] = self.clinical_data['TNM wg mnie'].str.extract(r'T([0-9]+[a-b]?|x|is)?')
        self.clinical_data['wmN'] = self.clinical_data['TNM wg mnie'].str.extract(r'N([0-9]+[a-c]?)?')
        self.clinical_data['wmM'] = self.clinical_data['TNM wg mnie'].str.extract(r'M([0-9]+)?')


    def extract_patches(self, image, mask):
        """Extracts balanced patches from a 3D image and segmentation mask."""
        H, W, D = image.shape
        h_size, w_size, d_size = self.patch_size

        h_idxs = list(range(0, H - h_size + 1, self.stride))
        w_idxs = list(range(0, W - w_size + 1, self.stride))
        d_idxs = list(range(0, D - d_size + 1, self.stride))

        patch_candidates = []

        for h in h_idxs:
            for w in w_idxs:
                for d in d_idxs:
                    img_patch = image[h:h+h_size, w:w+w_size, d:d+d_size]
                    mask_patch = mask[h:h+h_size, w:w+w_size, d:d+d_size]
                    if torch.mean((img_patch < 0.001).float()) < 0.3:
                        patch_candidates.append((img_patch, mask_patch))

        foreground_patches = [p for p in patch_candidates if torch.any(p[1] > 0)]
        background_patches = [p for p in patch_candidates if not torch.any(p[1] > 0)]
        
        min_samples = min(len(foreground_patches), len(background_patches))
        num_samples = min(min_samples, self.num_patches_per_sample // 2)

        if len(foreground_patches) == 0:
            num_samples = min(len(background_patches), self.num_patches_per_sample // 2)
            selected_patches = random.sample(background_patches, num_samples)
        else:
            selected_foreground = random.sample(foreground_patches, num_samples) 
            selected_background = random.sample(background_patches, num_samples)
            selected_patches = selected_foreground + selected_background

        return selected_patches