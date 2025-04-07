import os
import re
import torch
import random
import numpy as np
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
                 num_patches_per_sample: int = 50,
                 mode = '3d'):  
                         
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
        self.cut_filtered_image_path = []
        # self.cut_filtered_bodyMask = []
        self.cut_mask_path = []
        self.transforms = transforms
        self.train_mode = False
        self.mode = mode

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
                elif '_body.nii.gz' in f:
                    self.cut_filtered_image_path.append(f)
                elif 'cut_filterMask.nii.gz' in f:
                    continue
                elif 'instance_mask.nii.gz' in f:
                    self.instance_masks_path.append(f)
                elif 'nii.gz' in f:
                    self.images_path.append(f)
                elif 'mapping.pkl' in f:
                    self.mapping_path.append(f)



    def __getitem__(self, idx):
        cut = True
        filtered = True
        if not cut:
            image_path = self.images_path[idx]
            mask_path = self.masks_path[idx]
        else:
            if filtered:
                image_path = self.cut_filtered_image_path[idx]
            else:
                image_path = self.cut_images_path[idx]
            mask_path = self.cut_mask_path[idx]

        instance_mask_path = self.instance_masks_path[idx]

        image = np.asarray(nib.load(image_path).dataobj)
        image = torch.from_numpy(image)
        mask = np.asarray(nib.load(mask_path).dataobj)
        mask = torch.from_numpy(mask)

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
        image = image.permute(2, 0, 1)   # D, H, W
        mask = mask.permute(2, 0, 1)

        if self.mode == '3d':

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

            return img_patch, mask_patch, image, mask, self.get_patient_id(idx).strip("'")
        
        elif self.mode == '2d':
            if (self.transforms is not None) and self.train_mode:
                data_to_transform = {"image": image, "mask": mask}
                transformed = self.transforms[0](data_to_transform)  # train_transforms
                image, mask = transformed["image"], transformed["mask"]
            elif (self.transforms is not None) and not self.train_mode:
                data_to_transform = {"image": img_patch, "mask": mask_patch}
                transformed = self.transforms[1](data_to_transform)  # val_transforms
                image, mask = transformed["image"], transformed["mask"]

            return torch.zeros(1), torch.zeros(1), image, mask, self.get_patient_id(idx).strip("'")



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


    def extract_patches(self, image, mask):
        """Extracts balanced patches from a 3D image and segmentation mask."""
        H, W, D = image.shape
        h_size, w_size, d_size = self.patch_size

        h_idxs = list(range(0, H - h_size + 1, self.stride))
        w_idxs = list(range(0, W - w_size + 1, self.stride))
        d_idxs = list(range(0, D - d_size + 1, self.stride))

        if h_idxs[-1] != H - h_size:
            h_idxs.append(H - h_size)
        if w_idxs[-1] != W - w_size:
            w_idxs.append(W - w_size)
        if d_idxs[-1] != D - d_size:
            d_idxs.append(D - d_size)

        patch_candidates = []

        for h in h_idxs:
            for w in w_idxs:
                for d in d_idxs:
                    img_patch = image[h:h+h_size, w:w+w_size, d:d+d_size]
                    mask_patch = mask[h:h+h_size, w:w+w_size, d:d+d_size]
                    if torch.mean((img_patch < 0.001).float()) < 0.3 or torch.any(mask_patch > 0):
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