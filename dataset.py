import os
import pickle
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from connected_components import create_instance_level_mask

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


    def __len__(self):
        # todo
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        instance_mask_path = self.instance_masks_path[idx]
        mapping_path = self.mapping_path[idx]

        img = np.asarray(nib.load(image_path).dataobj)
        img = torch.from_numpy(img)

        mask = np.asarray(nib.load(mask_path).dataobj)
        mask = torch.from_numpy(mask)
        
        instance_mask = np.asarray(nib.load(instance_mask_path).dataobj)
        instance_mask = torch.from_numpy(instance_mask)
        
        mapping = {}
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)



        if len(img.shape) == 3:
            _, _, _ = img.shape
        elif len(img.shape) == 4:
            img = img[:, :, 0, :]
            mask = mask[:, :, 0, :]
            instance_mask = instance_mask[:, :, 0, :]
        img = img.permute(2, 1, 0)  # slice, height, width
        mask = mask.permute(2, 1, 0)
        instance_mask = instance_mask.permute(2, 1, 0)

        # Normalize image
        img = (img - img.min()) / (img.max() - img.min())

        instance_to_class = {}  # dict with keys as instance indices and values as (instance_label, class_label) tuples
        instance_counter = 0
        for info in mapping.values():
            if info['class_label'] != 0:  # Skip background
                for instance_label in info['instance_labels']:
                    instance_to_class[instance_counter] = (instance_label, info['class_label'])
                    instance_counter += 1

        mapped_masks = np.zeros((len(instance_to_class), *instance_mask.shape), dtype=int)

        for instance_idx, (instance_label, class_label) in instance_to_class.items():
            instance_class_mask = np.zeros_like(instance_mask, dtype=int)
            instance_class_mask[instance_mask == instance_label] = class_label

            mapped_masks[instance_idx] = instance_class_mask            

        return img, mask, instance_mask, mapped_masks
