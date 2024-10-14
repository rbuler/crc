import os
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage
from torch.utils.data import Dataset
from connected_components import create_instance_level_mask

class CRCDataset(Dataset):
    def __init__(self, root, transform=None, save_new_masks=True):
        self.root = root
        self.images_path = []
        self.masks_path = []
        self.image = None
        for root, dirs, files in os.walk(self.root, topdown=False):
            for name in files:
                f = os.path.join(root, name)
                if 'labels.nii.gz' in f:
                    self.masks_path.append(f)
                    save_dir = None if not save_new_masks else f
                    mapping = create_instance_level_mask(f, save_dir=save_dir)
                    print(mapping)
                elif 'nii.gz' in f:
                    self.images_path.append(f)

    def __len__(self):
        # todo
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        img = np.asarray(nib.load(image_path).dataobj)
        img = torch.from_numpy(img)
        # img = img / img.max()
        mask = np.asarray(nib.load(mask_path).dataobj)
        mask = torch.from_numpy(mask)
        
        if len(img.shape) == 3:
            h, w, s = img.shape
        elif len(img.shape) == 4:
            img = img[:, :, 0, :]
            mask = mask[:, :, 0, :]
            h, w, s = img.shape

        img = img.permute(2, 1, 0)  # slice, height, width
        mask = mask.permute(2, 1, 0)

        # Normalize image
        img = (img - img.min()) / (img.max() - img.min())

        mask_label_1 = (mask == 1.).float()
        mask_label_2 = (mask == 2.).float()
        mask_label_3 = (mask == 3.).float()
        mask_label_4 = (mask == 4.).float()
        mask_label_5 = (mask == 5.).float()
        mask_label_6 = (mask == 6.).float()

        masks_dict = {'colon_1': mask_label_1,
                  'node_1': mask_label_2,
                  'fat_1': mask_label_3,
                  'colon_0': mask_label_4,
                  'node_0': mask_label_5,
                  'fat_0': mask_label_6}
        
        mask_slice_1 = []
        mask_slice_2 = []
        mask_slice_3 = []
        mask_slice_4 = []
        mask_slice_5 = []
        mask_slice_6 = []

        for i in range(s):
            if torch.any(mask_label_1[i]):
                mask_slice_1.append(i)
            if torch.any(mask_label_2[i]):
                mask_slice_2.append(i)
            if torch.any(mask_label_3[i]):
                mask_slice_3.append(i)
            if torch.any(mask_label_4[i]):
                mask_slice_4.append(i)
            if torch.any(mask_label_5[i]):
                mask_slice_5.append(i)
            if torch.any(mask_label_6[i]):
                mask_slice_6.append(i)

        masks_slice_dict = {'colon_1': mask_slice_1,
                    'node_1': mask_slice_2,
                    'fat_1': mask_slice_3,
                    'colon_0': mask_slice_4,
                    'node_0': mask_slice_5,
                    'fat_0': mask_slice_6}
        
        self.images = img
        return img, mask, masks_dict, masks_slice_dict
