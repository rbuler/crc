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
        self.cut_filtered_bodyMask = []
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
                    self.cut_filtered_bodyMask.append(f)
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
                body_mask_path = self.cut_filtered_bodyMask[idx]
            else:
                image_path = self.cut_images_path[idx]
            mask_path = self.cut_mask_path[idx]

        image = np.asarray(nib.load(image_path).dataobj)
        image = torch.from_numpy(image)
        mask = np.asarray(nib.load(mask_path).dataobj)
        mask = torch.from_numpy(mask)
        body_mask = np.asarray(nib.load(body_mask_path).dataobj)
        body_mask = torch.from_numpy(body_mask)

        id = self.get_patient_id(idx).strip("'")


        if len(image.shape) == 4:
            image = image[:, :, 0, :]
        if len(mask.shape) == 4:
            mask = mask[:, :, 0, :]

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
        body_mask = body_mask.permute(2, 0, 1)

        if self.mode == '3d':

            patches = self.extract_patches(image, mask)
            num_foreground = sum(1 for p in patches if torch.any(p[1] > 0))
            num_to_select = min(36, num_foreground * 2)

            selected_patches = self.select_patches(patches, num_to_select)
            img_patch = torch.stack([p[0] for p in selected_patches])
            mask_patch = torch.stack([p[1] for p in selected_patches])

            
            if (self.transforms is not None) and self.train_mode:
                data_to_transform = {"image": img_patch, "mask": mask_patch}
                transformed_patches = self.transforms[0](data_to_transform)  # train_transforms
                img_patch, mask_patch = transformed_patches["image"], transformed_patches["mask"]
                return img_patch, mask_patch, torch.zeros(1), torch.zeros(1), id
            elif (self.transforms is not None) and not self.train_mode:
                data_to_transform = {"image": img_patch, "mask": mask_patch}
                transformed_patches = self.transforms[1](data_to_transform)  # val_transforms
                img_patch, mask_patch = transformed_patches["image"], transformed_patches["mask"]
                mask = {"mask": mask, "body_mask": body_mask}
                return img_patch, mask_patch, image, mask, id
        
        elif self.mode == '2d':

            if (self.transforms is not None) and self.train_mode:

                slice_indices = list(range(image.shape[0]))
                random.shuffle(slice_indices)
                selected_slices = []
                masked_count = 0
                unmasked_count = 0
            
                for i in slice_indices:
                    if torch.any(mask[i] > 0) and masked_count < 24:
                        selected_slices.append(i)
                        masked_count += 1
                    elif not torch.any(mask[i] > 0) and unmasked_count < 12:
                        selected_slices.append(i)
                        unmasked_count += 1
                    if masked_count == 24 and unmasked_count == 12:
                        break
                # if not enough slices are found, pad with remaining slices
                if len(selected_slices) < 36:
                    remaining_slices = [i for i in slice_indices if i not in selected_slices]
                    selected_slices.extend(remaining_slices[:36 - len(selected_slices)])

                image = image[selected_slices]
                mask = mask[selected_slices]

                data_to_transform = {"image": image, "mask": mask}
                transformed = self.transforms[0](data_to_transform)  # train_transforms
                image, mask = transformed["image"], transformed["mask"]

            elif (self.transforms is not None) and not self.train_mode:
                data_to_transform = {"image": image, "mask": mask}
                transformed = self.transforms[1](data_to_transform)  # val_transforms
                image, mask = transformed["image"], transformed["mask"]

            return torch.zeros(1), torch.zeros(1), image, mask, id


    def __len__(self):
    # todo
        return len(self.images_path)
    

    def set_mode(self, train_mode):
        self.train_mode = train_mode


    def get_patient_id(self, idx):
        patient_id = os.path.basename(self.images_path[idx]).split('_')[0].split(' ')[0]
        return ''.join(filter(str.isdigit, patient_id))
    
    
    def window_and_normalize_ct(self, ct_image, window_center=45, window_width=400):
        """Enhanced with multi-window support and patch-level normalization"""
        
        lower_bound = window_center - window_width / 2
        upper_bound = window_center + window_width / 2
        ct_windowed = np.clip(ct_image, lower_bound, upper_bound)
        
        normalized = (ct_windowed - lower_bound) / (upper_bound - lower_bound)
        
        return np.clip(normalized, 0, 1)


    def extract_patches(self, image, mask):
        """Enhanced 3D patch extraction with:
        - Multi-scale support
        - Guaranteed positive patches
        - Adaptive stride based on content
        - Patch quality filtering
        """
        H, W, D = image.shape
        patches = []
        patch_scales = [1.0] # TODO so far only 1.0 scale is used


        for scale in patch_scales:
            h_size, w_size, d_size = [int(s * scale) for s in self.patch_size]
            
            overlap = 0.5
            h_stride = int(h_size * (1 - overlap))
            w_stride = int(w_size * (1 - overlap))
            d_stride = int(d_size * (1 - overlap))
            
            h_idxs = list(range(0, H - h_size + 1, h_stride))
            w_idxs = list(range(0, W - w_size + 1, w_stride))
            d_idxs = list(range(0, D - d_size + 1, d_stride))
            
            if h_idxs[-1] != H - h_size:
                h_idxs.append(H - h_size)
            if w_idxs[-1] != W - w_size:
                w_idxs.append(W - w_size)
            if d_idxs[-1] != D - d_size:
                d_idxs.append(D - d_size)
            
            for h in h_idxs:
                for w in w_idxs:
                    for d in d_idxs:
                        img_patch = image[h:h+h_size, w:w+w_size, d:d+d_size]
                        mask_patch = mask[h:h+h_size, w:w+w_size, d:d+d_size]
                        
                        if self.is_valid_patch(img_patch, mask_patch):
                            patches.append((img_patch, mask_patch))
        
        patches.extend(self.extract_lesion_centered_patches(image, mask))

        return patches
    

    def is_valid_patch(self, img_patch, mask_patch):
        """Patch quality criteria"""

        if torch.mean((img_patch < 0.001).float()) > 0.2:
            return False
        if torch.var(img_patch) < 0.01:
            return False
        return True


    def extract_lesion_centered_patches(self, image, mask):
        """Extract patches centered on lesions to ensure coverage"""
        lesion_coords = torch.nonzero(mask > 0)
        patches = []
        h_size, w_size, d_size = self.patch_size
        
        num_samples = min(10, len(lesion_coords))
        sampled_coords = lesion_coords[torch.randperm(len(lesion_coords))[:num_samples]]
        
        for coord in sampled_coords:
            h, w, d = coord

            h_start = max(0, h - h_size // 2)
            w_start = max(0, w - w_size // 2)
            d_start = max(0, d - d_size // 2)
            h_end = min(image.shape[0], h_start + h_size)
            w_end = min(image.shape[1], w_start + w_size)
            d_end = min(image.shape[2], d_start + d_size)
            
            if h_end - h_start < h_size:
                h_start = h_end - h_size
            if w_end - w_start < w_size:
                w_start = w_end - w_size
            if d_end - d_start < d_size:
                d_start = d_end - d_size
            
            img_patch = image[h_start:h_end, w_start:w_end, d_start:d_end]
            mask_patch = mask[h_start:h_end, w_start:w_end, d_start:d_end]
            patches.append((img_patch, mask_patch))
        return patches


    def select_patches(self, patches, num_to_select):
        """Balanced patch selection with randomness"""
        foreground = [p for p in patches if torch.any(p[1] > 0)]
        background = [p for p in patches if not torch.any(p[1] > 0)]
        num_to_select = min(num_to_select, len(patches))
        
        if not foreground:
            return random.sample(background, num_to_select)
        
        num_foreground = min(len(foreground), num_to_select // 2)
        num_background = min(len(background), num_to_select - num_foreground)
        selected = (
            random.sample(foreground, num_foreground) +
            random.sample(background, num_background))
        
        return selected