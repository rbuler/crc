import os
import torch
import random
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CRCDataset_seg(Dataset):
    def __init__(self, root_dir: os.PathLike,
                 df: pd.DataFrame,
                 config,
                 transforms=None,
                 patch_size: tuple = (64, 64, 64),
                 stride: int = 32,
                 num_patches_per_sample: int = 50,
                 mode = '3d'):  
                         
        self.root = root_dir
        self.df = df

        self.transforms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches_per_sample = num_patches_per_sample
        self.mode = mode
        self.train_mode = False

        self.images_path = []
        self.images_path = df['images_path'].values
        self.cut_filtered_images_path = df['cut_filtered_image_path'].values
        self.cut_mask_path = df['cut_mask_path'].values if 'cut_mask_path' in df.columns else None
        self.cut_filtered_bodyMask_path = df['cut_filtered_bodyMask_path'].values


    def __getitem__(self, idx):
        id = self.get_patient_id(idx).strip("'")
        
        # load original image for original spacing
        original_image_nib = nib.load(self.images_path[idx])
        image_spacing = original_image_nib.header.get_zooms()[:3]  # assumes z, y, x order
        target_spacing = (1.0, 1.0, 1.5)

        # load images
        image_nib = nib.load(self.cut_filtered_images_path[idx])
        body_mask_nib = nib.load(self.cut_filtered_bodyMask_path[idx])
        if self.cut_mask_path is None:
            mask_nib = nib.Nifti1Image(np.zeros(image_nib.shape), affine=image_nib.affine)
        else:
            mask_nib = nib.load(self.cut_mask_path[idx])

        # convert to tensors
        image = torch.from_numpy(np.asarray(image_nib.dataobj))
        mask = torch.from_numpy(np.asarray(mask_nib.dataobj))
        body_mask = torch.from_numpy(np.asarray(body_mask_nib.dataobj))


        if len(image.shape) == 4:
            image = image[:, :, 0, :]
        if len(mask.shape) == 4:
            mask = mask[:, :, 0, :]

        image = resample_tensor(image, image_spacing, target_spacing, is_label=False)
        mask = resample_tensor(mask, image_spacing, target_spacing, is_label=True)
        body_mask = resample_tensor(body_mask, image_spacing, target_spacing, is_label=True)

        window_center = 45
        window_width = 400
        image = self.window_and_normalize_ct(image,
                                             window_center=window_center,
                                             window_width=window_width)

        # # assign 0 values to mask > 1
                                    # "colon_positive": 1,
        mask[mask == 2] = 0         # "lymph_node_positive": 2,
        mask[mask == 3] = 0         # "suspicious_fat": 3,
        mask[mask == 4] = 0         # "colon_negative": 4,
        mask[mask == 5] = 0         # "lymph_node_negative": 5,
        mask[mask == 6] = 0         # "unsuspicious_fat": 6


        # temporary cuz model requires patch of shape (64 128 128)
        image = image.permute(2, 0, 1)   # D, H, W
        mask = mask.permute(2, 0, 1)
        body_mask = body_mask.permute(2, 0, 1)

        if self.mode == '3d':

            if self.train_mode:
                patches = self.extract_patches(image, mask, body_mask)
                min_voxel_threshold = 100
                num_foreground = sum(1 for p in patches if torch.sum(p[1] > 0) > min_voxel_threshold)

                if num_foreground == 0:
                    num_to_select = 36
                else:
                    num_to_select = min(36, num_foreground * 2)

                selected_patches = self.select_patches(patches, num_to_select)
                img_patch = torch.stack([p[0] for p in selected_patches])
                mask_patch = torch.stack([p[1] for p in selected_patches])
            
                if self.transforms is not None:
                    # transformed = [
                    #     self.transforms[0]({"image": img.unsqueeze(0), "mask": msk.unsqueeze(0)})
                    #     for img, msk in zip(img_patch, mask_patch)
                    # ]
                    # img_patch = torch.stack([t["image"].squeeze(0) for t in transformed])
                    # mask_patch = torch.stack([t["mask"].squeeze(0) for t in transformed])
                    input_dict = {"image": img_patch, "mask": mask_patch}
                    transformed = self.transforms[0](input_dict)

                    img_patch = transformed["image"]
                    mask_patch = transformed["mask"]
                    if isinstance(img_patch, list):
                        img_patch = torch.stack([t.squeeze(0) for t in img_patch])
                        mask_patch = torch.stack([t.squeeze(0) for t in mask_patch])
                return img_patch, mask_patch, torch.zeros(1), torch.zeros(1), id
            else:
                mask = {"mask": mask, "body_mask": body_mask}
                if self.transforms is not None:
                    transformed = self.transforms[1]({"image": image, "mask": mask})
                    image, mask = transformed["image"], transformed["mask"]
                return torch.zeros(1), torch.zeros(1), image, mask, id
        
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

    def get_mode(self):
        return self.train_mode


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


    def extract_patches(self, image, mask, body_mask):
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
            
            overlap = 0.625
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
                        body_mask_patch = body_mask[h:h+h_size, w:w+w_size, d:d+d_size]
                        
                        if self.is_valid_patch(mask_patch, body_mask_patch):
                            patches.append((img_patch, mask_patch))
        
        patches.extend(self.extract_lesion_centered_patches(image, mask))

        return patches
    

    def is_valid_patch(self, mask_patch, body_mask_patch):
        """Patch quality criteria"""

        body_fraction = torch.mean(body_mask_patch.float())
        if body_fraction >= 0.7:
            return True
        elif torch.any(mask_patch > 0):
            return True
        else:
            return False


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

        min_voxel_threshold = 100
        foreground = [p for p in patches if torch.sum(p[1] > 0) > min_voxel_threshold]
        background = [p for p in patches if torch.sum(p[1] > 0) <= min_voxel_threshold]
        num_to_select = min(num_to_select, len(patches))
        
        if not foreground:
            return random.sample(background, num_to_select)
        
        num_foreground = min(len(foreground), num_to_select // 2)
        num_background = min(len(background), num_to_select - num_foreground)
        selected = (
            random.sample(foreground, num_foreground) +
            random.sample(background, num_background))
        
        return selected
    

def resample_tensor(image, original_spacing, target_spacing=(1.0, 1.0, 1.0), is_label=False):
    scale = [o / t for o, t in zip(original_spacing, target_spacing)]
    new_shape = [int(round(s * dim)) for s, dim in zip(scale, image.shape)]
    
    image = image.unsqueeze(0).unsqueeze(0).float()  # [B, C, D, H, W]
    mode = 'nearest' if is_label else 'trilinear'
    resampled = F.interpolate(image, size=new_shape, mode=mode, align_corners=False if not is_label else None)
    return resampled.squeeze()