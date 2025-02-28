# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    CropForegroundd,
    RandSpatialCropd,
    Spacingd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandZoomd,
    RandFlipd,
    RandRotate90d,
    MapTransform,
    ToTensord,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    RandCropBoxByPosNegLabeld,
    ConvertBoxToStandardModed,
    ConvertBoxModed,
    MaskToBoxd,
    StandardizeEmptyBoxd,
)
from monai.config import KeysCollection

def generate_detection_train_transform(
    image_key,
    box_key,
    label_key,
    mask_key,
    intensity_transform,
    patch_size,
    batch_size,
):

    train_transforms = Compose(
        [
            LoadImaged(keys=[image_key, mask_key], image_only=False, meta_key_postfix="meta_dict"),
            SqueezeAllDimsd(keys=[image_key, mask_key]),
            # EnsureChannelFirstd(keys=[image_key, mask_key]),
            EnsureTyped(keys=[image_key, mask_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            # StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            # Orientationd(keys=[image_key, mask_key], axcodes="RAS"),
            intensity_transform,
            RandCropBoxByPosNegLabeld(
                image_keys=[image_key, mask_key],
                box_keys=box_key,
                label_keys=[label_key],
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=batch_size,
                whole_box=True,
                # thresh_image_key=thresh_image_key,
                # image_threshold=image_threshold,
                # fg_indices_key=fg_indices_key,
                # bg_indices_key=bg_indices_key,
                meta_keys=["image_meta_dict", "mask_meta_dict"],
                # meta_key_postfix=meta_key_postfix,
                allow_smaller=False,
                # allow_missing_keys=allow_missing_keys,
            ),
                 # MatchBoxCoordinates(keys=["boxes"]), # x to H, y to W, z to D
            RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=[image_key],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.1),
            RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.9, 1.1)),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            DeleteItemsd(keys=["image_meta_dict", "mask_meta_dict"]),

        ]
    )
    return train_transforms


def generate_detection_val_transform(
    image_key,
    box_key,
    label_key,
    mask_key,
    intensity_transform,
):
    val_transforms = Compose(
        [
            LoadImaged(keys=[image_key, mask_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key, mask_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            # StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            # Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
        ]
    )
    return val_transforms


def generate_detection_inference_transform(
    image_key,
    pred_box_key,
    pred_label_key,
    pred_score_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        pred_box_key: the key to represent predicted boxes
        pred_label_key: the key to represent predicted box labels
        pred_score_key: the key to represent predicted classification scores
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        validation transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    test_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            # EnsureTyped(keys=[image_key], dtype=torch.float32),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            # EnsureTyped(keys=[image_key], dtype=compute_dtype),
        ]
    )
    post_transforms = Compose(
        [
            ClipBoxToImaged(
                box_keys=[pred_box_key],
                label_keys=[pred_label_key, pred_score_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            AffineBoxToWorldCoordinated(
                box_keys=[pred_box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            ConvertBoxModed(box_keys=[pred_box_key], src_mode="xyzxyz", dst_mode=gt_box_mode),
            DeleteItemsd(keys=[image_key]),
        ]
    )
    return test_transforms, post_transforms

class MatchBoxCoordinates(MapTransform):
    """
    Match x to H, y to W, z to D for the boxes in the data dictionary.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
        """
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = np.array([[box[1], box[0], box[2], box[4], box[3], box[5]] for box in d[key]])
        return d

class SqueezeAllDimsd(MapTransform):
    """
    Custom transform to remove all singleton dimensions from image and mask.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].squeeze().unsqueeze(0)  # Remove all dimensions with size=1
        return d