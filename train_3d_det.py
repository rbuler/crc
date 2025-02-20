# %%

import numpy as np
import torch
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

def interactive_slice_viewer(data, axis=2):
    """
    IPython interactive viewer for 3D medical images with masks and bounding boxes.

    Args:
        data (dict): Dictionary with keys ['image', 'mask', 'boxes']
        axis (int): Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
    """
    # Extract data
    image = data['image'].squeeze(0).numpy()  # (224, 224, 64)
    mask = data['mask'].squeeze(0).numpy()    # (224, 224, 64)
    boxes = data['boxes']           # (N, 6) -> (xmin, ymin, zmin, xmax, ymax, zmax)

    num_slices = image.shape[axis]  # Number of slices along chosen axis
    print(f"{num_slices=}")
    def get_slice(data, axis, idx):
        """Extract a 2D slice from a 3D volume."""
        if axis == 0:
            return data[idx, :, :]
        elif axis == 1:
            return data[:, idx, :]
        else:
            return data[:, :, idx]

    def get_2d_boxes(boxes, axis, idx):
        """Filter and transform 3D boxes to 2D for the current slice."""
        filtered_boxes = []
        for box in boxes:
            ymin, xmin, zmin, ymax, xmax, zmax = box
            if axis == 2 and zmin <= idx <= zmax:
                filtered_boxes.append([xmin, ymin, xmax, ymax])
            elif axis == 1 and ymin <= idx <= ymax:
                filtered_boxes.append([xmin, zmin, xmax, zmax])
            elif axis == 0 and xmin <= idx <= xmax:
                filtered_boxes.append([ymin, zmin, ymax, zmax])
        return filtered_boxes

    import matplotlib.patches as patches

    def plot_slice(idx):
        """Plot image, mask, and bounding boxes for a given slice."""
        img_slice = get_slice(image, axis, idx)
        mask_slice = get_slice(mask, axis, idx)
        boxes_2d = get_2d_boxes(boxes, axis, idx)

        # Plot Image
        plt.figure(figsize=(5, 5))
        plt.imshow(img_slice, cmap='gray', alpha=0.8)
        plt.title(f'Image (Slice {idx})')
        plt.show()

        # Plot Mask
        plt.figure(figsize=(5, 5))
        plt.imshow(mask_slice, cmap='jet', alpha=0.6)
        plt.title(f'Mask (Slice {idx})')
        plt.show()

        # Plot Bounding Boxes
        plt.figure(figsize=(5, 5))
        plt.title(f'Bounding Boxes (Slice {idx})')
        for box in boxes_2d:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=1.5, edgecolor='r', facecolor='none'
            )
            plt.gca().add_patch(rect)
        plt.xlim(0, img_slice.shape[1])
        plt.ylim(img_slice.shape[0], 0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


    interact(plot_slice, idx=IntSlider(min=0, max=num_slices-1, step=1, value=num_slices//2))

from monai.utils import set_determinism
from monai.transforms import ScaleIntensityRanged
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset
import os
import re
import glob
import json
from generate_transforms import generate_detection_train_transform, generate_detection_val_transform
import utils
import yaml
from my_tests import check_masks_inside_boxes

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
set_determinism(seed=seed)
amp = True
if amp:
    compute_dtype = torch.float16
else:
    compute_dtype = torch.float32


root_dir = config['dir']['root']
data_dir = os.path.join(root_dir, config['dir']['nii_images'])
pattern = re.compile(r'^\d+a$')  # take only those ##a
item_dirs = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if pattern.match(item)]
train_images = sorted(glob.glob(os.path.join(data_dir, '**', '*nii.gz'), recursive=True))
train_labels = sorted(glob.glob(os.path.join(data_dir, '**', '*boxes*.json'), recursive=True))

json_files = train_labels
data_dicts = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        sample = json.load(f)
    sample['boxes'] = np.array(sample['boxes'])
    sample['labels'] = np.array(sample['labels'])
    data_dicts.append(sample)

train_files, val_files = data_dicts[:-9], data_dicts[-9:]

intensity_transform = ScaleIntensityRanged(
    keys=["image"],
    a_min=-100,
    a_max=250,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)
train_transforms = generate_detection_train_transform(
    image_key="image",
    box_key="boxes",
    label_key="labels",
    mask_key="mask",
    intensity_transform=intensity_transform,
    patch_size=(224,512,64),
    batch_size=1,
)

val_transforms = generate_detection_val_transform(
    image_key="image",
    box_key="boxes",
    label_key="labels",
    mask_key="mask",
    intensity_transform=intensity_transform,
)

dataset = Dataset(data=data_dicts, transform=val_transforms)
data_loader = DataLoader(dataset, batch_size=1, num_workers=4)

check_masks_inside_boxes(data_loader)

check_data = train_transforms(train_files[0])[0]
image, labels = (check_data["image"], check_data["labels"])
mask = check_data["mask"]
print(check_data['boxes'])
print(f"image shape: {image.shape}, label shape: {labels.shape}, mask shape: {mask.shape}")
interactive_slice_viewer(check_data)
