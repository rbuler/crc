# %%
import os
import re
import yaml
import utils
import logging
import warnings
import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure, binary_fill_holes
import torch
from monai.utils import set_determinism

def extract_largest_body_mask(image, threshold=-400):
    """
    Extracts a coarse body mask from CT by:
    - Thresholding
    - Morphological smoothing
    - Keeping only the largest component per 2D slice
    - Filling internal air bubbles (e.g. in colon)
    """
    # Step 1: Threshold to exclude air
    mask = image > threshold

    # Step 2: Morphological and connected-component per slice
    structure2d = generate_binary_structure(2, 1)
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

    for i in range(mask.shape[-1]):
        slice_mask = mask[:, :, i]

        # Morphological cleaning
        structure_open = generate_binary_structure(2, 1)
        structure_close = generate_binary_structure(2, 2)  # or np.ones((5,5))

        slice_mask = binary_opening(slice_mask, structure=structure_open, iterations=4)
        slice_mask = binary_closing(slice_mask, structure=structure_close, iterations=6)
        slice_mask = binary_opening(slice_mask, structure=structure_open, iterations=4)
        slice_mask = binary_closing(slice_mask, structure=structure_close, iterations=10)
        # Connected components in 2D
        labeled, num = label(slice_mask, structure=structure2d)

        if num > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # ignore background
            largest_label = sizes.argmax()
            largest_region = (labeled == largest_label)

            # Fill internal holes (e.g., air in colon)
            largest_region = binary_fill_holes(largest_region)

            cleaned_mask[:, :, i] = largest_region.astype(np.uint8)

    return cleaned_mask


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# nii_pth = "/media/dysk_a/jr_buler/RJG-gumed/RJG_13-02-25_nii_labels"
nii_pth = "/users/project1/pt01191/CRC/Data/RJG_13-02-25_nii_labels"
image_paths = []
mask_paths = []
cut_image_paths = []
cut_filter_mask_paths = []
pattern = re.compile(r'^\d+a$') # take only those ##a

for root, dirs, files in os.walk(nii_pth, topdown=False):
    for name in files:
        f = os.path.join(root, name)
        folder_name = f.split('/')[-2]
        if not pattern.match(folder_name):
            continue
        if 'labels.nii.gz' in f:
            mask_paths.append(f)
        elif 'labels_cut.nii.gz' in f:
            continue
        elif '_cut.nii.gz' in f:
            cut_image_paths.append(f)
        elif '_body.nii.gz' in f:
            continue
        elif 'cut_filterMask.nii.gz' in f:
            continue
        elif 'instance_mask.nii.gz' in f:
            continue
        elif 'nii.gz' in f:
            image_paths.append(f)
        elif 'mapping.pkl' in f:
            continue
# %%
# z = 0

for iter in range(len(image_paths)):

    image = nib.load(cut_image_paths[iter]).get_fdata()
    body_mask = extract_largest_body_mask(image, threshold=-400)
    new_image = image.copy()
    new_image[body_mask == 0] = -1024.0

    # save image to file
    new_image_nii = nib.Nifti1Image(new_image, np.eye(4))
    print(cut_image_paths[iter].replace('.nii.gz', '_body.nii.gz'))
    nib.save(new_image_nii, cut_image_paths[iter].replace('.nii.gz', '_body.nii.gz'))
    new_image_nii = nib.Nifti1Image(body_mask, np.eye(4))
    print(cut_image_paths[iter].replace('.nii.gz', '_filterMask.nii.gz'))
    nib.save(new_image_nii, cut_image_paths[iter].replace('.nii.gz', '_filterMask.nii.gz'))
    # Loop over each slice and create air mask inside the body mask
    # for i in range(body_mask.shape[-1]):
    #     body_slice = body_mask[:, :, i]
    #     air_slice = air_mask[:, :, i]
    #     air_inside_body = np.logical_and(body_slice, air_slice)
    #     air_inside_body_mask[:, :, i] = air_inside_body.astype(np.uint8)


    # for i in range(body_mask.shape[-1]):
    #     # if i % 200 == 0:
    #     labeled_mask, num_objects = label(body_mask[:, :, i])
    #     if num_objects >2:
    #         print("Number of objects:", num_objects)
    #         plt.figure()
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(image[:, :, i], cmap='gray')
    #         plt.title(f"Image - Slice {i}")
    #         plt.axis('off')

    #         plt.subplot(1, 2, 2)
    #         plt.imshow(body_mask[:, :, i], cmap='gray')
    #         plt.title(f"Body Mask - Slice {i}")
    #         plt.axis('off')

    #         plt.tight_layout()
    #         plt.show()
    # for i in range(air_inside_body_mask.shape[-1]):
    #     plt.figure()
    #     plt.imshow(air_inside_body_mask[:, :, i], cmap='gray')
    #     plt.title(f"Air Inside Body - Slice {i}")
    #     plt.axis('off')
    #     plt.show()

    # air_per_slice = np.array([np.sum(air_inside_body_mask[:, :, i]) for i in range(air_inside_body_mask.shape[-1])])
    # window_size = 10
    # smoothed_air_per_slice = np.convolve(air_per_slice, np.ones(window_size)/window_size, mode='same')
    # air_per_slice_derivative = np.diff(smoothed_air_per_slice)

    # air_per_slice = air_per_slice / np.max(air_per_slice)
    # air_per_slice_derivative = air_per_slice_derivative / np.max(air_per_slice_derivative)
    # smoothed_air_per_slice = smoothed_air_per_slice / np.max(smoothed_air_per_slice)


    # nifti_mask = nib.load(mask_paths[iter])
    # mask = nifti_mask.get_fdata()
    # mask = np.squeeze(mask) if len(mask.shape) == 4 else mask
    # mask = mask > 0
    # mask_slices_sum = np.sum(mask, axis=(0, 1))[::-1]


    # plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(len(air_per_slice)), air_per_slice, label='Air Pixels per Slice', color='blue')
    # plt.plot(np.arange(1, len(air_per_slice_derivative) + 1), air_per_slice_derivative, label='Derivative of Air Pixels', color='red')
    # plt.plot(np.arange(1, len(smoothed_air_per_slice) + 1), smoothed_air_per_slice, label='Smoothed Air Pixels per Slice', color='black')
    # plt.plot(np.arange(len(mask_slices_sum)), mask_slices_sum / np.max(mask_slices_sum), label='Mask Pixels per Slice', color='orange')
    # plt.vlines(np.argmax(air_per_slice_derivative), 0, 1, colors='green', linestyles='dashed', label='Max Descend Index')
    # plt.title('Air Pixels per Slice and Its Derivative')
    # plt.xlabel('Slice Index')
    # plt.ylabel('Pixel Count / Derivative')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # break

# %% 