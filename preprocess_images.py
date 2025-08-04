# %%
import os
import re
import logging
import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure, binary_fill_holes
import pydicom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def segment_lung_and_bone(image_hu, body_mask, lung_hu_range=(-1000, -500), bone_hu_range=(300, 1200)):
    """
    Segments lung (air) and bone regions based on Hounsfield Unit ranges, constrained within the body mask.

    Parameters:
    - image_hu: The input 3D image array in Hounsfield Units.
    - body_mask: Binary mask where 1 indicates body and 0 indicates background.
    - lung_hu_range: HU range to segment lungs (air).
    - bone_hu_range: HU range to segment bones (thigh, etc.).

    Returns:
    - lung_mask: Binary mask for lung regions constrained within the body.
    - bone_mask: Binary mask for bone regions constrained within the body.
    """
    lung_mask = (image_hu >= lung_hu_range[0]) & (image_hu <= lung_hu_range[1]) & (body_mask > 0)
    bone_mask = (image_hu >= bone_hu_range[0]) & (image_hu <= bone_hu_range[1]) & (body_mask > 0)

    return lung_mask, bone_mask


def process_images(image_paths, mask_paths=None, window_size=10):
    """
    Process the given images to segment lung and bone regions, calculate statistics, and return summed values.

    Parameters:
    - image_paths: List of paths to the NIfTI images.
    - window_size: The size of the window for calculating the derivative.

    Returns:
    - results: A list of dictionaries containing summed values for bones and lungs for each image.
    """
    results = []

    for idx, image_path in enumerate(image_paths):
        print(f"{idx+1}/{len(image_paths)}")
        nifti_image = nib.load(image_path)
        image_hu = nifti_image.get_fdata()


        body_mask = extract_largest_body_mask(image_hu, threshold=-400)

        image_hu = np.squeeze(image_hu) if len(image_hu.shape) == 4 else image_hu
        lung_mask, bone_mask = segment_lung_and_bone(image_hu=image_hu, body_mask=body_mask)

        lung_slices_sum = np.sum(lung_mask, axis=(0, 1))[::-1]
        # bone_slices_sum = np.sum(bone_mask, axis=(0, 1))[::-1]

        lung_means = [np.mean(lung_slices_sum[i:i + window_size]) for i in range(0, len(lung_slices_sum), window_size)]
        half_length = len(lung_means) // 2

        max_descend = 0
        max_descend_index = 0
        for i in range(1, half_length):
            descend = lung_means[i - 1] - lung_means[i]
            if descend > max_descend:
                max_descend = descend
                max_descend_index = i
            if max_descend_index == 0:
                max_descend_index = i

        lung_end_index = max_descend_index * window_size

        """
        TODO - Implement a more robust method to find the end of the lung region.
        AND also implement a method to find the start of the bone (lower pelvis) region.
        """

        # how many nonzero pixels are in the mask slice
        if mask_paths is None or len(mask_paths) == 0:
          slice_indexes_to_cut = lung_end_index + window_size
          pass
        else:      
            nifti_mask = nib.load(mask_paths[idx])
            mask = nifti_mask.get_fdata()
            mask = np.squeeze(mask) if len(mask.shape) == 4 else mask
            mask = mask > 0
            mask_slices_sum = np.sum(mask, axis=(0, 1))[::-1]

            # normalize lung_slices_sum and mask_slices_sum
            lung_slices_sum = lung_slices_sum / np.max(lung_slices_sum)
            mask_slices_sum = mask_slices_sum / np.max(mask_slices_sum)

            if lung_end_index + window_size >= np.argmax(mask_slices_sum > 0):
                slice_indexes_to_cut = min(lung_end_index, np.argmax(mask_slices_sum > 0) - 10)
            else:
                slice_indexes_to_cut = min(lung_end_index + window_size, np.argmax(mask_slices_sum > 0) - 5) # index for each patient
        print(lung_end_index)

        results.append({
            "image_path": image_path,
            "mask_path": mask_paths[idx] if mask_paths is not None else None,
            "slice_index_to_cut": slice_indexes_to_cut,
        })

    return results

# %%

def convert_dicom_to_nii(healthy_people_path, output_base_path):

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    pattern = re.compile(r'^\d+a$')  # take only those ##a

    for root, dirs, files in os.walk(healthy_people_path, topdown=False):
        for name in dirs:
            if not pattern.match(name):
                logger.info(f"Skipping {name}. Pattern does not match.")
                continue

            logger.info(f"Processing {name}. Pattern matches.")
            person_path = os.path.join(root, name)
            output_folder = os.path.join(output_base_path, name)
            os.makedirs(output_folder, exist_ok=True)

            dicom_files = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith('.dcm') or f[0].isdigit()]
            if not dicom_files:
                logger.warning(f"No DICOM files found for {name} in {person_path}")
                continue

            def get_sort_key(fp):
                dcm = pydicom.dcmread(fp, stop_before_pixels=True)
                return float(dcm.ImagePositionPatient[2]) if 'ImagePositionPatient' in dcm else int(dcm.InstanceNumber)

            dicom_files.sort(key=get_sort_key)

            slices = [pydicom.dcmread(dcm_fp) for dcm_fp in dicom_files]
            pixel_arrays = [
                s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept)
                for s in slices
            ]
            volume = np.stack(pixel_arrays, axis=-1)
            spacing = (float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[0].SliceThickness))
            affine = np.diag(spacing + (1.0,))  # 4x4 affine matrix

            output_file = os.path.join(output_folder, f"{name}.nii.gz")
            nib.save(nib.Nifti1Image(volume, affine), output_file)
            logger.info(f"Saved NIfTI file for {name} at {output_file}")