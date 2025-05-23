# %%
import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure, binary_fill_holes


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


def process_images(image_paths, mask_paths, window_size=10):
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
        bone_slices_sum = np.sum(bone_mask, axis=(0, 1))[::-1]

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
            "image_index": idx + 1,
            "lung_slices_sum": lung_slices_sum,
            "bone_slices_sum": bone_slices_sum,
            "lung_end_index": lung_end_index,  # largest descend in means
            "mask_slices_sum": mask_slices_sum,
            "mask_first_nonzero": np.argmax(mask_slices_sum > 0),
            "image_path": image_path,
            "mask_path": mask_paths[idx],
            "slice_index_to_cut": slice_indexes_to_cut,
        })

    return results


# Example usage
if __name__ == '__main__':

    # nii_pth = "/media/dysk_a/jr_buler/RJG-gumed/RJG_13-02-25_nii_labels"
    nii_pth = "/users/project1/pt01191/CRC/Data/RJG_13-02-25_nii_labels"
    image_paths = []
    mask_paths = []
    cut_filtered_bodyMask = []
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
                continue
            elif '_body.nii.gz' in f:
                continue
            elif 'cut_filterMask.nii.gz' in f:
                cut_filtered_bodyMask.append(f)
            elif 'instance_mask.nii.gz' in f:
                continue
            elif 'nii.gz' in f:
                image_paths.append(f)
            elif 'mapping.pkl' in f:
                continue

    print("Processing...")
    results = process_images(image_paths, mask_paths, window_size=10)
    print("Saving...")


    # save new images and masks after cutting the slices
    # so image now is slice[slice_index_to_cut:]
    # and mask is mask[slice_index_to_cut:]
    for result in results:
        nifti_image = nib.load(result["image_path"])
        nifti_mask = nib.load(result["mask_path"])
        image_hu = nifti_image.get_fdata()
        image_hu = np.squeeze(image_hu) if len(image_hu.shape) == 4 else image_hu
        nifti_mask = nib.load(result["mask_path"])
        mask = nifti_mask.get_fdata()
        mask = np.squeeze(mask) if len(mask.shape) == 4 else mask
        image_hu = image_hu[:, :, :-result['slice_index_to_cut']]
        mask = mask[:, :, :-result['slice_index_to_cut']]

        new_image_path = result["image_path"].replace(".nii.gz", "_cut.nii.gz")
        new_mask_path = result["mask_path"].replace(".nii.gz", "_cut.nii.gz")
        nib.save(nib.Nifti1Image(image_hu, nifti_image.affine), new_image_path)
        nib.save(nib.Nifti1Image(mask, nifti_mask.affine), new_mask_path)
    draw = False
    print("Finished.")
    if draw:
        for result in results:
            plt.figure(figsize=(12, 6))
            plt.plot(result["lung_slices_sum"], label='Lung Slices Sum')
            plt.plot(result["mask_slices_sum"], label='Mask Slices Sum', linestyle='--')
            plt.axvline(x=result["lung_end_index"], color='r', linestyle='-', label='Lung End Index')
            plt.axvline(x=result["slice_index_to_cut"], color='b', linestyle='-', label='Slice to cut')

            plt.legend()
            plt.title(f"Lung and Mask Slices Sum - {result['image_path'][57:61]}")
            plt.xlabel('Slice Index')
            plt.ylabel('Sum of Pixels')
            plt.tight_layout()
            plt.show()
# %%