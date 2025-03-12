# %%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def segment_lung_and_bone(image_hu, lung_hu_range=(-1000, -500), bone_hu_range=(300, 1200)):
    """
    Segments lung (air) and bone regions based on Hounsfield Unit ranges.

    Parameters:
    - image_hu: The input 3D image array in Hounsfield Units.
    - lung_hu_range: HU range to segment lungs (air).
    - bone_hu_range: HU range to segment bones (thigh, etc.).

    Returns:
    - lung_mask: Binary mask for lung regions.
    - bone_mask: Binary mask for bone regions.
    """
    lung_mask = (image_hu >= lung_hu_range[0]) & (image_hu <= lung_hu_range[1])

    bone_mask = (image_hu >= bone_hu_range[0]) & (image_hu <= bone_hu_range[1])

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
        nifti_image = nib.load(image_path)
        image_hu = nifti_image.get_fdata()
        image_hu = np.squeeze(image_hu) if len(image_hu.shape) == 4 else image_hu
        lung_mask, bone_mask = segment_lung_and_bone(image_hu)

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

        results.append({
            "image_index": idx + 1,
            "lung_slices_sum": lung_slices_sum,
            "bone_slices_sum": bone_slices_sum,
            "lung_end_index": lung_end_index,  # largest descend in means
            "mask_slices_sum": mask_slices_sum,
            "mask_first_nonzero": np.argmax(mask_slices_sum > 0)
        })

    return results


# Example usage
if __name__ == '__main__':

    nii_pth = "/media/dysk_a/jr_buler/RJG-gumed/RJG_13-02-25_nii_labels"
    image_paths = []
    mask_paths = []

    for root, dirs, files in os.walk(nii_pth, topdown=False):
        for name in files:
            f = os.path.join(root, name)
            if 'labels.nii.gz' in f:
                mask_paths.append(f)
            elif 'instance_mask.nii.gz' in f:
                continue
            elif 'nii.gz' in f:
                image_paths.append(f)
            elif 'mapping.pkl' in f:
                continue

    results = process_images(image_paths, mask_paths, window_size=20)
    
    slice_indexes_to_cut = [min(result["lung_end_index"], result["mask_first_nonzero"]) for result in results] # index for each patient

    # save new images and masks after cutting the slices
    # so image now is slice[slice_index_to_cut:]
    # and mask is mask[slice_index_to_cut:]
    for idx, image_path in enumerate(image_paths):
        nifti_image = nib.load(image_path)
        image_hu = nifti_image.get_fdata()
        image_hu = np.squeeze(image_hu) if len(image_hu.shape) == 4 else image_hu
        nifti_mask = nib.load(mask_paths[idx])
        mask = nifti_mask.get_fdata()
        mask = np.squeeze(mask) if len(mask.shape) == 4 else mask
        image_hu = image_hu[:, :, slice_indexes_to_cut[idx]:]
        mask = mask[:, :, slice_indexes_to_cut[idx]:]

        new_image_path = image_path.replace(".nii.gz", "_cut.nii.gz")
        new_mask_path = mask_paths[idx].replace(".nii.gz", "_cut.nii.gz")
        nib.save(nib.Nifti1Image(image_hu, nifti_image.affine), new_image_path)
        nib.save(nib.Nifti1Image(mask, nifti_mask.affine), new_mask_path)
    
    draw = False
    if draw:
        for result in results:
            plt.figure(figsize=(12, 6))
            plt.plot(result["lung_slices_sum"], label='Lung Slices Sum')
            plt.plot(result["mask_slices_sum"], label='Mask Slices Sum', linestyle='--')
            plt.axvline(x=result["lung_end_index"], color='r', linestyle='-', label='Lung End Index')
            plt.legend()
            plt.title('Lung and Mask Slices Sum')
            plt.xlabel('Slice Index')
            plt.ylabel('Sum of Pixels')
            plt.tight_layout()
            plt.show()
# %%