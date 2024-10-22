# %%
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


def process_images(image_paths, window_size=10):
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

        
        results.append({
            "image_index": idx + 1,
            "lung_slices_sum": lung_slices_sum,
            "bone_slices_sum": bone_slices_sum,
            "lung_end_index": lung_end_index  # Index of the largest descend in means
        })

    return results


# Example usage
if __name__ == '__main__':

    image_paths = [
        '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version/8a/8 (M, 2023-11-22).nii.gz',
        '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version/75a/75 (2024-03-21).nii.gz'
    ]

    results = process_images(image_paths, window_size=5)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results[0]["bone_slices_sum"], label='Bone Slices Sum - Image 1')
    plt.plot(results[1]["bone_slices_sum"], label='Bone Slices Sum - Image 2')
    plt.legend()
    plt.title('Bone Slices Sum Comparison')
    plt.xlabel('Slice Index')
    plt.ylabel('Sum of Bone Pixels')

    plt.subplot(1, 2, 2)
    plt.plot(results[0]["lung_slices_sum"], label='Lung Slices Sum - Image 1')
    plt.plot(results[1]["lung_slices_sum"], label='Lung Slices Sum - Image 2')
    plt.axvline(x=results[0]["lung_end_index"], color='r', linestyle='--', label='Lung End Index - Image 1')
    plt.axvline(x=results[1]["lung_end_index"], color='g', linestyle='--', label='Lung End Index - Image 2')
    plt.legend()
    plt.title('Lung Slices Sum Comparison')
    plt.xlabel('Slice Index')
    plt.ylabel('Sum of Lung Pixels')
    plt.tight_layout()
    plt.show()

    print("Lung End Index - Image 1:", results[0]["lung_end_index"])
    print("Lung End Index - Image 2:", results[1]["lung_end_index"])
# %%