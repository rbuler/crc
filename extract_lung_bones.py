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


def calculate_statistics(mask, image_hu):
    """
    Calculate statistics (mean, standard deviation) of the pixel values in a specific region.

    Parameters:
    - mask: Binary mask for the region of interest.
    - image_hu: 3D image array in Hounsfield Units.

    Returns:
    - stats: A dictionary containing mean and standard deviation of pixel values in the masked region.
    """
    region_values = image_hu[mask]
    mean_value = np.mean(region_values)
    std_value = np.std(region_values)
    return {"mean": mean_value, "std": std_value}


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

        lung_mask, bone_mask = segment_lung_and_bone(image_hu)

        lung_slices_sum = np.sum(lung_mask, axis=(0, 1))[::-1]
        bone_slices_sum = np.sum(bone_mask, axis=(0, 1))[::-1]

        lung_stats = calculate_statistics(lung_mask, image_hu)
        bone_stats = calculate_statistics(bone_mask, image_hu)

        lung_slices_derivative = np.diff(lung_slices_sum)

        max_negative_change = np.min(lung_slices_derivative)
        min_value_index = np.where(lung_slices_derivative == max_negative_change)[0][0]  # Get the index


        """
        TODO - Implement a more robust method to find the end of the lung region.
        AND also implement a method to find the start of the bone (lower pelvis) region.
        """


        results.append({
            "image_index": idx + 1,
            "lung_slices_sum": lung_slices_sum,
            "bone_slices_sum": bone_slices_sum,
            "lung_stats": lung_stats,
            "bone_stats": bone_stats,
            "lung_end_index": min_value_index  # Index of the highest negative differentiation
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

    print("Lung Stats - Image 1:", results[0]["lung_stats"])
    print("Bone Stats - Image 1:", results[0]["bone_stats"])
    print("Lung Stats - Image 2:", results[1]["lung_stats"])
    print("Bone Stats - Image 2:", results[1]["bone_stats"])
    print("Lung End Index - Image 1:", results[0]["lung_end_index"])
    print("Lung End Index - Image 2:", results[1]["lung_end_index"])
# %%