# %%
import numpy as np


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