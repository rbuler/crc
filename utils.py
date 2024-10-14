import numpy as np

def find_unique_value_mapping(mask1, mask2) -> dict:
    """
    Find the mapping between unique values of two 3D masks, excluding zeros.
    
    Parameters:
    mask1 (np.ndarray): The first 3D numpy array (e.g., class mask).
    mask2 (np.ndarray): The second 3D numpy array (e.g., instance mask).
    
    Returns:
    dict: A dictionary mapping unique values from mask1 to corresponding values in mask2.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks should have the same shapes.")

    unique_values_mask1 = np.unique(mask1[mask1 != 0])

    class_mapping = {
          "background": {"class_label": 0, "instance_labels": [0]},
          "colon_positive": {"class_label": 1, "instance_labels": []},
          "lymph_node_positive": {"class_label": 2, "instance_labels": []},
          "suspicious_fat": {"class_label": 3, "instance_labels": []},
          "colon_negative": {"class_label": 4, "instance_labels": []},
          "lymph_node_negative": {"class_label": 5, "instance_labels": []},
          "unsuspicious_fat": {"class_label": 6, "instance_labels": []}
          }

    for k, v in class_mapping.items():
        if v["class_label"] in unique_values_mask1:
            v["instance_labels"] = list(np.unique(mask2[mask1 == v["class_label"]]))
            v["instance_labels"] = [x for x in v["instance_labels"] if x != 0]


    return class_mapping