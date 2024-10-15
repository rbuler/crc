import os
import typing
import argparse
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


def pretty_dict_str(d, key_only=False):
    #take empty string
    sorted_list = sorted(d.items())
    sorted_dict = {}
    for key, value in sorted_list:
        sorted_dict[key] = value
    pretty_dict = ''  
     
    #get items for dict
    if key_only:
        for k, _ in sorted_dict.items():
            pretty_dict += f'\n\t{k}'
    else:
        for k, v in sorted_dict.items():
            pretty_dict += f'\n\t{k}:\t{v}'
        #return result
    return pretty_dict


def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=path,
                        help=help)
    return parser