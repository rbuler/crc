import os
import torch
import typing
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider


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


def view_slices(image, stack, cmap='gray', title=''):    
    @interact
    def show_slice(slice_idx=IntSlider(min=0, max=stack.shape[0]-1, step=1, value=0)):
        plt.figure(figsize=(10, 10))
        plt.imshow(stack[slice_idx], cmap=cmap)
        
        # Add image overlay here
        plt.imshow(image[slice_idx], cmap='gray', alpha=0.5)        
        plt.title(f'{title} - Slice {slice_idx}')
        plt.axis('off')
        plt.show()


def generate_mil_bags(df, patient_col='patient_id',
                      features: torch.Tensor = None,
                      instance_label_col='class_label',
                      bag_label_col='bag_label'):
    
    bags = []
    
    for idx, row in df.iterrows():
        patient_id = row[patient_col]
        feature_vector = features[idx]
        instance_label = row[instance_label_col]
        bag_label = row[bag_label_col]
        
        # Find the bag for the current patient_id or create a new one
        bag = next((b for b in bags if b['patient_id'] == patient_id), None)
        if bag is None:
            bag = {'patient_id': patient_id, 'instances': [], 'instance_labels': [], 'bag_label': None}
            bags.append(bag)
        
        bag['instances'].append(feature_vector)
        bag['instance_labels'].append(instance_label)
        
        # Set bag label to 0 if bag_label is 0, otherwise set to 1
        if bag_label == 0 or bag_label == '0':
            bag['bag_label'] = torch.tensor(0, dtype=torch.long)
        else:
            bag['bag_label'] = torch.tensor(1, dtype=torch.long)
    
    return bags

    # TODO: Add multiclass label mapping
    # 0 -> 0
    # 1a -> 1
    # 1b -> 2
    # 2a -> 3
    # 2b -> 4


def summarize_bags(bags):
    positive_bags = sum(1 for bag in bags if bag['bag_label'] == 1)
    negative_bags = sum(1 for bag in bags if bag['bag_label'] == 0)
    return positive_bags, negative_bags