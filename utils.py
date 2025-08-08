import os
import torch
import typing
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import matplotlib.patches as patches
from monai.metrics import DiceMetric, MeanIoU
from scipy.ndimage import label
from medpy.metric.binary import hd95, assd

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
            bag['bag_label'] = torch.zeros(1, dtype=torch.float32)
        else:
            bag['bag_label'] = torch.ones(1, dtype=torch.float32)
    
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


def get_3d_bounding_boxes(segmentation, mapping_path):
    segmentation = segmentation.cpu().numpy()
    instances = np.unique(segmentation)
    instances = instances[instances > 0]

    bounding_boxes = []
    labels = []

    instance_to_class = {}

    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
        for category, data in mapping.items():
            class_label = data['class_label']
            for instance_label in data['instance_labels']:
                instance_to_class[instance_label] = class_label

    for instance in instances:
        indices = np.argwhere(segmentation == instance)

        if indices.size == 0:
            continue

        min_coords = indices.min(axis=0)
        max_coords = indices.max(axis=0)

        h, w, d = max_coords[0] - min_coords[0], max_coords[1] - min_coords[1], max_coords[2] - min_coords[2]

        if h <= 0 or w <= 0 or d <= 0:
            continue

        bounding_box = [*min_coords, *max_coords]
        label_ = int(instance_to_class[instance])
        bounding_boxes.append(bounding_box)
        labels.append(label_)

    return {'boxes': np.stack(bounding_boxes), 'labels': np.stack(labels)}


def get_2d_bounding_boxes(segmentation, mapping_path, plane='xy'):
    segmentation = segmentation.cpu().numpy()
    instances = np.unique(segmentation)
    instances = instances[instances > 0]
    instance_to_class = {}
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
        for category, data in mapping.items():
            class_label = data['class_label']
            for instance_label in data['instance_labels']:
                instance_to_class[instance_label] = class_label

    bounding_boxes_per_slice = {}

    for instance in instances:
        indices = np.argwhere(segmentation == instance)

        if indices.size == 0:
            continue

        class_label = int(instance_to_class.get(instance, -1))

        if plane == 'xy':
            slices = np.unique(indices[:, 0])
            for z in slices:
                slice_indices = indices[indices[:, 0] == z][:, 1:]
                min_coords = slice_indices.min(axis=0)
                max_coords = slice_indices.max(axis=0)
                if (max_coords[0] - min_coords[0] < 2) or (max_coords[1] - min_coords[1] < 2):
                    continue
                bbox = [min_coords[1], min_coords[0], max_coords[1], max_coords[0]]

                if z not in bounding_boxes_per_slice:
                    bounding_boxes_per_slice[z] = {'boxes': [], 'labels': []}
                bounding_boxes_per_slice[z]['boxes'].append(bbox)
                bounding_boxes_per_slice[z]['labels'].append(class_label)

        elif plane == 'xz':
            slices = np.unique(indices[:, 1])
            for y in slices:
                slice_indices = indices[indices[:, 1] == y][:, [0, 2]]
                min_coords = slice_indices.min(axis=0)
                max_coords = slice_indices.max(axis=0)
                if (max_coords[0] - min_coords[0] < 2) or (max_coords[1] - min_coords[1] < 2):
                    continue
                bbox = [min_coords[1], min_coords[0], max_coords[1], max_coords[0]]

                if y not in bounding_boxes_per_slice:
                    bounding_boxes_per_slice[y] = {'boxes': [], 'labels': []}
                bounding_boxes_per_slice[y]['boxes'].append(bbox)
                bounding_boxes_per_slice[y]['labels'].append(class_label)

        elif plane == 'yz':
            slices = np.unique(indices[:, 2])
            for x in slices:
                slice_indices = indices[indices[:, 2] == x][:, :2]
                min_coords = slice_indices.min(axis=0)
                max_coords = slice_indices.max(axis=0)
                if (max_coords[0] - min_coords[0] < 2) or (max_coords[1] - min_coords[1] < 2):
                    continue
                bbox = [min_coords[1], min_coords[0], max_coords[1], max_coords[0]]

                if x not in bounding_boxes_per_slice:
                    bounding_boxes_per_slice[x] = {'boxes': [], 'labels': []}
                bounding_boxes_per_slice[x]['boxes'].append(bbox)
                bounding_boxes_per_slice[x]['labels'].append(class_label)
    
    if plane == 'xy':
        total_slices = segmentation.shape[0]
    elif plane == 'xz':
        total_slices = segmentation.shape[1]
    elif plane == 'yz':
        total_slices = segmentation.shape[2]

    for slice_index in range(total_slices):
        if slice_index not in bounding_boxes_per_slice:
            bounding_boxes_per_slice[slice_index] = {'boxes': torch.empty((0, 4), dtype=torch.float32), 'labels': torch.empty((0,), dtype=torch.int64)}
        else:
            bounding_boxes_per_slice[slice_index]['boxes'] = torch.tensor(
                bounding_boxes_per_slice[slice_index]['boxes'], dtype=torch.float32
            )
            bounding_boxes_per_slice[slice_index]['labels'] = torch.tensor(
                bounding_boxes_per_slice[slice_index]['labels'], dtype=torch.int64
            )

    return bounding_boxes_per_slice


def interactive_slice_viewer(data, axis=2, label_=None):
    """
    IPython interactive viewer for 3D medical images with masks and bounding boxes.

    Args:
        data (dict): Dictionary with keys ['image', 'mask', 'boxes']
        axis (int): Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
    """
    # Extract data
    image = data['image'].squeeze(0).numpy()  # (224, 224, 64)
    if label_:
        mask = data['mask'].squeeze(0).numpy() == label
    else:
        mask = data['mask'].squeeze(0).numpy()    # (224, 224, 64)
    boxes = data['boxes']           # (N, 6) -> (xmin, ymin, zmin, xmax, ymax, zmax)

    num_slices = image.shape[axis]  # Number of slices along chosen axis
    print(f"{num_slices=}")
    def get_slice(data, axis, idx):
        """Extract a 2D slice from a 3D volume."""
        if axis == 0:
            return data[idx, :, :]
        elif axis == 1:
            return data[:, idx, :]
        else:
            return data[:, :, idx]

    def get_2d_boxes(boxes, axis, idx):
        """Filter and transform 3D boxes to 2D for the current slice."""
        filtered_boxes = []
        for box in boxes:
            Hmin, Wmin, Dmin, Hmax, Wmax, Dmax = box
            if axis == 2 and Dmin <= idx <= Dmax:
                filtered_boxes.append([Hmin, Wmin, Hmax, Wmax])
            elif axis == 1 and Wmin <= idx <= Wmax:
                filtered_boxes.append([Hmin, Dmin, Hmax, Dmax])
            elif axis == 0 and Hmin <= idx <= Hmax:
                filtered_boxes.append([Wmin, Dmin, Wmax, Dmax])
        return filtered_boxes

    def plot_slice(idx):
        """Plot image, mask, and bounding boxes for a given slice."""
        img_slice = get_slice(image, axis, idx)
        mask_slice = get_slice(mask, axis, idx)
        boxes_2d = get_2d_boxes(boxes, axis, idx)

        # Plot Image
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Image
        axes[0].imshow(img_slice, cmap='gray', alpha=0.8)
        axes[0].set_title(f'Image (Slice {idx})')

        # Plot Mask
        axes[1].imshow(mask_slice, cmap='jet', alpha=0.6)
        axes[1].set_title(f'Mask (Slice {idx})')

        # Plot Bounding Boxes
        axes[2].imshow(img_slice, cmap='gray', alpha=0.8)
        for box in boxes_2d:
            # matplotlib coords x-y
            ymin, xmin, ymax, xmax = box
            print(f"{xmin=}, {ymin=}, {xmax=}, {ymax=}")
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=1.5, edgecolor='r', facecolor='none'
            )
            axes[2].add_patch(rect)
        axes[2].set_xlim(0, img_slice.shape[1])
        axes[2].set_ylim(img_slice.shape[0], 0)
        axes[2].set_title(f'Bounding Boxes (Slice {idx})')
        axes[2].set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show()


    interact(plot_slice, idx=IntSlider(min=0, max=num_slices-1, step=1, value=num_slices//2))



def compute_false_positive_metrics(pred_labels, voxel_spacing=(1.0, 1.0, 1.0), min_volume_mm3=10.0):
    pred_binary = (pred_labels == 1).cpu().numpy().astype(np.uint8)
    filtered_mask = filter_small_components(pred_binary, min_volume_mm3=min_volume_mm3)
    voxel_volume = np.prod(voxel_spacing)
    cluster_volumes = []

    labeled_array, num_components = label(filtered_mask)
    for i in range(1, num_components + 1):
        cluster = (labeled_array == i)
        volume = np.sum(cluster) * voxel_volume
        cluster_volumes.append(volume)

    fpv = np.sum(filtered_mask) * voxel_volume
    fpr = 1.0 if np.sum(filtered_mask) > 0 else 0.0
    fpcv = np.mean(cluster_volumes) if cluster_volumes else 0.0

    return {
        "FPV": fpv,
        "FPR": fpr,
        "FPCV": fpcv
    }


def keep_largest_connected_component(binary_mask):
    labeled_array, num_features = label(binary_mask)
    if num_features == 0:
        return binary_mask  # empty mask, return as is
    largest_cc = (labeled_array == np.argmax(np.bincount(labeled_array.flat)[1:]) + 1)
    return largest_cc.astype(np.bool_)

def filter_small_components(binary_mask, voxel_spacing=(1.0, 1.0, 1.5), min_volume_mm3=10.0):
    voxel_volume = np.prod(voxel_spacing)
    labeled_array, num_features = label(binary_mask)
    kept_mask = np.zeros_like(binary_mask)

    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        volume = region.sum() * voxel_volume
        if volume >= min_volume_mm3:
            kept_mask[region] = 1
    return kept_mask.astype(np.bool_)


def evaluate_segmentation(pred, true_mask, epoch=None, num_classes=7, prob_thresh=0.5, logits_input=True):
    
    # target_spacing = (1.0, 1.0, 1.5)
    target_spacing = (1.5, 1.0, 1.0) # transpose in dataset
    if logits_input:
        pred_probs = torch.sigmoid(pred) if num_classes == 1 else torch.softmax(pred, dim=1)
    elif logits_input is False:
        pred_probs = pred
    
    pred_labels = torch.argmax(pred_probs, dim=1) if num_classes > 1 else (pred_probs > prob_thresh).long()


    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    mean_iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

    valid_pred_labels = []
    valid_true_masks = []

    for i in range(true_mask.shape[0]):
        if torch.any(true_mask[i] > 0):
            valid_pred_labels.append(pred_labels[i].unsqueeze(0))
            valid_true_masks.append(true_mask[i].unsqueeze(0))

    if valid_pred_labels:
        valid_pred_labels = torch.cat(valid_pred_labels, dim=0)
        valid_true_masks = torch.cat(valid_true_masks, dim=0)

        if valid_pred_labels.dim() == 5:  # 3D case B C H W D
            dice_metric(y_pred=valid_pred_labels, y=valid_true_masks)
            mean_iou_metric(y_pred=valid_pred_labels, y=valid_true_masks)
        elif valid_pred_labels.dim() == 4:  # 2D case B C H W or slidng window inferer
            dice_metric(y_pred=valid_pred_labels.unsqueeze(1), y=valid_true_masks.unsqueeze(1))
            mean_iou_metric(y_pred=valid_pred_labels.unsqueeze(1), y=valid_true_masks.unsqueeze(1))

        mean_dice = dice_metric.aggregate().item()
        mean_iou = mean_iou_metric.aggregate().item()

        dice_metric.reset()
        mean_iou_metric.reset()


        if epoch < 20:
            hd95_score = 0.0
            assd_score = 0.0
        else:

            pred_np = valid_pred_labels.cpu().numpy().astype(np.bool_)
            true_np = valid_true_masks.cpu().numpy().astype(np.bool_)
            hd95_scores = []
            assd_scores = []

            for i in range(pred_np.shape[0]):
                pred_i = filter_small_components(pred_np[i, 0], voxel_spacing=target_spacing)
                true_i = filter_small_components(true_np[i, 0], voxel_spacing=target_spacing)
                pred_i = keep_largest_connected_component(pred_i)
                true_i = keep_largest_connected_component(true_i)

                try:
                    hd = hd95(pred_i, true_i, voxelspacing=target_spacing)
                except Exception:
                    hd = float("nan")

                try:
                    assd_val = assd(pred_i, true_i, voxelspacing=target_spacing)
                except Exception:
                    assd_val = float("nan")

                hd95_scores.append(hd)
                assd_scores.append(assd_val)

            hd95_score = np.nanmean(hd95_scores)
            assd_score = np.nanmean(assd_scores)

        tp = torch.sum((valid_pred_labels == 1) & (valid_true_masks == 1)).item()
        fp = torch.sum((valid_pred_labels == 1) & (valid_true_masks == 0)).item()
        fn = torch.sum((valid_pred_labels == 0) & (valid_true_masks == 1)).item()
        tn = torch.sum((valid_pred_labels == 0) & (valid_true_masks == 0)).item()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "Dice": mean_dice,
            "IoU": mean_iou,
            "HD95": hd95_score,
            "ASSD": assd_score,
            "FPR": fpr,
            "TPR": recall,
            "Precision": precision,
        }
    else:
        
        fp_metrics = compute_false_positive_metrics(pred_labels, voxel_spacing=target_spacing)
        return {
            **fp_metrics
        }