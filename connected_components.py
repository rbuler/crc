# %%
import cc3d
import numpy as np
from utils import find_unique_value_mapping
import nibabel as nib
import pickle

def create_instance_level_mask(mask_path, connectivity=26, save_dir=None,
                               remove_dust=True, threshold=100,
                               verbose=True) -> None:
    """
    Create an instance-level mask from a given class mask.
    Args:
        mask: A numpy array representing the input mask.
        connectivity: The connectivity used for connected components labeling. Only 4, 8 (2D), 26, 18, and 6 (3D) are allowed. Default is 26.
        save_dir: The directory to save the instance mask. If None, the instance mask will not be saved. Default is None.
        remove_dust: Whether to remove dust from the input mask. Default is True.
        threshold: The threshold used for dust removal. Default is 100.
        verbose: Whether to print verbose information during the process. Default is True.
    Returns:
        None
    """
    
    #  only 4,8 (2D) and 26, 18, and 6 (3D) are allowed

    if verbose:
        print("\n----------------------------\n")
    labels_in = np.asarray(nib.load(mask_path).dataobj)
    if len(labels_in.shape) == 4:
        labels_in = np.squeeze(labels_in)
    if remove_dust:
        labels_in = cc3d.dust(labels_in, threshold=threshold,
                              connectivity=connectivity,
                              in_place=False)
    labels_out, N = cc3d.connected_components(labels_in,
                                              connectivity=connectivity,
                                              return_N=True)
    if verbose:
        print(f"Class labels:\t\t{np.unique(labels_in)}")
        print(f"Instance labels:\t{np.unique(labels_out)}")
        print(f"Number of instances:\t{N}")
    
    mapping = find_unique_value_mapping(labels_in, labels_out)

    if save_dir is not None:
        labels_out = nib.Nifti1Image(labels_out, affine=np.eye(4))
        instance_mask_path = save_dir.split('.')[0] + "_instance_mask.nii.gz"
        nib.save(labels_out, instance_mask_path)
        mapping_path = save_dir.split('.')[0] + "_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(mapping, f)
        if verbose:
            for k, v in mapping.items():
                print(f"{k:20}:\t{v['class_label']} -> {v['instance_labels']}")
            print(f"Instance mask saved as {instance_mask_path}.")
            print(f"Mapping saved as {mapping_path}.")
    else:
        if verbose:
            print("Instance mask not saved. Mapping created:")
            for k, v in mapping.items():
                print(f"{k:20}:\t{v['class_label']} -> {v['instance_labels']}")
    return