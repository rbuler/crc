import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from dataset import CRCDataset


def view_slices(stack, cmap='gray', title='', absolute_slice=None):    
    @interact
    def show_slice(slice_idx=IntSlider(min=0, max=stack.shape[0]-1, step=1, value=0)):
        plt.figure(figsize=(10, 10))
        plt.imshow(stack[slice_idx], cmap=cmap)
        if absolute_slice is not None:
            plt.title(f'{title} - Slice {absolute_slice[slice_idx]}')
        else:
            plt.title(f'{title} - Slice {slice_idx}')
        plt.axis('off')
        plt.show()


# %%
sample_idx = 0
root = '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version'
dataset = CRCDataset(root)
img, mask, masks_dict, masks_slice_dict = dataset[sample_idx]

mask_images = masks_dict['colon_0'][masks_slice_dict['colon_0']]
absolute_slice = masks_slice_dict['colon_0']


view_slices(mask_images, title='3D Mask Slices', absolute_slice=absolute_slice)
# %%
