import os
import torch
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ipywidgets import interact, IntSlider
from dataset import CRCDataset


def view_slices(stack, cmap='gray', title=''):    
    @interact
    def show_slice(slice_idx=IntSlider(min=0, max=stack.shape[0]-1, step=1, value=0)):
        plt.figure(figsize=(10, 10))
        plt.imshow(stack[slice_idx], cmap=cmap)
        plt.title(f'{title} - Slice {slice_idx}')
        plt.axis('off')
        plt.show()

# %%
def main():
    sample_idx = 0
    root = '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version'
    dataset = CRCDataset(root)
    img, mask, d1, d2, d3 = dataset[sample_idx]

    mask_images = mask[d2['colon_1']]
    view_slices(mask_images, title='3D Mask Slices')

    #l1  shape is num_object x slice x height x width  (e.g. 93keys x 376 x 512 x 512)
    for key in d3:
        single_object = d3[list(d3.keys())[1]]
        for slice in single_object:
            if sum(sum(slice)) > 0:
                plt.imshow(slice, cmap='gray')
                plt.show()
            else:
                continue
        break    
# %%
if __name__ == '__main__':
    main()
# %%

