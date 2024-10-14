import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from dataset import CRCDataset


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


# %%
sample_idx = 1
root = '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version'
dataset = CRCDataset(root, save_new_masks=False)

img, mask, instance_mask, mapped_masks = dataset[sample_idx]

# try:
#     view_slices(img, mask, title='3D Mask Slices')
# except Exception as e:
#     print(e)
# %%
