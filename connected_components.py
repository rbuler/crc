# %%
import cc3d
import numpy as np
from dataset import CRCDataset

sample_idx = 0
dir = '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version'
dataset = CRCDataset(dir)
img, mask, masks_dict, masks_slice_dict = dataset[sample_idx]

# %%
labels_in = np.ones((512, 512, 512), dtype=np.int32)
labels_out = cc3d.connected_components(labels_in) # 26-connected

connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

stats = cc3d.statistics(labels_out)

# Remove dust from the input image. Removes objects with
# fewer than `threshold` voxels.
labels_out = cc3d.dust(
  labels_in, threshold=100, 
  connectivity=26, in_place=False
)

# %%

