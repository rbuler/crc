import torch
import numpy as np
import yaml
import logging
import utils
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from dataset import CRCDataset
from reduce_dim_features import reduce_dim


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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# %%

if __name__ == '__main__':
    
    root = config['dir']['root']
    dataset = CRCDataset(root, transform=None,
                         save_new_masks=False)

    radiomics = dataset.radiomic_features
    reduce_dim(radiomics, comparison_type='all') # 'colon', 'node', 'fat', 'all'

# %%
# try:
#     view_slices(img, mask, title='3D Mask Slices')
# except Exception as e:
#     print(e)