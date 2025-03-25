# %%
import os
import sys
import ast
import yaml
import torch
import utils
import random
import logging
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader

import monai.transforms as mt
from monai.inferers import SlidingWindowInferer
# from pytorch3dunet.unet3d.model import UNet3D
from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from utils import evaluate_segmentation
from dataset_seg import CRCDataset_seg

# SET UP LOGGING -------------------------------------------------------------
logger = logging.getLogger(__name__)
logger_radiomics = logging.getLogger("radiomics")
logging.basicConfig(level=logging.ERROR)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# USE CONFIG PARAMS ----------------------------------------------------------
num_classes = config['training']['num_classes']
patch_size = ast.literal_eval(config['training']['patch_size'])
stride = config['training']['stride']
batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
train_transforms = mt.Compose([mt.ToTensord(keys=["image", "mask"])])
val_transforms = mt.Compose([mt.ToTensord(keys=["image", "mask"])])
transforms = [train_transforms, val_transforms]
# %%
dataset = CRCDataset_seg(root_dir=config['dir']['root'],
                         nii_dir=config['dir']['nii_images'],
                         clinical_data_dir=config['dir']['clinical_data'],
                         config=config,
                         transforms=transforms,
                         patch_size=patch_size,
                         stride=stride,
                         num_patches_per_sample=100)

ids = []
for i in range(len(dataset)):
    ids.append(int(dataset.get_patient_id(i)))

explicit_ids_test = [31, 32, 47, 54, 73, 78, 109, 197, 204]
ids_train_val_test = list(set(ids) - set(explicit_ids_test))

train_size = int(0.75 * len(ids_train_val_test))
val_size = int(0.2 * len(ids_train_val_test))
test_size = len(ids_train_val_test) - train_size - val_size + len(explicit_ids_test)

train_ids = random.sample(ids_train_val_test, train_size)
val_ids = random.sample(list(set(ids_train_val_test) - set(train_ids)), val_size)
test_ids = list((set(ids_train_val_test) - set(train_ids) - set(val_ids)) | set(explicit_ids_test))

train_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in train_ids])
val_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in val_ids])
test_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in test_ids])

def collate_fn(batch):
    img_patch, mask_patch, _, _ = zip(*batch)
    img_patch = torch.stack(img_patch)
    mask_patch = torch.stack(mask_patch)
    return img_patch, mask_patch

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

# %%


weights_path = 'best_model_03bd4070-76d5-4bb7-ae90-df83ced59d7b.pth'
weights_path = os.path.join(config['dir']['root'], 'models', weights_path)

model = UNETR_PP(in_channels=1, out_channels=14,
                 img_size=tuple(patch_size),
                 depths=[3, 3, 3, 3],
                 dims=[32, 64, 128, 256], do_ds=False)

model.out1.conv.conv = torch.nn.Conv3d(16, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
# %%
inferer = SlidingWindowInferer(roi_size=tuple(patch_size), sw_batch_size=1, overlap=0.5, device=torch.device('cpu'))
model.eval()

total_iou = 0
total_dice = 0
probs = 0.5

dataloader = test_dataloader
print(test_ids)
num_samples = len(dataloader)

with torch.no_grad():
    dataloader.dataset.dataset.set_mode(train_mode=False)
    for i, (_, _, image, mask) in enumerate(dataloader):
        image = image.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)
        image = image.unsqueeze(0)
        final_output = inferer(inputs=image, network=model)
        final_output = final_output.squeeze(0)
        metrics = evaluate_segmentation(final_output, mask.to(torch.device('cpu')), num_classes=num_classes, prob_thresh=probs)
        print(f"Patient {i + 1}/{num_samples}: IoU: {metrics['IoU']:.4f}, Dice: {metrics['Dice']:.4f}")
        id = dataloader.dataset.dataset.get_patient_id(i)
        final_output = (final_output.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
        
        
        mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        combined_output = np.zeros_like(mask)
        combined_output[mask == 1] = 1
        combined_output[final_output == 1] = 2
        combined_output[(mask == 1) & (final_output == 1)] = 3
        
        combined_output_nifti = nib.Nifti1Image(combined_output, np.eye(4))
        # nib.save(combined_output_nifti, f'temporary/final_output_{id}.nii.gz')
        
        total_iou += metrics["IoU"]
        total_dice += metrics["Dice"]

avg_iou = total_iou / num_samples
avg_dice = total_dice / num_samples

print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")
# %%
