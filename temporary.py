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
from sklearn.model_selection import KFold

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
fold = config['fold']

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                                                               # bad res <  #  > tx t0
explicit_ids_test = [1, 21, 57, 4, 40, 138, 17, 102, 180, 6, 199, 46, 59,  31, 32, 47, 54, 73, 78, 109, 197, 204]
ids_train_val_test = list(set(ids) - set(explicit_ids_test))


kf = KFold(n_splits=5, shuffle=True, random_state=seed)
folds = list(kf.split(ids_train_val_test))
# %%
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    if fold_idx + 1 != fold:
        continue

    train_ids = [ids_train_val_test[i] for i in train_idx]
    val_ids = [ids_train_val_test[i] for i in val_idx]
    test_ids = explicit_ids_test

    train_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in train_ids])
    val_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in val_ids])
    test_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in test_ids])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

# %%
weights_path = 'best_model_8e369b78-dfcf-4620-beff-fb1373eae30f.pth'

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
total_tpr = 0
total_precision = 0
probs = 0.5

dataloader = test_dataloader
print(test_ids)
num_samples = len(dataloader)

with torch.no_grad():
    dataloader.dataset.dataset.set_mode(train_mode=False)
    for i, (_, _, image, mask, id) in enumerate(dataloader):
        image = image.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)
        image = image.unsqueeze(0)
        final_output = inferer(inputs=image, network=model)
        final_output = final_output.squeeze(0)
        metrics = evaluate_segmentation(final_output, mask.to(torch.device('cpu')), num_classes=num_classes, prob_thresh=probs)
        print(f"Patient {i + 1}/{num_samples}/id={id}: "
              f"IoU: {metrics['IoU']:.4f}, Dice: {metrics['Dice']:.4f}, "
              f"TPR: {metrics['TPR']:.4f}, "
              f"Precision: {metrics['Precision']:.4f}")
        final_output = (final_output.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
        
        mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        combined_output = np.zeros_like(mask)
        combined_output[mask == 1] = 1
        combined_output[final_output == 1] = 2
        combined_output[(mask == 1) & (final_output == 1)] = 3
        
        combined_output_nifti = nib.Nifti1Image(combined_output, np.eye(4))
        # nib.save(combined_output_nifti, f'inference_output/final_output_{id}.nii.gz')
        
        total_iou += metrics["IoU"]
        total_dice += metrics["Dice"]
        total_tpr += metrics["TPR"]
        total_precision += metrics["Precision"]

    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_tpr = total_tpr / num_samples
    avg_precision = total_precision / num_samples

    print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}, "
          f"Average TPR: {avg_tpr:.4f}, "
          f"Average Precision: {avg_precision:.4f}")

# %%













