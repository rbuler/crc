import os
import sys
import ast
import yaml
import torch
import utils
import random
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import nibabel as nib
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import monai.transforms as mt
from utils import evaluate_segmentation
from monai.inferers import SlidingWindowInferer
from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from monai.networks.nets import UNet
from dataset_seg import CRCDataset_seg

# SET UP LOGGING -------------------------------------------------------------
logger = logging.getLogger(__name__)
logger_radiomics = logging.getLogger("radiomics")
logging.basicConfig(level=logging.ERROR)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
# parser.add_argument("--fold", type=int, default=None)
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# USE CONFIG PARAMS ----------------------------------------------------------
root = config['dir']['root']
patch_size = ast.literal_eval(config['training']['patch_size'])
stride = config['training']['stride']
batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']
num_classes = config['training']['num_classes']
mode = config['training']['mode']

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout3d):
            m.train()

@torch.no_grad()
def mc_forward(model, inputs, inferer, T=20, body_mask=None):
    model.eval()
    enable_mc_dropout(model)
    outputs = []

    for _ in range(T):
        logits = inferer(inputs=inputs, network=model)
        if body_mask is not None:
            logits[body_mask == 0] = -1e10
        outputs.append(torch.sigmoid(logits))

    outputs = torch.stack(outputs, dim=0)
    mean_pred = outputs.mean(dim=0)
    std_pred = outputs.std(dim=0)
    return mean_pred, std_pred

# %%
train_transforms = mt.Compose([mt.ToTensord(keys=["image", "mask"])])
val_transforms = mt.Compose([
    mt.CenterSpatialCropd(keys=["image", "mask"], roi_size=patch_size) if mode == '2d' else mt.Lambda(lambda x: x),
    mt.ToTensord(keys=["image", "mask"]),
])
transforms = [train_transforms, val_transforms]
# %%
df_path_u = os.path.join(root, config['dir']['processed'], 'unhealthy_df.pkl')
df_path_h = os.path.join(root, config['dir']['processed'], 'healthy_df.pkl')
# %%
df_u = pd.read_pickle(df_path_u)
df_h = pd.read_pickle(df_path_h)
stair_step_artifact_ids = [1, 19, 98]
slice_thickness_ids = [97, 128, 137]  # slice thickness > 5mm
colon_blockage_ids = [64, 77, 173]
explicit_ids_test = [31, 32, 47, 54, 78, 109, 73, 197, 204] # > tx t0
df_u = df_u[~df_u['id'].astype(int).isin(stair_step_artifact_ids + slice_thickness_ids + colon_blockage_ids)]
df_u_explicit_test = df_u[df_u['id'].astype(int).isin(explicit_ids_test)]
df_u = df_u[~df_u['id'].astype(int).isin(explicit_ids_test)].reset_index(drop=True)

dataset_u = CRCDataset_seg(root_dir=root,
                         df=df_u,
                         config=config,
                         transforms=transforms,
                         patch_size=patch_size,
                         stride=stride,
                         num_patches_per_sample=100,
                         mode=mode)
dataset_h = CRCDataset_seg(root_dir=root,
                            df=df_h,
                            config=config,
                            transforms=transforms,
                            patch_size=patch_size,
                            stride=stride,
                            num_patches_per_sample=100,
                            mode=mode)
# %%
paths = [
    "best_model_53bbcfec-9c65-4e5a-ae95-6fb395d9dc1d.pth",  # 1
    "best_model_96890d9d-d810-46e6-a898-4f9def279276.pth",  # 2
    "best_model_c17d8489-5a14-449c-b844-b7c01a28964f.pth",  # 3
    "best_model_33251dfb-5c62-4cda-90ec-abacc7c925ec.pth",  # 4
    "best_model_60e3c15c-3b7a-4afd-a195-c6bf93aef4b1.pth",  # 5
    "best_model_be650c7d-7af5-42d1-bb18-f8e08da05f9f.pth",  # 6
    "best_model_23d8c77a-64a7-43dd-a8fc-ac997706f226.pth",  # 7
    "best_model_3a3005cb-2c2d-4764-8ce7-07def61ef11f.pth",  # 8
    "best_model_442f564e-bcbb-4d7a-b77a-ea7ff322b300.pth",  # 9
    "best_model_15e7d4cc-99a8-4005-afab-331772f54726.pth"   # 10
]

ids_train_val = dataset_u.df['id'].astype(int).unique().tolist()
SPLITS = 10

kf = KFold(n_splits=SPLITS, shuffle=True, random_state=seed)
folds = list(kf.split(ids_train_val))

# %%
for i, path in enumerate(paths):
    fold = i+1
    weights_path = path

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx + 1 != fold:
            continue

        val_ids = [ids_train_val[i] for i in val_idx]
        test_ids = dataset_h.df['id'].astype(int).unique().tolist()
        val_dataset = torch.utils.data.Subset(dataset_u, [i for i in range(len(dataset_u)) if int(dataset_u.get_patient_id(i)) in val_ids])
        test_dataset = torch.utils.data.Subset(dataset_h, range(len(dataset_h)))
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        weights_path = os.path.join(config['dir']['root'], 'models', weights_path)
        if mode == '2d':
            model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=[32, 64, 128, 256, 512],
                strides=[2, 2, 2, 2],
                kernel_size=3,
                up_kernel_size=3,
                num_res_units=2,
                act=("PReLU", {}),
                norm=("instance", {"affine": False}),
                dropout=0.1,
                bias=True
            )
            model.load_state_dict(torch.load(weights_path))
        elif mode == '3d':
            model = UNETR_PP(
                in_channels=1,
                out_channels=14,
                img_size=tuple(patch_size),
                depths=[3, 3, 3, 3],
                dims=[32, 64, 128, 256],
                do_ds=False
            )
            model.out1.conv.conv = torch.nn.Conv3d(16, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
            model.load_state_dict(checkpoint)
        model = model.to(device)

        # gaussian or not?
        inferer = SlidingWindowInferer(
            roi_size=tuple(patch_size), sw_batch_size=1, overlap=0.75, mode="constant", device=torch.device('cpu')
        )
        model.eval()

        total_iou, total_dice, total_tpr, total_precision = 0, 0, 0, 0
        probs = 0.5

        dataloader = val_dataloader
        dataloader.dataset.dataset.set_mode(train_mode=False)

        num_samples = len(dataloader)

        with torch.no_grad():
            totals = None
            for i, (_, _, image, mask, id) in enumerate(dataloader):

                inputs = image.to(device, dtype=torch.float32)
                targets = mask["mask"].to(torch.device('cpu'), dtype=torch.long)
                body_mask = mask["body_mask"].to(torch.device('cpu'), dtype=torch.long)
                inputs = inputs.unsqueeze(0)

                targets = targets.unsqueeze(0)
                body_mask = body_mask.unsqueeze(0)

                use_mc = False
                if use_mc:
                    mean_pred, std_pred = mc_forward(model, inputs, inferer, T=20, body_mask=body_mask)
                    logits = mean_pred
                else:
                    logits = inferer(inputs=inputs, network=model)
                    logits[body_mask == 0] = -1e10
                
                metrics = evaluate_segmentation(logits, targets, num_classes=num_classes, prob_thresh=probs)
                metrics_str = " | ".join([f"{key}: {float(value):.3f}" for key, value in metrics.items()])
                print(f"Patient_ID: {str(id[0]):<7} | {metrics_str}")

                final_output = torch.sigmoid(logits)
                final_output = (final_output.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
                    
                
                mask = targets.squeeze(0).cpu().numpy().astype(np.uint8)
                combined_output = np.zeros_like(mask)
                combined_output[mask == 1] = 1
                combined_output[final_output == 1] = 2
                combined_output[(mask == 1) & (final_output == 1)] = 3

                output_dir = os.path.join("inference_output_last", f"patient_{id}")
                os.makedirs(output_dir, exist_ok=True)

                mask = mask.squeeze(0)
                combined_output = combined_output.squeeze(0)

                image_nifti = nib.Nifti1Image(image.squeeze(0).cpu().numpy(), np.eye(4))
                mask_nifti = nib.Nifti1Image(mask, np.eye(4))
                combined_output_nifti = nib.Nifti1Image(combined_output, np.eye(4))
                nib.save(image_nifti, os.path.join(output_dir, f"{id}_image.nii.gz"))
                nib.save(mask_nifti, os.path.join(output_dir, f"{id}_mask.nii.gz"))
                nib.save(combined_output_nifti, os.path.join(output_dir, f"{id}_result.nii.gz"))
                # TODO: Add functionality to save NIfTI files for 2D approach
                # nib.save(combined_output_nifti, f'inference_output/final_output_filtered_{fold}_{id}.nii.gz')

                if totals is None:
                    totals = {key: 0 for key in metrics.keys()}
                for key, value in metrics.items():
                    totals[key] += value

            averages = {key: total / num_samples for key, total in totals.items()}
            avg_metrics_str = ", ".join([f"Average {key}: {avg:.4f}" for key, avg in averages.items()])
            print(avg_metrics_str)
# %%
