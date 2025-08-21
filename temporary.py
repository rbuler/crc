import os
import re
import sys
import ast
import yaml
import json
import torch
import utils
import random
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

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
patch_mode = config['training']['patch_mode']

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_list(v) for v in obj]
    return obj


def list_to_numpy(obj):
    """Recursively convert lists in dict/list back to numpy arrays."""
    if isinstance(obj, list):
        # Only convert if it's a list of numbers (not a list of dicts)
        if all(isinstance(x, (int, float)) for x in obj):
            return np.array(obj)
        else:
            return [list_to_numpy(v) for v in obj]
    if isinstance(obj, dict):
        return {k: list_to_numpy(v) for k, v in obj.items()}
    return obj


def enable_last_mc_dropout(model):
    dropout3d_layers = [m for m in model.modules() if isinstance(m, nn.Dropout3d)]
    
    if dropout3d_layers:
        # Set only the last Dropout3d layer to train mode
        dropout3d_layers[-1].train()


def enable_mc_dropout(model):
    dropout3d_layers = [m for m in model.modules() if isinstance(m, nn.Dropout3d)]
    
    for layer in dropout3d_layers:
        layer.train()


@torch.no_grad()
def mc_forward(model, inputs, inferer, T=20, body_mask=None):
    model.eval()
    enable_mc_dropout(model)
    preds_list = []
    metrics_list = []

    for _ in range(T):
        logits = inferer(inputs=inputs, network=model)
        if body_mask is not None:
            logits[body_mask == 0] = -1e10
        preds = torch.sigmoid(logits)
        preds_list.append(preds)
        metrics = evaluate_segmentation(preds, targets, num_classes=num_classes, prob_thresh=probs, logits_input=False)
        metrics_list.append(metrics)

    preds_stack = torch.stack(preds_list, dim=0)  # Shape: [T, B, C, H, W, D]
    mean_pred = preds_stack.mean(dim=0)
    std_pred = preds_stack.std(dim=0)

    # Calculate mean and std for metrics
    metrics_mean = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0].keys()}
    metrics_std = {key: np.std([m[key] for m in metrics_list]) for key in metrics_list[0].keys()}

    return mean_pred, std_pred, metrics_mean, metrics_std
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
rectal_cancer_ids = [76]
explicit_ids_test = [31, 32, 47, 54, 78, 109, 73, 197, 204] # > tx t0
# drop specific ids
df_u = df_u[~df_u['id'].astype(int).isin(stair_step_artifact_ids + slice_thickness_ids + colon_blockage_ids + rectal_cancer_ids)]
df_u_explicit_test = df_u[df_u['id'].astype(int).isin(explicit_ids_test)]
df_u = df_u[~df_u['id'].astype(int).isin(explicit_ids_test)]
# %%
# load clinical data
def extract_T(value):
    if value is None:
        return None
    s = str(value).strip()
    m = re.match(r"^(T\d)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    print(f"Warning: Could not extract T from value '{value}'")
    return None

clinical_data_dir = os.path.join(root, config['dir']['clinical_data'])
default_missing = pd._libs.parsers.STR_NA_VALUES
clinical_data = pd.read_excel(
            clinical_data_dir,
            index_col=False,
            na_filter=True,
            na_values=default_missing)
tnm_data = clinical_data.rename(columns={'TNM wg mnie': 'TNM', 'Nr pacjenta': 'id'})[['id', 'TNM']].dropna(subset=['id'])
tnm_data = tnm_data[tnm_data['id'].astype(int).isin(df_u['id'].astype(int))]
tnm_data["T_extracted"] = tnm_data["TNM"].apply(extract_T)
most_frequent_T = tnm_data["T_extracted"].dropna().mode()[0]
tnm_data["T_clean"] = tnm_data["T_extracted"].fillna(most_frequent_T)
# %%
dataset_u = CRCDataset_seg(root_dir=root,
                         df=df_u,
                         config=config,
                         transforms=transforms,
                         patch_size=patch_size,
                         stride=stride,
                         num_patches_per_sample=100,
                         mode=mode,
                         patch_mode=patch_mode)
dataset_h = CRCDataset_seg(root_dir=root,
                            df=df_h,
                            config=config,
                            transforms=transforms,
                            patch_size=patch_size,
                            stride=stride,
                            num_patches_per_sample=100,
                            mode=mode,
                            patch_mode=patch_mode)
# %%
ids_train_val = dataset_u.df['id'].astype(int).unique().tolist()
SPLITS = 10

stratification_labels = tnm_data.set_index('id').reindex(ids_train_val)['T_clean'].values
skf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=seed)
folds = list(skf.split(ids_train_val, stratification_labels))

# %%
paths = [
    "best_model_7a7eb0f2-86b2-41bf-91d1-9eb44f7f027c.pth",  # 1
    "best_model_9b5cd086-538f-4dcc-a93c-63285ff96df2.pth",  # 2
    "best_model_e4251409-e654-4525-871d-045d8a422621.pth",  # 3
    "best_model_c61817a8-8539-4a1c-bf21-4528040468ec.pth",  # 4
    "best_model_44df5e2d-b7ab-4b5c-8eeb-1872f13b43ed.pth",  # 5
    "best_model_2f964ecd-4c1f-446d-a2c7-0b5663675581.pth",  # 6
    "best_model_30bea749-65f7-4cfa-9056-aa4cef380e73.pth",  # 7
    "best_model_9521c2a5-c48b-4150-8dca-f5e72677afe2.pth",  # 8
    "best_model_fd5da63f-4f2a-4819-8374-0aee5cecf4c5.pth",  # 9
    "best_model_e497c76b-6ea7-4a55-933b-a9b15eaf8810.pth"   # 10
]

all_patients_metrics = {}


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
        print(fold, val_ids, weights_path)
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
        inferer = SlidingWindowInferer(roi_size=tuple(patch_size), sw_batch_size=36, overlap=0.75, mode="constant", device=torch.device('cpu'))
        model.eval()
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
                    mean_pred, std_pred, metrics, metrics_std = mc_forward(model, inputs, inferer, T=20, body_mask=body_mask)
                    metrics_str = " | ".join([f"{key}: {float(value):.3f}" for key, value in metrics.items() if 'patient' not in key])
                    print(f"Patient_ID: {str(id[0]):<7} | {metrics_str}")
                    metrics_str = " | ".join([f"{key}: {float(value):.3f}" for key, value in metrics_std.items() if 'patient' not in key])
                    # print(f"Patient_ID: {str(id[0]):<7} | {metrics_str}")
                    final_output = mean_pred
                    final_output = (final_output.squeeze(0) > probs).cpu().numpy().astype(np.uint8)
                   
                else:
                    logits = inferer(inputs=inputs, network=model)
                    logits[body_mask == 0] = -1e10
                    metrics = evaluate_segmentation(logits, targets, num_classes=num_classes, prob_thresh=probs)
                    metrics_str = " | ".join([f"{key}: {float(value):.3f}" for key, value in metrics.items() if 'patient' not in key])
                    print(f"Patient_ID: {str(id[0]):<7} | {metrics_str}")
                    final_output = torch.sigmoid(logits)
                    final_output = (final_output.squeeze(0) > probs).cpu().numpy().astype(np.uint8)

                mask = targets.squeeze(0).cpu().numpy().astype(np.uint8)
                combined_output = np.zeros_like(mask)
                combined_output[mask == 1] = 1
                combined_output[final_output == 1] = 2
                combined_output[(mask == 1) & (final_output == 1)] = 3

                output_dir = os.path.join("inference_output_last", f"patient_{id}")
                if dataloader == val_dataloader:
                    output_dir = os.path.join("inference_output_last", "validation", f"patient_{id}")
                elif dataloader == test_dataloader:
                    output_dir = os.path.join("inference_output_last", "test", f"patient_{id}")
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
            
                all_patients_metrics[str(id[0])] = {
                    'fold': fold,
                    'metrics': metrics
                }
            averages = {key: total / num_samples for key, total in totals.items()}
            avg_metrics_str = ", ".join([f"Average {key}: {avg:.4f}" for key, avg in averages.items() if 'patient' not in key])
            print(avg_metrics_str)
# %%

cube_sizes = np.concatenate([np.arange(1, 20), np.arange(20, 55, 5)])

detections = []
max_detections = []
for pid, pdata in all_patients_metrics.items():
    det = pdata["metrics"]["patient_detection"]
    max_det = pdata["metrics"]["patient_max_detection"]
    detections.append(det)
    max_detections.append(max_det)

detections = np.vstack(detections)
max_detections = np.vstack(max_detections)

success_rate = detections.mean(axis=0)
max_success_rate = max_detections.mean(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(cube_sizes, success_rate, linestyle="-", label="Model Detection Success", color="blue")
plt.plot(cube_sizes, max_success_rate, linestyle="--", label="Maximum Possible Detection", color="orange")
plt.xlabel("Cube Size (Voxels per Side)", fontsize=12)
plt.ylabel("Success Rate (Fraction of Patients)", fontsize=12)
plt.ylim(0, 1)
plt.xlim(1)  # Set x-axis lower limit to 1
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.title("Patient Detection Success vs. Cube Size", fontsize=14, fontweight="bold")
plt.tight_layout()

save_dir = os.path.join("inference_output_last", "figures")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "patient_detection_success_vs_cube_size_mc.png" if use_mc else "patient_detection_success_vs_cube_size.png")
plt.savefig(save_path)
plt.show()
# %%

tpr_values = [pdata["metrics"]["TPR"] for pid, pdata in all_patients_metrics.items()]
precision_values = [pdata["metrics"]["Precision"] for pid, pdata in all_patients_metrics.items()]

tpr_values = np.array(tpr_values)  # shape (n_patients,)
precision_values = np.array(precision_values)  # shape (n_patients,)

tpr_percent = tpr_values * 100  
precision_percent = precision_values * 100  

thresholds = np.linspace(0, 100, 201)
tpr_success_fraction = [(tpr_percent >= th).mean() for th in thresholds]
precision_success_fraction = [(precision_percent >= th).mean() for th in thresholds]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, tpr_success_fraction, linestyle="-", color="green", label="TPR Distribution")
plt.plot(thresholds, precision_success_fraction, linestyle="--", color="red", label="Precision Distribution")
plt.xlabel("Threshold (%)", fontsize=12)
plt.ylabel("Fraction of Patients ≥ Threshold", fontsize=12)
plt.ylim(0, 1)
plt.xlim(0, 100)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.title("Fraction of Patients vs. Threshold (TPR and Precision)", fontsize=14, fontweight="bold")
plt.tight_layout()

save_path = os.path.join("inference_output_last", "figures", "tpr_prec_thresholds_mc.png" if use_mc else "tpr_prec_thresholds.png")
plt.savefig(save_path)
plt.show()
# %%
save_path = os.path.join("inference_output_last", "figures", "all_patients_metrics_mc.json" if use_mc else "all_patients_metrics.json")
with open(save_path, 'w') as f:
    json.dump(numpy_to_list(all_patients_metrics), f, indent=4)
# %%

# TODO
# overlay mc and no-mc results
# load json files and convert lists to numpy arrays
