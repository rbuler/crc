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
from monai.losses import TverskyLoss, FocalLoss, DiceCELoss, DiceFocalLoss, DiceLoss

import monai.transforms as mt
from monai.inferers import SlidingWindowInferer
# from pytorch3dunet.unet3d.model import UNet3D
from monai.networks.nets import UNet
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
mode = config['training']['mode']

alpha = config['training']['loss_alpha']
beta = config['training']['loss_beta']
pos_weight = config['training']['pos_weight']
if pos_weight == 'None':
    pos_weight = None
loss_fn = config['training']['loss_fn']
# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

class LossFn:
    def __init__(self, loss_fn, alpha=0.25, beta=0.75, weight=None, gamma=2.0, device=None):
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.gamma = gamma
        self.device = device

    def get_loss(self):
        if self.loss_fn == "hybrid":
            return self.HybridLoss(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        elif self.loss_fn == "hybrid_v2":
            return self.HybridLoss_v2()
        elif self.loss_fn == "tversky":
            return TverskyLoss(alpha=self.alpha, beta=self.beta, sigmoid=True)
        elif self.loss_fn == "dicece":
            if self.weight is None:
                return DiceCELoss(sigmoid=True, squared_pred=True)
            else:
                pos_weight = torch.tensor([self.weight]).to(self.device) if self.device else None
                return DiceCELoss(sigmoid=True, squared_pred=True, weight=pos_weight)
        elif self.loss_fn == "dice":
            return DiceLoss(sigmoid=True, squared_pred=True)
        elif self.loss_fn == "dicefocal":
            return DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=self.gamma)
        elif self.loss_fn == "focal":
            return FocalLoss(gamma=self.gamma)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    class HybridLoss(torch.nn.Module):
        def __init__(self, alpha, beta, weights=(0.7, 0.3), gamma=2):
            super().__init__()
            self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
            self.focal_loss = FocalLoss(gamma=gamma)
            self.weights = weights

        def forward(self, logits, targets):
            tversky_loss = self.tversky_loss(logits, targets)
            focal_loss = self.focal_loss(logits, targets)
            return (self.weights[0] * tversky_loss + self.weights[1] * focal_loss)

    class HybridLoss_v2(torch.nn.Module):
        def __init__(self, gamma=1.0):
            super().__init__()
            self.dicece = DiceCELoss(sigmoid=True, squared_pred=True)
            self.focal_loss = FocalLoss(gamma=gamma)

        def forward(self, logits, targets):

            dicece = self.dicece(logits, targets)
            focal = self.focal_loss(logits, targets)

            return (dicece + focal)
        

# %%
train_transforms = mt.Compose([mt.ToTensord(keys=["image", "mask"])])
val_transforms = mt.Compose([
    mt.CenterSpatialCropd(keys=["image", "mask"], roi_size=patch_size) if mode == '2d' else mt.Lambda(lambda x: x),
    mt.ToTensord(keys=["image", "mask"]),
])
transforms = [train_transforms, val_transforms]
# %%
dataset = CRCDataset_seg(root_dir=config['dir']['root'],
                         nii_dir=config['dir']['nii_images'],
                         clinical_data_dir=config['dir']['clinical_data'],
                         config=config,
                         transforms=transforms,
                         patch_size=patch_size,
                         stride=stride,
                         num_patches_per_sample=100,
                         mode=mode)

ids = []
for i in range(len(dataset)):
    ids.append(int(dataset.get_patient_id(i)))
                                                               # bad res <  #  > tx t0
# explicit_ids_test = [1, 21, 57, 4, 40, 138, 17, 102, 180, 6, 199, 46, 59,  31, 32, 47, 54, 73, 78, 109, 197, 204]
# explicit_ids_test = [31, 32, 47, 54, 73, 78, 109, 197, 204]
explicit_ids_test = [1, 19, 98, 64, 77, 173,  97, 128, 137,    31, 32, 47, 54, 78, 109, 73, 197, 204]

ids_train_val_test = list(set(ids) - set(explicit_ids_test))

SPLITS = 10

kf = KFold(n_splits=SPLITS, shuffle=True, random_state=seed)
folds = list(kf.split(ids_train_val_test))

# %%
# 2D
# best_model_c769763b-84d4-488d-a56c-abe28f1f148a.pth   # 1
# best_model_fcd998bc-0408-4c11-b513-5c5f2c6d5acb.pth   # 2
# best_model_c2a265bb-d93a-4649-b22e-0adaeaa598d7.pth   # 3
# best_model_454c2476-b016-4d18-94c1-548d9a2d6afc.pth   # 4
# best_model_14b34184-ac52-4cba-9444-0cf9de7cadae.pth   # 5
# best_model_fc9ac5b7-6e47-4c72-86ba-56f8caef0693.pth   # 6
# best_model_87268de5-ed83-4769-b038-ebe48e0bda6b.pth   # 7
# best_model_d9889b42-c83d-442b-93bc-96234eaa8dba.pth   # 8
# best_model_5884f305-eefb-42de-8766-faf7fe617ad7.pth   # 9
# best_model_95283dbf-f017-46d3-aa03-1b46a86ea5da.pth   # 10

# paths = [
#     "best_model_9e59cf21-59ed-4afd-bfd1-eca01db3054b.pth",  # 1
#     "best_model_055f8b0c-55ef-416e-8647-d590f36df813.pth",  # 2
#     "best_model_849d1d6d-b968-459f-971a-81b522deb1ec.pth",  # 3
#     "best_model_4f0a6d5c-82fc-40b3-8352-a5d8449425bc.pth",  # 4
#     "best_model_c65738af-1683-46ff-bcbb-551cb10ca2d0.pth",  # 5
#     "best_model_17ba2d78-55d2-4306-9a9c-314c096a099a.pth",  # 6
#     "best_model_bc8487ea-da2f-4abe-9830-f9cbae95e0a1.pth",  # 7
#     "best_model_a2333010-d211-408c-bafa-738c855daf30.pth",  # 8
#     "best_model_cd69bbea-88f2-4223-9651-589db3cab61c.pth",  # 9
#     "best_model_6a56b77f-4d6b-4218-9e5d-5735e59ba4d3.pth"   # 10
# ]

# paths = [
#     "best_model_2ac9be11-cdb3-4c9e-ba0b-d9f5d2ce1ffa.pth",  # 1
#     "best_model_cda91d49-fea9-48fc-b286-8114a1ed2fdf.pth",  # 2
#     "best_model_333520f1-f991-4441-9c86-133d257d5f88.pth",  # 3
#     "best_model_86f2cf62-ee7e-4350-9abf-a96e86eaf37e.pth",  # 4
#     "best_model_a42ca83a-01ed-43b6-a0a6-f6d9b5aa161f.pth",  # 5
#     "best_model_71da9539-db96-471f-9ac0-b852f01c2bb5.pth",  # 6
#     "best_model_9e5e77e8-49b5-4ea7-8a08-21435bbf55ac.pth",  # 7
#     "best_model_1cf73c94-0279-4604-96a9-31b86fe290ff.pth",  # 8
#     "best_model_745a871f-447e-4865-b9cb-bc4712b671ab.pth",  # 9
#     "best_model_92601926-7666-4642-80fc-a9787b4157d8.pth"   # 10
# ]

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


for i, path in enumerate(paths):
    fold = i+1
    weights_path = path

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx + 1 != fold:
            continue

        train_ids = [ids_train_val_test[i] for i in train_idx]
        val_ids = [ids_train_val_test[i] for i in val_idx]
        test_ids = explicit_ids_test

        train_dataset = torch.utils.data.Subset(
            dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in train_ids]
        )
        val_dataset = torch.utils.data.Subset(
            dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in val_ids]
        )
        test_dataset = torch.utils.data.Subset(
            dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in test_ids]
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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

        loss_factory = LossFn(loss_fn=loss_fn, alpha=alpha, beta=beta, weight=pos_weight, gamma=2.0, device=device)
        criterion = loss_factory.get_loss()
        # gaussian or not?
        inferer = SlidingWindowInferer(
            roi_size=tuple(patch_size), sw_batch_size=1, overlap=0.75, mode="constant", device=torch.device('cpu')
        )
        model.eval()

        total_iou, total_dice, total_tpr, total_precision = 0, 0, 0, 0
        probs = 0.5

        dataloader = val_dataloader
        dataloader.dataset.dataset.set_mode(train_mode=False)

        print(fold)
        print(val_ids)
        num_samples = len(dataloader)

        with torch.no_grad():
            for i, (_, _, image, mask, id) in enumerate(dataloader):
                if mode == '2d':
                    inputs = image.to(device, dtype=torch.float32)
                    targets = mask.to(device, dtype=torch.long)
                    inputs = inputs.permute(1, 0, 2, 3)
                    targets = targets.permute(1, 0, 2, 3)
                    logits = model(inputs)
                elif mode == '3d':
                    inputs = image.to(device, dtype=torch.float32)
                    targets = mask["mask"].to(torch.device('cpu'), dtype=torch.long)
                    body_mask = mask["body_mask"].to(torch.device('cpu'), dtype=torch.long)
                    inputs = inputs.unsqueeze(0)

                    original_depth = inputs.shape[2]
                    targets = targets.unsqueeze(0)
                    body_mask = body_mask.unsqueeze(0)
                    logits = inferer(inputs=inputs, network=model)
                    logits[body_mask == 0] = -1e10
                
                criterion = criterion.to(device=logits.device)
                loss = criterion(logits,  targets)
                metrics = evaluate_segmentation(logits, targets, num_classes=num_classes, prob_thresh=probs)
                print(f"Patient_ID: {str(id[0]):<7} | Loss: {float(loss):.4f} | IoU: {float(metrics['IoU']):.3f} | "
                      f"Dice: {float(metrics['Dice']):.3f} | TPR: {float(metrics['TPR']):.3f} | Precision: {float(metrics['Precision']):.3f}")

                final_output = torch.sigmoid(logits)
                if mode == '3d':
                    final_output = (final_output.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
                elif mode == '2d':
                    final_output = (final_output.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
                    
                
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

                total_iou += metrics["IoU"]
                total_dice += metrics["Dice"]
                total_tpr += metrics["TPR"]
                total_precision += metrics["Precision"]
            avg_iou = total_iou / num_samples
            avg_dice = total_dice / num_samples
            avg_tpr = total_tpr / num_samples
            avg_precision = total_precision / num_samples

            print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}, "
                f"Average TPR: {avg_tpr:.4f}, Average Precision: {avg_precision:.4f}")
# %%
