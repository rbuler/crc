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


# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
explicit_ids_test = [31, 32, 47, 54, 73, 78, 109, 197, 204]
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

paths = [
    "best_model_2ac9be11-cdb3-4c9e-ba0b-d9f5d2ce1ffa.pth",  # 1
    "best_model_cda91d49-fea9-48fc-b286-8114a1ed2fdf.pth",  # 2
    "best_model_333520f1-f991-4441-9c86-133d257d5f88.pth",  # 3
    "best_model_86f2cf62-ee7e-4350-9abf-a96e86eaf37e.pth",  # 4
    "best_model_a42ca83a-01ed-43b6-a0a6-f6d9b5aa161f.pth",  # 5
    "best_model_71da9539-db96-471f-9ac0-b852f01c2bb5.pth",  # 6
    "best_model_9e5e77e8-49b5-4ea7-8a08-21435bbf55ac.pth",  # 7
    "best_model_1cf73c94-0279-4604-96a9-31b86fe290ff.pth",  # 8
    "best_model_745a871f-447e-4865-b9cb-bc4712b671ab.pth",  # 9
    "best_model_92601926-7666-4642-80fc-a9787b4157d8.pth"   # 10
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
                
                metrics = evaluate_segmentation(logits, targets, num_classes=num_classes, prob_thresh=probs)
                print(f"Patient {i + 1}/{num_samples}/id={id}: "
                    f"IoU: {metrics['IoU']:.4f}, Dice: {metrics['Dice']:.4f}, "
                    f"TPR: {metrics['TPR']:.4f}, Precision: {metrics['Precision']:.4f}")

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
