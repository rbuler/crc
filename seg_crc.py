# %%
import os
import sys
import ast
import uuid
import json
import yaml
import time
import torch
import utils
import random
import neptune
import logging
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

import monai.transforms as mt
from monai.losses import TverskyLoss, FocalLoss
from monai.inferers import SlidingWindowInferer
# from pytorch3dunet.unet3d.model import UNet3D
from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from utils import evaluate_segmentation
from dataset import CRCDataset_seg

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
num_epochs = config['training']['epochs']
patience = config['training']['patience']
lr = config['training']['lr']
optimizer = config['training']['optimizer']

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if config['neptune']:
    run = neptune.init_run(project="ProjektMMG/CRC")
    run["parameters/config"] = config
    run["sys/group_tags"].add(["Seg3D"])
else:
    run = None

# %%
class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, beta=0.75, gamma=2.0):
        super(HybridLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, logits, targets):
        tversky_loss = self.tversky_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        return tversky_loss + focal_loss

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, beta=0.75, gamma=2.0):
        super(FocalTverskyLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
        self.gamma = gamma

    def forward(self, logits, targets):
        tversky_loss = self.tversky_loss(logits, targets)
        return torch.pow(tversky_loss, self.gamma)

# %%
train_transforms = mt.Compose([
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
    mt.RandRotated(keys=["image", "mask"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.3),
    mt.RandAffined(keys=["image", "mask"], prob=0.3, scale_range=(0.1, 0.1, 0.1), 
                   rotate_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
    mt.RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
    mt.RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.05),
    mt.RandGaussianSmoothd(keys="image", prob=0.2, sigma_x=(0.5, 1.5)),
    mt.ToTensord(keys=["image", "mask"]),
])

val_transforms = mt.Compose([
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

if run:
    run["dataset/train_size"] = len(train_dataset)
    run["dataset/val_size"] = len(val_dataset)
    run["dataset/test_size"] = len(test_dataset)
    transform_list = [str(t.__class__.__name__) for t in train_transforms.transforms]
    run["dataset/transformations"] = json.dumps(transform_list)

def collate_fn(batch):
    img_patch, mask_patch, _, _ = zip(*batch)
    img_patch = torch.stack(img_patch)
    mask_patch = torch.stack(mask_patch)
    return img_patch, mask_patch

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# %%

# model = UNet3D(in_channels=1, out_channels=num_classes, final_sigmoid=True).to(device)
model = UNETR_PP(in_channels=1, out_channels=14,
                 img_size=tuple(patch_size),
                 depths=[3, 3, 3, 3],
                 dims=[32, 64, 128, 256], do_ds=False)



weights_path = os.path.join(config['dir']['root'], "models", "model_final_checkpoint.model")
checkpoint = torch.load(weights_path, weights_only=False, map_location=device)

model.load_state_dict(checkpoint['state_dict'], strict=False)
model.out1.conv.conv = torch.nn.Conv3d(16, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
model = model.to(device)

# criterion = HybridLoss(alpha=0.25, beta=0.75, gamma=2.0)
criterion = TverskyLoss(alpha=0.9, beta=0.1, sigmoid=True)

if optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
elif optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

if run:
    run["train/model"] = model.__class__.__name__
    run["train/loss_fn"] = criterion.__class__.__name__
    run["train/optimizer"] = optimizer.__class__.__name__

# %%
best_val_loss = float('inf')
best_val_metrics = {"IoU": 0, "Dice": 0}

best_model_path = os.path.join(config['dir']['root'], "models", f"best_model_{uuid.uuid4()}.pth")
early_stopping_counter = 0

inferer = SlidingWindowInferer(roi_size=tuple(patch_size), sw_batch_size=1, overlap=0.5, device=torch.device('cpu'))


for epoch in range(num_epochs):
    start_time = time.time()
    
    model.train()
    train_dataloader.dataset.dataset.set_mode(train_mode=True)
    total_loss = 0
    total_iou = 0
    total_dice = 0
    num_batches = len(train_dataloader)
    
    for img_patch, mask_patch in train_dataloader:
        img_patch, mask_patch = img_patch.to(device, dtype=torch.float32), mask_patch.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        img_patch = img_patch.permute(1, 0, 2, 3, 4)
        mask_patch = mask_patch.permute(1, 0, 2, 3, 4)
        # outputs, logits = model(img_patch, return_logits=True)
        logits = model(img_patch)
        
        loss = criterion(logits, mask_patch)
        metrics = evaluate_segmentation(logits, mask_patch, num_classes=num_classes)
        total_loss += loss.detach().item()
        total_iou += metrics["IoU"]
        total_dice += metrics["Dice"]
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    if run:
        run["train/loss"].log(avg_loss)
        run["train/IoU"].log(avg_iou)
        run["train/Dice"].log(avg_dice)

    model.eval()
    val_dataloader.dataset.dataset.set_mode(train_mode=False)
    val_loss = 0
    val_iou = 0
    val_dice = 0
    num_val_batches = len(val_dataloader)
    
    with torch.no_grad():
        for _, _, image, mask in val_dataloader:
            image = image.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            image = image.unsqueeze(0)
            logits = inferer(inputs=image, network=model)
            logits = logits.squeeze(0)
            metrics = evaluate_segmentation(logits, mask.to(torch.device('cpu')), num_classes=num_classes, prob_thresh=0.5)
        
            loss = criterion(logits,  mask.to(torch.device('cpu')))
            val_loss += loss.detach().item()
            val_iou += metrics["IoU"]
            val_dice += metrics["Dice"]

    avg_val_loss = val_loss / num_val_batches
    avg_val_iou = val_iou / num_val_batches
    avg_val_dice = val_dice / num_val_batches

    if run:
        run["val/loss"].log(avg_val_loss)
        run["val/IoU"].log(avg_val_iou)
        run["val/Dice"].log(avg_val_dice)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_metrics = {"IoU": avg_val_iou, "Dice": avg_val_dice}
        best_model = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    if run:
        run["val/patience_counter"] = early_stopping_counter

    if early_stopping_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")

        break

    end_time = time.time()
    epoch_time = end_time - start_time

    epoch_time_hms = time.strftime("%H:%M:%S", time.gmtime(epoch_time))
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}, Train Dice: {avg_dice:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}, "
          f"Time: {epoch_time_hms}")

print(f"Saved best model with Val Loss: {best_val_loss:.4f}, Val IoU: {best_val_metrics['IoU']:.4f}, Val Dice: {best_val_metrics['Dice']:.4f}")
torch.save(best_model, best_model_path)
if run:
    run["model_filename"] = best_model_path   

# %%
inferer = SlidingWindowInferer(roi_size=tuple(patch_size), sw_batch_size=1, overlap=0.5, device=torch.device('cpu'))

load = True
if load:
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    model.eval()

total_iou = 0
total_dice = 0
num_samples = len(test_dataloader)
probs = 0.5

with torch.no_grad():
    test_dataloader.dataset.dataset.set_mode(train_mode=False)
    for i, (_, _, image, mask) in enumerate(test_dataloader):
        image = image.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)
        image = image.unsqueeze(0)
        final_output = inferer(inputs=image, network=model)
        final_output = final_output.squeeze(0)
        metrics = evaluate_segmentation(final_output, mask.to(torch.device('cpu')), num_classes=num_classes, prob_thresh=probs)
        
        total_iou += metrics["IoU"]
        total_dice += metrics["Dice"]

avg_iou = total_iou / num_samples
avg_dice = total_dice / num_samples

if run:
    run["test/avg_IoU"] = avg_iou
    run["test/avg_Dice"] = avg_dice

print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")

            # id = val_dataloader.dataset.dataset.get_patient_id(i)
            # final_output = (final_output.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
            # mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

            # combined_output = np.zeros_like(mask)
            # combined_output[mask == 1] = 1
            # combined_output[final_output == 1] = 2
            # combined_output[(mask == 1) & (final_output == 1)] = 3
            
            # combined_output_nifti = nib.Nifti1Image(combined_output, np.eye(4))
            # nib.save(combined_output_nifti, f'temporary/final_output_{id}.nii.gz')

# %%
