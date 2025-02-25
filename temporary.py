# %%
import os
import re
import sys
import yaml
import time
import torch
import random
import utils
import logging
import numpy as np
import nibabel as nib
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch3dunet.unet3d.model import UNet3D
import monai.transforms as mt

from monai.losses import TverskyLoss
from monai.metrics import DiceMetric, MeanIoU
# from pytorch3dunet.unet3d.losses import DiceLoss


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

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# %%
class CRCDataset(Dataset):
    def __init__(self, root_dir: os.PathLike,
                 nii_dir: os.PathLike,
                 transforms=None,
                 patch_size: tuple = (64, 64, 64),
                 stride: int = 32,
                 num_patches_per_sample: int = 50):  
        
        self.root = root_dir
        self.nii_dir = nii_dir
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches_per_sample = num_patches_per_sample
        self.images_path = []
        self.masks_path = []
        self.instance_masks_path = []
        self.mapping_path = []
        self.transforms = transforms
        self.train_mode = False

        nii_pth = os.path.join(self.root, self.nii_dir)

        pattern = re.compile(r'^\d+a$') # take only those ##a
        for root, dirs, files in os.walk(nii_pth, topdown=False):
            for name in files:
                f = os.path.join(root, name)
                folder_name = f.split('/')[-2]
                if not pattern.match(folder_name):
                    logger.info(f"Skipping {folder_name}. Pattern does not match. {pattern}")
                    continue
                if 'labels.nii.gz' in f:
                    self.masks_path.append(f)
                elif 'instance_mask.nii.gz' in f:
                    self.instance_masks_path.append(f)
                elif 'nii.gz' in f:
                    self.images_path.append(f)
                elif 'mapping.pkl' in f:
                    self.mapping_path.append(f)

    def __len__(self):
    # todo
        return len(self.images_path)
    

    def get_patient_id(self, idx):
        patient_id = os.path.basename(self.images_path[idx]).split('_')[0].split(' ')[0]
        return ''.join(filter(str.isdigit, patient_id))


    def extract_patches(self, image, mask):
        """Extracts balanced patches from a 3D image and segmentation mask."""
        H, W, D = image.shape
        h_size, w_size, d_size = self.patch_size

        h_idxs = list(range(0, H - h_size + 1, self.stride))
        w_idxs = list(range(0, W - w_size + 1, self.stride))
        d_idxs = list(range(0, D - d_size + 1, self.stride))

        patch_candidates = []

        for h in h_idxs:
            for w in w_idxs:
                for d in d_idxs:
                    img_patch = image[h:h+h_size, w:w+w_size, d:d+d_size]
                    mask_patch = mask[h:h+h_size, w:w+w_size, d:d+d_size]
                    if torch.mean((img_patch < 0.1).float()) < 0.9:
                        patch_candidates.append((img_patch, mask_patch))

        foreground_patches = [p for p in patch_candidates if torch.any(p[1] > 0)]
        background_patches = [p for p in patch_candidates if not torch.any(p[1] > 0)]
        
        min_samples = min(len(foreground_patches), len(background_patches))
        num_samples = min(min_samples, self.num_patches_per_sample // 2)

        if len(foreground_patches) == 0:
            num_samples = min(len(background_patches), self.num_patches_per_sample // 2)
            selected_patches = random.sample(background_patches, num_samples)
        else:
            selected_foreground = random.sample(foreground_patches, num_samples)
            selected_background = random.sample(background_patches, num_samples)
            selected_patches = selected_foreground + selected_background

        return selected_patches


    def set_mode(self, train_mode):
        self.train_mode = train_mode


    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        instance_mask_path = self.instance_masks_path[idx]

        image = np.asarray(nib.load(image_path).dataobj)
        image = torch.from_numpy(image)

        mask = np.asarray(nib.load(mask_path).dataobj)
        mask = torch.from_numpy(mask)

        instance_mask = np.asarray(nib.load(instance_mask_path).dataobj)
        instance_mask = torch.from_numpy(instance_mask)

        if len(image.shape) == 4:
            image = image[:, :, 0, :]
        if len(mask.shape) == 4:
            mask = mask[:, :, 0, :]
        if len(instance_mask.shape) == 4:
            instance_mask = instance_mask[:, :, 0, :]
        # instance_mask = instance_mask.permute(1, 0, 2) # need to permute to get correct bounding boxes
        
        threshold_min = -125  # lower bound for soft tissue
        threshold_max = 225  # upper bound for soft tissue
        
# 
        image = image - threshold_min
        image[(image <= 0) & (image >= threshold_max - threshold_min)] = 0
        image = image / (threshold_max - threshold_min)

        # # assign 0 values to mask > 1
        mask[mask == 2] = 0
        mask[mask == 3] = 0
        mask[mask == 5] = 0
        mask[mask == 6] = 0
        mask[mask == 4] = 0

        # bboxes = get_3d_bounding_boxes(instance_mask, self.mapping_path[idx])
        # bboxes = get_2d_bounding_boxes(instance_mask, self.mapping_path[idx], plane='xy')

        patches = self.extract_patches(image, mask)
        # img_patch, mask_patch = random.choice(patches)
        selected_patches = random.sample(patches, 16)
        img_patch = torch.stack([p[0] for p in selected_patches])
        mask_patch = torch.stack([p[1] for p in selected_patches])
        # img_patch = img_patch.unsqueeze(0)
        # import json
        # save to json file 
        # image path, seg path, bboxes
        # file_name = image_path.replace('.nii.gz', 'boxes.json')
        # with open(file_name, 'w') as f:
        #     json.dump({'image': image_path, 'mask': mask_path, 'boxes': bboxes['boxes'].tolist(), 'labels': bboxes['labels'].tolist() }, f)
        # return image_path, mask_path, bboxes
        
        if (self.transforms is not None) and self.train_mode:
            data_to_transform = {"image": img_patch, "mask": mask_patch}
            transformed_patches = self.transforms(data_to_transform)
            img_patch, mask_patch = transformed_patches["image"], transformed_patches["mask"]

        return img_patch, mask_patch, image, mask
        # return image, mask , bboxes, instance_mask


def evaluate_segmentation(pred_logits, true_mask, num_classes=7):

    pred_probs = torch.sigmoid(pred_logits) if num_classes == 1 else torch.softmax(pred_logits, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1) if num_classes > 1 else (pred_probs > 0.5).long()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    mean_iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

    dice_metric(y_pred=pred_labels, y=true_mask)
    mean_iou_metric(y_pred=pred_labels, y=true_mask)

    mean_dice = dice_metric.aggregate().item()
    mean_iou = mean_iou_metric.aggregate().item()

    dice_metric.reset()
    mean_iou_metric.reset()

    return {
        "IoU": mean_iou,
        "Dice": mean_dice,
    }
# %%
transforms = mt.Compose([
    mt.RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
    mt.RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    mt.RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
    mt.RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
    mt.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
    mt.RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.9, 1.1)),
    mt.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
    mt.RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
    mt.Rand3DElasticd(keys=["image", "mask"], prob=0.2, sigma_range=(1, 2), magnitude_range=(1, 3)),
    mt.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    # mt.ToTensord(keys=["image", "mask"])
])

patch_size = (96, 96, 32)  # (H, W, D)
stride = 32
batch_size = 1

dataset = CRCDataset(root_dir=config['dir']['root'],
                        nii_dir=config['dir']['nii_images'],
                        transforms=transforms,
                        patch_size=patch_size,
                        stride=stride,
                        num_patches_per_sample=100)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
def collate_fn(batch):
    img_patch, mask_patch, _, _ = zip(*batch)
    img_patch = torch.stack(img_patch)
    mask_patch = torch.stack(mask_patch)
    return img_patch, mask_patch


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

num_classes = 1
model = UNet3D(in_channels=1, out_channels=num_classes, final_sigmoid=True).to(device)
criterion = TverskyLoss(alpha=0.25, beta=0.75, softmax=False, sigmoid=True, to_onehot_y=False)

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# %%
num_epochs = 2000
best_val_loss = float('inf')
best_val_metrics = {"IoU": 0, "Dice": 0}
best_model_path = "best_model.pth"
patience = 20
early_stopping_counter = 0

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
        outputs, logits = model(img_patch, return_logits=True)
        loss = criterion(logits, mask_patch)
        metrics = evaluate_segmentation(logits, mask_patch, num_classes=num_classes)
        total_loss += loss.item()
        total_iou += metrics["IoU"]
        total_dice += metrics["Dice"]
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    model.eval()
    val_dataloader.dataset.dataset.set_mode(train_mode=False)
    val_loss = 0
    val_iou = 0
    val_dice = 0
    num_val_batches = len(val_dataloader)
    
    with torch.no_grad():
        for img_patch, mask_patch in val_dataloader:
            img_patch, mask_patch = img_patch.to(device, dtype=torch.float32), mask_patch.to(device, dtype=torch.long)
            img_patch = img_patch.permute(1, 0, 2, 3, 4)
            mask_patch = mask_patch.permute(1, 0, 2, 3, 4)
            outputs, logits = model(img_patch, return_logits=True)
            loss = criterion(logits, mask_patch)
            metrics = evaluate_segmentation(logits, mask_patch, num_classes=num_classes)
            val_loss += loss.item()
            val_iou += metrics["IoU"]
            val_dice += metrics["Dice"]
    
    avg_val_loss = val_loss / num_val_batches
    avg_val_iou = val_iou / num_val_batches
    avg_val_dice = val_dice / num_val_batches

    if avg_val_loss < best_val_loss and (avg_val_iou > best_val_metrics["IoU"] or avg_val_dice > best_val_metrics["Dice"]):
        best_val_loss = avg_val_loss
        best_val_metrics = {"IoU": avg_val_iou, "Dice": avg_val_dice}
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with Val Loss: {best_val_loss:.4f}, Val IoU: {best_val_metrics['IoU']:.4f}, Val Dice: {best_val_metrics['Dice']:.4f}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

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

# %%
# inference test with sliding window

def sliding_window_inference(image, model, patch_size, stride, device):
    model.eval()
    _, H, W, D = image.shape
    h_size, w_size, d_size = patch_size

    h_idxs = list(range(0, H - h_size + 1, stride))
    w_idxs = list(range(0, W - w_size + 1, stride))
    d_idxs = list(range(0, D - d_size + 1, stride))

    output = torch.zeros((1, H, W, D), device=device)
    count_map = torch.zeros((1, H, W, D), device=device)

    with torch.no_grad():
        for h in h_idxs:
            for w in w_idxs:
                for d in d_idxs:
                    img_patch = image[:, h:h+h_size, w:w+w_size, d:d+d_size].to(device, dtype=torch.float32)
                    img_patch = img_patch.unsqueeze(0)
                    _, logits = model(img_patch, return_logits=True)
                    output[:, h:h+h_size, w:w+w_size, d:d+d_size] += logits.squeeze(0)
                    count_map[:, h:h+h_size, w:w+w_size, d:d+d_size] += 1

    output /= count_map
    return output

model.eval()
test_dataloader.dataset.dataset.set_mode(train_mode=False)
for _, _, image, mask in test_dataloader:
    image = image.to(device, dtype=torch.float32)
    pred_logits = sliding_window_inference(image, model, patch_size=patch_size,  stride=stride, device=device)
    metrics = evaluate_segmentation(pred_logits, mask.to(device, dtype=torch.long), num_classes=num_classes)
    print(f"Test IoU: {metrics['IoU']:.4f}, Test Dice: {metrics['Dice']:.4f}")

# draw 3d prediction mask with ground truth mask



# %%

# x = dataset[22]
# image = x[0]  # Shape (512, 512, 264)
# mask = x[1]   # Shape (512, 512, 264)
# bboxes = x[2]["boxes"]
# labels = x[2]["labels"]
# instance_mask = x[3].type(torch.uint8)
# mask = instance_mask

# threshold_min = -125  # Lower bound for soft tissue
# threshold_max = 225  # Upper bound for soft tissue
# image = torch.clamp(image, threshold_min, threshold_max)
# image = image - threshold_min  # Shift to start from 0
# image = image / (threshold_max - threshold_min)


# sum_per_slice = (mask > 0).sum(dim=(0, 1))
# max_slice_idx = sum_per_slice.argmax().item()

# image_slice = image[:, :, max_slice_idx]
# mask_slice = mask[:, :, max_slice_idx]
# plt.figure(figsize=(8, 8))
# plt.imshow(image_slice, cmap="gray")
# plt.imshow(mask_slice, cmap="jet", alpha=0.4)



# for i, box in enumerate(bboxes):
#     y_min, x_min, z_min, y_max, x_max, z_max = box

#     if z_min <= max_slice_idx <= z_max:
#         print(box)
#         rect = plt.Rectangle(
#             (x_min, y_min),
#             x_max - x_min,
#             y_max - y_min,
#             linewidth=2,
#             edgecolor="red",
#             facecolor="none",
#         )
#         plt.gca().add_patch(rect)
#         plt.text(
#             x_min,
#             y_min - 5,
#             f"Label: {labels[i]}",
#             color="red",
#             fontsize=12,
#             bbox=dict(facecolor="white", alpha=0.5),
#         )

# # draw line at x min and y min
# plt.axhline(y=y_min, color='r', linestyle='--')
# plt.axvline(x=x_min, color='g', linestyle='--')
# plt.axhline(y=y_max, color='b', linestyle='--')
# plt.axvline(x=x_max, color='c', linestyle='--')


# plt.title(f"Slice {max_slice_idx} with Mask and 2D Bounding Boxes")
# plt.axis("off")
# plt.show()

# # %%
# x = dataset[0]
# image = x[0]  # Shape (patch_, patch_, patch_)
# mask = x[1]   # Shape (patch_, patch_, patch_)

# mid_slice = image.shape[2] // 2
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(image[:, :, mid_slice], cmap="gray")
# axes[0].imshow(mask[:, :, mid_slice], cmap="jet", alpha=0.25)
# axes[0].set_title("Axial View (Top-Down)")

# axes[1].imshow(image[:, mid_slice, :], cmap="gray")
# axes[1].imshow(mask[:, mid_slice, :], cmap="jet", alpha=0.25)
# axes[1].set_title("Coronal View (Front)")

# axes[2].imshow(image[mid_slice, :, :], cmap="gray")
# axes[2].imshow(mask[mid_slice, :, :], cmap="jet", alpha=0.25)
# axes[2].set_title("Sagittal View (Side)")

# for ax in axes:
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

# %%
