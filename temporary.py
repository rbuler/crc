# %%
import os
import re
import sys
import ast
import uuid
import yaml
import time
import torch
import utils
import random
import neptune
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch3dunet.unet3d.model import UNet3D
import monai.transforms as mt

from monai.losses import TverskyLoss, FocalLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import SlidingWindowInferer
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


num_classes = config['training']['num_classes']
# patch_size = tuple(map(int, config['training']['patch_size']))
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

def window_and_normalize_ct(ct_image, window_center=45, window_width=400):

    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    ct_windowed = np.clip(ct_image, lower_bound, upper_bound)
    normalized = (ct_windowed - lower_bound) / (upper_bound - lower_bound)
    
    return normalized




class CRCDataset(Dataset):
    def __init__(self, root_dir: os.PathLike,
                 nii_dir: os.PathLike,
                 clinical_data_dir: os.PathLike,
                 config,
                 transforms=None,
                 patch_size: tuple = (64, 64, 64),
                 stride: int = 32,
                 num_patches_per_sample: int = 50):  
                         
               
        
        self.root = root_dir
        self.nii_dir = nii_dir
        self.clinical_data = clinical_data_dir

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


        clinical_data_dir = os.path.join(self.root, self.clinical_data)
        default_missing = pd._libs.parsers.STR_NA_VALUES
        self.clinical_data = pd.read_csv(
            clinical_data_dir,
            delimiter=';',
            encoding='utf-8',
            index_col=False,
            na_filter=True,
            na_values=default_missing)
        
        self.clinical_data.columns = self.clinical_data.columns.str.strip()
        self.clinical_data = self.clinical_data[config['clinical_data_attributes'].keys()]
        self.clinical_data.dropna(subset=['Nr pacjenta'], inplace=True)

        for column, dtype in config['clinical_data_attributes'].items():
            self.clinical_data[column] = self.clinical_data[column].astype(dtype)

        self.clinical_data = self.clinical_data.reset_index(drop=True)        
        self.clinical_data.rename(columns={'Nr pacjenta': 'patient_id'}, inplace=True)
        self._clean_tnm_clinical_data()



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
        window_center = 45
        window_width = 400
        image = window_and_normalize_ct(image,
                                        window_center=window_center,
                                        window_width=window_width)

        # # assign 0 values to mask > 1
        mask[mask == 2] = 0
        mask[mask == 3] = 0
        mask[mask == 5] = 0
        mask[mask == 6] = 0
        mask[mask == 4] = 0

        patches = self.extract_patches(image, mask)
        # img_patch, mask_patch = random.choice(patches)
        selected_patches = random.sample(patches, 8)
        img_patch = torch.stack([p[0] for p in selected_patches])
        mask_patch = torch.stack([p[1] for p in selected_patches])

        
        if (self.transforms is not None) and self.train_mode:
            data_to_transform = {"image": img_patch, "mask": mask_patch}
            transformed_patches = self.transforms[0](data_to_transform)  # train_transforms
            img_patch, mask_patch = transformed_patches["image"], transformed_patches["mask"]
        elif (self.transforms is not None) and not self.train_mode:
            data_to_transform = {"image": img_patch, "mask": mask_patch}
            transformed_patches = self.transforms[1](data_to_transform)  # val_transforms
            img_patch, mask_patch = transformed_patches["image"], transformed_patches["mask"]

        return img_patch, mask_patch, image, mask


    def _clean_tnm_clinical_data(self):

        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].notna() & (self.clinical_data['TNM wg mnie'] != '')]
        
        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.startswith('T')]
        
        self.clinical_data.dropna(how='all', axis=0, inplace=True)
        self.clinical_data.dropna(subset=['TNM wg mnie'], inplace=True)

        self.clinical_data = self.clinical_data[
            self.clinical_data['TNM wg mnie'].str.contains(r'T', regex=True) &
            self.clinical_data['TNM wg mnie'].str.contains(r'N', regex=True)
        ]


        def make_lower_case(tnm_string):
            return ''.join([char.lower() if char in ['A', 'B', 'C', 'X'] else char for char in tnm_string])

        self.clinical_data['TNM wg mnie'] = self.clinical_data['TNM wg mnie'].apply(make_lower_case)
    
        self.clinical_data['wmT'] = self.clinical_data['TNM wg mnie'].str.extract(r'T([0-9]+[a-b]?|x|is)?')
        self.clinical_data['wmN'] = self.clinical_data['TNM wg mnie'].str.extract(r'N([0-9]+[a-c]?)?')
        self.clinical_data['wmM'] = self.clinical_data['TNM wg mnie'].str.extract(r'M([0-9]+)?')



def evaluate_segmentation(pred_logits, true_mask, num_classes=7, prob_thresh=0.5):
    pred_probs = torch.sigmoid(pred_logits) if num_classes == 1 else torch.softmax(pred_logits, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1) if num_classes > 1 else (pred_probs > prob_thresh).long()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    mean_iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

    valid_pred_labels = []
    valid_true_masks = []

    for i in range(true_mask.shape[0]):
        if torch.any(true_mask[i] > 0):
            valid_pred_labels.append(pred_labels[i].unsqueeze(0))
            valid_true_masks.append(true_mask[i].unsqueeze(0))

    if valid_pred_labels:
        valid_pred_labels = torch.cat(valid_pred_labels, dim=0)
        valid_true_masks = torch.cat(valid_true_masks, dim=0)

        dice_metric(y_pred=valid_pred_labels, y=valid_true_masks)
        mean_iou_metric(y_pred=valid_pred_labels, y=valid_true_masks)

        mean_dice = dice_metric.aggregate().item()
        mean_iou = mean_iou_metric.aggregate().item()

        dice_metric.reset()
        mean_iou_metric.reset()

        return {
            "IoU": mean_iou,
            "Dice": mean_dice,
        }
    else:
        return {
            # rare case if all batch samples have no foreground pixels
            "IoU": 0.0,
            "Dice": 0.0,
        }
    
class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, beta=0.75, gamma=2.0):
        super(HybridLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
        self.focal_loss = FocalLoss()
        self.gamma = gamma

    def forward(self, logits, targets):
        tversky_loss = self.tversky_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        return tversky_loss + focal_loss

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
])

val_transforms = mt.Compose([
    mt.ToTensord(keys=["image", "mask"]),
])
transforms = [train_transforms, val_transforms]

# %%
dataset = CRCDataset(root_dir=config['dir']['root'],
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

# %%
def collate_fn(batch):
    img_patch, mask_patch, _, _ = zip(*batch)
    img_patch = torch.stack(img_patch)
    mask_patch = torch.stack(mask_patch)
    return img_patch, mask_patch

# train data set = subset of dataset with train_dataset indices
# val data set = subset of dataset with val_dataset indices
# test data set = subset of dataset with test_dataset indices


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = UNet3D(in_channels=1, out_channels=num_classes, final_sigmoid=True).to(device)
criterion = HybridLoss(alpha=0.25, beta=0.75, gamma=2.0)

if optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

if run:
    run["train/loss_fn"] = criterion.__class__.__name__
    run["train/optimizer"] = optimizer.__class__.__name__


# %%
best_val_loss = float('inf')
best_val_metrics = {"IoU": 0, "Dice": 0}

# join root from config dir root and /models
best_model_path = os.path.join(config['dir']['root'], "models", f"best_model_{uuid.uuid4()}.pth")
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
        for img_patch, mask_patch in val_dataloader:
            img_patch, mask_patch = img_patch.to(device, dtype=torch.float32), mask_patch.to(device, dtype=torch.float32)
            img_patch = img_patch.permute(1, 0, 2, 3, 4)
            mask_patch = mask_patch.permute(1, 0, 2, 3, 4)
            outputs, logits = model(img_patch, return_logits=True)
            loss = criterion(logits, mask_patch)
            metrics = evaluate_segmentation(logits, mask_patch, num_classes=num_classes)
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
        print(f"Saved best model with Val Loss: {best_val_loss:.4f}, Val IoU: {best_val_metrics['IoU']:.4f}, Val Dice: {best_val_metrics['Dice']:.4f}")
        torch.save(best_model, best_model_path)
        if run:
            run["model/best"].upload(best_model_path)
        break

    end_time = time.time()
    epoch_time = end_time - start_time

    epoch_time_hms = time.strftime("%H:%M:%S", time.gmtime(epoch_time))
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}, Train Dice: {avg_dice:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}, "
          f"Time: {epoch_time_hms}")
    

# %%
inferer = SlidingWindowInferer(roi_size=(96,96,64), sw_batch_size=1, overlap=0.5, device=torch.device('cpu'))

model = UNet3D(in_channels=1, out_channels=num_classes, final_sigmoid=True).to(device)
load = True
if load:
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    model.eval()

total_iou = 0
total_dice = 0
num_samples = len(test_dataloader)

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
