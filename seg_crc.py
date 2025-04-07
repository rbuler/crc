# %%
import os
import sys
import ast
import json
import yaml
import torch
import utils
import random
import neptune
import logging
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import monai.transforms as mt
from monai.losses import TverskyLoss, FocalLoss, DiceCELoss, DiceFocalLoss, DiceLoss
from monai.inferers import SlidingWindowInferer
from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from monai.networks.nets import UNet
from dataset_seg import CRCDataset_seg
from net_utils import train_net, test_net

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
root = config['dir']['root']
num_classes = config['training']['num_classes']
patch_size = ast.literal_eval(config['training']['patch_size'])
stride = config['training']['stride']
batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']
num_epochs = config['training']['epochs']
patience = config['training']['patience']
lr = config['training']['lr']
weight_decay = config['training']['wd']
optimizer = config['training']['optimizer']
alpha = config['training']['loss_alpha']
beta = config['training']['loss_beta']
pos_weight = config['training']['pos_weight']
if pos_weight == 'None':
    pos_weight = None
loss_fn = config['training']['loss_fn']
mode = config['training']['mode']
fold = config['fold']

if config['neptune']:
    run = neptune.init_run(project="ProjektMMG/CRC")
    run["parameters/config"] = config
    run["sys/group_tags"].add([mode])
else:
    run = None

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
        elif self.loss_fn == "tversky":
            return TverskyLoss(alpha=self.alpha, beta=self.beta, sigmoid=True)
        elif self.loss_fn == "dicece":
            if self.weight is None:
                return DiceCELoss(sigmoid=True)
            else:
                pos_weight = torch.tensor([self.weight]).to(self.device) if self.device else None
                return DiceCELoss(sigmoid=True, weight=pos_weight)
        elif self.loss_fn == "dice":
            return DiceLoss(sigmoid=True)
        elif self.loss_fn == "dicefocal":
            return DiceFocalLoss(sigmoid=True, gamma=self.gamma)
        elif self.loss_fn == "focal":
            return FocalLoss(gamma=self.gamma)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    class HybridLoss(torch.nn.Module):
        def __init__(self, alpha, beta, gamma):
            super().__init__()
            self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, sigmoid=True)
            self.focal_loss = FocalLoss(gamma=gamma)

        def forward(self, logits, targets):
            tversky_loss = self.tversky_loss(logits, targets)
            focal_loss = self.focal_loss(logits, targets)
            return tversky_loss + focal_loss


def get_optimizer(optimizer_name, model_params, lr, weight_decay):
    if optimizer_name == "adam":
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# %%
train_transforms = mt.Compose([
    # center crop if mode 2d
    mt.CenterSpatialCropd(keys=["image", "mask"], roi_size=patch_size) if mode == '2d' else mt.Lambda(lambda x: x),
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2) if mode == '3d' else mt.Lambda(lambda x: x),
    mt.RandGaussianNoised(keys=['image'], prob=0.25, mean=0.0, std=0.01),
    mt.RandShiftIntensityd(keys=['image'], offsets=0.05, prob=0.25),
    mt.RandStdShiftIntensityd(keys=['image'], factors=0.05, prob=0.25),
    mt.RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.25),
    mt.RandScaleIntensityFixedMeand(keys=['image'], factors=0.05, prob=0.25),
    mt.RandGaussianSmoothd(keys=['image'], sigma_x=(0.25, .5), sigma_y=(0.25, .5), sigma_z=(0.25, .5), prob=0.25),
    mt.ToTensord(keys=["image", "mask"]),
])

val_transforms = mt.Compose([
    mt.CenterSpatialCropd(keys=["image", "mask"], roi_size=patch_size) if mode == '2d' else mt.Lambda(lambda x: x),
    mt.ToTensord(keys=["image", "mask"]),
])

transforms = [train_transforms, val_transforms]
# %%
dataset = CRCDataset_seg(root_dir=root,
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
explicit_ids_test = [1, 21, 57, 4, 40, 138, 17, 102, 180, 6, 199, 46, 59,  31, 32, 47, 54, 73, 78, 109, 197, 204]
ids_train_val_test = list(set(ids) - set(explicit_ids_test))
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
folds = list(kf.split(ids_train_val_test))

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    if fold_idx + 1 != fold:
        continue

    train_ids = [ids_train_val_test[i] for i in train_idx]
    val_ids = [ids_train_val_test[i] for i in val_idx]
    test_ids = explicit_ids_test

    train_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in train_ids])
    val_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in val_ids])
    test_dataset = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if int(dataset.get_patient_id(i)) in test_ids])

if run:
    run["dataset/train_size"] = len(train_dataset)
    run["dataset/val_size"] = len(val_dataset)
    run["dataset/test_size"] = len(test_dataset)
    transform_list = [str(t.__class__.__name__) for t in train_transforms.transforms]
    run["dataset/transformations"] = json.dumps(transform_list)
    run["dataset/val_ids"] = str(val_ids)
    run["dataset/test_ids"] = str(test_ids)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

# %%
if mode == '2d':
    model = UNet(spatial_dims=2,
                 in_channels=1,
                 out_channels=1,
                 channels=(4, 8, 16, 32, 64),
                 strides=(2, 2, 2, 2),
        )

if mode == '3d':
    model = UNETR_PP(in_channels=1, out_channels=14,
                    img_size=tuple(patch_size),
                    depths=[3, 3, 3, 3],
                    dims=[32, 64, 128, 256], do_ds=False)
    weights_path = os.path.join(root, "models", "model_final_checkpoint.model")
    checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.out1.conv.conv = torch.nn.Conv3d(16, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))

model = model.to(device)

# %%
loss_factory = LossFn(loss_fn=loss_fn, alpha=alpha, beta=beta, weight=pos_weight, gamma=2.0, device=device)
criterion = loss_factory.get_loss()
optimizer = get_optimizer(optimizer, model.parameters(), lr, weight_decay)

if run:
    run["train/model"] = model.__class__.__name__
    run["train/loss_fn"] = criterion.__class__.__name__
    run["train/wd"] = weight_decay
    run["train/optimizer"] = optimizer.__class__.__name__

# %%
inferer = SlidingWindowInferer(roi_size=tuple(patch_size), sw_batch_size=1, overlap=0.5, device=torch.device('cpu'))
best_model_path = train_net(mode, root, model, criterion, optimizer, dataloaders=[train_dataloader, val_dataloader],
                            num_epochs=num_epochs, patience=patience, device=device, run=run, inferer=inferer)
test_net(mode, model, best_model_path, test_dataloader, device=device, run=run, inferer=inferer)

# %%