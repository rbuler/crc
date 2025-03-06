# %%
import os, sys, uuid, gc, re, glob, time, json, copy, warnings, yaml, logging, neptune
import numpy as np
import torch
import torch.optim as optim
import utils
from monai.utils import set_determinism
from monai.transforms import ScaleIntensityRanged
from monai.data import DataLoader, Dataset
from monai.data.box_utils import box_iou 
from monai.losses import FocalLoss
from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from sklearn.model_selection import train_test_split

from generate_transforms import generate_detection_train_transform, generate_detection_val_transform
from my_tests import check_masks_inside_boxes
from det_model import retinanet_resnet_fpn_detector
from dataset import CRCDataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if config["neptune"] and not any("ipykernel" in arg for arg in sys.argv):
    run = neptune.init_run(project="ProjektMMG/CRC")
    run["parameters/config"] = config
    run["sys/group_tags"].add(["Det3D"])
    run["sys/group_tags"].add(["val_neg_0"])
else:
    run = None


# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
set_determinism(seed=seed)


data_dir = os.path.join(config['dir']['root'], config['dir']['nii_images'])
pattern = re.compile(r'^\d+a$')  # take only those ##a
item_dirs = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if pattern.match(item)]
json_files = []
for item_dir in item_dirs:
    json_files.extend(sorted(glob.glob(os.path.join(item_dir, '**', '*boxes.json'), recursive=True)))

data_dicts = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        sample = json.load(f)
    sample['boxes'] = np.array(sample['boxes'])
    sample['labels'] = np.array(sample['labels']) - 1
    sample['id'] = json_file.split('/')[-2]
    data_dicts.append(sample)

for sample in data_dicts:
    valid_indices = sample['labels'] == 0
    sample['boxes'] = sample['boxes'][valid_indices]
    sample['labels'] = sample['labels'][valid_indices]

dataset = CRCDataset(root_dir=config['dir']['root'],
                        clinical_data_dir=config['dir']['clinical_data'],
                        nii_dir=config['dir']['nii_images'],
                        dcm_dir=config['dir']['dcm_images'],
                        config=config,
                        transform=None,
                        save_new_masks=False)

clinical_data = dataset.clinical_data
# Get IDs that have Tx or T0 in wmT column
# IDs to be moved to the test set (Tx or T0 Clinical Diagnosis)
Tx_T0_ids = clinical_data[clinical_data['wmT'].isin(['x', '0'])]['patient_id'].tolist()
Tx_T0_ids = [f"{id}a" for id in Tx_T0_ids]
print(f"IDs with Tx or T0 in wmT column: {Tx_T0_ids}")
test_ids = Tx_T0_ids
test_files = [sample for sample in data_dicts if sample['id'] in test_ids]
remaining_data = [sample for sample in data_dicts if sample['id'] not in test_ids]
additional_test_files = remaining_data[:6]
test_files.extend(additional_test_files)
remaining_data = remaining_data[6:]
train_files, val_files = train_test_split(remaining_data, test_size=0.2, random_state=seed)

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

def get_box_counts(data_files):
    box_counts = [len(sample['boxes']) for sample in data_files]
    empty_images = sum(1 for count in box_counts if count == 0)
    return box_counts, empty_images

train_box_counts, train_empty_images = get_box_counts(train_files)
val_box_counts, val_empty_images = get_box_counts(val_files)
test_box_counts, test_empty_images = get_box_counts(test_files)

print(f"Train set: {len(train_files)} images, {sum(train_box_counts)} boxes, {train_empty_images} empty images")
print(f"Val set: {len(val_files)} images, {sum(val_box_counts)} boxes, {val_empty_images} empty images")
print(f"Test set: {len(test_files)} images, {sum(test_box_counts)} boxes, {test_empty_images} empty images")
# %%
intensity_transform = ScaleIntensityRanged(
    keys=["image"],
    a_min=-200,
    a_max=500,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)
train_transforms = generate_detection_train_transform(
    image_key="image",
    box_key="boxes",
    label_key="labels",
    mask_key="mask",
    intensity_transform=intensity_transform,
    patch_size=tuple(config['training']['patch_size']),
    batch_size=config['training']['batch_size'],
)
val_transforms = generate_detection_val_transform(
    image_key="image",
    box_key="boxes",
    label_key="labels",
    mask_key="mask",
    intensity_transform=intensity_transform,
    patch_size=tuple(config['training']['patch_size']),
    batch_size=config['training']['batch_size'],
)

# val_transforms = generate_detection_val_transform(
#     image_key="image",
#     box_key="boxes",
#     label_key="labels",
#     mask_key="mask",
#     intensity_transform=intensity_transform,
# )
def detection_collate_fn(batch):
    images = []
    targets = []

    for item in batch:
        for i in range(len(item)):
            images.append(item[i]["image"])

            target = {
                "boxes": item[i]["boxes"],
                "labels": item[i]["labels"],
                # "mask": item[i]["mask"],
                # "id": item[i]["id"],
            }
            targets.append(target)

    return images, targets

train_dataset = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_dataset,
                          batch_size=1, num_workers=4,
                          collate_fn=detection_collate_fn,
                          shuffle=True)

val_dataset = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_dataset,
                        batch_size=1, num_workers=4,
                        collate_fn=detection_collate_fn,
                        shuffle=False)
# check_masks_inside_boxes(train_loader, label=1) # 197a 204a no mask
# check_masks_inside_boxes(val_loader, label=1)
# %%
# for item in train_files:
#     if item["id"] == '47a':
#         iitem = Dataset(data=[item], transform=train_transforms)[0]
#         interactive_slice_viewer(iitem, label=1)
#         print(iitem["id"])
#         print(iitem["boxes"])
#         print(iitem["labels"])
#         print(iitem["mask"].shape)
#         print(iitem["image"].shape)
#         break
# check_data = train_dataset[0]
# interactive_slice_viewer(check_data[0])
# %%
# sizes for 3 feature map levels of FPN (but len(returned_layers)+1 is the number of extracted feature maps)
sizes = (
    (32, 32, 12),   # Small
    (48, 48, 16),   # Medium-small
    (64, 64, 24),   # Medium-small
    (80, 80, 32),   # Medium-large
)

aspect_ratios = (
    ((1., 1.), (1.5, 1.), (1., 1.5), (1.5, 1.5)),
    ((1., 1.), (1.5, 1.), (1., 1.5), (1.5, 1.5)),
    ((1., 1.), (1.5, 1.), (1., 1.5), (1.5, 1.5)),
    ((1., 1.), (1.5, 1.), (1., 1.5), (1.5, 1.5)),
)

anchor_generator = AnchorGenerator(sizes, aspect_ratios, indexing="ij")

model =  retinanet_resnet_fpn_detector(
    num_classes=config['training']['num_classes'],
    n_input_channels=1,
    spatial_dims=3,
    feed_forward=False,
    shortcut_type='A',
    bias_downsample=True,
    anchor_generator=anchor_generator,
    returned_layers = (1, 2, 3),
    pretrained=True,
    progress=True,
)
# matcher for GTboxes and anchors
# If an anchor is unassigned, which may happen with overlap
# in [0.4, 0.5)
model.set_regular_matcher(fg_iou_thresh=0.5,bg_iou_thresh=0.4)
# model.set_balanced_sampler(512, 0.5)
model.set_cls_loss(FocalLoss(reduction="mean", gamma=2.0))
model.set_sliding_window_inferer(roi_size=config['training']['patch_size'],
                                 sw_batch_size=1,
                                 overlap=0.5,
                                 device="cpu")
model.set_box_selector_parameters(
    score_thresh=0.05,
    topk_candidates_per_level=100,
    nms_thresh=0.5,
    detections_per_img=1,
    apply_sigmoid=True
)
# When saving the model, only self.network contains trainable parameters and needs to be saved.
# only detector.network may contain trainable parameters.
if "cuda" in config["device"]:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)

NUM_EPOCHS = config['training']['epochs']
LR = config['training']['lr']
if config['training']['optimizer'] == 'adam':
    optimizer = optim.Adam(model.network.parameters(), lr=LR)
elif config['training']['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.network.parameters(), lr=LR)
else:
    raise NotImplementedError(f"Optimizer {config['training']['optimizer']} not implemented")

# %%
training_start_time = time.time()
best_iou = 0
best_model = None
patience_counter = 0
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    torch.cuda.empty_cache()
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        del images, targets, loss_dict, loss
        torch.cuda.empty_cache()
        gc.collect()
    avg_loss = epoch_loss / len(train_loader)
    if run:
        run["train/loss"].log(float(avg_loss))
    epoch_end_time = time.time()
    epoch_elapsed_time = epoch_end_time - epoch_start_time
    total_elapsed_time = epoch_end_time - training_start_time
    total_minutes, total_seconds = divmod(total_elapsed_time, 60)    
    
    print(f"Epoch {epoch+1:3}/{NUM_EPOCHS}, Loss: {avg_loss:.4f} ", end="")
    if (epoch+1) % 10 == 0:
        print(f"in Total {int(total_minutes):4}m {int(total_seconds):2}s ", end="")
    else:
        print(f"in Total {int(total_minutes):4}m {int(total_seconds):2}s ")
    
    if (epoch+1) % 5 == 0:
        model.eval()
        iou_scores = []
        with torch.no_grad():
            # for images, targets in val_loader:
            #     images = [img.to(device) for img in images]
            #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            #     outputs = model(input_images=images, use_inferer=False)
            for batch in val_loader:
                images, boxes, labels = batch["image"], batch["boxes"], batch["labels"]
                targets = [{"boxes": boxes[0], "labels": labels[0]}]
                images = images.to(device)
                outputs = model(input_images=images, use_inferer=True)
                pred_boxes = outputs[0]["boxes"]
                target_boxes = targets[0]["boxes"]
                if pred_boxes.shape[0] == 0 and target_boxes.shape[0] > 0:
                    iou_scores.append(0)
                elif target_boxes.shape[0] == 0:
                    iou_scores.append(0)
                else:
                    ious = box_iou(pred_boxes, target_boxes)
                    if ious.numel() > 0 and not torch.isnan(ious).any():
                        iou_scores.append(ious.max(dim=0).values.mean().item())
                del outputs, pred_boxes, target_boxes
                torch.cuda.empty_cache()
                gc.collect()
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        print(f"IoU avg: {mean_iou:.3f} min: {np.min(iou_scores):.3f} max: {np.max(iou_scores):.3f}")
        if best_iou < mean_iou:
            best_iou = mean_iou
            patience_counter = config['training']['patience'] # reset
            best_model = copy.deepcopy(model.network.state_dict())
        else:
            patience_counter = max(0, patience_counter - 1)
        if run:
            run["val/IoU"].log(mean_iou)
            run["val/patience_counter"].log(patience_counter)
        if not patience_counter:
            print(f"Early stopping at epoch {epoch+1}, best IoU: {best_iou:.3f}")
            break
        
filename = f"model_{uuid.uuid4().hex}.pth"
models_dir = os.path.join(config['dir']['root'], "models")
os.makedirs(models_dir, exist_ok=True)
filepath = os.path.join(models_dir, filename)
torch.save(best_model, filepath)
if run:
    run["model_filename"] = filepath
print(f"Model saved at {filepath}")
run.stop()
# %%
