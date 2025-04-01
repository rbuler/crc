import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure, binary_fill_holes
import nibabel as nib
import os, re, glob, json, warnings, yaml, logging
import numpy as np
import torch
import utils
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
from dataset import CRCDataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
additional_test_files = remaining_data[-5:]
test_files.extend(additional_test_files)
remaining_data = remaining_data[:-5]
train_files, val_files = train_test_split(remaining_data, test_size=0.1, random_state=seed)



for iter in range(len(train_files)):
    image = train_files[iter]['image']
    image = nib.load(image).get_fdata()

    BODY_THRESHOLD = -300
    body_mask = image > BODY_THRESHOLD

    LUNG_THRESHOLD = -500
    air_mask = image < LUNG_THRESHOLD

    structure = generate_binary_structure(2, 2)
    for i in range(body_mask.shape[-1]):
        body_mask[:,:,i] = body_mask[:,:,i]
        body_mask[:,:,i] = binary_opening(body_mask[:,:,i], structure=structure, iterations=4)
        body_mask[:,:,i] = binary_fill_holes(body_mask[:,:,i])

    air_inside_body_mask = np.zeros_like(body_mask, dtype=np.uint8)

    # Loop over each slice and create air mask inside the body mask
    for i in range(body_mask.shape[-1]):
        body_slice = body_mask[:, :, i]
        air_slice = air_mask[:, :, i]
        air_inside_body = np.logical_and(body_slice, air_slice)
        air_inside_body_mask[:, :, i] = air_inside_body.astype(np.uint8)

    # for i in range(air_inside_body_mask.shape[-1]):
    #     plt.figure()
    #     plt.imshow(air_inside_body_mask[:, :, i], cmap='gray')
    #     plt.title(f"Air Inside Body - Slice {i}")
    #     plt.axis('off')
    #     plt.show()

    air_per_slice = np.array([np.sum(air_inside_body_mask[:, :, i]) for i in range(air_inside_body_mask.shape[-1])])
    window_size = 10
    smoothed_air_per_slice = np.convolve(air_per_slice, np.ones(window_size)/window_size, mode='same')
    air_per_slice_derivative = np.diff(smoothed_air_per_slice)

    air_per_slice = air_per_slice / np.max(air_per_slice)
    air_per_slice_derivative = air_per_slice_derivative / np.max(air_per_slice_derivative)
    smoothed_air_per_slice = smoothed_air_per_slice / np.max(smoothed_air_per_slice)


    nifti_mask = nib.load(train_files[iter]['mask'])
    mask = nifti_mask.get_fdata()
    mask = np.squeeze(mask) if len(mask.shape) == 4 else mask
    mask = mask > 0
    mask_slices_sum = np.sum(mask, axis=(0, 1))[::-1]


    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(air_per_slice)), air_per_slice, label='Air Pixels per Slice', color='blue')
    plt.plot(np.arange(1, len(air_per_slice_derivative) + 1), air_per_slice_derivative, label='Derivative of Air Pixels', color='red')
    plt.plot(np.arange(1, len(smoothed_air_per_slice) + 1), smoothed_air_per_slice, label='Smoothed Air Pixels per Slice', color='black')
    plt.plot(np.arange(len(mask_slices_sum)), mask_slices_sum / np.max(mask_slices_sum), label='Mask Pixels per Slice', color='orange')
    plt.vlines(np.argmax(air_per_slice_derivative), 0, 1, colors='green', linestyles='dashed', label='Max Descend Index')
    plt.title('Air Pixels per Slice and Its Derivative')
    plt.xlabel('Slice Index')
    plt.ylabel('Pixel Count / Derivative')
    plt.legend()
    plt.grid(True)
    plt.show()
    break
