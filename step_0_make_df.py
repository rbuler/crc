import os
import re
import pandas as pd
import logging
import yaml
import utils
import nibabel as nib
import numpy as np
from preprocess_images import process_images, extract_largest_body_mask
from preprocess_images import convert_dicom_to_nii
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# %%


# %%
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

root_path = config['dir']['root']
# %%
nii_pth = os.path.join(root_path, config['dir']['nii_images'])
masks_path = []
cut_mask_path = []
cut_images_path = []
cut_filtered_image_path = []
cut_filtered_bodyMask = []
instance_masks_path = []
images_path = []
mapping_path = []
ids = []
pattern = re.compile(r'^\d+a$') # take only those ##a

for root, dirs, files in os.walk(nii_pth, topdown=False):
    for name in files:
        f = os.path.join(root, name)
        folder_name = f.split('/')[-2]
        if not pattern.match(folder_name):
            print(f"Skipping {folder_name}. Pattern does not match. {pattern}")
            logger.info(f"Skipping {folder_name}. Pattern does not match. {pattern}")
            continue
        else:
            if folder_name not in ids:
                ids.append(folder_name)
                logger.info(f"Processing {folder_name}. Pattern matches. {pattern}")
            else:
                logger.info(f"Skipping {folder_name}. Already processed.")
        if 'labels.nii.gz' in f:
            masks_path.append(f)
        elif 'nii.gz' in f:
            images_path.append(f)


data = []
for i in ids:
    numeric_id = re.sub(r'[a-zA-Z]', '', i)
    data.append({
        "id": numeric_id,
        "images_path": next((p for p in images_path if f"/{i}/" in p), None),
        "masks_path": next((p for p in masks_path if f"/{i}/" in p), None),
    })

df = pd.DataFrame(data).sort_values(by='id', key=lambda x: x.astype(int)).reset_index(drop=True)
# drop row if all paths are missing (.imf file extension)
df = df.dropna(how='all', subset=df.columns[1:])
stair_step_artifact_ids = [1, 19, 98]
slice_thickness_ids = [97, 128, 137]  # slice thickness > 5mm
df = df[~df['id'].astype(int).isin(stair_step_artifact_ids + slice_thickness_ids)]
# %%
paths_u = config['dir']['processed_images_u']
results = process_images(df['images_path'].values, df['masks_path'].values, window_size=10)

# save new images and masks after cutting the slices
# so image now is slice[slice_index_to_cut:]
# and mask is mask[slice_index_to_cut:]
for result in results:
    nifti_image = nib.load(result["image_path"])
    orientation = nib.aff2axcodes(nifti_image.affine)
    logger.info(f"Image orientation: {orientation}")
    image_hu = nifti_image.get_fdata()
    image_hu = np.squeeze(image_hu) if len(image_hu.shape) == 4 else image_hu
    image_hu = image_hu[:, :, :-result['slice_index_to_cut']]

    body_mask = extract_largest_body_mask(image_hu, threshold=-400)
    new_image = image_hu.copy()
    new_image[body_mask == 0] = -1024.0
    filtered_image = nib.Nifti1Image(new_image, nifti_image.affine)
    body_mask = nib.Nifti1Image(body_mask, nifti_image.affine)

    nifti_mask = nib.load(result["mask_path"])
    orientation = nib.aff2axcodes(nifti_mask.affine)
    logger.info(f"Image orientation: {orientation}")
    mask = nifti_mask.get_fdata()
    mask = np.squeeze(mask) if len(mask.shape) == 4 else mask
    mask = mask[:, :, :-result['slice_index_to_cut']]
    mask = nib.Nifti1Image(mask, nifti_mask.affine)

    id = result['image_path'].split('/')[-2]
    id_numeric = re.sub(r'[a-zA-Z]', '', id)
    savepath = os.path.join(root_path, paths_u, id)
    os.makedirs(savepath, exist_ok=True)
    
    filtered_image_path = os.path.join(savepath, f"{id}_cut_filtered.nii.gz")
    body_mask_path = os.path.join(savepath, f"{id}_cut_body_mask.nii.gz")
    cut_mask_path = os.path.join(savepath, f"{id}_cut_mask.nii.gz")
    
    nib.save(filtered_image, filtered_image_path)
    nib.save(body_mask, body_mask_path)
    nib.save(mask, cut_mask_path)

    # df is currently id and paths to images and masks
    # add paths to df for corresponding id
    # the paths of filtered image, body mask and cut mask
    df.loc[df['id'] == id_numeric, 'cut_filtered_image_path'] = filtered_image_path
    df.loc[df['id'] == id_numeric, 'cut_filtered_bodyMask_path'] = body_mask_path
    df.loc[df['id'] == id_numeric, 'cut_mask_path'] = cut_mask_path

# %%
convert = False
if convert:
    convert_dicom_to_nii(os.path.join(root_path, config["dir"]["dcm_healthy"]),
                        os.path.join(root_path, config["dir"]["nii_healthy"]))

healthy_people_data = []
healthy_people_path = os.path.join(root_path, config["dir"]["nii_healthy"])

for root, dirs, files in os.walk(healthy_people_path, topdown=False):
    for name in dirs:
        folder_path = os.path.join(root, name)
        if not pattern.match(name):
            logger.info(f"Skipping {name}. Pattern does not match.")
            continue

        logger.info(f"Processing {name}. Pattern matches.")
        nii_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
        for nii_file in nii_files:
            healthy_people_data.append({
                "id": re.sub(r'[a-zA-Z]', '', name),
                "images_path": nii_file
            })
# 73a and 86a are missing
healthy_people_df = pd.DataFrame(healthy_people_data).sort_values(by='id', key=lambda x: x.astype(int)).reset_index(drop=True)
paths_h = config['dir']['processed_images_h']
results = process_images(healthy_people_df['path'].values, None, window_size=10)

for result in results:
    nifti_image = nib.load(result["image_path"])
    orientation = nib.aff2axcodes(nifti_image.affine)
    logger.info(f"Image orientation: {orientation}")
    image_hu = nifti_image.get_fdata()
    image_hu = np.squeeze(image_hu) if len(image_hu.shape) == 4 else image_hu
    image_hu = image_hu[:, :, :-result['slice_index_to_cut']]

    body_mask = extract_largest_body_mask(image_hu, threshold=-400)
    new_image = image_hu.copy()
    new_image[body_mask == 0] = -1024.0
    filtered_image = nib.Nifti1Image(new_image, nifti_image.affine)
    body_mask = nib.Nifti1Image(body_mask, nifti_image.affine)

    id = result['image_path'].split('/')[-2]
    id_numeric = re.sub(r'[a-zA-Z]', '', id)
    savepath = os.path.join(root_path, paths_h, id)
    os.makedirs(savepath, exist_ok=True)
    
    filtered_image_path = os.path.join(savepath, f"{id}_cut_filtered.nii.gz")
    body_mask_path = os.path.join(savepath, f"{id}_cut_body_mask.nii.gz")
    
    nib.save(filtered_image, filtered_image_path)
    nib.save(body_mask, body_mask_path)

    healthy_people_df.loc[healthy_people_df['id'] == id_numeric, 'cut_filtered_image_path'] = filtered_image_path
    healthy_people_df.loc[healthy_people_df['id'] == id_numeric, 'cut_filtered_bodyMask_path'] = body_mask_path

# %%
# save dataframes to pkl
df.to_pickle(os.path.join(root_path, config['dir']['processed'], 'unhealthy_df.pkl'))
healthy_people_df.to_pickle(os.path.join(root_path, config['dir']['processed'], 'healthy_df.pkl'))

# %%
