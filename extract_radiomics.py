# %%
import os
import yaml
import pickle
import logging
import datetime
import numpy as np
import torch
import utils
from RadiomicsExtractor import RadiomicsExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

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

mode = config['radiomics']['mode']
extract = config['radiomics']['extract']
save_radiomics = config['radiomics']['save']
n_processes = config['radiomics']['n_processes']
# %%
images_path = []
masks_path = []
instance_masks_path = []
mapping_path = []
root = '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version'
for root, dirs, files in os.walk(root, topdown=False):
    for name in files:
        f = os.path.join(root, name)
        if 'labels.nii.gz' in f:
            masks_path.append(f)
        elif 'instance_mask.nii.gz' in f:
            instance_masks_path.append(f)
        elif 'nii.gz' in f:
            images_path.append(f)
        elif 'mapping.pkl' in f:
            mapping_path.append(f)

list_of_dicts = []

for img, mask, instance_mask, mapping_p in zip(images_path, masks_path, instance_masks_path, mapping_path):
    patient_id = os.path.basename(img).split('_')[0].split(' ')[0]
   
    mapping = {}
    instance_to_class = {}
    instance_counter = 0

    with open(mapping_p, 'rb') as f:
        mapping = pickle.load(f)
        for info in mapping.values():
            if info['class_label'] != 0:  # Skip background
                for instance_label in info['instance_labels']:
                    instance_to_class[instance_counter] = (instance_label, info['class_label'])
                    instance_counter += 1
        for instance_label, class_label in instance_to_class.values():
            d = {
                'image': img,
                'segmentation': instance_mask,
                'label': instance_label,
                'class_label': class_label,
                'patient_id': patient_id
            }
            list_of_dicts.append(d)


transforms = None
radiomics_extractor = RadiomicsExtractor('params.yml')

if extract and mode in ['serial', 'parallel']:
    if mode == 'serial':
        results = radiomics_extractor.serial_extraction(list_of_dicts)
    elif mode == 'parallel':
        results = radiomics_extractor.parallell_extraction(list_of_dicts, n_processes=n_processes)
    
    if save_radiomics:
        with open(config['dir']['pkl_radiomics'], 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Saved radiomics features in {config['dir']['pkl_radiomics']}")
        image_types = radiomics_extractor.get_enabled_image_types()
        feature_types = radiomics_extractor.get_enabled_features()
        with open(config['dir']['inf'], 'w') as file:
            current_datetime = datetime.datetime.now()
            file.write(f"Modified: {current_datetime}\n")
            file.write(yaml.dump(config))
            file.write('\n\nEnabled Image Types:\n')
            file.write('\n'.join(image_types))
            file.write('\n\nEnabled Features:\n')
            file.write('\n'.join(feature_types))
            file.write('\n\nTransforms:\n' + str(transforms))
            logger.info(f"Saved extraction details in {config['dir']['inf']}")
elif extract and (mode not in ['parallel', 'serial']):
    raise ValueError('Invalid mode')
else:
    results = None
# %%
