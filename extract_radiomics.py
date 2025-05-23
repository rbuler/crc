# %%
import os
import yaml
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
import torch
import utils
from RadiomicsExtractor import RadiomicsExtractor
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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
# %%
def get_radiomics(images_path, masks_path, instance_masks_path, mapping_path):
    radiomics = None
    
    if config['radiomics']['action'] == 'extract':
        radiomics = extract_radiomics(images_path, masks_path, instance_masks_path, mapping_path)
    
    elif config['radiomics']['action'] == 'load':
        pth = os.path.join(config['dir']['root'], config['dir']['pkl_radiomics'])
        with open(pth, 'rb') as f:
            radiomics = pickle.load(f)
        logger.info(f"Loaded radiomics features from {pth}")

    return radiomics


def extract_radiomics(images_path, masks_path, instance_masks_path, mapping_path, transform=None):
    
    list_of_dicts = []

    for img, _, instance_mask, mapping_p in zip(images_path, masks_path, instance_masks_path, mapping_path):
        patient_id = os.path.basename(img).split('_')[0].split(' ')[0]
        patient_id =  ''.join(filter(str.isdigit, patient_id))
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
                'patient_id': patient_id,
                'instance_label': instance_label,
                'class_label': class_label
                
                # TODO
                #  
                # ADD FEATURE FOR EXTRACTING FEATURES with SPECIFIC INSTANCE LABELS
                # SO THAT THERE IS NO NEED TO EXTRACT FEATURES FOR ALL INSTANCES
                # if instance_label in instance_labels_to_be_analysed_:
                #
            }
            list_of_dicts.append(d)
    transform = None
    radiomics_extractor = RadiomicsExtractor('params.yml')

    if config['radiomics']['multiple_binWidth']['if_multi']:
        binWidths = config['radiomics']['multiple_binWidth']['binWidths']
    else:
        binWidths = []
        binWidths.append(radiomics_extractor.extractor.settings['binWidth'])

    results = []

    for binWidth in binWidths:
        radiomics_extractor.extractor.settings['binWidth'] = binWidth
        
        if config['radiomics']['mode'] in ['serial', 'parallel']:
            if config['radiomics']['mode'] == 'serial':
                extraction_results = radiomics_extractor.serial_extraction(list_of_dicts)
            elif config['radiomics']['mode'] == 'parallel':
                extraction_results = radiomics_extractor.parallel_extraction(list_of_dicts, n_processes=config['radiomics']['n_processes'])        
            results.extend(extraction_results)
        elif config['radiomics']['mode'] not in ['parallel', 'serial']:
            results = None
            raise ValueError('Invalid radiomics extraction mode')

    if results is not None:
        results = pd.DataFrame(results)
        id_cols = ['class_label', 'instance_label', 'patient_id']

        df_grouped = results.groupby(id_cols).agg(lambda x: list(x)).reset_index()
        val_cols = df_grouped.drop(columns=id_cols)
        idx_cols = df_grouped[id_cols]
        new_cols_names = []
        new_results = pd.DataFrame(idx_cols)

        for col in val_cols.columns:
            for binWidth_idx, binWidth in enumerate(binWidths):
                new_col_name = f"{col}_binWidth{binWidth}"
                new_cols_names.append(new_col_name)
                new_results[new_col_name] = val_cols[col].apply(lambda x: x[binWidth_idx])
        
        # results = new_results.T.drop_duplicates().T
        results = new_results

        if config['radiomics']['save']:
            image_types = radiomics_extractor.get_enabled_image_types()
            feature_types = radiomics_extractor.get_enabled_features()
            pth = os.path.join(config['dir']['root'], config['dir']['inf'])
            with open(pth, 'w') as file:
                current_datetime = datetime.datetime.now()
                file.write(f"Modified: {current_datetime}\n")
                file.write(yaml.dump(config))
                file.write('\n\nEnabled Image Types:\n')
                file.write('\n'.join(image_types))
                file.write('\n\nEnabled Features:\n')
                file.write('\n'.join(feature_types))
                file.write('\n\nTransforms:\n' + str(transform))
                file.write(f"\n\nBin Widths: {binWidths}")
                logger.info(f"Saved extraction details in {config['dir']['inf']}")

            with open(config['dir']['pkl_radiomics'], 'wb') as f:
                pickle.dump(results, f)
                logger.info(f"Saved radiomics features in {config['dir']['pkl_radiomics']}")

    return results
# %%
