# %%
import os
import pickle
from RadiomicsExtractor import RadiomicsExtractor

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

# %%

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


radiomics_extractor = RadiomicsExtractor('params.yml')
results = radiomics_extractor.parallell_extraction(list_of_dicts, n_processes=8)
# results = radiomics_extractor.serial_extraction(list_of_dicts)

# patient_data = {}
# for result in results:
#     patient_id = result['patient_id']
#     if patient_id not in patient_data:
#         patient_data[patient_id] = []
#     patient_data[patient_id].append(result)


# %%

save_path = '/media/dysk_a/jr_buler/RJG-gumed/RJG-6_labels_version/radiomics.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(results, f)

# %%