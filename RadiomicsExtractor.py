import time
import logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import multiprocessing
# import SimpleITK as sitk
from multiprocessing import Pool
from tqdm import tqdm, trange
from utils import pretty_dict_str
from radiomics import featureextractor


logger = logging.getLogger(__name__)


class RadiomicsExtractor():
    """
    A class for extracting radiomics features from medical (gray scale) images.
    Only 2D images are supported. Only single label segmentations are supported.

    Args:
        param_file (str): The path to the parameter file used by the RadiomicsFeatureExtractor.
        transforms (optional): A transformation function or pipeline to apply to the images and segmentations before feature extraction.

    Attributes:
        extractor (RadiomicsFeatureExtractor): The RadiomicsFeatureExtractor object used for feature extraction.
        transforms (optional): The transformation function or pipeline applied to the images and segmentations.

    Methods:
        extract_radiomics: Extracts radiomics features from an image and its corresponding segmentation.
        parallell_extraction: Performs parallel extraction of radiomics features from a list of images.
        serial_extraction: Performs serial extraction of radiomics features from a list of images.
    """

    def  __init__(self, param_file: str, transforms=None):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
        msg = "\n\nEnabled Image Types:"
        msg += pretty_dict_str(self.extractor.enabledImagetypes, key_only=True)
        msg += "\n\nEnabled Features:"
        msg += pretty_dict_str(self.extractor.enabledFeatures, key_only=True)
        logger.info(msg)
        self.transforms = transforms
        if self.transforms:
            logger.info(f"Transforms: {self.transforms}")
    
    def get_enabled_image_types(self):
        return list(self.extractor.enabledImagetypes.keys())
    
    def get_enabled_features(self):
        return list(self.extractor.enabledFeatures.keys())

    def extract_radiomics(self, d:dict):
        image = d['image']
        segmentation = d['segmentation']
        instance_to_class = d['instance_to_class']
        patient_id = d['patient_id']


        image = np.asarray(nib.load(image).dataobj)
        segmentation = np.asarray(nib.load(segmentation).dataobj)

        if len(image.shape) == 3:
            _, _, _ = image.shape
        elif len(image.shape) == 4: 
            image = image[:, :, 0, :]

        if len(segmentation.shape) == 3:
            _, _, _ = segmentation.shape
        elif len(segmentation.shape) == 4:
            segmentation = segmentation[:, :, 0, :]

        image = sitk.GetImageFromArray(image)
        segmentation = sitk.GetImageFromArray(segmentation)

        all_features = []
        for instance_label, class_label in instance_to_class.values():

            # TODO 
            # ADD FEATURE FOR EXTRACTING FEATURES with SPECIFIC INSTANCE LABELS
            # SO THAT THERE IS NO NEED TO EXTRACT FEATURES FOR ALL INSTANCES
            # if instance_label (not) in [1, 2, 3, 4, 5]:

            features = self.extractor.execute(image, segmentation, label=instance_label)
            features['class_label'] = class_label
            features['patient_id'] = patient_id
            all_features.append(features)

        return all_features
    

    def parallell_extraction(self, list_of_dicts: list, n_processes = None):
        logger.info("Extraction mode: parallel")
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1
        start_time = time.time()
        with Pool(n_processes) as pool:
            results = list(tqdm(pool.imap(self.extract_radiomics, list_of_dicts),
                                 total=len(list_of_dicts)))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")

        return results
    

    def serial_extraction(self, list_of_dicts: list):
        logger.info("Extraction mode: serial")
        all_results = []
            # for item in trange(len(train_df)):
        start_time = time.time()
        for item in trange(len(list_of_dicts)):
            all_results.append(self.extract_radiomics(list_of_dicts[item]))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")
        return all_results


    def _convert_time(self, start_time, end_time):
        '''
        Converts time in seconds to hours, minutes and seconds.
        '''
        dt = end_time - start_time
        h, m, s = int(dt // 3600), int((dt % 3600 ) // 60), int(dt % 60)
        return h, m, s