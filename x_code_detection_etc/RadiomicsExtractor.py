import time
import logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import multiprocessing
from pyinstrument import Profiler
# import SimpleITK as sitk
from multiprocessing import Pool
from tqdm import trange
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
        profiler1 = Profiler()
        profiler1.start()

        image = d['image']
        segmentation = d['segmentation']
        class_label = d['class_label']
        instance_label = d['instance_label']
        patient_id = d['patient_id']


        image = np.asarray(nib.load(image).dataobj)
        image = np.squeeze(image) if len(image.shape) == 4 else image
        image = sitk.GetImageFromArray(image)
        segmentation = np.asarray(nib.load(segmentation).dataobj)
        segmentation = np.squeeze(segmentation) if len(segmentation.shape) == 4 else segmentation
        segmentation = sitk.GetImageFromArray(segmentation)

        logger.info(f"Extracting radiomics features for class={class_label} & instance={instance_label} of id={patient_id}")
        
        features = self.extractor.execute(image, segmentation, label=instance_label)
        profiler1.stop()
        features['class_label'] = class_label
        features['instance_label'] = instance_label
        features['patient_id'] = patient_id
        
        # logger.info(f"Profiler OUTPUT{profiler1.output_text(unicode=True, color=True)}")
        logger.info(f"Extraction FINISHED for class={class_label} & instance={instance_label} of id={patient_id}")
        return features
    

    def parallel_extraction(self, list_of_dicts: list, n_processes = None):
        logger.info("Extraction mode: parallel")
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1
        start_time = time.time()
        with Pool(n_processes) as pool:
            results = list(pool.map(self.extract_radiomics, list_of_dicts))
            # results = list(tqdm(pool.imap(self.extract_radiomics, list_of_dicts),
            #                      total=len(list_of_dicts)))
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