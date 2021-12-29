import os

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.brain as fob 


# Dataset Zoo
## list of str / None for download all
DATASET_ZOO_NAME   = 'open-images-v6'
DATASET_MOD_NAME   = DATASET_ZOO_NAME + '_mod'
## list of str / None for download all
DATASET_ZOO_SPLITS = ["train", 'test', 'validation']
## list of str / None for default dir
DATASET_ZOO_DIR    = os.path.expanduser('~/fiftyone')
# Filter configs
CLASSES_OF_INTEREST = [
    "Motorcycle", "Car", "Truck",
    "Bus", "Taxi", "Van", "Land vehicle",
    # "Tank", "Train", "Vehicle", 
]

LABEL_TYPES = [
    "detections", 
    # "classifications", 
    # "segmentations", 
    # "relationships",
]

# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
# LABEL_FIELD = "ground_truth"
LABEL_FIELD = "detections"

## Aug
DATASET_AUG_NAME   = DATASET_ZOO_NAME + "_aug"
DATASET_AUG_SPLITS = DATASET_ZOO_SPLITS + [i+"_aug" for i in DATASET_ZOO_SPLITS]

EXPORT_DIR = "export"

################################################################################
################################################################################


# Model Zoo
ZOO_MODEL_NAME     = ""
DETECTION_FIELD    = ""


################################################################################
################################################################################

LIGHTNESS_THRESHOLD = 10
CONTRAST_THRESHOLD  = 10

################################################################################
################################################################################


# The directory to which to write the exported dataset
## Filter_data.ipynb


# dataset_type = fo.types.COCODetectionDataset  # COCO
EXPORT_DATASET_TYPE = fo.types.dataset_types.YOLOv5Dataset # YOLOv5


